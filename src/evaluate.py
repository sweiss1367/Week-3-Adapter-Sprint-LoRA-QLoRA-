"""
Evaluates base / QLoRA / QLoRA+DPO on the held-out test complaints. Every metric is
computed from generations produced in this run -- nothing here is a pre-filled number.

Two modes:

1. Single-model eval (original mode): scores one model variant against the full held-out
   test set. Usage:
       python src/evaluate.py --tag base   --model Qwen/Qwen2.5-0.5B-Instruct
       python src/evaluate.py --tag qlora  --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qlora_adapter
       python src/evaluate.py --tag dpo    --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/dpo_adapter

2. Controlled 10-example comparison (--comparison): the specific behavioral comparison
   this assignment asks for -- base vs. QLoRA vs. DPO on the SAME deterministic 10
   held-out complaints (5 gold-null-address + 5 gold-addressed controls), loaded and run
   sequentially so GPU memory never holds more than one model at a time. Usage:
       python src/evaluate.py --comparison

Metrics (single-model mode):
    - json_valid_rate: fraction of generations that parse as JSON
    - field_complete_rate: fraction with all required schema keys present
    - category_accuracy: fraction with category matching ground truth
    - address_null_precision / address_null_recall: specifically scores the missing-address
      behavior -- of generations that output null, how many should have (precision), and of
      gold examples that should be null, how many the model actually output null for (recall)

Metrics (comparison mode), see compute_comparison_metrics() for exact definitions:
    - json_valid_rate, field_complete_rate, category_accuracy, address_exact_match_accuracy
      (all over the fixed n=10)
    - missing_address_accuracy, hallucinated_address_count (over the fixed 5 null-gold examples)
    - addressed_control_accuracy (over the fixed 5 addressed-gold examples)
"""
import argparse
import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate_instruction_data import build_chat_prompt
from resource_monitor import check_hardware, resolve_precision

REQUIRED_KEYS = {"category", "department", "priority", "address_or_null", "description", "requested_action"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_model_for_eval(model_name: str, adapter_path: str = None, qlora_config_path: str = "configs/qlora_config.json"):
    with open(qlora_config_path) as f:
        qlora_cfg = json.load(f)
    q = qlora_cfg["quantization"]
    compute_dtype = getattr(torch, q["bnb_4bit_compute_dtype"])  # float16 on the T4 this project targets

    check_hardware(resolve_precision(qlora_cfg["training"]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}, dtype=compute_dtype,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_ticket(model, tokenizer, complaint: str, max_new_tokens: int = 120) -> str:
    prompt = build_chat_prompt(tokenizer, complaint)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def evaluate_model(model, tokenizer, examples, tag: str, out_dir: str):
    rows = []
    for ex in examples:
        raw = generate_ticket(model, tokenizer, ex["complaint"])
        rows.append({"complaint": ex["complaint"], "gold": ex["ticket"], "raw_output": raw})

    n = len(rows)
    valid_json = complete_fields = category_correct = 0
    null_predicted_correctly = null_predicted_total = null_gold_total = 0

    for r in rows:
        try:
            parsed = json.loads(r["raw_output"])
        except Exception:
            r["parsed"] = None
            continue
        r["parsed"] = parsed
        valid_json += 1
        if REQUIRED_KEYS.issubset(parsed.keys()):
            complete_fields += 1
        if parsed.get("category") == r["gold"]["category"]:
            category_correct += 1

        gold_is_null = r["gold"]["address_or_null"] is None
        pred_is_null = parsed.get("address_or_null", "MISSING_KEY") is None
        if gold_is_null:
            null_gold_total += 1
        if pred_is_null:
            null_predicted_total += 1
        if gold_is_null and pred_is_null:
            null_predicted_correctly += 1

    metrics = {
        "tag": tag,
        "n": n,
        "json_valid_rate": round(valid_json / n, 3) if n else None,
        "field_complete_rate": round(complete_fields / n, 3) if n else None,
        "category_accuracy": round(category_correct / n, 3) if n else None,
        "address_null_recall": round(null_predicted_correctly / null_gold_total, 3) if null_gold_total else None,
        "address_null_precision": round(null_predicted_correctly / null_predicted_total, 3) if null_predicted_total else None,
        "n_gold_null_address": null_gold_total,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def select_comparison_examples(examples, n_null: int = 5, n_addressed: int = 5):
    """
    Deterministic, decided-before-inference selection -- no cherry-picking after seeing
    any model's output:

        - the first `n_null` examples in test-file order whose gold address_or_null is null
        - the first `n_addressed` examples in test-file order whose gold address_or_null is not null

    Original file order is preserved within each subgroup. Returns a list of
    (original_test_index, example) tuples, null subgroup first.
    """
    null_selected, addressed_selected = [], []
    for idx, ex in enumerate(examples):
        is_null = ex["ticket"]["address_or_null"] is None
        if is_null and len(null_selected) < n_null:
            null_selected.append((idx, ex))
        elif not is_null and len(addressed_selected) < n_addressed:
            addressed_selected.append((idx, ex))
        if len(null_selected) >= n_null and len(addressed_selected) >= n_addressed:
            break

    if len(null_selected) < n_null or len(addressed_selected) < n_addressed:
        raise SystemExit(
            f"Could not find {n_null} null-address and {n_addressed} addressed examples in "
            f"the test set (found {len(null_selected)} null, {len(addressed_selected)} addressed)."
        )
    return null_selected + addressed_selected


def compute_comparison_metrics(records, tag: str):
    """
    All rates are computed over the fixed n=10 unless noted:
        - json_valid_rate, field_complete_rate, category_accuracy, address_exact_match_accuracy: over all 10
        - missing_address_accuracy: over the 5 gold-null examples; success = predicted
          address_or_null is JSON null
        - hallucinated_address_count: a raw count (not a rate) over the same 5 gold-null
          examples -- predicted address_or_null is a non-null value (an invented address),
          counted only when the output actually parsed as JSON (a parse failure is a
          formatting failure, not a hallucination)
        - addressed_control_accuracy: over the 5 gold-addressed examples; success =
          predicted address_or_null exactly equals the gold address
    """
    n = len(records)
    null_records = [r for r in records if r["gold"]["address_or_null"] is None]
    addressed_records = [r for r in records if r["gold"]["address_or_null"] is not None]

    valid_json = complete_fields = category_correct = address_exact_match = 0
    for r in records:
        parsed = r[f"{tag}_parsed"]
        if parsed is None:
            continue
        valid_json += 1
        if REQUIRED_KEYS.issubset(parsed.keys()):
            complete_fields += 1
        if parsed.get("category") == r["gold"]["category"]:
            category_correct += 1
        if parsed.get("address_or_null", "MISSING_KEY") == r["gold"]["address_or_null"]:
            address_exact_match += 1

    null_correct = hallucinated = 0
    for r in null_records:
        parsed = r[f"{tag}_parsed"]
        if parsed is None:
            continue
        pred_addr = parsed.get("address_or_null", "MISSING_KEY")
        if pred_addr is None:
            null_correct += 1
        elif pred_addr != "MISSING_KEY":
            hallucinated += 1

    addressed_correct = 0
    for r in addressed_records:
        parsed = r[f"{tag}_parsed"]
        if parsed is None:
            continue
        if parsed.get("address_or_null", "MISSING_KEY") == r["gold"]["address_or_null"]:
            addressed_correct += 1

    return {
        "tag": tag,
        "n": n,
        "json_valid_rate": round(valid_json / n, 3) if n else None,
        "field_complete_rate": round(complete_fields / n, 3) if n else None,
        "category_accuracy": round(category_correct / n, 3) if n else None,
        "address_exact_match_accuracy": round(address_exact_match / n, 3) if n else None,
        "n_null_address_examples": len(null_records),
        "missing_address_accuracy": round(null_correct / len(null_records), 3) if null_records else None,
        "hallucinated_address_count": hallucinated,
        "n_addressed_control_examples": len(addressed_records),
        "addressed_control_accuracy": round(addressed_correct / len(addressed_records), 3) if addressed_records else None,
    }


def run_comparison(args):
    test_examples = load_jsonl(args.test_data)
    selected = select_comparison_examples(test_examples)

    print(f"Selected {len(selected)} comparison examples from {args.test_data} "
          "(deterministic, decided before any inference):")
    for idx, ex in selected:
        gold_is_null = ex["ticket"]["address_or_null"] is None
        print(f"  test_index={idx} gold_address_is_null={gold_is_null} complaint={ex['complaint']!r}")

    n_null = sum(1 for _, ex in selected if ex["ticket"]["address_or_null"] is None)
    n_addressed = len(selected) - n_null
    print(f"n_null={n_null} n_addressed={n_addressed}")
    if n_null != 5 or n_addressed != 5:
        raise SystemExit(f"Expected exactly 5 null and 5 addressed examples, got {n_null} null / {n_addressed} addressed.")

    with open(args.qlora_config) as f:
        qlora_cfg = json.load(f)

    variants = [
        ("base", None),
        ("qlora", args.qlora_adapter),
        ("dpo", args.dpo_adapter),
    ]

    generations = {}
    for tag, adapter_path in variants:
        print(f"\nLoading variant '{tag}' (adapter={adapter_path})...")
        model, tokenizer = load_model_for_eval(args.model, adapter_path, args.qlora_config)
        outputs = {}
        for idx, ex in selected:
            outputs[idx] = generate_ticket(model, tokenizer, ex["complaint"], max_new_tokens=args.max_new_tokens)
        generations[tag] = outputs

        # Delete the actual references held here, not a helper's local parameter --
        # gc.collect()/empty_cache() can't release the model while this scope still
        # holds a strong reference to it.
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
            reserved_gb = round(torch.cuda.memory_reserved() / 1e9, 3)
            print(f"Unloaded '{tag}'. Post-unload GPU memory: allocated={allocated_gb}GB reserved={reserved_gb}GB")
        else:
            print(f"Unloaded '{tag}'.")

    records = []
    for idx, ex in selected:
        record = {"test_index": idx, "complaint": ex["complaint"], "gold": ex["ticket"]}
        for tag, _ in variants:
            raw = generations[tag][idx]
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            record[f"{tag}_raw_output"] = raw
            record[f"{tag}_parsed"] = parsed
        records.append(record)

    metrics = {tag: compute_comparison_metrics(records, tag) for tag, _ in variants}

    artifact = {
        "selection_rule": {
            "description": (
                "Deterministic, decided before any inference: the first 5 examples in "
                "data/sft_test.jsonl file order whose gold address_or_null is null, then "
                "the first 5 examples in file order whose gold address_or_null is not "
                "null. Original file order is preserved within each subgroup."
            ),
            "test_data_file": args.test_data,
            "n_null_selected": n_null,
            "n_addressed_selected": n_addressed,
            "selected_test_indices": [idx for idx, _ in selected],
            "selected_test_indices_null": [idx for idx, ex in selected if ex["ticket"]["address_or_null"] is None],
            "selected_test_indices_addressed": [idx for idx, ex in selected if ex["ticket"]["address_or_null"] is not None],
        },
        "generation_settings": {
            "model_name": args.model,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "same_prompt_construction_for_all_variants": True,
            "qlora_adapter_path": args.qlora_adapter,
            "dpo_adapter_path": args.dpo_adapter,
            "quantization": qlora_cfg["quantization"],
        },
        "records": records,
        "metrics": metrics,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "comparison_10.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved combined comparison artifact to {out_path}")

    print("\nSide-by-side metric summary:")
    metric_keys = [
        "json_valid_rate", "field_complete_rate", "category_accuracy",
        "address_exact_match_accuracy", "missing_address_accuracy",
        "hallucinated_address_count", "addressed_control_accuracy",
    ]
    tags = [tag for tag, _ in variants]
    print(f"{'metric':32}" + "".join(f"{tag:>12}" for tag in tags))
    for key in metric_keys:
        print(f"{key:32}" + "".join(f"{str(metrics[tag][key]):>12}" for tag in tags))

    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="store_true",
                         help="run the deterministic 10-example base/qlora/dpo comparison instead of a single-tag eval")
    parser.add_argument("--tag", default=None, help="base | qlora | dpo (single-tag mode only)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="single-tag mode: adapter dir, or omit for the raw base model")
    parser.add_argument("--qlora-adapter", default="outputs/qlora_adapter", help="comparison mode: QLoRA adapter dir")
    parser.add_argument("--dpo-adapter", default="outputs/dpo_adapter", help="comparison mode: DPO adapter dir")
    parser.add_argument("--qlora-config", default="configs/qlora_config.json")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    if args.comparison:
        run_comparison(args)
        return

    if not args.tag:
        parser.error("--tag is required unless --comparison is given")
    examples = load_jsonl(args.test_data)
    model, tokenizer = load_model_for_eval(args.model, args.adapter, args.qlora_config)
    evaluate_model(model, tokenizer, examples, args.tag, args.out_dir)


if __name__ == "__main__":
    main()
