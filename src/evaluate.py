"""
Runs the held-out test complaints through a given model (base / QLoRA / QLoRA+DPO) and
scores the generations against ground truth. Every metric is computed from generations
produced in this run -- nothing here is a pre-filled number.

Metrics:
    - json_valid_rate: fraction of generations that parse as JSON
    - field_complete_rate: fraction with all required schema keys present
    - category_accuracy: fraction with category matching ground truth
    - address_null_precision / address_null_recall: specifically scores the missing-address
      behavior -- of generations that output null, how many should have (precision), and of
      gold examples that should be null, how many the model actually output null for (recall)

Usage examples (run in Colab):
    python src/evaluate.py --tag base   --model Qwen/Qwen2.5-0.5B-Instruct
    python src/evaluate.py --tag qlora  --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qlora_adapter
    python src/evaluate.py --tag dpo    --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/dpo_adapter
"""
import argparse
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
        model_name, quantization_config=bnb_config, device_map={"": 0}, torch_dtype=compute_dtype,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="base | qlora | dpo (used for output filenames)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter directory, or omit for the raw base model")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    examples = load_jsonl(args.test_data)
    model, tokenizer = load_model_for_eval(args.model, args.adapter)
    evaluate_model(model, tokenizer, examples, args.tag, args.out_dir)


if __name__ == "__main__":
    main()
