"""
Structural + behavioral audit for the balanced DPO v2 dataset (data/dpo_pairs_v2.jsonl,
built by generate_address_dpo_data_v2.py). Extends the v1 audit's structural checks
(chosen/rejected differ in exactly address_or_null; no leakage/duplication) with the two
checks that matter specifically for the balanced design: every missing-address prompt
genuinely lacks a street, and every addressed-control prompt genuinely contains its own
gold street -- otherwise the "addressed control" pairs wouldn't actually teach the
conditional rule this dataset exists to teach.

Character-length statistics are always computed. Real token-length statistics --
including a check that every prompt+chosen / prompt+rejected sequence stays within the
approved max_length=256 -- additionally require the Hugging Face Hub (to fetch the Qwen
tokenizer); if that isn't reachable, those fields are left null with an explicit note,
never substituted with character counts or estimates.

Usage:
    python src/audit_address_dpo_data_v2.py --pairs data/dpo_pairs_v2.jsonl \
        --train-data data/sft_train.jsonl --val-data data/sft_val.jsonl --test-data data/sft_test.jsonl \
        --v1-data data/dpo_pairs.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import os
import statistics

from generate_instruction_data import STREETS, build_chat_prompt


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def length_summary(lengths, unit: str):
    return {
        "unit": unit,
        "n": len(lengths),
        "min": min(lengths),
        "median": round(statistics.median(lengths), 1),
        "p90": round(percentile(lengths, 90), 1),
        "p95": round(percentile(lengths, 95), 1),
        "max": max(lengths),
    }


def try_load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"NOTE: could not load tokenizer '{model_name}' ({e}). "
              "Reporting character-length statistics only -- run this in Colab for real token counts.")
        return None


def check_single_field_diff(chosen_str: str, rejected_str: str):
    """Returns (same_keys_and_order, non_address_fields_identical), parsed order-preserving
    so key ORDER is actually checked, not just key membership. Direction-agnostic -- works
    the same whether the pair's address_or_null goes null->street or street->null."""
    chosen_pairs = json.loads(chosen_str, object_pairs_hook=list)
    rejected_pairs = json.loads(rejected_str, object_pairs_hook=list)

    chosen_keys = [k for k, _ in chosen_pairs]
    rejected_keys = [k for k, _ in rejected_pairs]
    same_keys_and_order = chosen_keys == rejected_keys

    chosen_dict = dict(chosen_pairs)
    rejected_dict = dict(rejected_pairs)
    non_address_identical = all(
        chosen_dict[k] == rejected_dict.get(k)
        for k in chosen_dict
        if k != "address_or_null"
    )
    return same_keys_and_order, non_address_identical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo_pairs_v2.jsonl")
    parser.add_argument("--train-data", default="data/sft_train.jsonl")
    parser.add_argument("--val-data", default="data/sft_val.jsonl")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--v1-data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="data/dpo_pairs_v2_audit.json")
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    train_complaints = {e["complaint"] for e in load_jsonl(args.train_data)}
    val_complaints = {e["complaint"] for e in load_jsonl(args.val_data)}
    test_complaints = {e["complaint"] for e in load_jsonl(args.test_data)}
    v1_complaints = {p["prompt"] for p in load_jsonl(args.v1_data)} if os.path.exists(args.v1_data) else set()

    n = len(pairs)
    prompts = [p["prompt"] for p in pairs]
    n_duplicate_prompts = len(prompts) - len(set(prompts))

    overlap_train = sum(1 for p in prompts if p in train_complaints)
    overlap_val = sum(1 for p in prompts if p in val_complaints)
    overlap_test = sum(1 for p in prompts if p in test_complaints)
    overlap_v1 = sum(1 for p in prompts if p in v1_complaints)

    n_missing = sum(1 for p in pairs if p["pair_type"] == "missing_address")
    n_addressed = sum(1 for p in pairs if p["pair_type"] == "addressed_control")

    key_order_violations = non_address_violations = 0
    missing_address_street_leaks = []
    addressed_missing_gold_street = []

    for p in pairs:
        same_keys, non_address_identical = check_single_field_diff(p["chosen"], p["rejected"])
        if not same_keys:
            key_order_violations += 1
        if not non_address_identical:
            non_address_violations += 1

        complaint_lower = p["prompt"].lower()
        if p["pair_type"] == "missing_address":
            for s in STREETS:
                if s.lower() in complaint_lower:
                    missing_address_street_leaks.append({"prompt": p["prompt"], "leaked_street": s})
                    break
        elif p["pair_type"] == "addressed_control":
            gold_addr = p.get("address_or_null_in_gold")
            if not gold_addr or gold_addr.lower() not in complaint_lower:
                addressed_missing_gold_street.append({"prompt": p["prompt"], "gold_address": gold_addr})

    chosen_char_lens = [len(p["chosen"]) for p in pairs]
    rejected_char_lens = [len(p["rejected"]) for p in pairs]
    pct_diffs = [100 * abs(c - r) / max(c, r, 1) for c, r in zip(chosen_char_lens, rejected_char_lens)]
    n_over_15 = sum(1 for d in pct_diffs if d > 15)

    report = {
        "n_pairs": n,
        "n_missing_address_pairs": n_missing,
        "n_addressed_control_pairs": n_addressed,
        "n_duplicate_prompts_within_dataset": n_duplicate_prompts,
        "overlap_with_sft_train": overlap_train,
        "overlap_with_sft_val": overlap_val,
        "overlap_with_sft_test": overlap_test,
        "overlap_with_dpo_v1": overlap_v1,
        "n_key_order_violations": key_order_violations,
        "all_pairs_same_keys_and_order": key_order_violations == 0,
        "n_non_address_field_violations": non_address_violations,
        "all_pairs_identical_except_address": non_address_violations == 0,
        "n_missing_address_prompts_with_street_leak": len(missing_address_street_leaks),
        "missing_address_street_leaks": missing_address_street_leaks,
        "n_addressed_prompts_missing_gold_street": len(addressed_missing_gold_street),
        "addressed_prompts_missing_gold_street": addressed_missing_gold_street,
        "chosen_length_chars": length_summary(chosen_char_lens, unit="characters"),
        "rejected_length_chars": length_summary(rejected_char_lens, unit="characters"),
        "pct_pairs_differing_by_more_than_15pct_chars": round(100 * n_over_15 / n, 2) if n else None,
    }

    tokenizer = try_load_tokenizer(args.model)
    if tokenizer is not None:
        prompt_texts = [build_chat_prompt(tokenizer, p["prompt"]) for p in pairs]
        prompt_tok_lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in prompt_texts]
        chosen_tok_lens = [len(tokenizer(p["chosen"], add_special_tokens=False)["input_ids"]) for p in pairs]
        rejected_tok_lens = [len(tokenizer(p["rejected"], add_special_tokens=False)["input_ids"]) for p in pairs]
        combined_chosen = [pt + ct for pt, ct in zip(prompt_tok_lens, chosen_tok_lens)]
        combined_rejected = [pt + rt for pt, rt in zip(prompt_tok_lens, rejected_tok_lens)]

        report["prompt_tokens"] = length_summary(prompt_tok_lens, unit="tokens")
        report["prompt_plus_chosen_tokens"] = length_summary(combined_chosen, unit="tokens")
        report["prompt_plus_rejected_tokens"] = length_summary(combined_rejected, unit="tokens")

        n_exceeding_chosen = sum(1 for l in combined_chosen if l > args.max_length)
        n_exceeding_rejected = sum(1 for l in combined_rejected if l > args.max_length)
        report["max_length_check"] = {
            "max_length": args.max_length,
            "n_prompt_plus_chosen_exceeding": n_exceeding_chosen,
            "n_prompt_plus_rejected_exceeding": n_exceeding_rejected,
            "all_within_max_length": n_exceeding_chosen == 0 and n_exceeding_rejected == 0,
        }
    else:
        report["prompt_tokens"] = None
        report["prompt_plus_chosen_tokens"] = None
        report["prompt_plus_rejected_tokens"] = None
        report["max_length_check"] = None
        report["token_length_note"] = (
            "Real Qwen tokenizer unavailable in this run -- character-length statistics above "
            "are not a substitute for token counts, and the max_length=256 check could not be "
            "run here. Rerun in Colab for real token-length numbers and the real max_length check."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
