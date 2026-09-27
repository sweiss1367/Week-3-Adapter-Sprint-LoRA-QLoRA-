"""
Structural audit for the controlled missing-address DPO dataset (data/dpo_pairs.jsonl,
built by generate_address_dpo_data.py). Verifies the design constraint directly -- chosen
and rejected must differ in exactly one field, address_or_null -- rather than assuming the
generator got it right, plus duplication/leakage checks against every SFT split.

Character-length statistics are always computed. Real token-length statistics additionally
require the Hugging Face Hub (to fetch the Qwen tokenizer); if that isn't reachable, those
fields are left null with an explicit note -- never substituted with character counts.

Usage:
    python src/audit_address_dpo_data.py --pairs data/dpo_pairs.jsonl \
        --train-data data/sft_train.jsonl --val-data data/sft_val.jsonl --test-data data/sft_test.jsonl
"""
import argparse
import json
import os
import statistics


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
    so key ORDER is actually checked, not just key membership."""
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
    parser.add_argument("--pairs", default="data/dpo_pairs.jsonl")
    parser.add_argument("--train-data", default="data/sft_train.jsonl")
    parser.add_argument("--val-data", default="data/sft_val.jsonl")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="data/dpo_pairs_audit.json")
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    train_complaints = {e["complaint"] for e in load_jsonl(args.train_data)}
    val_complaints = {e["complaint"] for e in load_jsonl(args.val_data)}
    test_complaints = {e["complaint"] for e in load_jsonl(args.test_data)}

    n = len(pairs)
    prompts = [p["prompt"] for p in pairs]
    n_duplicate_prompts = len(prompts) - len(set(prompts))

    overlap_train = sum(1 for p in prompts if p in train_complaints)
    overlap_val = sum(1 for p in prompts if p in val_complaints)
    overlap_test = sum(1 for p in prompts if p in test_complaints)

    address_leaks = 0
    key_order_violations = 0
    non_address_violations = 0
    for p in pairs:
        rejected_ticket = json.loads(p["rejected"])
        invented = rejected_ticket.get("address_or_null")
        if invented and invented.lower() in p["prompt"].lower():
            address_leaks += 1

        same_keys, non_address_identical = check_single_field_diff(p["chosen"], p["rejected"])
        if not same_keys:
            key_order_violations += 1
        if not non_address_identical:
            non_address_violations += 1

    chosen_char_lens = [len(p["chosen"]) for p in pairs]
    rejected_char_lens = [len(p["rejected"]) for p in pairs]
    pct_diffs = [100 * abs(c - r) / max(c, r, 1) for c, r in zip(chosen_char_lens, rejected_char_lens)]
    n_over_15 = sum(1 for d in pct_diffs if d > 15)

    report = {
        "n_pairs": n,
        "n_duplicate_prompts_within_dataset": n_duplicate_prompts,
        "overlap_with_sft_train": overlap_train,
        "overlap_with_sft_val": overlap_val,
        "overlap_with_sft_test": overlap_test,
        "n_invented_address_leaks_into_source_complaint": address_leaks,
        "n_key_order_violations": key_order_violations,
        "all_pairs_same_keys_and_order": key_order_violations == 0,
        "n_non_address_field_violations": non_address_violations,
        "all_pairs_identical_except_address": non_address_violations == 0,
        "chosen_length_chars": length_summary(chosen_char_lens, unit="characters"),
        "rejected_length_chars": length_summary(rejected_char_lens, unit="characters"),
        "pct_pairs_differing_by_more_than_15pct_chars": round(100 * n_over_15 / n, 2) if n else None,
    }

    tokenizer = try_load_tokenizer(args.model)
    if tokenizer is not None:
        chosen_tok_lens = [len(tokenizer(p["chosen"], add_special_tokens=False)["input_ids"]) for p in pairs]
        rejected_tok_lens = [len(tokenizer(p["rejected"], add_special_tokens=False)["input_ids"]) for p in pairs]
        report["chosen_length_tokens"] = length_summary(chosen_tok_lens, unit="tokens")
        report["rejected_length_tokens"] = length_summary(rejected_tok_lens, unit="tokens")
    else:
        report["chosen_length_tokens"] = None
        report["rejected_length_tokens"] = None
        report["token_length_note"] = (
            "Real Qwen tokenizer unavailable in this run -- character-length statistics above "
            "are not a substitute for token counts. Rerun in Colab for real token-length numbers."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
