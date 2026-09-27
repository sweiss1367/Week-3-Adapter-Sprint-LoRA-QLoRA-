"""
Structural audit for the controlled missing-address DPO dataset (data/dpo_pairs.jsonl,
built by generate_address_dpo_data.py). Verifies the design constraint directly -- chosen
and rejected must differ in exactly one field, address_or_null -- rather than assuming the
generator got it right, plus duplication/leakage checks against every SFT split.

Character-length statistics (chosen/rejected response only) are always computed. Real
token-length statistics -- including the full DPO prompt and prompt+response sequences
that DPOTrainer will actually see -- additionally require the Hugging Face Hub (to fetch
the Qwen tokenizer); if that isn't reachable, all token-length fields are left null with
an explicit note, never substituted with character counts or estimates.

The prompt is built with generate_instruction_data.build_chat_prompt(), the exact same
function train_dpo.py calls (`dataset.map(lambda row: {"prompt": build_chat_prompt(tokenizer,
row["prompt"])})`), so prompt_tokens/prompt_plus_chosen_tokens/prompt_plus_rejected_tokens
reflect what DPOTrainer actually receives, not an approximation of it.

Usage:
    python src/audit_address_dpo_data.py --pairs data/dpo_pairs.jsonl \
        --train-data data/sft_train.jsonl --val-data data/sft_val.jsonl --test-data data/sft_test.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import os
import statistics

from generate_instruction_data import build_chat_prompt

CANDIDATE_LENGTHS = [128, 192, 256, 384, 512]


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


def summarize_full(lengths, unit: str):
    """min/median/p90/p95/p99/max -- used for prompt/combined sequence lengths, where a
    p99 is meaningful given the dataset size."""
    return {
        "unit": unit,
        "n": len(lengths),
        "min": min(lengths),
        "median": round(statistics.median(lengths), 1),
        "p90": round(percentile(lengths, 90), 1),
        "p95": round(percentile(lengths, 95), 1),
        "p99": round(percentile(lengths, 99), 1),
        "max": max(lengths),
    }


def truncation_report(lengths, candidates=CANDIDATE_LENGTHS):
    n = len(lengths)
    return [
        {
            "candidate_len": c,
            "n_exceeding": sum(1 for l in lengths if l > c),
            "pct_exceeding": round(100 * sum(1 for l in lengths if l > c) / n, 2) if n else None,
        }
        for c in candidates
    ]


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

        # Same construction train_dpo.py uses: build_chat_prompt(tokenizer, row["prompt"]).
        prompt_texts = [build_chat_prompt(tokenizer, p["prompt"]) for p in pairs]
        prompt_tok_lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in prompt_texts]
        combined_chosen_lens = [pt + ct for pt, ct in zip(prompt_tok_lens, chosen_tok_lens)]
        combined_rejected_lens = [pt + rt for pt, rt in zip(prompt_tok_lens, rejected_tok_lens)]

        report["prompt_tokens"] = summarize_full(prompt_tok_lens, unit="tokens")
        report["prompt_plus_chosen_tokens"] = summarize_full(combined_chosen_lens, unit="tokens")
        report["prompt_plus_rejected_tokens"] = summarize_full(combined_rejected_lens, unit="tokens")

        report["max_prompt_length_truncation"] = {
            "basis": "prompt_tokens",
            "candidates": truncation_report(prompt_tok_lens),
        }
        report["max_length_truncation"] = {
            "basis_prompt_plus_chosen": truncation_report(combined_chosen_lens),
            "basis_prompt_plus_rejected": truncation_report(combined_rejected_lens),
        }
    else:
        report["chosen_length_tokens"] = None
        report["rejected_length_tokens"] = None
        report["prompt_tokens"] = None
        report["prompt_plus_chosen_tokens"] = None
        report["prompt_plus_rejected_tokens"] = None
        report["max_prompt_length_truncation"] = None
        report["max_length_truncation"] = None
        report["token_length_note"] = (
            "Real Qwen tokenizer unavailable in this run -- character-length statistics above "
            "are not a substitute for token counts, and no estimate was substituted for the "
            "prompt/prompt+response token fields. Rerun in Colab for real token-length numbers."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
