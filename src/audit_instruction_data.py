"""
Token-length audit -- run this BEFORE choosing max_seq_len. Do not hardcode a sequence
length assumption; measure the actual generated data with the real tokenizer and pick a
value from the observed distribution.

Uses the same chat template and prompt/target construction as train_qlora.py's
build_sft_dataset() (both import build_chat_prompt/target_json from
generate_instruction_data.py), so what this script measures is exactly what training will
see -- not an approximation of it.

Requires network access to the Hugging Face Hub to fetch the tokenizer, and the
`transformers` package. If this script cannot reach the Hub (e.g. this dev container's
proxy denies huggingface.co), it must be run in Colab instead -- do not substitute an
estimate for a real measurement.

Usage:
    python src/audit_instruction_data.py --data data/sft_train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import os
import statistics

from generate_instruction_data import build_chat_prompt, target_json

CANDIDATE_MAX_SEQ_LENS = [128, 192, 256, 384, 512]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(name, lengths):
    return {
        "field": name,
        "n": len(lengths),
        "min": min(lengths),
        "median": round(statistics.median(lengths), 1),
        "p90": round(percentile(lengths, 90), 1),
        "p95": round(percentile(lengths, 95), 1),
        "p99": round(percentile(lengths, 99), 1),
        "max": max(lengths),
    }


def candidate_truncation_report(prompt_lens, combined_lens, candidates=CANDIDATE_MAX_SEQ_LENS):
    """
    For each candidate max_seq_len: how many examples would be truncated at all (combined
    length > candidate), and -- specifically -- how many would lose their ENTIRE assistant
    target (prompt alone already reaches the candidate length, so none of the JSON survives).

    Targets are appended after the prompt in build_sft_dataset(), so truncation always cuts
    from the tail first: any example counted in n_truncated_any loses part of its target
    (at minimum, the closing brace and/or the EOS token) even when n_target_fully_lost is 0.
    """
    n = len(combined_lens)
    rows = []
    for c in candidates:
        n_truncated_any = sum(1 for l in combined_lens if l > c)
        n_target_fully_lost = sum(1 for p in prompt_lens if p >= c)
        rows.append({
            "max_seq_len": c,
            "n_truncated_any": n_truncated_any,
            "pct_truncated_any": round(100 * n_truncated_any / n, 2) if n else None,
            "n_target_fully_lost": n_target_fully_lost,
            "note": (
                "no examples truncated at this length"
                if n_truncated_any == 0
                else "every truncated example loses part of its assistant target (the JSON "
                     "gets cut, possibly before the closing brace or the EOS token) because "
                     "the target is appended after the prompt and truncation cuts from the end"
            ),
        })
    return rows


def recommend_max_seq_len(prompt_lens, combined_lens):
    """
    Recommends the smallest max_seq_len (rounded up to a multiple of 32, with a small
    safety margin) that preserves essentially all training data -- not a conventional
    round number chosen ahead of time.

    Basis is the measured maximum combined length whenever that maximum isn't a large
    outlier relative to p99 (outlier_ratio <= 1.25): in that normal case, rounding up from
    the true max truncates zero examples, so there's no tradeoff to make.

    If the max IS a large outlier (one or two unusually long examples dragging the
    recommendation up for everyone), basis switches to p99 instead, and the resulting
    truncation count/pct at that shorter length is reported explicitly -- accepting a
    small number of truncated outliers in exchange for not padding every batch to
    accommodate them is the actual memory/truncation tradeoff, and it's reported as a
    real count, not asserted.
    """
    n = len(combined_lens)
    max_len = max(combined_lens)
    p99 = percentile(combined_lens, 99)

    outlier_ratio = (max_len / p99) if p99 else None
    is_outlier = outlier_ratio is not None and outlier_ratio > 1.25

    if not is_outlier:
        basis = "max"
        basis_value = max_len
    else:
        basis = "p99"
        basis_value = p99

    recommended = int(-(-(basis_value * 1.05) // 32) * 32)  # +5% margin, ceil to multiple of 32

    n_truncated = sum(1 for l in combined_lens if l > recommended)
    n_target_fully_lost = sum(1 for p in prompt_lens if p >= recommended)

    return {
        "recommended_max_seq_len": recommended,
        "basis": basis,
        "basis_value_tokens": round(basis_value, 1),
        "max_combined_length": max_len,
        "p99_combined_length": p99,
        "outlier_ratio_max_over_p99": round(outlier_ratio, 2) if outlier_ratio else None,
        "n_truncated_at_recommended": n_truncated,
        "pct_truncated_at_recommended": round(100 * n_truncated / n, 2) if n else None,
        "n_target_fully_lost_at_recommended": n_target_fully_lost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sft_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="data/token_length_audit.json")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "transformers is required for this audit (pip install transformers). "
            f"Original error: {e}"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        raise SystemExit(
            f"Could not load tokenizer '{args.model}' from the Hugging Face Hub. "
            "This audit must be run somewhere with Hub access (e.g. Colab) -- "
            f"do not substitute an estimated length. Original error: {e}"
        )

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))

    prompt_lens, target_lens, combined_lens = [], [], []
    for ex in examples:
        prompt_text = build_chat_prompt(tokenizer, ex["complaint"])
        target_text = target_json(ex["ticket"]) + tokenizer.eos_token
        p_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        t_len = len(tokenizer(target_text, add_special_tokens=False)["input_ids"])
        prompt_lens.append(p_len)
        target_lens.append(t_len)
        combined_lens.append(p_len + t_len)

    candidate_report = candidate_truncation_report(prompt_lens, combined_lens)
    recommendation = recommend_max_seq_len(prompt_lens, combined_lens)

    report = {
        "model": args.model,
        "data_file": args.data,
        "n_examples": len(examples),
        "prompt_tokens": summarize("prompt_tokens", prompt_lens),
        "target_tokens": summarize("target_tokens", target_lens),
        "combined_tokens": summarize("combined_tokens", combined_lens),
        "truncation_at_candidate_lengths": candidate_report,
        "recommendation": recommendation,
        "recommendation_rationale": (
            f"basis={recommendation['basis']} ({recommendation['basis_value_tokens']} tokens); "
            f"recommended_max_seq_len={recommendation['recommended_max_seq_len']} adds a 5% "
            f"margin over the basis value and rounds up to the nearest multiple of 32. At this "
            f"length, {recommendation['n_truncated_at_recommended']} of {len(examples)} training "
            f"examples ({recommendation['pct_truncated_at_recommended']}%) would still be "
            f"truncated, and {recommendation['n_target_fully_lost_at_recommended']} would lose "
            f"their entire assistant target. Human approval required before writing this value "
            f"into configs/qlora_config.json or configs/dpo_config.json."
        ),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
