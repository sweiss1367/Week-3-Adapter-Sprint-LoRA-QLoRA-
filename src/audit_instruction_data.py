"""
Token-length audit -- run this BEFORE choosing max_seq_len. Do not hardcode a sequence
length assumption; measure the actual generated data with the real tokenizer and pick a
value from the observed distribution.

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
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "p90": round(percentile(lengths, 90), 1),
        "p95": round(percentile(lengths, 95), 1),
        "p99": round(percentile(lengths, 99), 1),
    }


def recommend_seq_len(combined_p99: float) -> int:
    """Round the measured p99 combined length up to the next multiple of 32, with a small
    safety margin, rather than guessing a round number ahead of time."""
    padded = combined_p99 * 1.05
    return int(-(-padded // 32) * 32)  # ceil to multiple of 32


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

    report = {
        "model": args.model,
        "data_file": args.data,
        "n_examples": len(examples),
        "prompt_tokens": summarize("prompt_tokens", prompt_lens),
        "target_tokens": summarize("target_tokens", target_lens),
        "combined_tokens": summarize("combined_tokens", combined_lens),
    }
    recommended = recommend_seq_len(report["combined_tokens"]["p99"])
    report["recommended_max_seq_len"] = recommended
    report["recommendation_rationale"] = (
        f"p99 combined (prompt+target) length is {report['combined_tokens']['p99']} tokens; "
        f"recommended value adds a 5% margin and rounds up to the nearest multiple of 32 "
        f"({recommended}). This covers the observed distribution without padding every "
        "batch to a size the data never needs. Human approval required before use in "
        "configs/qlora_config.json."
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
