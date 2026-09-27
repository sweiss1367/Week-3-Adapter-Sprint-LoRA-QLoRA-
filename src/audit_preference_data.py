"""
Audits the DPO preference dataset (data/dpo_pairs.jsonl) before training: composition,
length statistics, and specifically whether the dataset contains detectable shortcuts
(e.g. "rejected is always shorter") that DPO could learn instead of the intended
behavior, or any leakage from the held-out test set.

Character-length statistics are always computed (no dependency needed). Real token-length
statistics additionally require the Hugging Face Hub (to fetch the Qwen tokenizer) -- if
that isn't reachable, the character-length numbers are reported on their own and clearly
labeled as a proxy, never presented as token counts.

Usage:
    python src/audit_preference_data.py --pairs data/dpo_pairs.jsonl --held-out data/sft_test.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct
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


def length_delta_report(pairs, chosen_lens, rejected_lens, unit: str, threshold_pct: float = 15.0):
    deltas = [c - r for c, r in zip(chosen_lens, rejected_lens)]
    pct_diffs = []
    n_over_threshold = 0
    for c, r in zip(chosen_lens, rejected_lens):
        base = max(c, r, 1)
        pct_diff = 100 * abs(c - r) / base
        pct_diffs.append(pct_diff)
        if pct_diff > threshold_pct:
            n_over_threshold += 1

    # Per-corruption-type breakdown: does one corruption type make `rejected` predictably
    # shorter/longer than `chosen`? That's a length-based shortcut DPO could exploit
    # instead of learning the intended content difference.
    by_type = {}
    for pair, c, r in zip(pairs, chosen_lens, rejected_lens):
        t = pair["corruption_type"]
        by_type.setdefault(t, []).append(c - r)

    by_type_summary = {
        t: {
            "n": len(deltas_for_type),
            "mean_chosen_minus_rejected": round(statistics.mean(deltas_for_type), 2),
            "all_same_sign": len(set(d > 0 for d in deltas_for_type)) == 1 if deltas_for_type else None,
        }
        for t, deltas_for_type in by_type.items()
    }

    return {
        "unit": unit,
        "mean_chosen_minus_rejected": round(statistics.mean(deltas), 2),
        "pct_pairs_differing_by_more_than_15pct": round(100 * n_over_threshold / len(pairs), 2),
        "by_corruption_type": by_type_summary,
    }


def check_address_leak(pairs):
    """For hallucinate_address pairs specifically: does the invented address in `rejected`
    accidentally already appear somewhere in the source complaint? It shouldn't, since
    no-address complaints are generated to never mention a street -- verify rather than
    assume."""
    leaks = []
    for pair in pairs:
        if pair["corruption_type"] != "hallucinate_address":
            continue
        rejected_ticket = json.loads(pair["rejected"])
        invented_address = rejected_ticket.get("address_or_null")
        if invented_address and invented_address.lower() in pair["prompt"].lower():
            leaks.append({"prompt": pair["prompt"], "invented_address": invented_address})
    return leaks


def check_held_out_overlap(pairs, held_out_complaints):
    return [p["prompt"] for p in pairs if p["prompt"] in held_out_complaints]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo_pairs.jsonl")
    parser.add_argument("--held-out", default="data/sft_test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="data/dpo_pairs_audit.json")
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    held_out_complaints = {ex["complaint"] for ex in load_jsonl(args.held_out)}

    n = len(pairs)
    corruption_counts = {}
    for p in pairs:
        corruption_counts[p["corruption_type"]] = corruption_counts.get(p["corruption_type"], 0) + 1
    corruption_pct = {k: round(100 * v / n, 2) for k, v in corruption_counts.items()}

    n_hallucinate = corruption_counts.get("hallucinate_address", 0)

    chosen_char_lens = [len(p["chosen"]) for p in pairs]
    rejected_char_lens = [len(p["rejected"]) for p in pairs]

    report = {
        "n_pairs": n,
        "corruption_type_counts": corruption_counts,
        "corruption_type_pct": corruption_pct,
        "n_hallucinate_address_pairs": n_hallucinate,
        "pct_hallucinate_address_pairs": round(100 * n_hallucinate / n, 2),
        "chosen_length_chars": length_summary(chosen_char_lens, unit="characters"),
        "rejected_length_chars": length_summary(rejected_char_lens, unit="characters"),
        "length_delta_chars": length_delta_report(pairs, chosen_char_lens, rejected_char_lens, unit="characters"),
    }

    tokenizer = try_load_tokenizer(args.model)
    if tokenizer is not None:
        chosen_tok_lens = [len(tokenizer(p["chosen"], add_special_tokens=False)["input_ids"]) for p in pairs]
        rejected_tok_lens = [len(tokenizer(p["rejected"], add_special_tokens=False)["input_ids"]) for p in pairs]
        report["chosen_length_tokens"] = length_summary(chosen_tok_lens, unit="tokens")
        report["rejected_length_tokens"] = length_summary(rejected_tok_lens, unit="tokens")
        report["length_delta_tokens"] = length_delta_report(pairs, chosen_tok_lens, rejected_tok_lens, unit="tokens")
    else:
        report["chosen_length_tokens"] = None
        report["rejected_length_tokens"] = None
        report["length_delta_tokens"] = None
        report["token_length_note"] = (
            "Real Qwen tokenizer unavailable in this run -- only character-length statistics "
            "above are real measurements. Do not treat them as token counts; rerun in Colab "
            "for real token-length numbers."
        )

    address_leaks = check_address_leak(pairs)
    report["address_leak_check"] = {
        "n_hallucinate_address_pairs_checked": n_hallucinate,
        "n_leaks_found": len(address_leaks),
        "leaks": address_leaks,
    }

    held_out_overlap = check_held_out_overlap(pairs, held_out_complaints)
    report["held_out_overlap_check"] = {
        "n_test_complaints": len(held_out_complaints),
        "n_dpo_prompts_found_in_test_set": len(held_out_overlap),
        "overlapping_prompts": held_out_overlap,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
