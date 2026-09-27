"""
Generates the controlled, single-signal DPO preference dataset for missing-address
hallucination -- the only behavior this DPO experiment is meant to isolate.

This REPLACES the earlier mixed-corruption dataset (malformed JSON, dropped fields,
wrong category, wrong address, plus hallucinate_address), which was rejected on manual
review for introducing confounding preference signals and detectable surface-level
shortcuts (e.g. rejected being systematically shorter/longer by corruption type -- see
README). Every pair here differs from its chosen counterpart in exactly one field:
`address_or_null`. See generate_preference_data.py for the earlier, superseded design.

Every prompt is a freshly generated complaint with no street address, built with
generate_instruction_data.make_example(force_no_address=True) -- the same categories,
tone/priority logic, and ticket schema as the approved instruction dataset, but new
examples, never copies of the SFT train/val/test complaints. Generation aborts if it
cannot produce enough unique complaints, and again (as an independent safety check) if
any duplicate or cross-split overlap is found after generation.

Usage:
    python src/generate_address_dpo_data.py --n 150 --seed 101
"""
import argparse
import json
import os
import random

from generate_instruction_data import STREETS, make_example, target_json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def generate_unique_no_address_examples(n: int, seed: int, excluded_complaints: set, max_attempts_multiplier: int = 50):
    rng = random.Random(seed)
    seen = set(excluded_complaints)
    examples = []
    attempts = 0
    max_attempts = n * max_attempts_multiplier
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        ex = make_example(rng, force_no_address=True)
        if ex["complaint"] in seen:
            continue
        seen.add(ex["complaint"])
        examples.append(ex)
    if len(examples) < n:
        raise SystemExit(
            f"Could only generate {len(examples)}/{n} unique no-address complaints after "
            f"{attempts} attempts. Increase --max-attempts-multiplier or reduce --n."
        )
    return examples


def build_pair(ex: dict, rng: random.Random) -> dict:
    chosen_ticket = dict(ex["ticket"])
    assert chosen_ticket["address_or_null"] is None, "generator invariant violated: expected a no-address example"

    invented_street = rng.choice(STREETS)
    rejected_ticket = dict(chosen_ticket)  # same keys, same order; only this value changes
    rejected_ticket["address_or_null"] = invented_street

    return {
        "prompt": ex["complaint"],
        "chosen": target_json(chosen_ticket),
        "rejected": target_json(rejected_ticket),
        "corruption_type": "hallucinate_address",
        "address_or_null_in_gold": None,
        "invented_address": invented_street,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-attempts-multiplier", type=int, default=50)
    parser.add_argument("--train-data", default="data/sft_train.jsonl")
    parser.add_argument("--val-data", default="data/sft_val.jsonl")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--out", default="data/dpo_pairs.jsonl")
    args = parser.parse_args()

    train_complaints = {e["complaint"] for e in load_jsonl(args.train_data)}
    val_complaints = {e["complaint"] for e in load_jsonl(args.val_data)}
    test_complaints = {e["complaint"] for e in load_jsonl(args.test_data)}
    excluded = train_complaints | val_complaints | test_complaints

    examples = generate_unique_no_address_examples(args.n, args.seed, excluded, args.max_attempts_multiplier)

    pair_rng = random.Random(args.seed + 1)
    pairs = [build_pair(ex, pair_rng) for ex in examples]

    # Independent safety check -- do not rely solely on the generation-time dedup above.
    prompts = [p["prompt"] for p in pairs]
    if len(set(prompts)) != len(prompts):
        raise SystemExit("Duplicate complaint found within the generated DPO set -- aborting, nothing written.")
    train_overlap = train_complaints & set(prompts)
    val_overlap = val_complaints & set(prompts)
    test_overlap = test_complaints & set(prompts)
    if train_overlap or val_overlap or test_overlap:
        raise SystemExit(
            f"Overlap detected with existing SFT splits (train={len(train_overlap)}, "
            f"val={len(val_overlap)}, test={len(test_overlap)}) -- aborting, nothing written."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    summary = {
        "design": "controlled single-signal missing-address dataset (replaces the earlier mixed-corruption set)",
        "n_pairs": len(pairs),
        "corruption_type_counts": {"hallucinate_address": len(pairs)},
        "seed": args.seed,
        "excluded_complaints_checked": len(excluded),
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "dpo_pairs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
