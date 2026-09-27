"""
Builds the BALANCED v2 DPO preference dataset: 75 missing-address pairs plus 75
addressed-control pairs, so DPO sees both directions of the conditional rule --
preserve null when no address is given, preserve the given address when one is given --
instead of only ever training "prefer null".

Root cause this fixes: v1 (generate_address_dpo_data.py, data/dpo_pairs.jsonl) used 150
missing-address pairs and nothing else. The resulting DPO v1 adapter over-generalized to
"always output null": on the deterministic 10-example held-out comparison it scored
missing_address_accuracy=1.0 but addressed_control_accuracy=0.0 (it output null on every
one of the 5 addressed-control examples too).

- 75 missing-address pairs: same design as v1 -- chosen preserves address_or_null=null,
  rejected invents a plausible street.
- 75 addressed-control pairs: NEW -- complaints that explicitly give a street. chosen
  preserves that exact address; rejected sets address_or_null to null (the mirror-image
  corruption of the missing-address pairs), so the preference signal explicitly penalizes
  dropping a real address, not just penalizes inventing one.

Every complaint is freshly generated (generate_instruction_data.make_example), never
copied from any SFT split or from data/dpo_pairs.jsonl (v1) -- v2 is a distinct dataset.

Usage:
    python src/generate_address_dpo_data_v2.py --seed 42
"""
import argparse
import json
import os
import random

from generate_instruction_data import STREETS, make_example, target_json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def generate_unique_examples(n: int, seed: int, force_no_address: bool, excluded_complaints: set, max_attempts_multiplier: int = 50):
    rng = random.Random(seed)
    seen = set(excluded_complaints)
    examples = []
    attempts = 0
    max_attempts = n * max_attempts_multiplier
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        ex = make_example(rng, force_no_address=force_no_address)
        if ex["complaint"] in seen:
            continue
        seen.add(ex["complaint"])
        examples.append(ex)
    if len(examples) < n:
        raise SystemExit(
            f"Could only generate {len(examples)}/{n} unique examples "
            f"(force_no_address={force_no_address}) after {attempts} attempts. "
            "Increase --max-attempts-multiplier or reduce the target count."
        )
    return examples


def build_missing_address_pair(ex: dict, rng: random.Random) -> dict:
    chosen_ticket = dict(ex["ticket"])
    assert chosen_ticket["address_or_null"] is None, "generator invariant violated: expected a no-address example"

    invented_street = rng.choice(STREETS)
    rejected_ticket = dict(chosen_ticket)  # same keys, same order; only this value changes
    rejected_ticket["address_or_null"] = invented_street

    return {
        "prompt": ex["complaint"],
        "chosen": target_json(chosen_ticket),
        "rejected": target_json(rejected_ticket),
        "pair_type": "missing_address",
        "address_or_null_in_gold": None,
        "invented_address": invented_street,
    }


def build_addressed_control_pair(ex: dict) -> dict:
    chosen_ticket = dict(ex["ticket"])
    assert chosen_ticket["address_or_null"] is not None, "generator invariant violated: expected an addressed example"

    rejected_ticket = dict(chosen_ticket)  # same keys, same order; only this value changes
    rejected_ticket["address_or_null"] = None

    return {
        "prompt": ex["complaint"],
        "chosen": target_json(chosen_ticket),
        "rejected": target_json(rejected_ticket),
        "pair_type": "addressed_control",
        "address_or_null_in_gold": chosen_ticket["address_or_null"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-missing", type=int, default=75)
    parser.add_argument("--n-addressed", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts-multiplier", type=int, default=50)
    parser.add_argument("--train-data", default="data/sft_train.jsonl")
    parser.add_argument("--val-data", default="data/sft_val.jsonl")
    parser.add_argument("--test-data", default="data/sft_test.jsonl")
    parser.add_argument("--v1-data", default="data/dpo_pairs.jsonl", help="v1 prompts are also excluded, so v2 is a distinct dataset")
    parser.add_argument("--out", default="data/dpo_pairs_v2.jsonl")
    args = parser.parse_args()

    train_complaints = {e["complaint"] for e in load_jsonl(args.train_data)}
    val_complaints = {e["complaint"] for e in load_jsonl(args.val_data)}
    test_complaints = {e["complaint"] for e in load_jsonl(args.test_data)}
    v1_complaints = {p["prompt"] for p in load_jsonl(args.v1_data)} if os.path.exists(args.v1_data) else set()
    excluded = train_complaints | val_complaints | test_complaints | v1_complaints

    missing_examples = generate_unique_examples(args.n_missing, args.seed, True, excluded, args.max_attempts_multiplier)
    # Also exclude the missing-address complaints just generated, so the addressed batch
    # can't accidentally collide with them.
    excluded_for_addressed = excluded | {e["complaint"] for e in missing_examples}
    addressed_examples = generate_unique_examples(args.n_addressed, args.seed + 1, False, excluded_for_addressed, args.max_attempts_multiplier)

    pair_rng = random.Random(args.seed + 2)
    missing_pairs = [build_missing_address_pair(ex, pair_rng) for ex in missing_examples]
    addressed_pairs = [build_addressed_control_pair(ex) for ex in addressed_examples]
    pairs = missing_pairs + addressed_pairs

    # Independent safety checks -- do not rely solely on generation-time dedup above.
    prompts = [p["prompt"] for p in pairs]
    if len(set(prompts)) != len(prompts):
        raise SystemExit("Duplicate complaint found within DPO v2 -- aborting, nothing written.")
    overlaps = {
        "sft_train": train_complaints & set(prompts),
        "sft_val": val_complaints & set(prompts),
        "sft_test": test_complaints & set(prompts),
        "dpo_v1": v1_complaints & set(prompts),
    }
    if any(overlaps.values()):
        raise SystemExit(
            f"Overlap detected: { {k: len(v) for k, v in overlaps.items()} } -- aborting, nothing written."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    summary = {
        "design": "balanced v2: 75 missing-address + 75 addressed-control pairs (replaces v1's all-missing-address design, which over-generalized to always predicting null)",
        "n_pairs": len(pairs),
        "n_missing_address_pairs": len(missing_pairs),
        "n_addressed_control_pairs": len(addressed_pairs),
        "seed": args.seed,
        "excluded_complaints_checked": len(excluded_for_addressed),
        "v1_data_checked_for_overlap": args.v1_data,
        "overlap_found": {k: len(v) for k, v in overlaps.items()},
    }
    summary_path = os.path.join(os.path.dirname(args.out) or ".", "dpo_pairs_v2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
