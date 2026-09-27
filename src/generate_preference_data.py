"""
Builds the chosen/rejected preference dataset for DPO from the SFT instruction data.

Every `rejected` response is a rule-based corruption of the correct ticket -- a realistic
failure mode for this task, not a random string. One corruption type is the specific
behavior this assignment's DPO stage is meant to suppress: for complaints where the
resident never gave an address (address_or_null is null in the ground truth), `chosen`
preserves null and `rejected` invents a plausible-looking street. Every null-address
training example gets this corruption so the DPO signal on that behavior is never diluted
by other corruption types.

Complaints that do carry a real address get one of the other corruption types (dropped
field, wrong category, malformed JSON, or a wrong/hallucinated address swapped in) so the
preference set also teaches general schema correctness, not just the address behavior.
"""
import argparse
import json
import os
import random

from generate_instruction_data import CATEGORIES, STREETS


def target_json(ticket: dict) -> str:
    return json.dumps(ticket, ensure_ascii=False)


def hallucinate_address(ticket: dict, rng: random.Random) -> dict:
    corrupted = dict(ticket)
    corrupted["address_or_null"] = rng.choice(STREETS)
    return corrupted


def drop_field(ticket: dict, rng: random.Random) -> dict:
    corrupted = dict(ticket)
    key = rng.choice(["priority", "department", "requested_action"])
    del corrupted[key]
    return corrupted


def wrong_category(ticket: dict, rng: random.Random) -> dict:
    corrupted = dict(ticket)
    other = rng.choice([c for c in CATEGORIES if c != corrupted["category"]])
    corrupted["category"] = other
    corrupted["department"] = CATEGORIES[other]["department"]
    return corrupted


def wrong_address(ticket: dict, rng: random.Random) -> dict:
    corrupted = dict(ticket)
    other_streets = [s for s in STREETS if s != corrupted["address_or_null"]]
    corrupted["address_or_null"] = rng.choice(other_streets)
    return corrupted


def build_rejected(ticket: dict, rng: random.Random):
    """Returns (rejected_json_str, corruption_type)."""
    if ticket["address_or_null"] is None:
        corrupted = hallucinate_address(ticket, rng)
        return target_json(corrupted), "hallucinate_address"

    corruption_type = rng.choice(["drop_field", "wrong_category", "malformed_json", "wrong_address"])
    if corruption_type == "drop_field":
        corrupted = drop_field(ticket, rng)
        return target_json(corrupted), corruption_type
    if corruption_type == "wrong_category":
        corrupted = wrong_category(ticket, rng)
        return target_json(corrupted), corruption_type
    if corruption_type == "wrong_address":
        corrupted = wrong_address(ticket, rng)
        return target_json(corrupted), corruption_type
    # malformed_json: drop the closing brace so it's genuinely invalid JSON
    return target_json(ticket)[:-1], corruption_type


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sft_train.jsonl", help="source examples -- must be the train split only")
    parser.add_argument("--held-out-data", default="data/sft_test.jsonl", help="checked for leakage, never sampled from")
    parser.add_argument("--out", default="data/dpo_pairs.jsonl")
    parser.add_argument("--target-total", type=int, default=165, help="~150-180 total pairs")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    examples = load_jsonl(args.data)
    held_out_complaints = {ex["complaint"] for ex in load_jsonl(args.held_out_data)}

    leaked = [ex for ex in examples if ex["complaint"] in held_out_complaints]
    if leaked:
        raise SystemExit(
            f"{len(leaked)} example(s) from {args.data} also appear in {args.held_out_data}. "
            "The held-out test set must never be used to build the DPO preference dataset -- aborting."
        )

    rng = random.Random(args.seed)

    # The main behavioral target is missing-address hallucination: every no-address
    # example in the source data gets a pair, not a sampled subset of them, so the
    # behavior is fully represented rather than diluted by chance.
    no_address_examples = [ex for ex in examples if ex["ticket"]["address_or_null"] is None]
    addressed_examples = [ex for ex in examples if ex["ticket"]["address_or_null"] is not None]

    n_addressed_needed = max(0, args.target_total - len(no_address_examples))
    rng.shuffle(addressed_examples)
    selected_addressed = addressed_examples[:n_addressed_needed]

    selected = no_address_examples + selected_addressed
    rng.shuffle(selected)

    pairs = []
    for ex in selected:
        chosen = target_json(ex["ticket"])
        rejected, corruption_type = build_rejected(ex["ticket"], rng)
        pairs.append({
            "prompt": ex["complaint"],
            "chosen": chosen,
            "rejected": rejected,
            "corruption_type": corruption_type,
            "address_or_null_in_gold": ex["ticket"]["address_or_null"],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    counts = {}
    for pair in pairs:
        counts[pair["corruption_type"]] = counts.get(pair["corruption_type"], 0) + 1
    summary = {
        "n_pairs": len(pairs),
        "n_no_address_source_examples": len(no_address_examples),
        "n_no_address_pairs_included": sum(1 for p in pairs if p["corruption_type"] == "hallucinate_address"),
        "corruption_type_counts": counts,
        "source_data": args.data,
        "held_out_data_checked": args.held_out_data,
        "held_out_leakage_found": len(leaked),
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "dpo_pairs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
