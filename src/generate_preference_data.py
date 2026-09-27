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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sft_train.jsonl")
    parser.add_argument("--out", default="data/dpo_pairs.jsonl")
    parser.add_argument("--max-pairs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[: args.max_pairs]

    rng = random.Random(args.seed)
    pairs = []
    for ex in examples:
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
    summary = {"n_pairs": len(pairs), "corruption_type_counts": counts}
    with open(os.path.join(os.path.dirname(args.out) or ".", "dpo_pairs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
