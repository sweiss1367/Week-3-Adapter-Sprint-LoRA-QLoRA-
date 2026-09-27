"""
Generates the synthetic 311 complaint -> structured ticket instruction dataset.

Fully offline: template + randomized slot-filling, no external API calls, so it is free
and reproducible from a fixed seed.

Ticket schema:
    {category, department, priority, address_or_null, description, requested_action}

`address_or_null` is deliberately null for a slice of complaints that never mention a
street (see NO_ADDRESS_RATE). This is the ground truth the preference dataset (see
generate_preference_data.py) uses to build the missing-address hallucination DPO pairs:
chosen responses must preserve null when no address was given; rejected responses invent
one. Getting this right during generation matters more than getting it right during DPO --
if every example carries an address, there is nothing for that DPO signal to teach.
"""
import argparse
import json
import os
import random

CATEGORIES = {
    "pothole": {"department": "Public Works", "action": "Dispatch pothole repair crew"},
    "missed_trash_pickup": {"department": "Sanitation", "action": "Schedule makeup collection"},
    "illegal_dumping": {"department": "Sanitation", "action": "Dispatch cleanup crew and investigate"},
    "noise_complaint": {"department": "Code Enforcement", "action": "Send officer to assess noise violation"},
    "water_leak": {"department": "Water Utility", "action": "Dispatch emergency water crew"},
    "graffiti": {"department": "Public Works", "action": "Schedule graffiti removal"},
    "downed_tree": {"department": "Parks & Forestry", "action": "Dispatch crew to clear tree/limb"},
    "abandoned_vehicle": {"department": "Code Enforcement", "action": "Tag vehicle and initiate towing process"},
    "streetlight_outage": {"department": "Public Works", "action": "Dispatch electrician to repair streetlight"},
    "animal_control": {"department": "Animal Control", "action": "Dispatch animal control officer"},
}

STREETS = [
    "Maple St", "5th Ave", "Union Blvd", "Birch Rd", "Harbor Dr", "Elm St",
    "Lincoln Ave", "Cedar Ln", "Franklin St", "Riverside Dr", "Oak St", "Grant Ave",
]

# Each category has multiple noun-phrase variants (no leading article baked in twice --
# these are the object of "there's"/"there is", e.g. "there's a huge pothole...",
# "there's trash that hasn't been picked up...", "there's graffiti covering...").
ISSUE_PHRASES = {
    "pothole": [
        "a huge pothole that's going to wreck someone's tire",
        "a deep pothole that opened up in the middle of the road",
    ],
    "missed_trash_pickup": [
        "trash that hasn't been picked up in over a week",
        "garbage still sitting out from the last two collection days",
    ],
    "illegal_dumping": [
        "a pile of dumped furniture and trash",
        "a bunch of construction debris someone dumped",
    ],
    "noise_complaint": [
        "a neighbor blasting music late into the night",
        "constant loud noise coming from a nearby apartment",
    ],
    "water_leak": [
        "water gushing out of a broken pipe in the street",
        "a burst water main flooding the road",
    ],
    "graffiti": [
        "graffiti covering the wall of a building",
        "fresh spray paint tags all over a storefront",
    ],
    "downed_tree": [
        "a fallen tree branch blocking the sidewalk",
        "a large tree limb that came down after the wind",
    ],
    "abandoned_vehicle": [
        "an abandoned car with no license plates",
        "a car that's been parked in the same spot with flat tires for weeks",
    ],
    "streetlight_outage": [
        "a streetlight that's been out for days",
        "a broken streetlight leaving the block dark at night",
    ],
    "animal_control": [
        "a stray dog wandering around aggressively",
        "a loose dog that's been chasing people on the sidewalk",
    ],
}

# Opener + tone phrasing are keyed to the SAME priority tier so urgency language never
# contradicts the assigned priority (the earlier generator picked these independently,
# which could pair an "URGENT!!!" opener with a low-priority tone phrase).
OPENERS = {
    "low": [
        "hi there,", "hello,", "just wanted to mention something,",
        "no big deal, but,", "whenever you get a chance,",
    ],
    "medium": [
        "hi,", "wanted to report an issue,", "following up on something,",
        "not sure who else to tell,", "reaching out about a problem,",
    ],
    "high": [
        "URGENT:", "please help,", "this needs attention right away,",
        "calling to report an emergency,", "need help immediately,",
    ],
}

TONE_PHRASES = {
    "low": [
        "whenever someone gets a chance to look into it",
        "not urgent, just wanted it on record",
        "no rush on this one",
        "figured i'd mention it, no big deal",
    ],
    "medium": [
        "this has been going on for a few days now",
        "would appreciate someone looking into it soon",
        "please take care of this when you can",
        "hoping someone can check on it this week",
    ],
    "high": [
        "this is dangerous and needs attention ASAP",
        "someone could get hurt, please send help right away",
        "this can't wait, please respond urgently",
        "please send someone out immediately",
    ],
}

# Inserted after the issue phrase when a street is given.
ADDRESS_PHRASES = [
    " on {street}", " near {street}", " by {street}", " over by {street}", " close to {street}",
]

# Inserted after the issue phrase when no street is given -- address_or_null stays null.
NO_ADDRESS_PHRASES = [
    " somewhere near my block", " around my neighborhood",
    " near my house, i don't know the exact address",
    " somewhere nearby, i forgot the street name",
    " a couple blocks from my place, not sure of the address",
]

NO_ADDRESS_RATE = 0.15

SYSTEM_PROMPT = (
    "You are a municipal 311 dispatch assistant. Read the resident's complaint and output ONLY a JSON "
    "object with exactly these keys: category, department, priority, address_or_null, description, "
    "requested_action. priority must be one of: low, medium, high. address_or_null must be the street "
    "mentioned by the resident, or the JSON value null if no address was given -- never invent one. "
    "Output valid JSON and nothing else."
)


def make_example(rng: random.Random, force_no_address: bool = None):
    category = rng.choice(list(CATEGORIES.keys()))
    priority = rng.choice(["low", "medium", "high"])
    days = rng.randint(1, 14)

    issue_phrase = rng.choice(ISSUE_PHRASES[category])
    opener = rng.choice(OPENERS[priority])
    tone = rng.choice(TONE_PHRASES[priority])
    days_mention = rng.random() < 0.5

    no_address = rng.random() < NO_ADDRESS_RATE if force_no_address is None else force_no_address
    if no_address:
        location_phrase = rng.choice(NO_ADDRESS_PHRASES)
        address_or_null = None
    else:
        street = rng.choice(STREETS)
        location_phrase = rng.choice(ADDRESS_PHRASES).format(street=street)
        address_or_null = street

    days_clause = f", it's been like this for {days} days" if days_mention else ""
    order = rng.choice(["issue_first", "tone_first"])
    if order == "issue_first":
        complaint = f"{opener} there's {issue_phrase}{location_phrase}{days_clause}. {tone}"
    else:
        complaint = f"{opener} {tone} -- there's {issue_phrase}{location_phrase}{days_clause}"

    if rng.random() < 0.15:
        complaint = complaint.replace(".", "")

    meta = CATEGORIES[category]
    ticket = {
        "category": category,
        "department": meta["department"],
        "priority": priority,
        "address_or_null": address_or_null,
        "description": issue_phrase[0].upper() + issue_phrase[1:],
        "requested_action": meta["action"],
    }
    return {"complaint": complaint, "ticket": ticket}


def build_dataset(n: int, seed: int):
    rng = random.Random(seed)
    seen = set()
    examples = []
    while len(examples) < n:
        ex = make_example(rng)
        key = (ex["complaint"], json.dumps(ex["ticket"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)
    return examples


def build_prompt_text(complaint: str) -> str:
    """Plain-text fallback representation, only used if no tokenizer/chat template is available."""
    return f"{SYSTEM_PROMPT}\n\nUser: {complaint}\nAssistant:"


def build_chat_prompt(tokenizer, complaint: str) -> str:
    """Canonical prompt construction, shared by the audit, training, and evaluation
    scripts so every stage scores/trains on identically formatted input."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def target_json(ticket: dict) -> str:
    return json.dumps(ticket, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=360, help="total examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    all_examples = build_dataset(args.n, seed=args.seed)
    random.Random(args.seed).shuffle(all_examples)

    test_examples = all_examples[: args.n_test]
    val_examples = all_examples[args.n_test: args.n_test + args.n_val]
    train_examples = all_examples[args.n_test + args.n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {"train": train_examples, "val": val_examples, "test": test_examples}
    for name, data in splits.items():
        path = os.path.join(args.out_dir, f"sft_{name}.jsonl")
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    null_share = sum(1 for ex in all_examples if ex["ticket"]["address_or_null"] is None) / len(all_examples)
    priority_counts = {}
    for ex in all_examples:
        p = ex["ticket"]["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1

    summary = {
        "total": len(all_examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
        "no_address_share": round(null_share, 3),
        "priority_counts": priority_counts,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
