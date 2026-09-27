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

ISSUE_PHRASES = {
    "pothole": "a huge pothole thats gonna wreck someones tire",
    "missed_trash_pickup": "my trash wasnt picked up again this week",
    "illegal_dumping": "someone dumped a bunch of furniture and trash",
    "noise_complaint": "the neighbors have had a party blasting music since last night",
    "water_leak": "water just gushing out of a pipe in the street",
    "graffiti": "graffiti all over the wall of the building",
    "downed_tree": "a big tree branch fell and its blocking the sidewalk",
    "abandoned_vehicle": "a car with no plates thats been sitting there for weeks",
    "streetlight_outage": "the streetlight has been out and its pitch black at night",
    "animal_control": "a stray dog thats been wandering around aggressively",
}

SEVERITY_WORDS = {
    "low": ["whenever someone gets a chance", "not urgent but", "at some point"],
    "medium": ["please take care of this soon", "this has been going on for a few days"],
    "high": ["this is dangerous", "someone could get hurt", "please send someone ASAP"],
}

# Templates that mention a specific street -- address_or_null will be filled in.
ADDRESS_TEMPLATES = [
    "hi there theres a {issue} on {street} near the corner its been like this for {days} days {sev}",
    "URGENT!!! {issue} at {street}, {sev} can someone come out",
    "not sure who to tell but {issue} by {street}. {sev}",
    "so this is the third time im calling about {issue} on {street}... {sev} thanks",
    "{issue} on {street} sorry for typos im on my phone. {sev}",
]

# Templates that never mention a street -- address_or_null must stay null for these.
NO_ADDRESS_TEMPLATES = [
    "hi theres a {issue} somewhere around my block, been like this for {days} days {sev}",
    "URGENT!!! {issue} near my house, not sure of the exact address, {sev}",
    "not sure who to tell but {issue} nearby, i dont know the street name {sev}",
    "{issue}, sorry i dont know the address off the top of my head. {sev}",
    "calling about {issue} in my neighborhood, forgot to check the street sign {sev}",
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
    days = rng.randint(1, 14)
    priority = rng.choice(["low", "medium", "high"])
    sev_phrase = rng.choice(SEVERITY_WORDS[priority])

    no_address = rng.random() < NO_ADDRESS_RATE if force_no_address is None else force_no_address

    if no_address:
        template = rng.choice(NO_ADDRESS_TEMPLATES)
        complaint = template.format(issue=ISSUE_PHRASES[category], days=days, sev=sev_phrase)
        address_or_null = None
    else:
        street = rng.choice(STREETS)
        template = rng.choice(ADDRESS_TEMPLATES)
        complaint = template.format(issue=ISSUE_PHRASES[category], street=street, days=days, sev=sev_phrase)
        address_or_null = street

    meta = CATEGORIES[category]
    ticket = {
        "category": category,
        "department": meta["department"],
        "priority": priority,
        "address_or_null": address_or_null,
        "description": ISSUE_PHRASES[category][0].upper() + ISSUE_PHRASES[category][1:],
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
    summary = {
        "total": len(all_examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples),
        "no_address_share": round(null_share, 3),
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
