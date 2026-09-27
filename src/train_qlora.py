"""
QLoRA supervised fine-tuning on the synthetic 311 instruction dataset, plus the
deliberate-OOM demonstration and its fix.

NOT executed as part of building this repo structure -- requires a CUDA GPU (bitsandbytes
4-bit loading) that this dev environment doesn't have. Run in Colab.

Refuses to start if configs/qlora_config.json still has max_seq_len: null -- run
src/audit_instruction_data.py first and get the value approved (see that file's docstring
and the README).
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from generate_instruction_data import build_chat_prompt, target_json
from resource_monitor import StepTimingCallback, profile_run


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    if config.get("max_seq_len") is None:
        raise SystemExit(
            "configs/qlora_config.json has max_seq_len: null. Run "
            "src/audit_instruction_data.py against data/sft_train.jsonl, review the "
            "recommended value in data/token_length_audit.json, get it approved, and set "
            "max_seq_len before training. Do not guess a number here."
        )
    return config


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def verify_target_modules(model, proposed_targets):
    """Cross-check the config's proposed LoRA target modules against the model's actual
    Linear layer names before attaching adapters -- do not assume names from memory."""
    linear_leaf_names = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ in ("Linear4bit", "Linear8bitLt", "Linear"):
            linear_leaf_names.add(name.split(".")[-1])

    missing = [t for t in proposed_targets if t not in linear_leaf_names]
    if missing:
        raise SystemExit(
            f"Proposed LoRA target modules {missing} were not found among this model's "
            f"linear layer names ({sorted(linear_leaf_names)}). Update "
            "configs/qlora_config.json to match the model actually loaded."
        )
    print(f"Verified target modules against model: {proposed_targets}")
    print(f"All linear layer leaf names found in model: {sorted(linear_leaf_names)}")


def report_trainable_parameters(model):
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    return {"trainable_params": trainable, "total_params": total, "trainable_pct": round(pct, 4)}


def build_sft_dataset(examples, tokenizer, max_seq_len: int):
    input_ids_list, labels_list = [], []
    for ex in examples:
        prompt = build_chat_prompt(tokenizer, ex["complaint"])
        target = target_json(ex["ticket"]) + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_seq_len]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})


def make_pad_collate(tokenizer):
    def pad_collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
            attn.append([1] * len(x["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
    return pad_collate


def build_training_args(cfg: dict, out_dir: str) -> TrainingArguments:
    t = cfg["training"]
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=t["seed"],
    )


def run_safe_training(cfg: dict, data_dir: str, out_dir: str, log_path: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map={"": 0},
        torch_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )
    verify_target_modules(base_model, cfg["lora"]["target_modules_proposed"])

    model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules_proposed"],
    )
    model = get_peft_model(model, lora_cfg)
    param_report = report_trainable_parameters(model)

    train_examples = load_jsonl(os.path.join(data_dir, "sft_train.jsonl"))
    val_examples = load_jsonl(os.path.join(data_dir, "sft_val.jsonl"))
    max_seq_len = cfg["max_seq_len"]
    train_ds = build_sft_dataset(train_examples, tokenizer, max_seq_len)
    val_ds = build_sft_dataset(val_examples, tokenizer, max_seq_len)

    step_timer = StepTimingCallback()
    args = build_training_args(cfg, os.path.join(out_dir, "sft_ckpt"))

    def do_train():
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=make_pad_collate(tokenizer),
            callbacks=[step_timer],
        )
        trainer.train()
        return trainer

    run_config = {**cfg["training"], "max_seq_len": max_seq_len, **param_report}
    trainer, record = profile_run("qlora_sft_safe_config", do_train, config=run_config, step_timer=step_timer, log_path=log_path)

    adapter_path = os.path.join(out_dir, "qlora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Saved adapter to {adapter_path}")
    return model, tokenizer, record


def run_deliberate_oom(cfg: dict, log_path: str):
    """Genuinely tries to exceed GPU memory with an unsafe batch/seq_len/checkpointing
    config, captures the real traceback if it OOMs, then reruns the safe config for a
    single step to show the fix works. Does not fabricate a result if no OOM occurs --
    it reports what actually happened and tells you to raise the unsafe values further."""
    import traceback

    oom_cfg = cfg["deliberate_oom"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map={"": 0},
        torch_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"], task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules_proposed"],
    )

    def build_padded_batch(n, seq_len, device):
        input_ids = torch.randint(0, tokenizer.vocab_size, (n, seq_len), device=device)
        return {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}

    def run_unsafe_step():
        unsafe_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
        unsafe_model = get_peft_model(unsafe_model, lora_cfg)
        unsafe_model.gradient_checkpointing_disable()
        unsafe_model.train()
        optimizer = torch.optim.AdamW(unsafe_model.parameters(), lr=2e-4)
        batch = build_padded_batch(oom_cfg["unsafe_batch_size"], oom_cfg["unsafe_seq_len"], unsafe_model.device)
        outputs = unsafe_model(**batch)
        outputs.loss.backward()
        optimizer.step()
        return outputs.loss.item()

    oom_captured = False
    try:
        _, _ = profile_run(
            "deliberate_oom_attempt", run_unsafe_step,
            config={"batch_size": oom_cfg["unsafe_batch_size"], "seq_len": oom_cfg["unsafe_seq_len"], "gradient_checkpointing": False},
            log_path=log_path,
        )
    except torch.cuda.OutOfMemoryError:
        oom_captured = True
        tb = traceback.format_exc()
        os.makedirs("logs", exist_ok=True)
        with open("logs/oom_traceback.txt", "w") as f:
            f.write(tb)
        print("Captured a genuine CUDA OutOfMemoryError -- see logs/oom_traceback.txt")
        torch.cuda.empty_cache()

    if not oom_captured:
        print(
            "No OOM occurred with this configuration on this GPU allocation. Raise "
            "deliberate_oom.unsafe_batch_size / unsafe_seq_len in configs/qlora_config.json "
            "and rerun -- do not report an OOM that didn't happen."
        )

    def run_fixed_step():
        fixed_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        fixed_model = get_peft_model(fixed_model, lora_cfg)
        fixed_model.train()
        optimizer = torch.optim.AdamW(fixed_model.parameters(), lr=2e-4)
        batch = build_padded_batch(cfg["training"]["per_device_train_batch_size"], cfg["max_seq_len"], fixed_model.device)
        outputs = fixed_model(**batch)
        outputs.loss.backward()
        optimizer.step()
        return outputs.loss.item()

    profile_run(
        "post_fix_step", run_fixed_step,
        config={"batch_size": cfg["training"]["per_device_train_batch_size"], "seq_len": cfg["max_seq_len"], "gradient_checkpointing": True},
        log_path=log_path,
    )
    return oom_captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora_config.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--log-path", default="logs/experiment_log.json")
    parser.add_argument("--run-oom-demo", action="store_true", help="also run the deliberate OOM + fix")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_safe_training(cfg, args.data_dir, args.out_dir, args.log_path)
    if args.run_oom_demo:
        run_deliberate_oom(cfg, args.log_path)


if __name__ == "__main__":
    main()
