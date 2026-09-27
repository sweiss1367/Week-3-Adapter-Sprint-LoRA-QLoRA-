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
from resource_monitor import StepTimingCallback, check_hardware, profile_run, resolve_precision


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


def print_tokenizer_info(tokenizer):
    print("Tokenizer configuration:")
    print(f"  pad_token: {tokenizer.pad_token!r}")
    print(f"  pad_token_id: {tokenizer.pad_token_id!r}")
    print(f"  eos_token: {tokenizer.eos_token!r}")
    print(f"  eos_token_id: {tokenizer.eos_token_id!r}")
    print(f"  padding_side: {tokenizer.padding_side!r}")


def validate_labels(train_ds, val_ds):
    """Validation only -- does not modify either dataset or max_seq_len. Aborts before
    training if any example has zero supervised (non -100) label tokens, since that
    example would contribute no training signal and risks a nan loss."""
    train_counts = [sum(1 for l in labels if l != -100) for labels in train_ds["labels"]]
    val_counts = [sum(1 for l in labels if l != -100) for labels in val_ds["labels"]]
    all_counts = train_counts + val_counts

    n_zero = sum(1 for c in all_counts if c == 0)
    summary = {
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
        "min_supervised_target_tokens": min(all_counts),
        "max_supervised_target_tokens": max(all_counts),
        "n_examples_with_zero_supervised_tokens": n_zero,
    }
    print("Label integrity check (train + val combined for min/max/zero-count):")
    print(json.dumps(summary, indent=2))

    if n_zero > 0:
        raise SystemExit(
            f"{n_zero} example(s) have zero supervised (non -100) label tokens -- these "
            "would contribute no training signal or risk a nan loss. Aborting before "
            "training rather than silently training on them."
        )
    return summary


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
        fp16=t.get("fp16", False),
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=t["seed"],
    )


def run_safe_training(cfg: dict, data_dir: str, out_dir: str, log_path: str):
    check_hardware(resolve_precision(cfg["training"]))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print_tokenizer_info(tokenizer)

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
    if cfg["training"]["gradient_checkpointing"]:
        # use_cache=True is incompatible with gradient checkpointing (it stores
        # activations needed for backward specifically because caching is off).
        model.config.use_cache = False
    param_report = report_trainable_parameters(model)

    train_examples = load_jsonl(os.path.join(data_dir, "sft_train.jsonl"))
    val_examples = load_jsonl(os.path.join(data_dir, "sft_val.jsonl"))
    max_seq_len = cfg["max_seq_len"]
    train_ds = build_sft_dataset(train_examples, tokenizer, max_seq_len)
    val_ds = build_sft_dataset(val_examples, tokenizer, max_seq_len)
    validate_labels(train_ds, val_ds)

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


def _fresh_quantized_base(cfg: dict):
    """Loads a brand-new base model instance. Called separately for the unsafe attempt
    and the fixed rerun so the recovery run never reuses a base model/wrapper that the
    failed attempt already modified (prepare_model_for_kbit_training and get_peft_model
    both mutate the module tree they're given)."""
    return AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map={"": 0},
        torch_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )


def _lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"], task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules_proposed"],
    )


def _build_padded_batch(tokenizer, n, seq_len, device):
    input_ids = torch.randint(0, tokenizer.vocab_size, (n, seq_len), device=device)
    return {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}


def run_deliberate_oom(cfg: dict, log_path: str):
    """Genuinely tries to exceed GPU memory with an unsafe batch/seq_len/checkpointing
    config on a freshly loaded base model, captures the real traceback if it OOMs, tears
    down every reference the failed attempt created, then reloads an entirely fresh base
    model + fresh adapter for the safe rerun. The recovery run never reuses a base model,
    optimizer, or batch that the failed attempt already wrapped or touched -- a PEFT-
    wrapped, partially-mutated module tree is not a safe starting point to resume from.
    Does not fabricate a result if no OOM occurs -- it reports what actually happened and
    tells you to raise the unsafe values further."""
    import gc
    import traceback

    check_hardware(resolve_precision(cfg["training"]))

    oom_cfg = cfg["deliberate_oom"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Deliberate OOM attempt: fresh base model, fresh adapter, checkpointing off ---
    unsafe_base = _fresh_quantized_base(cfg)
    unsafe_model = prepare_model_for_kbit_training(unsafe_base, use_gradient_checkpointing=False)
    unsafe_model = get_peft_model(unsafe_model, _lora_config(cfg))
    unsafe_model.gradient_checkpointing_disable()
    # use_cache is intentionally left at its default here -- gradient checkpointing is
    # off for this path on purpose, so there's no use_cache/checkpointing conflict to guard against.
    unsafe_model.train()
    unsafe_optimizer = torch.optim.AdamW(unsafe_model.parameters(), lr=2e-4)
    unsafe_batch = None

    oom_captured = False
    try:
        unsafe_batch = _build_padded_batch(tokenizer, oom_cfg["unsafe_batch_size"], oom_cfg["unsafe_seq_len"], unsafe_model.device)

        def run_unsafe_step():
            outputs = unsafe_model(**unsafe_batch)
            outputs.loss.backward()
            unsafe_optimizer.step()
            return outputs.loss.item()

        profile_run(
            "deliberate_oom_attempt", run_unsafe_step,
            config={"batch_size": oom_cfg["unsafe_batch_size"], "seq_len": oom_cfg["unsafe_seq_len"], "gradient_checkpointing": False},
            log_path=log_path,
        )
    except torch.cuda.OutOfMemoryError:
        oom_captured = True
        # profile_run already logged peak memory/elapsed time/status="failed" to
        # log_path; this traceback file captures the actual stack trace separately.
        tb = traceback.format_exc()
        os.makedirs("logs", exist_ok=True)
        with open("logs/oom_traceback.txt", "w") as f:
            f.write(tb)
        print("Captured a genuine CUDA OutOfMemoryError -- see logs/oom_traceback.txt")
    finally:
        # Tear down every reference to the failed attempt before reloading anything --
        # never reuse a model/optimizer/batch that was wrapped or touched during the
        # failure, even if the exception wasn't a CUDA OOM.
        del unsafe_batch, unsafe_optimizer, unsafe_model, unsafe_base
        gc.collect()
        torch.cuda.empty_cache()

    if not oom_captured:
        print(
            "No OOM occurred with this configuration on this GPU allocation. Raise "
            "deliberate_oom.unsafe_batch_size / unsafe_seq_len in configs/qlora_config.json "
            "and rerun -- do not report an OOM that didn't happen."
        )

    # --- Recovery run: entirely fresh base model + fresh adapter, safe config ---
    fixed_base = _fresh_quantized_base(cfg)
    fixed_model = prepare_model_for_kbit_training(fixed_base, use_gradient_checkpointing=True)
    fixed_model = get_peft_model(fixed_model, _lora_config(cfg))
    # This recovery run has gradient checkpointing on -- disable KV caching to avoid the
    # same use_cache/checkpointing conflict guarded against in run_safe_training.
    fixed_model.config.use_cache = False
    fixed_model.train()
    fixed_optimizer = torch.optim.AdamW(fixed_model.parameters(), lr=2e-4)

    def run_fixed_step():
        batch = _build_padded_batch(tokenizer, cfg["training"]["per_device_train_batch_size"], cfg["max_seq_len"], fixed_model.device)
        outputs = fixed_model(**batch)
        outputs.loss.backward()
        fixed_optimizer.step()
        return outputs.loss.item()

    profile_run(
        "post_fix_step", run_fixed_step,
        config={"batch_size": cfg["training"]["per_device_train_batch_size"], "seq_len": cfg["max_seq_len"], "gradient_checkpointing": True},
        log_path=log_path,
    )
    del fixed_model, fixed_base, fixed_optimizer
    gc.collect()
    torch.cuda.empty_cache()
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
