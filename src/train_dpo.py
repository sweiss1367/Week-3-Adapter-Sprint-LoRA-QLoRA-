"""
DPO on top of the QLoRA-tuned model.

Reference-policy design (see configs/dpo_config.json "reference_policy_design" for the
full rationale), verified against the exact installed runtime (TRL 1.14.0, PEFT 0.20.0)
rather than assumed:

    - The trained SFT adapter loads as the PeftModel's normal "default" adapter (no
      custom adapter_name).
    - DPOTrainer is given ref_model=None. TRL 1.14.0's DPOTrainer, when the supplied
      model is already a PeftModel with a "default" adapter and no ref_model is given,
      reads model.peft_config["default"], creates a second adapter named "ref", and
      copies every .default. parameter into the matching .ref. parameter -- inside its
      own constructor. This gives a frozen copy of the instruction-tuned state without
      loading a second full model or a second adapter ourselves.
    - There is no model_adapter_name / ref_adapter_name on this DPOConfig -- an earlier
      design assumed TRL exposed those and manually created two named adapters; that
      assumption didn't hold against the actual installed TRL 1.14.0 / PEFT 0.20.0 API
      and has been removed.

verify_reference_setup() runs AFTER DPOTrainer construction (since TRL creates "ref"
inside its own __init__) and BEFORE trainer.train(), and fails loudly if "default"/"ref"
aren't both present, "ref" isn't fully frozen, "default" isn't trainable, or their
starting tensors aren't exact copies -- this is checked, not assumed.

NOT executed as part of building this repo structure -- requires a CUDA GPU. Run in Colab.
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from generate_instruction_data import build_chat_prompt
from resource_monitor import GpuMemoryMonitorCallback, StepTimingCallback, check_hardware, profile_run, resolve_precision


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("max_length") is None or cfg.get("max_prompt_length") is None:
        raise SystemExit(
            "configs/dpo_config.json has max_length/max_prompt_length: null. Both must be "
            "approved from the real tokenizer audit (see data/dpo_pairs_audit.json) before "
            "training -- max_length is what's actually passed to DPOConfig; max_prompt_length "
            "is kept as audit documentation only since the installed TRL 1.14.0 DPOConfig "
            "doesn't expose it. Do not guess a number here."
        )
    return cfg


def build_bnb_config(qlora_cfg: dict) -> BitsAndBytesConfig:
    q = qlora_cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_policy_model(cfg: dict, qlora_cfg: dict):
    """Loads the 4-bit base (same quantization config approved for SFT), prepares it for
    kbit training, and loads the trained SFT adapter onto it as "default" -- the name TRL
    1.14.0 expects when it builds its automatic "ref" copy."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(qlora_cfg),
        device_map={"": 0},
        dtype=getattr(torch, qlora_cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )

    gradient_checkpointing = cfg["training"]["gradient_checkpointing"]
    prepared_base = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )

    # No adapter_name given -- this loads as "default", not a custom name.
    model = PeftModel.from_pretrained(prepared_base, cfg["sft_adapter_path"], is_trainable=True)

    if gradient_checkpointing:
        model.config.use_cache = False

    return model, tokenizer


def restore_fp32_policy_for_fp16_t4(model, cfg):
    """
    Hardware-compatibility fix for this T4 runtime, not a design change to the
    reference-policy setup.

    TRL 1.14.0's DPOTrainer constructor, having just copied "default" into "ref", later
    in that same constructor casts every trainable ("default") parameter to bfloat16;
    "ref" is frozen and keeps its original (FP32) dtype. With fp16=True, Accelerate's
    GradScaler then fails to unscale BF16 gradients on this T4
    (NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented
    for 'BFloat16'). BF16 trainable parameters are simply incompatible with fp16 mixed
    precision here.

    Restores each trainable ".default." parameter to an FP32 CLONE OF ITS FROZEN ".ref."
    COUNTERPART -- not a cast of the already-BF16 default tensor (that would keep the
    BF16 rounding loss), but the original SFT adapter values straight from "ref",
    promoted to FP32. This both restores the exact pre-BF16-cast initialization and gives
    the FP16 GradScaler FP32 trainable parameters, which it supports.

    Runs only when this run is configured for fp16 (and not bf16); a no-op otherwise. The
    frozen "ref" adapter and every base-model parameter are left untouched.
    """
    if not (cfg["training"].get("fp16") and not cfg["training"].get("bf16")):
        return

    peft_config = getattr(model, "peft_config", {})
    if "default" not in peft_config or "ref" not in peft_config:
        raise RuntimeError(
            "restore_fp32_policy_for_fp16_t4 requires both 'default' and 'ref' adapters to "
            "already exist on the model -- call this after DPOTrainer construction, which "
            "is what creates 'ref'."
        )

    ref_by_key = {
        name.replace(".ref.", ".<adapter>."): param
        for name, param in model.named_parameters()
        if ".ref." in name
    }

    n_tensors, n_params = 0, 0
    for name, param in model.named_parameters():
        if ".default." not in name or not param.requires_grad:
            continue
        key = name.replace(".default.", ".<adapter>.")
        ref_param = ref_by_key.get(key)
        if ref_param is None:
            raise RuntimeError(
                f"No matching '.ref.' tensor found for trainable parameter '{name}' -- "
                "cannot restore it from the frozen reference. Aborting."
            )
        with torch.no_grad():
            param.data = ref_param.detach().to(torch.float32).clone()
        n_tensors += 1
        n_params += param.numel()

    default_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if ".default." in n})
    ref_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if ".ref." in n})

    print(
        f"restore_fp32_policy_for_fp16_t4: restored {n_tensors} tensor(s) "
        f"({n_params:,} parameters) in 'default' to FP32 clones of 'ref'."
    )
    print(f"Resulting 'default' dtypes: {default_dtypes}")
    print(f"Resulting 'ref' dtypes: {ref_dtypes}")


def verify_reference_setup(model):
    """
    Must be called after DPOTrainer's constructor has run (TRL 1.14.0 creates the "ref"
    adapter there, copying "default" into it) and before trainer.train() (so the exact-
    copy check below is checking TRL's copy, not a policy that has already been updated
    by an optimizer step). Fails loudly (RuntimeError) rather than assuming any of this
    worked.
    """
    peft_config = model.peft_config
    if "default" not in peft_config:
        raise RuntimeError("model.peft_config is missing the 'default' (policy) adapter.")
    if "ref" not in peft_config:
        raise RuntimeError(
            "model.peft_config is missing the 'ref' adapter. TRL 1.14.0 was expected to "
            "create it automatically from 'default' inside DPOTrainer.__init__ (ref_model=None) "
            "-- it did not. Aborting rather than training against an unverified reference."
        )

    default_trainable, default_total = 0, 0
    ref_trainable, ref_total = 0, 0
    default_tensors, ref_tensors = {}, {}

    for name, param in model.named_parameters():
        if ".default." in name:
            default_total += param.numel()
            default_trainable += param.numel() if param.requires_grad else 0
            default_tensors[name.replace(".default.", ".<adapter>.")] = param
        elif ".ref." in name:
            ref_total += param.numel()
            ref_trainable += param.numel() if param.requires_grad else 0
            ref_tensors[name.replace(".ref.", ".<adapter>.")] = param

    report = {
        "default_adapter": {"total_params": default_total, "trainable_params": default_trainable},
        "ref_adapter": {"total_params": ref_total, "trainable_params": ref_trainable},
    }
    print("Reference setup check:")
    print(json.dumps(report, indent=2))

    if default_trainable == 0:
        raise RuntimeError("'default' adapter has zero trainable parameters -- nothing would train.")
    if ref_trainable != 0:
        raise RuntimeError(f"'ref' adapter has {ref_trainable} trainable parameters -- it must be frozen.")

    if set(default_tensors) != set(ref_tensors):
        raise RuntimeError(
            "'default' and 'ref' adapters do not have matching parameter names -- cannot "
            "verify they start as exact copies."
        )

    # TRL 1.14.0 casts trainable (i.e. "default") parameters to bfloat16 later in the same
    # constructor that copies "default" into "ref"; "ref" is frozen so it keeps its
    # original dtype. A dtype mismatch here is therefore expected, not a bug -- compare in
    # the policy tensor's dtype (cast the frozen ref tensor to match) rather than loosening
    # the comparison to an allclose tolerance.
    dtype_report = {key: {"default_dtype": str(dp.dtype), "ref_dtype": str(ref_tensors[key].dtype)} for key, dp in default_tensors.items()}
    print("Adapter tensor dtypes (diagnostic):")
    print(json.dumps(dtype_report, indent=2))

    mismatches = [
        key for key, dp in default_tensors.items()
        if not torch.equal(dp.detach(), ref_tensors[key].detach().to(dp.dtype))
    ]
    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} adapter tensor(s) differ between 'default' and 'ref' before "
            f"training started (e.g. {mismatches[0]}), even after casting 'ref' to "
            "'default''s dtype -- TRL's automatic reference copy did not produce an exact "
            "copy of the SFT state. Aborting."
        )

    print("Verified: 'default' and 'ref' are exact copies before training; "
          "'default' is trainable, 'ref' is frozen.")
    return report


def save_and_print_log_history(trainer, log_history_path: str):
    """Saves the real TRL log_history (which includes rewards/chosen, rewards/rejected,
    rewards/margins, per the installed TRL 1.14.0 source) and prints every entry that
    contains a reward metric so the real DPO preference effect is visible in Colab
    output. Never invents or prefills a reward value -- only reports what TRL logged."""
    log_history = trainer.state.log_history
    os.makedirs(os.path.dirname(log_history_path) or ".", exist_ok=True)
    with open(log_history_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Saved {len(log_history)} log_history entries to {log_history_path}")

    reward_entries = [entry for entry in log_history if "rewards/margins" in entry]
    print(f"{len(reward_entries)} entries contain rewards/margins:")
    for entry in reward_entries:
        print(json.dumps(entry, indent=2))
    return log_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo-config", default="configs/dpo_config.json")
    parser.add_argument("--qlora-config", default="configs/qlora_config.json")
    parser.add_argument("--data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--adapter-out-dir", default=None,
                         help="where to save the final policy adapter; defaults to <out-dir>/dpo_adapter. "
                              "Set this to a distinct path (e.g. outputs/dpo_adapter_v2) to keep a prior "
                              "run's adapter intact instead of overwriting it.")
    parser.add_argument("--log-path", default="logs/experiment_log.json")
    parser.add_argument("--log-history-path", default="logs/dpo_trainer_log_history.json")
    args = parser.parse_args()

    cfg = load_config(args.dpo_config)
    with open(args.qlora_config) as f:
        qlora_cfg = json.load(f)

    check_hardware(resolve_precision(cfg["training"]))

    model, tokenizer = load_policy_model(cfg, qlora_cfg)

    dataset = load_dataset("json", data_files=args.data)["train"]
    dataset = dataset.map(lambda row: {"prompt": build_chat_prompt(tokenizer, row["prompt"])})

    dpo_args = DPOConfig(
        output_dir=os.path.join(args.out_dir, "dpo_ckpt"),
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        beta=cfg["beta"],
        max_length=cfg["max_length"],
        optim=cfg["training"]["optim"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"].get("fp16", False),
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["training"]["gradient_checkpointing"] else None,
        logging_steps=10,
        report_to=[],
        seed=cfg["training"]["seed"],
    )

    step_timer = StepTimingCallback()
    memory_monitor = GpuMemoryMonitorCallback(check_every_n_steps=10, ceiling_gb=14.0)

    def do_train():
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[step_timer, memory_monitor],
        )
        # TRL's constructor above also casts trainable ("default") params to bfloat16,
        # which this T4 + fp16 combination can't back-propagate through (GradScaler
        # can't unscale BF16 gradients) -- restore default to an FP32 copy of the
        # still-FP32 "ref" before verifying or training.
        restore_fp32_policy_for_fp16_t4(model, cfg)
        # TRL creates the "ref" adapter inside the constructor above -- verify now,
        # before any optimizer step can make "default" and "ref" diverge.
        verify_reference_setup(model)
        trainer.train()
        return trainer

    run_config = {**cfg["training"], "beta": cfg["beta"], "max_length": cfg["max_length"]}
    trainer, record = profile_run("dpo_training", do_train, config=run_config, step_timer=step_timer, log_path=args.log_path)

    save_and_print_log_history(trainer, args.log_history_path)

    adapter_out = args.adapter_out_dir or os.path.join(args.out_dir, "dpo_adapter")
    model.save_pretrained(adapter_out, selected_adapters=["default"])
    tokenizer.save_pretrained(adapter_out)
    print(f"Saved DPO policy adapter (default only, not ref) to {adapter_out}")


if __name__ == "__main__":
    main()
