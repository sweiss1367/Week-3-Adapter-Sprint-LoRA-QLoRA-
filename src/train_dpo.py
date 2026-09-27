"""
DPO on top of the QLoRA-tuned model.

Reference-policy design (see configs/dpo_config.json "reference_policy_design" for the
full rationale): the reference must represent the instruction-tuned adapter state
immediately before DPO, not the untouched raw base model. To get that without loading a
second full copy of the base weights, we load the SAME trained SFT adapter twice onto one
shared 4-bit base, under two adapter names:

    - "policy"    -- trainable, this is what DPO updates
    - "reference" -- frozen immediately after loading (requires_grad=False, eval mode)

DPOTrainer is then given `ref_model=None` with `model_adapter_name="policy"` and
`ref_adapter_name="reference"` (DPOConfig fields), so TRL computes reference logprobs by
switching the shared base to the frozen adapter rather than running a second model.

This mechanism depends on the installed TRL/PEFT versions supporting model_adapter_name /
ref_adapter_name on DPOConfig -- verify_reference_setup() below checks this against the
actually-installed package before training starts and fails loudly if it's missing,
rather than silently falling back to an incorrect reference.

NOT executed as part of building this repo structure -- requires a CUDA GPU. Run in Colab.
"""
import argparse
import inspect
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from generate_instruction_data import build_chat_prompt
from resource_monitor import StepTimingCallback, profile_run


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("max_length") is None or cfg.get("max_prompt_length") is None:
        raise SystemExit(
            "configs/dpo_config.json has max_length/max_prompt_length: null. These must "
            "be derived from the approved configs/qlora_config.json max_seq_len -- see "
            "README for how to set them. Do not guess a number here."
        )
    return cfg


def verify_reference_setup():
    """Confirm the installed DPOConfig actually supports model_adapter_name/ref_adapter_name
    before relying on it -- this is exactly the assumption the earlier plan got wrong by
    guessing at TRL behavior instead of checking the installed version."""
    sig = inspect.signature(DPOConfig.__init__)
    missing = [p for p in ("model_adapter_name", "ref_adapter_name") if p not in sig.parameters]
    if missing:
        raise SystemExit(
            f"Installed trl.DPOConfig is missing parameter(s) {missing}. The frozen-adapter "
            "reference design in this script requires them. Check `pip show trl` in this "
            "Colab runtime and either upgrade trl or adapt this script to that version's "
            "documented way of using a PEFT adapter as a frozen reference -- do not silently "
            "fall back to ref_model=None with a single adapter (that reference would be the "
            "raw base model, not the instruction-tuned state)."
        )
    print("Verified: installed trl.DPOConfig supports model_adapter_name / ref_adapter_name.")


def build_bnb_config(qlora_cfg: dict) -> BitsAndBytesConfig:
    q = qlora_cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_dual_adapter_model(cfg: dict, qlora_cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=build_bnb_config(qlora_cfg),
        device_map={"": 0},
        torch_dtype=getattr(torch, qlora_cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )

    adapter_path = cfg["sft_adapter_path"]
    policy_name = cfg["policy_adapter_name"]
    reference_name = cfg["reference_adapter_name"]

    model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=policy_name, is_trainable=True)
    model.load_adapter(adapter_path, adapter_name=reference_name, is_trainable=False)

    # Freeze every parameter that belongs to the reference adapter.
    for name, param in model.named_parameters():
        if f".{reference_name}." in name or name.endswith(f".{reference_name}"):
            param.requires_grad = False

    model.set_adapter(policy_name)
    return model, tokenizer


def verify_trainable_vs_frozen(model, policy_name: str, reference_name: str):
    """Explicit proof, not an assumption: which parameters are trainable (policy) and
    which are frozen (reference), printed before training starts."""
    policy_trainable, policy_total = 0, 0
    reference_trainable, reference_total = 0, 0
    other_trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if f".{policy_name}." in name:
            policy_total += n
            policy_trainable += n if param.requires_grad else 0
        elif f".{reference_name}." in name:
            reference_total += n
            reference_trainable += n if param.requires_grad else 0
        elif param.requires_grad:
            other_trainable += n

    report = {
        "policy_adapter": {"total_params": policy_total, "trainable_params": policy_trainable},
        "reference_adapter": {"total_params": reference_total, "trainable_params": reference_trainable},
        "other_trainable_params_outside_named_adapters": other_trainable,
    }
    print(json.dumps(report, indent=2))

    if reference_trainable != 0:
        raise SystemExit(
            f"Reference adapter '{reference_name}' has {reference_trainable} trainable "
            "parameters -- it must be fully frozen. Aborting before training."
        )
    if policy_trainable == 0:
        raise SystemExit(f"Policy adapter '{policy_name}' has zero trainable parameters -- nothing would train.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo-config", default="configs/dpo_config.json")
    parser.add_argument("--qlora-config", default="configs/qlora_config.json")
    parser.add_argument("--data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--log-path", default="logs/experiment_log.json")
    args = parser.parse_args()

    verify_reference_setup()

    cfg = load_config(args.dpo_config)
    with open(args.qlora_config) as f:
        qlora_cfg = json.load(f)

    model, tokenizer = load_dual_adapter_model(cfg, qlora_cfg)
    param_report = verify_trainable_vs_frozen(model, cfg["policy_adapter_name"], cfg["reference_adapter_name"])

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
        max_prompt_length=cfg["max_prompt_length"],
        optim=cfg["training"]["optim"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        model_adapter_name=cfg["policy_adapter_name"],
        ref_adapter_name=cfg["reference_adapter_name"],
        logging_steps=10,
        report_to=[],
        seed=cfg["training"]["seed"],
    )

    step_timer = StepTimingCallback()

    def do_train():
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[step_timer],
        )
        trainer.train()
        return trainer

    run_config = {**cfg["training"], "beta": cfg["beta"], "max_length": cfg["max_length"], **param_report}
    profile_run("dpo_training", do_train, config=run_config, step_timer=step_timer, log_path=args.log_path)

    adapter_out = os.path.join(args.out_dir, "dpo_adapter")
    model.save_pretrained(adapter_out)
    print(f"Saved DPO policy adapter to {adapter_out}")


if __name__ == "__main__":
    main()
