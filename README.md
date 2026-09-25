# Week 3 Adapter Sprint — Municipal 311 Assistant (QLoRA + DPO, 15GB ceiling)

Fine-tunes a sub-1B open model to rewrite messy resident 311 complaints into structured dispatch
tickets, under a free-tier Colab T4 (~15GB) memory budget.

## Model

`Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 0.5B params). Chosen because it's already
instruction-tuned (a meaningful baseline), small enough that 4-bit NF4 quantization leaves nearly
all of the T4's memory for activations/optimizer state, and fully supported by current
`transformers`/`peft`/`bitsandbytes`/`trl`.

## How to run

Open `311_adapter_sprint.ipynb` in Google Colab, set the runtime to a T4 GPU, and run all cells
top to bottom. It:

1. Generates ~360 synthetic complaint/ticket pairs from templates (offline, no external API), and a
   preference dataset for DPO.
2. Evaluates the base model on a held-out test split.
3. Fine-tunes with QLoRA (4-bit NF4 + LoRA, gradient checkpointing, `paged_adamw_8bit`).
4. Deliberately triggers and captures a real `CUDA OutOfMemoryError` with an unsafe batch/seq-len/
   checkpointing config, diagnoses the cause, then reverts to the safe config and reruns.
5. Runs a short DPO pass on top of the QLoRA model, reusing the same base weights as the frozen
   reference (`ref_model=None` with a `PeftModel`) instead of loading a second copy.
6. Re-evaluates the same held-out complaints on base / QLoRA / QLoRA+DPO and reports JSON validity,
   field completeness, and category accuracy.

All memory (`torch.cuda.max_memory_allocated`) and timing numbers are captured by the notebook's
`profile_run` helper into `outputs/results.json` as each stage actually runs — none are pre-filled.
The comparison table under "Results" is likewise built from `outputs/eval_*.json`, which only exist
once you've run the corresponding cell.

## Repo contents

- `311_adapter_sprint.ipynb` — the full pipeline (data generation through evaluation).
- `outputs/` — created on first run; holds datasets, adapters, logs, and results (not committed).

## Notes

- This was authored and syntax-checked without GPU access, so no measurements from the notebook are
  reported here — running it in Colab produces the real numbers.
- The deliberate-OOM cell's exact batch size / sequence length may need to be raised further
  depending on the free-tier GPU actually allocated; the goal is a genuine `torch.cuda.OutOfMemoryError`,
  not those specific numbers.
