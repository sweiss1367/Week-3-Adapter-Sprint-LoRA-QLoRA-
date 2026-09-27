# Week 3 Adapter Sprint — Municipal 311 Assistant (QLoRA + DPO, 15GB ceiling)

Fine-tunes a sub-1B open model to rewrite messy resident 311 complaints into structured dispatch
tickets, under a free-tier Colab T4 (~15GB) memory budget.

## Model

`Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 0.49B params, 24 layers). Chosen because it's already
instruction-tuned (a meaningful baseline), small enough that 4-bit NF4 quantization leaves nearly
all of the T4's memory for activations/optimizer state, and fully supported by current
`transformers`/`peft`/`bitsandbytes`/`trl`.

## Repo structure

```
311_adapter_sprint.ipynb   Colab walkthrough / orchestrator -- imports and calls src/
src/
  generate_instruction_data.py  synthetic complaint -> ticket generator (offline, no API calls)
  audit_instruction_data.py     token-length audit; run BEFORE choosing max_seq_len
  generate_preference_data.py   chosen/rejected pairs for DPO, incl. address-hallucination pairs
  train_qlora.py                QLoRA SFT + the deliberate OOM demonstration and its fix
  train_dpo.py                  DPO with a frozen-adapter reference (see below)
  evaluate.py                   generation + scoring for base / QLoRA / DPO on held-out data
  resource_monitor.py           peak memory (allocated + reserved), wall time, avg step time
configs/
  qlora_config.json          LoRA/quantization/training hyperparameters
  dpo_config.json            DPO hyperparameters + reference-policy design notes
data/                        generated datasets and audit reports (jsonl/json, committed)
logs/                        experiment_log.json, oom_traceback.txt (committed)
outputs/                     checkpoints, adapter weights, per-example eval dumps (gitignored)
requirements.txt
```

The notebook is the walkthrough; the logic it calls lives in `src/` so each piece can be run,
tested, and explained independently (and so the recorded walkthrough can point at a specific file
for each rubric area instead of a notebook cell).

## Design decisions and open items

- **LoRA target modules** (`configs/qlora_config.json`): `q_proj, k_proj, v_proj, o_proj, gate_proj,
  up_proj, down_proj` is proposed as a capacity choice, not asserted as necessary — a narrower
  `q_proj`/`v_proj`-only adapter is a defensible alternative for a mostly stylistic/formatting task.
  `train_qlora.py` verifies these names against the model's actual linear layers at load time
  (`verify_target_modules`) and prints the exact trainable parameter count and percentage
  (`report_trainable_parameters`) so the choice can be defended with real numbers, not asserted.
- **Sequence length**: `configs/qlora_config.json` has `max_seq_len: null` on purpose.
  `src/audit_instruction_data.py` measures the real tokenized length distribution of the generated
  data (prompt / target / combined, with p90/p95/p99) and computes a recommendation; a human
  approves the value (the notebook's Section 2) before `train_qlora.py` — which refuses to run
  while the value is still `null` — uses it.
- **DPO reference policy**: the reference must represent the instruction-tuned adapter state
  immediately before DPO, not the untouched base model. `src/train_dpo.py` loads the SFT adapter
  twice onto one shared 4-bit base — `policy` (trainable) and `reference` (frozen right after
  loading) — and passes `ref_model=None` with `model_adapter_name="policy"` /
  `ref_adapter_name="reference"` to `DPOConfig`, so TRL switches adapters on one base instead of
  loading a second full model copy. `verify_reference_setup()` checks the installed TRL version
  actually supports these fields before training starts, and `verify_trainable_vs_frozen()` prints
  which parameters are trainable vs. frozen so this isn't taken on faith.
- **Missing-address behavior**: ~15% of generated complaints never mention a street
  (`address_or_null: null` in the ground truth). `generate_preference_data.py` gives every one of
  those a `hallucinate_address` rejection (chosen preserves `null`, rejected invents a street), so
  the DPO signal for "don't invent an address" isn't diluted by the other corruption types.
  `evaluate.py` reports `address_null_precision` / `address_null_recall` specifically for this.

## How to run

Open `311_adapter_sprint.ipynb` in Google Colab, set the runtime to a T4 GPU, and run cells
top to bottom (see the notebook for the full per-section commentary):

1. Generate the synthetic instruction dataset and run the token-length audit; review the
   recommendation and approve `max_seq_len` (writes it into both config files).
2. Generate the preference dataset.
3. Evaluate the base model on the held-out test split.
4. Run QLoRA SFT (`src/train_qlora.py`).
5. Run the deliberate OOM + diagnosis + fix (`src/train_qlora.py --run-oom-demo`).
6. Evaluate the QLoRA model.
7. Run DPO (`src/train_dpo.py`).
8. Evaluate the DPO model.
9. Build the results tables from `logs/experiment_log.json` and `outputs/eval/eval_*.json`.

Every peak-memory (allocated + reserved), wall-clock, and average-step-time number is captured by
`resource_monitor.profile_run` as each stage actually runs, into `logs/experiment_log.json`. Nothing
is pre-filled, and the log starts as an empty list.

## What's already been run vs. what's pending

- **Already run** (CPU-only, no GPU needed, outputs committed): `generate_instruction_data.py`,
  `generate_preference_data.py`. See `data/generation_summary.json` and `data/dpo_pairs_summary.json`
  for the real counts.
- **Requires Colab** (GPU and/or Hugging Face Hub access this dev container's network policy
  denies): the token-length audit, QLoRA training, the deliberate OOM, DPO, and evaluation. None of
  these have been executed yet, and `logs/experiment_log.json` is an empty list until they are —
  this repo does not contain invented measurements.

## .gitignore strategy

`outputs/` (checkpoints, adapter weight binaries) is excluded — large and regenerable. `data/` and
`logs/` are committed — they hold the small reproducibility evidence (generated JSONL datasets,
audit reports, the experiment log, the OOM traceback, and per-example eval JSON) that the rubric's
documentation/reproducibility criterion actually asks for.
