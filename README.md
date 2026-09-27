# Week 3 Adapter Sprint — Municipal 311 Assistant (QLoRA + DPO, 15GB ceiling)

Fine-tunes a sub-1B open model to rewrite messy resident 311 complaints into structured dispatch
tickets, under a free-tier Colab T4 (~15GB) memory budget.

## Model

`Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 0.49B params, 24 layers). Chosen because it's already
instruction-tuned (a meaningful baseline), small enough that 4-bit NF4 quantization leaves nearly
all of the T4's memory for activations/optimizer state, and fully supported by current
`transformers`/`peft`/`bitsandbytes`/`trl`.

## Precision: FP16, not BF16

The NVIDIA T4 has no native BF16 support. `configs/qlora_config.json` and `configs/dpo_config.json`
set `bnb_4bit_compute_dtype: "float16"`, `bf16: false`, `fp16: true`. `src/resource_monitor.py`'s
`check_hardware()` is called at the start of `train_qlora.py`, `train_dpo.py`, and
`evaluate.py::load_model_for_eval()`; it prints the detected GPU name, its CUDA compute capability,
`torch.cuda.is_bf16_supported()`, and the configured precision, and raises `SystemExit` if the
configured precision is `bf16` on a GPU that doesn't support it.

## Repo structure

```
311_adapter_sprint.ipynb   Colab walkthrough / orchestrator -- imports and calls src/
src/
  generate_instruction_data.py  synthetic complaint -> ticket generator (offline, no API calls)
  audit_instruction_data.py     token-length audit; run BEFORE choosing max_seq_len
  generate_preference_data.py   chosen/rejected pairs for DPO, incl. address-hallucination pairs
  audit_preference_data.py      preference-dataset audit; run BEFORE DPO training
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
- **Sequence length: APPROVED, `max_seq_len: 256`**. Measured with the real
  `Qwen/Qwen2.5-0.5B-Instruct` tokenizer in Colab against `data/sft_train.jsonl` (280 examples;
  see `data/token_length_audit.json`): combined prompt+target length min 175 / median 193 / p90 204 /
  p95 207 / p99 211.2 / max 214. 256 preserves all 280 training examples and their complete
  assistant target (0 truncated at 256, vs. 142/280 at 192 and 280/280 at 128); 384 and 512 add
  activation-memory cost with no data-preservation benefit over 256. Human-approved before being
  written into `configs/qlora_config.json`.
- **DPO reference policy**: the reference must represent the instruction-tuned adapter state
  immediately before DPO, not the untouched base model. `src/train_dpo.py` loads the SFT adapter
  twice onto one shared 4-bit base — `policy` (trainable) and `reference` (frozen right after
  loading) — and passes `ref_model=None` with `model_adapter_name="policy"` /
  `ref_adapter_name="reference"` to `DPOConfig`, so TRL switches adapters on one base instead of
  loading a second full model copy. `verify_reference_setup()` checks the installed TRL version
  actually supports these fields before training starts, and `verify_trainable_vs_frozen()` prints
  which parameters are trainable vs. frozen so this isn't taken on faith.
- **Failed-run logging**: `resource_monitor.profile_run()` wraps every run in a try/except. A run
  that raises (e.g. a genuine CUDA OOM) still gets a record written before the exception is
  re-raised: `status: "failed"`, the exception type, peak allocated/reserved memory and elapsed wall
  time at the moment of failure, and the run's config. `avg_step_time_sec` stays `null` if no
  optimizer step completed. The actual traceback is written separately to `logs/oom_traceback.txt`
  by the caller (`train_qlora.py::run_deliberate_oom`), since `profile_run` only owns numeric
  measurements.
- **OOM/recovery isolation**: `run_deliberate_oom()` loads a fresh quantized base model and a fresh
  LoRA adapter for the unsafe attempt. After a genuine OOM (or any exception), it deletes the failed
  model/optimizer/batch, runs `gc.collect()` and `torch.cuda.empty_cache()`, then loads an entirely
  new base model and a new adapter for the safe rerun — the recovery run never resumes a base model
  that `prepare_model_for_kbit_training`/`get_peft_model` already mutated during the failed attempt.
- **Missing-address behavior**: ~15% of generated complaints never mention a street
  (`address_or_null: null` in the ground truth). `generate_preference_data.py` gives **every** such
  training example (42 of 280, not a sampled subset) a `hallucinate_address` rejection (chosen
  preserves `null`, rejected invents a street), so the DPO signal for "don't invent an address" is
  fully represented rather than diluted. `evaluate.py` reports `address_null_precision` /
  `address_null_recall` specifically for this.
- **Preference dataset audit** (`src/audit_preference_data.py`, run against `data/dpo_pairs.jsonl`
  before DPO training): 165 pairs, 42 (25.45%) `hallucinate_address`, 0 leakage into
  `data/sft_test.jsonl`, 0 instances of an invented address accidentally already appearing in its
  source complaint. One real finding worth flagging rather than silently accepting: `malformed_json`
  and `drop_field` rejections are *always* shorter than `chosen` (by a consistent 1 char and ~35
  chars respectively), and `hallucinate_address` rejections are consistently longer (~7 chars) --
  three of five corruption types have a same-sign length delta across every example of that type,
  which is a length-based shortcut DPO could exploit instead of learning the actual content
  distinction. Not fixed in this pass (per instruction, this dataset is for review, not retraining
  yet) -- worth revisiting if evaluation later shows the DPO model discriminating on response length
  rather than on content.

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

- **Already run** (CPU-only, no GPU needed, outputs committed):
  - `generate_instruction_data.py` — see `data/generation_summary.json` (360 examples, 13.6%
    no-address, 120/124/116 low/medium/high).
  - The real token-length audit (`src/audit_instruction_data.py`), run in Colab with the actual
    `Qwen/Qwen2.5-0.5B-Instruct` tokenizer — see `data/token_length_audit.json`. Human-approved;
    `max_seq_len: 256` is now set in `configs/qlora_config.json`.
  - `generate_preference_data.py`, regenerated against the current (approved) `data/sft_train.jsonl`
    — see `data/dpo_pairs.jsonl` / `data/dpo_pairs_summary.json`.
  - `src/audit_preference_data.py` against the regenerated preference dataset — see
    `data/dpo_pairs_audit.json`. Token-length statistics in that file are character-length only
    (this container can't reach the Hub for the real tokenizer); rerun in Colab for real token
    counts on the preference pairs specifically.
- **Requires Colab** (GPU, and/or Hugging Face Hub access this dev container's network policy
  denies): QLoRA training, the deliberate OOM, DPO, and evaluation. None of these have been executed
  yet, and `logs/experiment_log.json` is an empty list until they are — this repo does not contain
  invented measurements. `configs/dpo_config.json`'s `max_length`/`max_prompt_length` are still
  `null`, pending approval to derive them from `max_seq_len`.

### Note: the instruction dataset was regenerated after manual review

The first generated dataset paired template selection with priority independently, so a
high-urgency opener (e.g. "URGENT!!!") could land on a low-priority example ("...whenever someone
gets a chance"), and issue phrasing was written as mixed noun/verb clauses that read awkwardly when
substituted into a shared template (e.g. "there's a my trash wasnt picked up"). That dataset was
replaced: `src/generate_instruction_data.py` now draws its opener and tone phrasing from the same
priority tier (so low-priority text never contains "URGENT"/"ASAP", and high-priority text always
carries urgency language), uses consistent noun-phrase issue descriptions, and composes complaints
from more independently-varying parts (opener, issue phrase, location phrase, tone, sentence order)
so 360 examples don't read as five templates with swapped fields. `data/sft_*.jsonl` reflect this
regenerated version; the previous files were overwritten, not kept alongside it.

## .gitignore strategy

`outputs/` (checkpoints, adapter weight binaries) is excluded — large and regenerable. `data/` and
`logs/` are committed — they hold the small reproducibility evidence (generated JSONL datasets,
audit reports, the experiment log, the OOM traceback, and per-example eval JSON) that the rubric's
documentation/reproducibility criterion actually asks for.
