# Week 3 Adapter Sprint — Municipal 311 Assistant (QLoRA + DPO, 15GB ceiling)

Fine-tunes a sub-1B open model to rewrite messy resident 311 complaints into structured dispatch
tickets, under a free-tier Colab T4 (~15GB) memory budget. QLoRA SFT, a deliberate OOM + fix, and
two DPO passes were all actually run in Colab; every number below is quoted from a tracked file in
`data/` or `logs/`, not estimated. **Headline result**: the first DPO pass (v1) trained only on
missing-address preference pairs and overgeneralized into "always output null" — it broke perfectly
correct addressed-ticket behavior that QLoRA SFT alone already had. A balanced v2 preference dataset
(missing-address *and* addressed-control pairs) fixed the regression without losing the
missing-address behavior. That failure-and-fix is the actual experimental finding of this project.

## 1. Model and environment

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`, Apache-2.0, 0.49B params, 24 layers. Chosen because it's
  already instruction-tuned (a meaningful baseline), small enough that 4-bit NF4 quantization leaves
  nearly all of the T4's memory for activations/optimizer state, and fully supported by the installed
  `transformers`/`peft`/`bitsandbytes`/`trl`.
- **GPU**: Tesla T4 (compute capability 7.5), confirmed via `src/resource_monitor.check_hardware()`
  at the start of every training/eval run. Real capacity observed during the OOM experiment: 14.56 GiB
  (see `logs/oom_traceback.txt`) — the "15GB ceiling" this project targets is a slight overstatement of
  the actual hardware.
- **Precision: FP16, not BF16.** The T4 has no native BF16 Tensor Core support (`torch.cuda.is_bf16_supported()`
  can return `True` on it — that check reflects emulated support, not native throughput). `configs/qlora_config.json`
  and `configs/dpo_config.json` set `bnb_4bit_compute_dtype: "float16"`, `bf16: false`, `fp16: true`.
  `check_hardware()` prints the GPU name, CUDA capability, `torch.cuda.is_bf16_supported()`, and the
  configured precision, and raises `SystemExit` if `bf16` is requested on a GPU that doesn't support it.
- **Package versions actually used**, captured where each mattered rather than in one preflight file:
  `transformers 5.16.1` (its `TrainingArguments` rejects `warmup_ratio`, uses `warmup_steps` instead —
  see `configs/qlora_config.json`'s `warmup_steps_status`), `trl 1.14.0` / `peft 0.20.0` (their exact
  `DPOTrainer`/`DPOConfig`/`PeftModel` behavior is documented inline in `src/train_dpo.py` and
  `configs/dpo_config.json`'s `reference_policy_design`). `torch 2.11.0+cu128`, `bitsandbytes 0.50.2`,
  `accelerate`, and `datasets` were confirmed via a live Colab preflight check during development but
  were not saved to a separate tracked file — if you need the exact pins, rerun the preflight block
  (`import torch, transformers, peft, trl, bitsandbytes, accelerate, datasets; print(...)`) before
  training.

## 2. Repo structure

```
311_adapter_sprint.ipynb   Colab walkthrough documenting the full pipeline, incl. the v1 failure and v2 fix
src/
  generate_instruction_data.py     synthetic complaint -> ticket generator (offline, no API calls)
  audit_instruction_data.py        SFT token-length audit; basis for max_seq_len=256
  generate_preference_data.py      SUPERSEDED mixed-corruption DPO generator -- reference only
  audit_preference_data.py         audit for the superseded design
  generate_address_dpo_data.py     DPO v1 generator -- 150 missing-address-only pairs
  audit_address_dpo_data.py        DPO v1 preference-dataset audit
  generate_address_dpo_data_v2.py  DPO v2 generator -- balanced 75 missing-address + 75 addressed-control pairs
  audit_address_dpo_data_v2.py     DPO v2 preference-dataset audit (incl. real token/max_length check)
  train_qlora.py                   QLoRA SFT + the deliberate OOM demonstration and its fix
  train_dpo.py                     DPO (used for both v1 and v2 runs, via --data/--adapter-out-dir)
  evaluate.py                      single-tag eval, and the deterministic 10-example comparison mode
  resource_monitor.py              peak memory (allocated + reserved), wall time, step time, GPU-memory callback
configs/
  qlora_config.json          LoRA/quantization/training hyperparameters (as actually run)
  dpo_config.json            DPO hyperparameters + reference-policy design notes (as actually run)
data/                        generated datasets and audit reports -- tracked, real, committed
logs/                        experiment logs, DPO trainer log histories, OOM traceback, comparison JSON -- tracked, real, committed
outputs/                     checkpoints and adapter weights -- gitignored; regenerated locally by running the scripts (see §8)
requirements.txt
```

## 3. SFT and QLoRA

**Data**: `src/generate_instruction_data.py` produced 360 synthetic complaint→ticket examples
(offline, template-based, no external API), split 280/30/50 train/val/test
(`data/sft_train.jsonl` / `sft_val.jsonl` / `sft_test.jsonl`, seed 42; see `data/generation_summary.json` —
13.6% no-address, 120/124/116 low/medium/high priority).

**Sequence length**: `max_seq_len=256` is not a guess. `src/audit_instruction_data.py` measured the
real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer against all 280 training examples
(`data/token_length_audit.json`): combined prompt+target length min 175 / median 193 / p90 204 /
p95 207 / p99 211.2 / max 214. 256 preserves every example's complete assistant target (0 truncated,
vs. 142/280 truncated at 192 and 280/280 at 128); 384/512 add memory cost with no data-preservation
benefit. This is recorded as `max_seq_len_status` in `configs/qlora_config.json`.

**Label integrity and prompt masking**: `src/train_qlora.py::build_sft_dataset()` masks every prompt
token with `-100` so loss is computed only on the assistant's JSON continuation.
`validate_labels()` then checks every training and validation example has at least one non-masked
label token and aborts before training otherwise — this is a pre-training validation gate, not
something that needs a separate log, since the real QLoRA run reached training (see below), which is
only possible if this check passed.

**LoRA configuration**: r=16, alpha=32, dropout=0.05, target modules `q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj` (`configs/qlora_config.json`). These names are verified against the
model's actual linear layers at load time (`verify_target_modules()`) rather than assumed. Base model:
4-bit NF4 with double quantization (`bnb_4bit_use_double_quant: true`).

**Real trainable-parameter count** (from `logs/experiment_log_qlora_final.json`):
**8,798,208 / 323,917,696 trainable (2.7162%)**.

**Real QLoRA training run** (`logs/experiment_log_qlora_final.json`, `stage: qlora_sft_safe_config`,
`status: success`):

| wall time | avg step time | peak allocated | peak reserved |
|---|---|---|---|
| 123.2 s | 2.2114 s | 3.393 GB | 8.867 GB |

3 epochs, batch size 4, gradient accumulation 4, `paged_adamw_8bit`, fp16, gradient checkpointing on
(`gradient_checkpointing_kwargs={"use_reentrant": False}`), `warmup_steps=2` (converted at runtime
from the originally-approved `warmup_ratio=0.03` — `transformers 5.16.1`'s `TrainingArguments` does
not expose `warmup_ratio`). **Final training loss is not present in a tracked file for this run** —
`train_qlora.py` does not save `trainer.state.log_history` the way `train_dpo.py` does, so no loss
curve was persisted; see the rubric audit below for this as a real, acknowledged gap.

## 4. Intentional OOM experiment

`src/train_qlora.py::run_deliberate_oom()` deliberately runs an unsafe configuration on a freshly
loaded, quantized base model — **micro-batch 32, sequence length 1024, gradient checkpointing off**
(`configs/qlora_config.json`'s `deliberate_oom` block) — to genuinely exceed the T4's memory via
activation memory scaling with `batch_size × seq_len² × num_layers`.

**Real failure** (`logs/experiment_log_oom_final.json`, `stage: deliberate_oom_attempt`,
`status: failed`, `exception_type: OutOfMemoryError`):

| wall time | peak allocated | peak reserved |
|---|---|---|
| 0.9 s | 12.855 GB | 13.883 GB |

The real traceback is saved at `logs/oom_traceback.txt`:
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 608.00 MiB. GPU 0 has a total capacity
of 14.56 GiB ... Including non-PyTorch memory, this process has 13.05 GiB memory in use.` — a genuine
CUDA allocator failure inside the LoRA `down_proj` matmul, not a simulated one.

**Recovery changed exactly two variables**, per the "cheapest fixes first" guidance:

1. micro-batch: 32 → 4
2. gradient checkpointing: off → on

**Sequence length stayed at 1024** — the same as the unsafe attempt, deliberately not also reduced,
so the recovery isolates which of the two cheap fixes actually resolves the OOM rather than changing
three variables at once. Before the recovery step, `run_deliberate_oom()` reloads an entirely fresh
quantized base model and a fresh LoRA adapter (`_fresh_quantized_base()` called again) — the recovery
never resumes the base model that `prepare_model_for_kbit_training`/`get_peft_model` already mutated
during the failed attempt.

**Real recovery result** (`logs/experiment_log_oom_final.json`, `stage: post_fix_step`,
`status: success`):

| wall time | peak allocated | peak reserved |
|---|---|---|
| 3.3 s | 11.1 GB | 11.927 GB |

## 5. DPO v1

**Data**: `src/generate_address_dpo_data.py` built 150 pairs, all missing-address complaints.
`chosen` preserves `address_or_null: null`; `rejected` is byte-for-byte identical except
`address_or_null` is set to a fabricated street. `src/audit_address_dpo_data.py`
(`data/dpo_pairs_audit.json`) verified every pair is identical outside `address_or_null` (0 key-order
violations, 0 non-address-field violations), 0 duplicate prompts, 0 overlap with `sft_train`/`sft_val`/`sft_test`,
0 invented addresses accidentally already present in their source complaint.

**Reference-policy setup**: the SFT adapter (`outputs/qlora_adapter`) loads once as `PeftModel`'s
normal `"default"` adapter. `train_dpo.py` passes `ref_model=None`; the installed `trl 1.14.0`
`DPOTrainer` then creates a second `"ref"` adapter automatically inside its own constructor and copies
every `.default.` parameter into the matching `.ref.` parameter — an exact frozen copy of the SFT
adapter, not the untouched raw base model. `verify_reference_setup()` checks this rather than
assuming it: both adapters exist, `"default"` is trainable, `"ref"` has zero trainable parameters, and
every corresponding tensor pair is an exact copy before training starts.

**Real compatibility failure and fix**: the first real DPO run crashed on its first optimizer step
with `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for
'BFloat16'`. Root cause (confirmed against the installed TRL source): the same constructor that
copies `"default"` into `"ref"` later casts every trainable (`"default"`) parameter to `bfloat16`;
`"ref"` stays frozen and FP32. With `fp16=True`, Accelerate's GradScaler can't unscale BF16 gradients
on this T4. `restore_fp32_policy_for_fp16_t4()` fixes this by replacing each trainable `.default.`
tensor with an FP32 clone of its still-FP32 `.ref.` counterpart (not a cast of the already-BF16
tensor, which would keep the rounding loss) — restoring the exact pre-cast SFT values in a dtype the
FP16 GradScaler supports.

**Real DPO v1 training run** (`logs/experiment_log_dpo_v1_final.json` + `logs/dpo_trainer_v1_log_history.json`):

| wall time (resource_monitor) | avg step time | peak allocated | peak reserved | TRL `train_runtime` | `rewards/margins` | `rewards/accuracies` |
|---|---|---|---|---|---|---|
| 58.6 s | 5.3578 s | 2.551 GB | 7.074 GB | 54.6821 s | 4.732575474580129 | 0.8933333333333333 |

(`beta=0.1`, `max_length=256`, batch size 2, gradient accumulation 8, 1 epoch, `lr=5e-5`, `paged_adamw_8bit`, fp16.)

**Real held-out failure** (`logs/eval_comparison_dpo_v1.json`, same deterministic 10 examples as
everything below): **`missing_address_accuracy = 1.0`, `addressed_control_accuracy = 0.0`**. DPO v1
output `address_or_null: null` on all 5 addressed-control examples, not just the 5 null-gold ones.
Because the entire 150-pair preference set only ever showed DPO "prefer null," it overgeneralized that
into an unconditional rule instead of the conditional one the task actually needs.

## 6. DPO v2

**Data**: `src/generate_address_dpo_data_v2.py` built a balanced 150-pair replacement — 75
missing-address pairs (same design as v1) plus 75 new addressed-control pairs, where `chosen`
preserves a real, complaint-stated street and `rejected` is the same ticket with `address_or_null` set
to `null` (the mirror-image corruption). Every complaint was freshly generated and checked for **zero
overlap** with `sft_train`/`sft_val`/`sft_test` **and** with DPO v1's own prompts
(`data/dpo_pairs_v2_summary.json`: `overlap_found: {sft_train: 0, sft_val: 0, sft_test: 0, dpo_v1: 0}`).

**Real structural audit** (`data/dpo_pairs_v2_audit.json`): 150 pairs (75/75), 0 duplicate prompts,
0 overlap on any axis, **0 key-order violations, 0 non-address-field violations** (every pair verified
identical outside `address_or_null`, regardless of which direction that field changes), 0 missing-address
prompts containing a leaked street, 0 addressed-control prompts missing their own gold street.

**Real Colab tokenizer audit** (same file, `prompt_tokens` / `prompt_plus_chosen_tokens` /
`prompt_plus_rejected_tokens` / `max_length_check`):

| prompt max | prompt+chosen max | prompt+rejected max | within `max_length=256`? |
|---|---|---|---|
| 156 | 215 | 218 | yes — `all_within_max_length: true`, 0 exceeding either direction |

**Same DPO training configuration as v1** (`logs/experiment_log_dpo_v2_final.json`'s `config` block is
identical to v1's: `beta=0.1`, `max_length=256`, batch size 2, gradient accumulation 8, 1 epoch,
`lr=5e-5`, `paged_adamw_8bit`, fp16, gradient checkpointing on, seed 42) — only the preference data
changed.

**Real DPO v2 training run** (`logs/experiment_log_dpo_v2_final.json` + `logs/dpo_trainer_v2_log_history.json`):

| wall time | avg step time | peak allocated | peak reserved | TRL `train_runtime` | `rewards/margins` | `rewards/accuracies` |
|---|---|---|---|---|---|---|
| 60.5 s | 5.4879 s | 2.557 GB | 8.055 GB | 56.0163 s | 2.24270688533783 | 0.88 |

## 7. Final deterministic held-out comparison

Same 10 held-out complaints for every model (`selected_test_indices: [2, 4, 8, 12, 20, 0, 1, 3, 5, 6]`
— 5 gold-null-address + 5 gold-addressed, selected deterministically before any inference; see
`src/evaluate.py::select_comparison_examples`). Source: `logs/eval_comparison_dpo_v1.json` (base,
QLoRA, DPO v1) and `logs/eval_comparison_dpo_v2.json` (base, QLoRA, DPO v2) — base and QLoRA numbers
are identical between the two files, confirming determinism.

| metric | Base | QLoRA | DPO v1 | DPO v2 |
|---|---|---|---|---|
| json_valid_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| field_complete_rate | 1.0 | 1.0 | 1.0 | 1.0 |
| category_accuracy | 0.0 | 1.0 | 1.0 | 1.0 |
| address_exact_match_accuracy | 0.4 | 1.0 | 0.5 | 1.0 |
| missing_address_accuracy (5 null) | 0.0 | 1.0 | 1.0 | 1.0 |
| hallucinated_address_count (5 null) | 5 | 0 | 0 | 0 |
| addressed_control_accuracy (5 addressed) | 0.8 | 1.0 | **0.0** | 1.0 |

**Interpretation** — read this carefully, the result is narrower than it might look:

- QLoRA SFT alone already reaches perfect scores on every metric on this 10-example set. DPO v2 also
  reaches perfect scores. **This is not evidence that DPO v2 outperformed QLoRA** — there's no headroom
  left for DPO to show improvement over on this small sample; QLoRA got there first.
- The real, demonstrated result is: **DPO v1 introduced a regression** (`addressed_control_accuracy`
  0.0, down from QLoRA's 1.0) by overgeneralizing a 100%-missing-address preference set into "always
  output null," and **the balanced DPO v2 preference set eliminated that regression while preserving
  the missing-address behavior** (`missing_address_accuracy` stayed 1.0 in both v1 and v2). The
  finding is about preference-dataset design causing and then fixing a specific behavioral failure,
  not about DPO improving on QLoRA.
- 10 examples is a small, fixed diagnostic set — it's designed to directly probe one behavior
  (missing- vs. addressed-address handling), not to support a general claim about overall model quality.

## 8. AI collaboration

This project was built through iterative, Claude-assisted development with the user driving every
experimental decision and every real Colab run. What Claude did was code, design, and diagnostic work
against evidence the user supplied back from actual runs — not independent experimentation. Concretely,
Claude-assisted work included: the synthetic instruction-dataset generator and its later revision after
manual review found template/priority mismatches; the token-length audit script (and revising it twice
as new requirements — per-candidate truncation, then real DPO prompt/combined lengths — were specified);
the FP16-not-BF16 decision and the `check_hardware()` guard for the T4's lack of native BF16; a series
of real-API-verified compatibility fixes (`eval_strategy` vs `evaluation_strategy`, `dtype` vs
deprecated `torch_dtype`, `gradient_checkpointing_kwargs`, and the `warmup_ratio`→`warmup_steps`
conversion after the user reported the exact installed `transformers 5.16.1` behavior); the
`GpuMemoryMonitorCallback` memory-ceiling instrumentation; the deliberate-OOM/two-cheapest-fixes
recovery design; investigating and implementing the TRL `"default"`/`"ref"` automatic reference-adapter
behavior after the user reported the exact installed `trl`/`peft` API surface; the FP32-policy
restoration fix after the user reported the real `NotImplementedError` from a live Colab run; the
deterministic 10-example held-out evaluation design (fixed selection rule, sequential model
load/unload); diagnosing the DPO v1 addressed-control regression from the real comparison JSON the
user reported back; and designing the balanced DPO v2 preference dataset as the fix.

**What Claude did not do**: run any Colab cell, observe any GPU, or produce any of the numbers quoted
in §§3–7. Every metric, traceback, and reward value in this document was reported back by the user
from an actual Colab execution and is stored in the tracked files under `data/` and `logs/` cited
throughout. Where a number could not be found in a tracked file (the QLoRA final training loss — see
§3), it is reported as missing here, not filled in from memory of an earlier chat message.

## How to run / reproducibility

Open `311_adapter_sprint.ipynb` in Colab (T4 runtime) to follow the pipeline in the order it was
actually run. Every script writes its real output to `data/` or `logs/`, which are committed — that's
the evidence this document cites. `outputs/` (model checkpoints and adapter weights, e.g.
`outputs/qlora_adapter`, `outputs/dpo_adapter`, `outputs/dpo_adapter_v2`) is **gitignored** — those
directories are not in GitHub and must be regenerated locally by actually running `train_qlora.py`
and `train_dpo.py`; they are not something a reader can fetch from the repo.

```bash
# SFT data + audit
python src/generate_instruction_data.py --n 360 --seed 42 --out-dir data
python src/audit_instruction_data.py --data data/sft_train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct

# QLoRA SFT, then the deliberate OOM + fix
python src/train_qlora.py --config configs/qlora_config.json --data-dir data --out-dir outputs --log-path logs/experiment_log.json
python src/train_qlora.py --config configs/qlora_config.json --data-dir data --out-dir outputs --log-path logs/experiment_log.json --run-oom-demo

# DPO v1 (missing-address only) -- reproduces the overgeneralization failure
python src/generate_address_dpo_data.py --seed 101
python src/audit_address_dpo_data.py --pairs data/dpo_pairs.jsonl
python src/train_dpo.py --dpo-config configs/dpo_config.json --qlora-config configs/qlora_config.json \
  --data data/dpo_pairs.jsonl --out-dir outputs --log-path logs/experiment_log.json

# DPO v2 (balanced) -- the fix, saved to a distinct adapter path so v1 evidence is never overwritten
python src/generate_address_dpo_data_v2.py --seed 42
python src/audit_address_dpo_data_v2.py --pairs data/dpo_pairs_v2.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
python src/train_dpo.py --dpo-config configs/dpo_config.json --qlora-config configs/qlora_config.json \
  --data data/dpo_pairs_v2.jsonl --out-dir outputs --adapter-out-dir outputs/dpo_adapter_v2 \
  --log-path logs/experiment_log_v2.json

# Deterministic 10-example comparison, base vs QLoRA vs a given DPO adapter
python src/evaluate.py --comparison --dpo-adapter outputs/dpo_adapter --out-dir outputs/eval
python src/evaluate.py --comparison --dpo-adapter outputs/dpo_adapter_v2 --out-dir outputs/eval_v2
```

## .gitignore strategy

`outputs/` (checkpoints, adapter weight binaries) is excluded — large, regenerable, and not needed to
verify the results, since every real measurement they'd produce is already captured in `data/` and
`logs/`. `data/` and `logs/` are committed in full — they hold the actual reproducibility evidence
(generated JSONL datasets, audit reports, experiment logs, DPO trainer log histories, the OOM
traceback, and the deterministic comparison JSON) this document is built from.
