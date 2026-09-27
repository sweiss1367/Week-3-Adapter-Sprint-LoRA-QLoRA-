"""
Shared instrumentation for every GPU run in this project.

Records, per run: peak allocated GPU memory, peak reserved GPU memory, total wall-clock
time, average per-step time, and the exact run configuration -- and nothing else. This
module never invents or pre-fills a measurement; every field is either a number this
process actually measured on this machine, or `None` if the run hasn't happened yet
(e.g. no CUDA device present, as in local development off of Colab).
"""
import gc
import json
import os
import time
from typing import Callable, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import TrainerCallback
except ImportError:  # allows importing this module before transformers is installed
    class TrainerCallback:
        pass


class StepTimingCallback(TrainerCallback):
    """Attach to a Trainer's `callbacks=[...]` to record wall-clock time per optimizer
    step, so we can report average step time in addition to total wall-clock time."""

    def __init__(self):
        self.step_times = []
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self.step_times.append(time.time() - self._step_start)
            self._step_start = None

    @property
    def avg_step_time_sec(self):
        if not self.step_times:
            return None
        return sum(self.step_times) / len(self.step_times)


class GpuMemoryMonitorCallback(TrainerCallback):
    """Attach to a Trainer's `callbacks=[...]` to print peak GPU memory periodically
    during training and enforce a hard ceiling on peak allocated memory.

    Reads CUDA's high-water marks (max_memory_allocated/max_memory_reserved) directly on
    every optimizer step rather than tracking separate state, and never resets them --
    profile_run() already resets peak stats once at the start of the measured run, so
    this callback's readings stay consistent with the final peak_allocated_gb/
    peak_reserved_gb that profile_run() reports for the whole run.

    Reusable as-is for DPO training, not just QLoRA SFT.
    """

    def __init__(self, check_every_n_steps: int = 10, ceiling_gb: float = 14.0):
        self.check_every_n_steps = check_every_n_steps
        self.ceiling_gb = ceiling_gb

    def on_step_end(self, args, state, control, **kwargs):
        if not _cuda_available():
            return

        step = state.global_step
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

        if step % self.check_every_n_steps == 0:
            print(
                f"[gpu_memory] step={step} "
                f"peak_allocated_gb={round(peak_allocated_gb, 3)} "
                f"peak_reserved_gb={round(peak_reserved_gb, 3)}"
            )

        if peak_allocated_gb > self.ceiling_gb:
            raise RuntimeError(
                f"Peak allocated GPU memory {round(peak_allocated_gb, 3)}GB exceeded the "
                f"{self.ceiling_gb}GB ceiling at optimizer step {step}."
            )


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def reset_cuda_stats():
    if _cuda_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_stats() -> dict:
    if not _cuda_available():
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


def _avg_step_time(step_timer: Optional[StepTimingCallback]):
    if step_timer is not None and step_timer.avg_step_time_sec is not None:
        return round(step_timer.avg_step_time_sec, 4)
    return None


def profile_run(
    stage_name: str,
    fn: Callable,
    config: Optional[dict] = None,
    step_timer: Optional[StepTimingCallback] = None,
    log_path: str = "logs/experiment_log.json",
):
    """
    Run fn(), measure it, log the record, and return (fn's return value, record).

    `config` should be the actual dict of hyperparameters used for this call (batch size,
    seq len, lora rank, etc.) so the log is self-describing without cross-referencing code.
    `step_timer` should be the same StepTimingCallback instance passed into the Trainer
    that fn() drives, if any, so average step time gets captured alongside peak memory.

    A run that raises (e.g. a genuine CUDA OOM) is logged too, before the exception is
    re-raised: status="failed", the exception type, peak allocated/reserved memory at the
    moment of failure, and the elapsed wall-clock time up to that point. avg_step_time_sec
    stays null if no optimizer step completed before the failure -- never fabricated. The
    caller is still responsible for writing the actual traceback to its own log file if it
    wants one (this function only records numbers, not the traceback text).
    """
    reset_cuda_stats()
    t0 = time.time()
    try:
        result = fn()
    except Exception as exc:
        wall_time_sec = round(time.time() - t0, 1)
        record = {
            "stage": stage_name,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "wall_time_sec": wall_time_sec,
            "avg_step_time_sec": _avg_step_time(step_timer),
            "config": config or {},
        }
        record.update(cuda_peak_stats())
        log_experiment(record, log_path)
        print(f"[{stage_name}] FAILED: {record}")
        raise

    if _cuda_available():
        torch.cuda.synchronize()
    wall_time_sec = round(time.time() - t0, 1)

    record = {
        "stage": stage_name,
        "status": "success",
        "wall_time_sec": wall_time_sec,
        "avg_step_time_sec": _avg_step_time(step_timer),
        "config": config or {},
    }
    record.update(cuda_peak_stats())
    log_experiment(record, log_path)
    print(f"[{stage_name}] {record}")
    return result, record


def resolve_precision(training_cfg: dict) -> str:
    """fp16 takes priority if both are somehow set; falls back to fp32 if neither is."""
    if training_cfg.get("fp16"):
        return "fp16"
    if training_cfg.get("bf16"):
        return "bf16"
    return "fp32"


def check_hardware(expected_precision: str):
    """
    Prints the detected GPU, its CUDA compute capability, whether this GPU supports
    bf16 (torch.cuda.is_bf16_supported()), and the precision this run is configured to
    use. Fails clearly (SystemExit) if the configured precision cannot run on the
    detected GPU -- in particular, the T4 required by this assignment has no native
    bf16 support, so a bf16-configured run must not silently proceed on it.
    """
    if not _cuda_available():
        raise SystemExit("No CUDA GPU detected. This project requires a GPU runtime (e.g. a Colab T4).")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    bf16_supported = torch.cuda.is_bf16_supported()

    print(f"GPU: {name}")
    print(f"CUDA capability: {major}.{minor}")
    print(f"torch.cuda.is_bf16_supported(): {bf16_supported}")
    print(f"Selected training precision: {expected_precision}")

    if expected_precision == "bf16" and not bf16_supported:
        raise SystemExit(
            f"Configured precision is bf16, but '{name}' does not support it "
            "(torch.cuda.is_bf16_supported() is False). The T4 this project targets has "
            "no native bf16 support -- set fp16: true and bf16: false in the config "
            "instead of proceeding on an incompatible precision."
        )

    print("Hardware check passed.")


def log_experiment(record: dict, log_path: str = "logs/experiment_log.json"):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    records = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            records = json.load(f)
    records.append(record)
    with open(log_path, "w") as f:
        json.dump(records, f, indent=2)
    return record
