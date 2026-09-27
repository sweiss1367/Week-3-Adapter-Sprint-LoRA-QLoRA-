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
    """
    reset_cuda_stats()
    t0 = time.time()
    result = fn()
    if _cuda_available():
        torch.cuda.synchronize()
    wall_time_sec = round(time.time() - t0, 1)

    record = {
        "stage": stage_name,
        "wall_time_sec": wall_time_sec,
        "avg_step_time_sec": (
            round(step_timer.avg_step_time_sec, 4)
            if step_timer is not None and step_timer.avg_step_time_sec is not None
            else None
        ),
        "config": config or {},
    }
    record.update(cuda_peak_stats())
    log_experiment(record, log_path)
    print(f"[{stage_name}] {record}")
    return result, record


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
