import sys
import threading
import time
from typing import Callable, Optional

import jax
from tqdm_loggable.auto import tqdm

import levanter.tracker
from levanter.callbacks._core import Callback, CBInfo, JitCallback, LambdaCallback, M, S, StepInfo
from levanter.callbacks._metrics import (
    _tqdm_logging_one_time_setup,
    get_total_dataset_tokens,
    log_epoch_progress,
    log_performance_stats,
    log_step_info,
    logger,
    pbar_logger,
)
from levanter.data import DataLoader
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import barrier_sync
from levanter.utils.logging import save_xla_dumps_to_wandb
from levanter.utils.stat_utils import MeanScalar
from levanter.utils.types import ComputeLossFunction, ExtraData


def eval_loss_loop(
    loss_fn: ComputeLossFunction, model, dataset, max_batches: Optional[int] = None, name: Optional[str] = None
):
    loss = MeanScalar.zero()
    extras: ExtraData = {}

    if name is not None:
        desc = f"eval {name}"
    else:
        desc = "eval"

    _tqdm_logging_one_time_setup()
    pbar = tqdm(dataset, desc=desc, position=1, leave=False, total=max_batches)

    iter_ = iter(pbar)
    n = 0
    while True:
        n += 1
        batch = next(iter_, None)
        if batch is None:
            break
        losses_i, where_i, extras_i = loss_fn(model, batch)
        loss += MeanScalar.init(losses_i, where=where_i)
        for k, v in extras_i.loggable.items():
            if k not in extras:
                extras[k] = v
            else:
                extras[k] += v

        pbar.set_postfix(loss=loss.item())

        if max_batches is not None and n >= max_batches:
            break

    return loss.item(), {k: v.item() for k, v in extras.items()}


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataset: DataLoader,
    max_batches: Optional[int] = None,
    name: Optional[str] = None,
):
    def compute_loss(info: StepInfo):
        loss, extras = eval_loss_loop(loss_fn, info.eval_model, dataset, max_batches=max_batches, name=name)

        prefix = "eval"
        if name:
            prefix += "/" + name
        levanter.tracker.log(
            {f"{prefix}/loss": loss} | {f"{prefix}/{k}": v for k, v in extras.items()}, step=info.step
        )

        if name:
            logger.info(f"{name} validation loss: {loss:.3f}")
        else:
            logger.info(f"validation loss: {loss:.3f}")

        return loss

    return compute_loss


def wandb_xla_logger(config: WandbConfig):
    import wandb

    last_mtime = wandb.run and wandb.run.start_time or time.time()

    def log_xla_to_wandb(step: StepInfo):
        nonlocal last_mtime
        save_xla_dumps_to_wandb(last_mtime)
        # update time to now
        last_mtime = time.time()

    if config.save_xla_dumps:
        return log_xla_to_wandb
    else:
        return lambda x: None


def profile(path: str, start_step: int, num_steps: int, create_perfetto_link: bool) -> Callable[[StepInfo], None]:
    def profiler_callback_fn(step: StepInfo):
        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
            logger.info(f"Starting profiler until step {start_step + num_steps}.")
            jax.profiler.start_trace(path, create_perfetto_link=_create_perfetto_link, create_perfetto_trace=True)
        elif step.step == start_step + num_steps - 1:
            if create_perfetto_link:
                logger.info(
                    f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}"
                )
            else:
                logger.info("Stopping profiler.")
            # so, annoyingly, gcloud ssh doesn't reliably flush stdout here, so we need to spin up
            # a thread to flush and print periodically until we make it past stop_trace
            # (note: stop_trace blocks if perfetto is enabled)
            event = threading.Event()
            if create_perfetto_link and jax.process_index() == 0:
                _flush_while_waiting(event)

            jax.profiler.stop_trace()

            if create_perfetto_link and jax.process_index() == 0:
                event.set()

            levanter.tracker.current_tracker().log_artifact(path, type="jax_profile")
            barrier_sync()

    return profiler_callback_fn


def _flush_while_waiting(event):
    def flush_stdout():
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)
        while not event.is_set():
            print("Waiting...", flush=True)
            print("\n", file=sys.stderr, flush=True)
            time.sleep(5)

    thread = threading.Thread(target=flush_stdout)
    thread.start()


__all__ = [
    "eval_loss_loop",
    "compute_validation_loss",
    "get_total_dataset_tokens",
    "wandb_xla_logger",
    "profile",
    "Callback",
    "JitCallback",
    "LambdaCallback",
    "StepInfo",
]
