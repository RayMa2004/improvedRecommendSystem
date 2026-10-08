"""Shared CPU-worker loading and batch transfer for ranking training."""

import os
import time

import torch
from torch.utils.data import DataLoader


def training_loader_settings(prefix, device):
    batch_size = int(os.getenv(f"{prefix}_BATCH_SIZE", "1024"))
    num_workers = int(os.getenv(f"{prefix}_NUM_WORKERS", "8"))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError(f"{prefix}_BATCH_SIZE must be positive and {prefix}_NUM_WORKERS must be non-negative")
    settings = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.device(device).type == "cuda",
    }
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return settings


def build_training_loader(dataset, collate_fn, prefix, device):
    settings = training_loader_settings(prefix, device)
    print(f"{prefix}: device={device}, loader_settings={settings}", flush=True)
    return DataLoader(dataset, collate_fn=collate_fn, **settings)


def move_training_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def iter_training_batches(loader, device, stage, epoch):
    """Time CPU batch waits; synchronize CUDA only at the epoch boundary."""
    device = torch.device(device)
    started = time.perf_counter()
    previous_step_end = started
    data_wait_seconds = 0.0
    for index, host_batch in enumerate(loader, start=1):
        data_wait_seconds += time.perf_counter() - previous_step_end
        batch = move_training_batch(host_batch, device)
        if epoch == 0 and index == 1:
            print(f"{stage}: input_device={batch['user_ids'].device}", flush=True)
            if device.type == "cuda":
                print(
                    f"{stage}: GPU={torch.cuda.get_device_name(device)}, "
                    f"allocated_memory_MB={torch.cuda.memory_allocated(device) / (1024 ** 2):.1f}",
                    flush=True,
                )
        yield batch
        if index % 50 == 0:
            print(
                f"{stage}: epoch={epoch + 1}, batch={index}/{len(loader)}, "
                f"elapsed_s={time.perf_counter() - started:.1f}, data_wait_s={data_wait_seconds:.1f}",
                flush=True,
            )
        previous_step_end = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        f"{stage}: epoch={epoch + 1} finished, elapsed_s={time.perf_counter() - started:.1f}, "
        f"data_wait_s={data_wait_seconds:.1f}",
        flush=True,
    )
