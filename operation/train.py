# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import Dataset
from models.MCNet import mc_net
from operation.batch_collate import spine_collater
from operation.loss import SpineLoss


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device_name: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def _require_labeled_split(data_dir: str, split: str) -> None:
    split_root = os.path.join(data_dir, split)
    image_dir = os.path.join(split_root, "images")
    label_dir = os.path.join(split_root, "labels")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing {split} image directory: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing {split} label directory: {label_dir}")


def _image_files(data_dir: str, split: str):
    image_dir = os.path.join(data_dir, split, "images")
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(extensions)
    ]


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_train_val_disjoint(data_dir: str) -> None:
    """Reject sample leakage between the only two splits used during training."""

    _require_labeled_split(data_dir, "train")
    _require_labeled_split(data_dir, "val")
    train_files = _image_files(data_dir, "train")
    val_files = _image_files(data_dir, "val")

    train_ids = {os.path.splitext(os.path.basename(path))[0].casefold() for path in train_files}
    val_ids = {os.path.splitext(os.path.basename(path))[0].casefold() for path in val_files}
    repeated_ids = sorted(train_ids & val_ids)
    if repeated_ids:
        preview = ", ".join(repeated_ids[:5])
        raise RuntimeError(
            "train/ and val/ contain overlapping sample identifiers: "
            f"{preview}{' ...' if len(repeated_ids) > 5 else ''}"
        )

    train_hashes = {_file_sha256(path) for path in train_files}
    val_hashes = {_file_sha256(path) for path in val_files}
    repeated_content = train_hashes & val_hashes
    if repeated_content:
        raise RuntimeError(
            "train/ and val/ contain identical image content under different names "
            f"({len(repeated_content)} duplicate file(s))."
        )


def _build_loader(args, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    _require_labeled_split(args.data_dir, split)
    dataset = Dataset(args, phase=split)
    num_workers = max(0, int(args.num_workers))
    loader_args = {
        "dataset": dataset,
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "drop_last": False,
        "collate_fn": spine_collater,
    }
    if num_workers > 0:
        loader_args["persistent_workers"] = bool(
            getattr(args, "loader_persistent_workers", True)
        )
        loader_args["prefetch_factor"] = int(
            getattr(args, "loader_prefetch_factor", 4)
        )
    return DataLoader(**loader_args)


def _move_batch(batch, device: torch.device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _accumulate_stats(total: Dict[str, float], stats: Dict[str, float]) -> None:
    for key, value in stats.items():
        total[key] += float(value)


def _mean_stats(total: Dict[str, float], count: int) -> Dict[str, float]:
    denom = max(int(count), 1)
    mean = {key: float(value) / denom for key, value in total.items()}
    mean["detection_objective"] = sum(
        mean.get(key, 0.0)
        for key in ("hm_loss", "base_hm_loss", "p2_support_hm_loss", "p2_hm_loss")
    )
    mean["geometry_objective"] = sum(
        mean.get(key, 0.0)
        for key in (
            "center_reg_loss",
            "corner_reg_loss",
            "p2_direct_center_reg_loss",
            "p2_direct_corner_reg_loss",
        )
    )
    mean["structure_objective"] = sum(
        mean.get(key, 0.0)
        for key in (
            "centerline_loss",
            "axis_visible_loss",
            "row_coverage_loss",
            "hm_row_recall_loss",
        )
    )
    return mean


def _format_progress(stats: Dict[str, float]) -> Dict[str, str]:
    keys = (
        "total_loss",
        "detection_objective",
        "geometry_objective",
        "structure_objective",
    )
    return {key: f"{stats[key]:.4f}" for key in keys if key in stats}


def _train_one_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    accumulation_steps: int,
    grad_clip: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    running: Dict[str, float] = defaultdict(float)
    steps = 0
    accumulation_steps = max(1, int(accumulation_steps))

    progress = tqdm(loader, ncols=120, desc=f"[train] epoch {epoch}")
    for iteration, batch in enumerate(progress, start=1):
        if batch is None:
            continue
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch=batch, return_intermediates=False)
            loss, stats = criterion(outputs, batch, epoch=epoch)
            scaled_loss = loss / float(accumulation_steps)

        scaler.scale(scaled_loss).backward()
        should_step = iteration % accumulation_steps == 0 or iteration == len(loader)
        if should_step:
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        _accumulate_stats(running, stats)
        steps += 1
        progress.set_postfix(_format_progress(_mean_stats(running, steps)))

    if steps == 0:
        raise RuntimeError("The training loader produced no valid batches.")
    return _mean_stats(running, steps)


@torch.no_grad()
def _validate_one_epoch(
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()
    running: Dict[str, float] = defaultdict(float)
    steps = 0

    progress = tqdm(loader, ncols=120, desc=f"[val] epoch {epoch}")
    for batch in progress:
        if batch is None:
            continue
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch=batch, return_intermediates=False)
            _, stats = criterion(outputs, batch, epoch=epoch)
        _accumulate_stats(running, stats)
        steps += 1
        progress.set_postfix(_format_progress(_mean_stats(running, steps)))

    if steps == 0:
        raise RuntimeError("The validation loader produced no valid batches.")
    return _mean_stats(running, steps)


def _load_resume_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> int:
    if not checkpoint_path:
        return 1
    checkpoint_path = os.path.abspath(os.path.normpath(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True)
    if isinstance(state, dict):
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        return int(state.get("epoch", 0) or 0) + 1
    return 1


def _checkpoint_payload(args, model, optimizer, scheduler, scaler, epoch, validation):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "training_complete": bool(int(epoch) >= int(args.epochs)),
        "train_split": "train",
        "validation_split": "val",
        "test_accessed": False,
        "validation": validation,
        "args": vars(args),
    }


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_train(args) -> Dict[str, object]:
    """Train on ``train`` and monitor convergence only on ``val``.

    This function never constructs or reads the ``test`` split. Final reporting is
    intentionally delegated to the independent ``eval`` command.
    """

    set_random_seed(int(args.seed))
    device = _resolve_device(args.device)
    use_amp = bool(device.type == "cuda" and args.amp)
    checkpoint_dir = os.path.abspath(os.path.normpath(args.checkpoint_dir))
    train_output_dir = os.path.abspath(os.path.normpath(args.output_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)

    _assert_train_val_disjoint(args.data_dir)

    train_loader = _build_loader(
        args,
        split="train",
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = _build_loader(
        args,
        split="val",
        batch_size=int(args.val_batch_size),
        shuffle=False,
    )

    model = mc_net(args).to(device)
    criterion = SpineLoss(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        foreach=bool(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(args.lr_gamma),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = _load_resume_checkpoint(
        args.resume_checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
    )

    _write_json(os.path.join(train_output_dir, "run_config.json"), vars(args))
    history_path = os.path.join(train_output_dir, "history.jsonl")
    final_epoch = int(args.epochs)
    if start_epoch > final_epoch:
        raise ValueError(
            f"Resume checkpoint starts at epoch {start_epoch}, beyond requested epoch {final_epoch}."
        )

    last_validation = None
    for epoch in range(start_epoch, final_epoch + 1):
        train_stats = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            accumulation_steps=int(args.accumulation_steps),
            grad_clip=float(args.grad_clip),
            epoch=epoch,
        )

        should_validate = epoch % int(args.val_interval) == 0 or epoch == final_epoch
        if should_validate:
            last_validation = _validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
            )

        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_stats,
            "val": last_validation if should_validate else None,
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        checkpoint = _checkpoint_payload(
            args,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            last_validation,
        )
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_model.pth"))
        if epoch % int(args.save_interval) == 0 or epoch == final_epoch:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_{epoch}.pth"))

    summary = {
        "phase": "train",
        "train_split": "train",
        "validation_split": "val",
        "epochs": final_epoch,
        "final_checkpoint": os.path.join(checkpoint_dir, "latest_model.pth"),
        "validation": last_validation,
        "train_val_integrity_checked": True,
        "test_accessed": False,
    }
    _write_json(os.path.join(train_output_dir, "summary.json"), summary)
    return summary

