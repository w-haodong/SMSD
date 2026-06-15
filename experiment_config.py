# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os

# Change these two values when the dataset moves or when switching datasets.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.environ.get("SCOLIOSIS_DATASET_DIR", os.path.join(REPO_ROOT, "data"))
DEFAULT_DATASET_NAME = os.environ.get("SCOLIOSIS_DATASET_NAME", "fh_data_bs")
DEFAULT_NUM_EPOCH = int(os.environ.get("SCOLIOSIS_NUM_EPOCH", "200"))
DEFAULT_OPEN_PTH_DIR = os.environ.get("SCOLIOSIS_OPEN_PTH_DIR", "")

DEFAULT_INPUT_SIZE = (1280, 512)
DATASET_INPUT_SIZES = {
    "fh_data_lc": (1664, 512),
    "fh_data_bs": (1280, 512),
}

# Eval accepts "val", "test", "val,test", "test,val", "both", or "all".
DEFAULT_EVAL_DATA = os.environ.get("SCOLIOSIS_EVAL_DATA", "val,test")


def _norm_abs(path: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


def open_pth_dir() -> str:
    if DEFAULT_OPEN_PTH_DIR:
        return _norm_abs(DEFAULT_OPEN_PTH_DIR)
    return _norm_abs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_pth"))


def open_pth_path(filename: str) -> str:
    return os.path.join(open_pth_dir(), str(filename))


def default_dataset_dir() -> str:
    return _norm_abs(DEFAULT_DATASET_DIR)


def default_dataset_name() -> str:
    return str(DEFAULT_DATASET_NAME).strip()


def default_data_dir() -> str:
    return _norm_abs(os.path.join(default_dataset_dir(), default_dataset_name()))


def default_img_dir() -> str:
    return default_data_dir()


def default_eval_data() -> str:
    return str(DEFAULT_EVAL_DATA).strip() or "val,test"


def default_num_epoch() -> int:
    return int(DEFAULT_NUM_EPOCH)


def resolve_dataset_meta(data_dir: str) -> tuple[str, str, str]:
    data_dir = _norm_abs(data_dir)
    dataset_name = os.path.basename(data_dir)
    dataset_dir = os.path.dirname(data_dir)
    return data_dir, dataset_dir, dataset_name


def input_size_for_dataset(dataset_name: str) -> tuple[int, int]:
    return DATASET_INPUT_SIZES.get(str(dataset_name).strip(), DEFAULT_INPUT_SIZE)


def apply_dataset_input_size(args):
    if hasattr(args, "input_h") and hasattr(args, "input_w"):
        if bool(getattr(args, "_input_size_from_cli", False)):
            args.input_h = int(args.input_h)
            args.input_w = int(args.input_w)
        else:
            input_h, input_w = input_size_for_dataset(getattr(args, "dataset_name", ""))
            args.input_h = int(input_h)
            args.input_w = int(input_w)
    return args


def _resolve_data_dir_from_args(args) -> str:
    data_dir = str(getattr(args, "data_dir", "") or "").strip()
    if data_dir:
        return data_dir

    dataset_dir = str(getattr(args, "dataset_dir", "") or "").strip() or default_dataset_dir()
    dataset_name = str(getattr(args, "dataset_name", "") or "").strip() or default_dataset_name()
    if dataset_dir and dataset_name:
        return os.path.join(dataset_dir, dataset_name)

    img_dir = str(getattr(args, "img_dir", "") or "").strip() or default_img_dir()
    has_split_layout = (
        os.path.isdir(img_dir)
        and any(os.path.isdir(os.path.join(img_dir, split)) for split in ("train", "val", "test"))
    )
    if has_split_layout:
        return img_dir
    return img_dir + str(getattr(args, "num_train_f_sample", 80))


def apply_dataset_config(args):
    args.work_dir = _norm_abs(getattr(args, "work_dir", os.getcwd()))
    args.data_dir, args.dataset_dir, args.dataset_name = resolve_dataset_meta(_resolve_data_dir_from_args(args))
    args = apply_dataset_input_size(args)
    if hasattr(args, "img_dir"):
        args.img_dir = args.data_dir
    args.pth_dir = os.path.join(args.work_dir, "pth", args.dataset_name)
    args.vis_dir = os.path.join(args.work_dir, "vis", args.dataset_name)
    args.logs_dir = os.path.join(args.work_dir, "logs", args.dataset_name)
    return args


def parse_eval_splits(raw: str | None = None) -> list[str]:
    value = str(default_eval_data() if raw is None else raw).strip().lower()
    if not value:
        value = default_eval_data().lower()
    if value in {"both", "all", "val+test", "test+val"}:
        parts = ["val", "test"]
    else:
        for sep in (";", "+", "|", " "):
            value = value.replace(sep, ",")
        parts = [item.strip() for item in value.split(",") if item.strip()]

    aliases = {
        "validation": "val",
        "valid": "val",
        "val": "val",
        "testing": "test",
        "test": "test",
    }
    splits: list[str] = []
    for item in parts:
        split = aliases.get(item)
        if split and split not in splits:
            splits.append(split)
    return splits or ["val"]


def iter_eval_args(args):
    for split in parse_eval_splits(getattr(args, "eval_data", None)):
        split_args = copy.copy(args)
        split_args.eval_data = split
        yield split_args
