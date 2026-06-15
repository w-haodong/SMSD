# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.dataset import (
    EVAL_PREPROCESS_CLAHE_ENV,
    EVAL_PREPROCESS_NORMALIZE_ENV,
    EVAL_PREPROCESS_PERCENTILE_ENV,
    EVAL_PERCENTILE_RANGE,
    IMG_EXTS,
    INPUT_NORM_MODE,
    INPUT_ZSCORE_CLIP,
    LETTERBOX_DOWNSAMPLE_SHARPEN,
    TRAIN_CLAHE_CLIP_LIMIT,
    TRAIN_CLAHE_TILE_GRID,
    USE_TRAIN_CLAHE,
    _eval_switch_enabled,
    imread_gray_unicode_safe,
    letterbox_image_gray,
)


class InferenceDataset(Dataset):
    def __init__(self, args, image_dir: str):
        super().__init__()
        self.args = args
        self.phase = "test"
        self.image_dir = os.path.abspath(os.path.normpath(image_dir))
        self.input_h = int(args.input_h)
        self.input_w = int(args.input_w)
        self.letterbox_downsample_sharpen = float(LETTERBOX_DOWNSAMPLE_SHARPEN)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")

        self.image_files = [
            fn for fn in sorted(os.listdir(self.image_dir))
            if fn.lower().endswith(IMG_EXTS)
        ]
        max_samples = int(getattr(args, "max_samples", 0) or 0)
        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def _apply_xray_preprocess(self, img_gray):
        img_gray = np.asarray(img_gray)
        if img_gray.ndim != 2:
            raise ValueError(f"Expected grayscale image, got shape {tuple(img_gray.shape)}")

        img_proc = img_gray.astype(np.float32)

        if _eval_switch_enabled(
            self.phase,
            self.args,
            "eval_preprocess_percentile",
            EVAL_PREPROCESS_PERCENTILE_ENV,
            default=True,
        ):
            low = float(EVAL_PERCENTILE_RANGE[0])
            high = float(EVAL_PERCENTILE_RANGE[1])
            low = max(0.0, min(low, 100.0))
            high = max(low + 1e-3, min(high, 100.0))
            lo = float(np.percentile(img_proc, low))
            hi = float(np.percentile(img_proc, high))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-6:
                img_proc = np.clip(img_proc, lo, hi)
                img_proc = (img_proc - lo) / (hi - lo)
                img_proc = img_proc * 255.0

        img_proc = np.clip(img_proc, 0.0, 255.0).astype(np.uint8)

        if USE_TRAIN_CLAHE and _eval_switch_enabled(
            self.phase,
            self.args,
            "eval_preprocess_clahe",
            EVAL_PREPROCESS_CLAHE_ENV,
            default=True,
        ):
            clip_limit = max(float(TRAIN_CLAHE_CLIP_LIMIT), 0.01)
            grid_size = max(int(TRAIN_CLAHE_TILE_GRID), 1)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            img_proc = clahe.apply(img_proc)

        return img_proc

    def _normalize_input_image(self, img_3c):
        img = img_3c.astype(np.float32)
        if not _eval_switch_enabled(
            self.phase,
            self.args,
            "eval_preprocess_normalize",
            EVAL_PREPROCESS_NORMALIZE_ENV,
            default=True,
        ):
            return img

        img = img / 255.0
        norm_mode = str(INPUT_NORM_MODE).lower()

        if norm_mode == "minus_half":
            return img - 0.5

        if norm_mode == "zscore_clip":
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True)
            std = np.maximum(std, 1e-6)
            img = (img - mean) / std
            z_clip = max(float(INPUT_ZSCORE_CLIP), 1e-3)
            img = np.clip(img, -z_clip, z_clip) / (2.0 * z_clip)
            return img

        raise ValueError(f"Unsupported input_norm_mode: {norm_mode}")

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        img = imread_gray_unicode_safe(image_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        img = self._apply_xray_preprocess(img)

        ori_h, ori_w = img.shape[:2]
        img_pad, scale, pad_x, pad_y, new_w, new_h = letterbox_image_gray(
            img,
            self.input_h,
            self.input_w,
            pad_value=0,
            downsample_sharpen=self.letterbox_downsample_sharpen,
        )
        img_pad = np.stack([img_pad, img_pad, img_pad], axis=-1)
        img_norm = self._normalize_input_image(img_pad)
        img_3c = np.transpose(img_norm, (2, 0, 1))

        return {
            "input_image": torch.from_numpy(img_3c).float(),
            "image_id": image_name,
            "ori_size": torch.tensor([ori_h, ori_w], dtype=torch.float32),
            "input_size": torch.tensor([self.input_h, self.input_w], dtype=torch.float32),
            "scale": torch.tensor([scale], dtype=torch.float32),
            "pad": torch.tensor([pad_x, pad_y], dtype=torch.float32),
            "resized_size": torch.tensor([new_h, new_w], dtype=torch.float32),
        }
