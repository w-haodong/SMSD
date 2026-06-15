# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset import Dataset
from datasets.inference_dataset import InferenceDataset
from eval_cobb_calm import (
    circular_angle_error_deg,
    cobb_angle_calc,
    compute_total_cobb_smape_percent,
)
from experiment_config import input_size_for_dataset
from models.MCNet import mc_net
from operation.batch_collate import spine_collater
from operation.decode import DecDecoder


def _default_hrnet_pretrained(variant: str = "w32") -> str:
    variant = str(variant).lower()
    filename_map = {
        "w18": "faster_rcnn_hrnetv2p_w18_mstrain_syncbn_2x.pth",
        "w32": "faster_rcnn_hrnetv2p_w32_mstrain_syncbn_1x.pth",
    }
    if variant not in filename_map:
        raise ValueError(f"Unsupported HRNet variant: {variant}")
    return os.path.join(ROOT, "weights", filename_map[variant])


def _hrnet_profile(variant: str):
    variant = str(variant).lower()
    if variant in {"w18", "w32"}:
        base = [18, 36, 72, 144] if variant == "w18" else [32, 64, 128, 256]
        return (
            base,
            [1, 4, 3],
            {
                "stage1": [4],
                "stage2": [4, 4],
                "stage3": [4, 4, 4],
                "stage4": [4, 4, 4, 4],
            },
        )
    raise ValueError(f"Unsupported HRNet variant: {variant}")


def _apply_internal_profile(args):
    args.log_schema = "release_infer_only"
    args.brief_init = True
    args.decoder_dim = 128
    args.head_dim = 128
    args.decode_candidate_topk = 32
    args.decoder_use_centerline = True
    args.decoder_strip_samples = 17
    args.decoder_strip_keep_ratio = 0.72
    args.decoder_strip_wide_context_scale = 1.70
    args.decoder_axis_min_width_ratio = 0.035
    args.decoder_axis_max_width_ratio = 0.24
    args.decoder_p4_row_keep_ratio = 0.78
    args.decoder_p2_refine_keep_ratio = 0.025
    args.lambda_hm = 1.0
    args.lambda_base_hm = 0.30
    args.lambda_p2_support_hm = 0.25
    args.lambda_p2_hm = 0.15
    args.lambda_center_reg = 1.0
    args.lambda_corner_reg = 0.1
    args.lambda_p2_direct_reg = 0.25
    args.lambda_centerline = 0.40
    args.lambda_axis_visible = 0.10
    args.lambda_row_coverage = 0.05
    args.lambda_hm_row_recall = 0.08
    args.lambda_progress = 0.0
    args.lambda_ortho = 0.0
    args.lambda_grad = 0.0
    args.lambda_pinn_curv = 0.0
    args.center_disk_frac_x = 0.18
    args.center_disk_frac_y = 0.18
    args.min_center_disk_r = 1
    args.max_center_disk_r = 4
    args.spine_mask_lateral_expand_ratio = 0.20
    args.spine_mask_endcap_expand_ratio = 0.10
    args.runtime_profile_on_setup = False
    args.loader_persistent_workers = True
    args.loader_prefetch_factor = 4
    args.loader_pin_memory = True
    args.vis_every = 0
    args.vis_every_val = 0
    args.vis_panel_h = 640
    args.debug_center_collision = False
    args.first_preview = False
    args.log_first_iter_timing = False
    args.hrnet_variant = str(getattr(args, "hrnet_variant", "w32")).lower()
    stage_channels, stage_modules, stage_blocks = _hrnet_profile(args.hrnet_variant)
    args.hrnet_stage_channels = stage_channels
    args.hrnet_stage_modules = stage_modules
    args.hrnet_stage_blocks = stage_blocks
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight public release for ours_g")
    parser.add_argument("--phase", choices=["predict", "eval"], default="predict")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dataset_name", type=str, default="fh_data_bs")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("weights", "latest_model.pth"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "predict"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_mat", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--hrnet_variant", choices=["w18", "w32"], default="w32")
    parser.add_argument("--hrnet_pretrained", type=str, default="")
    parser.add_argument("--use_hrnet_pretrained", action="store_true")
    parser.add_argument("--input_h", type=int, default=0)
    parser.add_argument("--input_w", type=int, default=0)
    parser.add_argument("--K", type=int, default=17)
    parser.add_argument("--hm_h_r", type=int, default=4)
    parser.add_argument("--hm_w_r", type=int, default=4)
    args = parser.parse_args()

    if args.phase == "predict" and not args.image_dir:
        parser.error("--image_dir is required when --phase predict")
    if args.phase == "eval" and not args.data_dir:
        parser.error("--data_dir is required when --phase eval")

    args.work_dir = ROOT
    args.checkpoint = os.path.abspath(os.path.normpath(args.checkpoint))
    args.output_dir = os.path.abspath(os.path.normpath(args.output_dir))
    if args.image_dir:
        args.image_dir = os.path.abspath(os.path.normpath(args.image_dir))
    if args.data_dir:
        args.data_dir = os.path.abspath(os.path.normpath(args.data_dir))
        args.dataset_name = os.path.basename(args.data_dir.rstrip("\\/")) or args.dataset_name

    if args.input_h <= 0 or args.input_w <= 0:
        input_h, input_w = input_size_for_dataset(args.dataset_name)
        args.input_h = int(input_h)
        args.input_w = int(input_w)

    args = _apply_internal_profile(args)
    if args.use_hrnet_pretrained:
        args.hrnet_pretrained = (
            os.path.abspath(os.path.normpath(args.hrnet_pretrained))
            if args.hrnet_pretrained
            else _default_hrnet_pretrained(args.hrnet_variant)
        )
    else:
        args.hrnet_pretrained = ""

    args.backbone_name = f"hrnet_{args.hrnet_variant}_axis_height_narrowband_sparse"
    args.amp = bool(torch.cuda.is_available() and not args.disable_amp)
    args.hm_h = int(math.ceil(args.input_h / float(args.hm_h_r)))
    args.hm_w = int(math.ceil(args.input_w / float(args.hm_w_r)))
    return args


def _build_loader(dataset, args):
    num_workers = max(0, int(args.num_workers))
    kwargs = {
        "dataset": dataset,
        "batch_size": max(1, int(args.batch_size)),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "drop_last": False,
        "collate_fn": spine_collater,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(args, "loader_persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(args, "loader_prefetch_factor", 4))
    return torch.utils.data.DataLoader(**kwargs)


def _move_batch(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _to_prob_tensor(x: torch.Tensor) -> torch.Tensor:
    xmin = float(x.min().detach().cpu())
    xmax = float(x.max().detach().cpu())
    return x.float() if xmin >= 0.0 and xmax <= 1.0 else torch.sigmoid(x.float())


def _sort_quads_by_center_y(quads: np.ndarray) -> np.ndarray:
    quads = np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2)
    return quads[np.argsort(quads.mean(axis=1)[:, 1])]


def _recover_quads_to_original(quads, scale, pad_x, pad_y, ori_h, ori_w):
    pts = np.asarray(quads, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = (pts[:, 0] - float(pad_x)) / (float(scale) + 1e-12)
    pts[:, 1] = (pts[:, 1] - float(pad_y)) / (float(scale) + 1e-12)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(ori_w) - 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(ori_h) - 1.0)
    return _sort_quads_by_center_y(pts.reshape(-1, 4, 2))


def _extract_pred_quads_abs(dets, hm_w_r, hm_h_r):
    dets = np.asarray(dets)
    if dets.ndim == 3:
        dets = dets[0]
    if dets.ndim != 2 or dets.shape[1] < 10 or len(dets) == 0:
        return None
    quads = dets[:, 2:10].reshape(-1, 4, 2)[:, [0, 1, 3, 2], :].astype(np.float32)
    quads[..., 0] *= float(hm_w_r)
    quads[..., 1] *= float(hm_h_r)
    return _sort_quads_by_center_y(quads)


def _single_angle_smape(pred_val, gt_val, eps: float = 1e-5):
    pred = np.asarray([pred_val], dtype=np.float32)
    gt = np.asarray([gt_val], dtype=np.float32)
    diff = np.abs(circular_angle_error_deg(pred, gt))
    denom = pred + gt
    denom[np.abs(denom) < float(eps)] = float(eps)
    return float(np.mean(diff / denom * 100.0))


def _load_model(args, device):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = mc_net(args).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _predict_batches(loader, model, args, device):
    decoder = DecDecoder(
        K=int(args.K),
        candidate_topk=int(getattr(args, "decode_candidate_topk", 32)),
        conf_thresh=0.05,
    )
    use_amp = bool(torch.cuda.is_available() and args.amp)
    rows = []
    mats = {}

    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"[{args.phase}]")
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch(batch, device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch=batch, return_intermediates=False)
        pred_hm = _to_prob_tensor(outputs["pred_global_hm"])
        dets = decoder.ctdet_decode(
            pred_hm,
            outputs["pred_corner_offsets"],
            outputs["pred_center_offsets"],
            topk=int(args.K),
        )

        batch_size = int(pred_hm.shape[0])
        for b in range(batch_size):
            pred_quads_abs = _extract_pred_quads_abs(dets[b:b + 1], args.hm_w_r, args.hm_h_r)
            if pred_quads_abs is None:
                continue
            scale = float(batch["scale"][b].view(-1)[0].detach().cpu().item())
            pad_x = float(batch["pad"][b].view(-1)[0].detach().cpu().item())
            pad_y = float(batch["pad"][b].view(-1)[1].detach().cpu().item())
            ori_h = float(batch["ori_size"][b].view(-1)[0].detach().cpu().item())
            ori_w = float(batch["ori_size"][b].view(-1)[1].detach().cpu().item())
            pred_quads_ori = _recover_quads_to_original(pred_quads_abs, scale, pad_x, pad_y, ori_h, ori_w)
            image_name = batch["image_id"][b] if isinstance(batch["image_id"], (list, tuple)) else str(batch["image_id"])
            mats[image_name] = pred_quads_ori.reshape(-1, 2).astype(np.float32)
            for vertebra_idx, quad in enumerate(pred_quads_ori):
                point_names = ["TL", "TR", "BL", "BR"]
                for point_idx, point in enumerate(quad):
                    rows.append([
                        image_name,
                        vertebra_idx,
                        point_names[point_idx],
                        float(point[0]),
                        float(point[1]),
                    ])
    return rows, mats


def run_predict(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = InferenceDataset(args, args.image_dir)
    loader = _build_loader(dataset, args)
    model = _load_model(args, device)
    rows, mats = _predict_batches(loader, model, args, device)

    pred_csv = os.path.join(args.output_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "vertebra_index", "point_name", "x", "y"])
        writer.writerows(rows)

    if args.save_mat:
        mat_dir = os.path.join(args.output_dir, "mat")
        os.makedirs(mat_dir, exist_ok=True)
        for image_name, pts in mats.items():
            base = os.path.splitext(os.path.basename(image_name))[0]
            sio.savemat(os.path.join(mat_dir, f"{base}.mat"), {"pr_landmarks": pts})

    summary = {
        "phase": "predict",
        "images": len(mats),
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.batch_size = 1
    dataset = Dataset(SimpleNamespace(**vars(args)), phase=args.split)
    loader = _build_loader(dataset, args)
    model = _load_model(args, device)
    decoder = DecDecoder(
        K=int(args.K),
        candidate_topk=int(getattr(args, "decode_candidate_topk", 32)),
        conf_thresh=0.05,
    )
    use_amp = bool(torch.cuda.is_available() and args.amp)

    angle_rows = []
    pred_rows = []
    point_px_vals = []
    cobb_smape_vals = []
    cobb_mae_vals = []
    pt_mae_vals = []
    mt_mae_vals = []
    tl_mae_vals = []
    pt_smape_vals = []
    mt_smape_vals = []
    tl_smape_vals = []
    vertebrae_labels = [
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
        "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5",
    ]
    vertebra_distance_errors = []

    pbar = tqdm(loader, total=len(loader), ncols=120, desc=f"[eval:{args.split}]")
    for batch in pbar:
        if batch is None:
            continue
        batch = _move_batch(batch, device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch=batch, return_intermediates=False)

        pred_hm = _to_prob_tensor(outputs["pred_global_hm"])
        dets = decoder.ctdet_decode(
            pred_hm,
            outputs["pred_corner_offsets"],
            outputs["pred_center_offsets"],
            topk=int(args.K),
        )

        pred_quads_abs = _extract_pred_quads_abs(dets[0:1], args.hm_w_r, args.hm_h_r)
        if pred_quads_abs is None:
            continue

        gt_quads_abs = batch["p_gt"][0].view(-1, 4, 2).detach().cpu().numpy().astype(np.float32)
        gt_quads_abs = _sort_quads_by_center_y(gt_quads_abs)
        scale = float(batch["scale"][0].view(-1)[0].detach().cpu().item())
        pad_x = float(batch["pad"][0].view(-1)[0].detach().cpu().item())
        pad_y = float(batch["pad"][0].view(-1)[1].detach().cpu().item())
        ori_h = float(batch["ori_size"][0].view(-1)[0].detach().cpu().item())
        ori_w = float(batch["ori_size"][0].view(-1)[1].detach().cpu().item())
        gt_quads_ori = _recover_quads_to_original(gt_quads_abs, scale, pad_x, pad_y, ori_h, ori_w)
        pred_quads_ori = _recover_quads_to_original(pred_quads_abs, scale, pad_x, pad_y, ori_h, ori_w)

        image_name = batch["image_id"][0] if isinstance(batch["image_id"], (list, tuple)) else str(batch["image_id"])
        base_name = os.path.splitext(os.path.basename(image_name))[0]

        gt_pts = gt_quads_ori.reshape(-1, 2)
        pred_pts = pred_quads_ori.reshape(-1, 2)
        point_px_vals.append(float(np.mean(np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=1)))))

        gt_centers = gt_pts.reshape(17, 4, 2).mean(axis=1)
        pred_centers = pred_pts.reshape(17, 4, 2).mean(axis=1)
        for vi, label in enumerate(vertebrae_labels):
            vertebra_distance_errors.append([base_name, label, float(np.linalg.norm(pred_centers[vi] - gt_centers[vi]))])

        gt_ca1, gt_ca2, gt_ca3, _ = cobb_angle_calc(gt_pts, image=None, is_train=False)
        pr_ca1, pr_ca2, pr_ca3, _ = cobb_angle_calc(pred_pts, image=None, is_train=False)
        gt_cobb = np.asarray([gt_ca1, gt_ca2, gt_ca3], dtype=np.float32)
        pred_cobb = np.asarray([pr_ca1, pr_ca2, pr_ca3], dtype=np.float32)

        abs_diff = np.abs(circular_angle_error_deg(pred_cobb, gt_cobb))
        cobb_smape_vals.append(compute_total_cobb_smape_percent(pred_cobb, gt_cobb))
        cobb_mae_vals.append(float(np.mean(abs_diff)))
        pt_mae_vals.append(float(abs_diff[0]))
        mt_mae_vals.append(float(abs_diff[1]))
        tl_mae_vals.append(float(abs_diff[2]))
        pt_smape_vals.append(_single_angle_smape(pred_cobb[0], gt_cobb[0]))
        mt_smape_vals.append(_single_angle_smape(pred_cobb[1], gt_cobb[1]))
        tl_smape_vals.append(_single_angle_smape(pred_cobb[2], gt_cobb[2]))
        angle_rows.append([base_name, float(gt_cobb[0]), float(gt_cobb[1]), float(gt_cobb[2]), float(pred_cobb[0]), float(pred_cobb[1]), float(pred_cobb[2])])

        point_names = ["TL", "TR", "BL", "BR"]
        for vertebra_idx, quad in enumerate(pred_quads_ori):
            for point_idx, point in enumerate(quad):
                gt_point = gt_quads_ori[vertebra_idx, point_idx]
                pred_rows.append([
                    base_name,
                    vertebra_idx,
                    point_names[point_idx],
                    float(point[0]),
                    float(point[1]),
                    float(gt_point[0]),
                    float(gt_point[1]),
                ])

        if args.save_mat:
            mat_dir = os.path.join(args.output_dir, "mat")
            os.makedirs(mat_dir, exist_ok=True)
            sio.savemat(
                os.path.join(mat_dir, f"{base_name}.mat"),
                {
                    "pr_landmarks": pred_pts.astype(np.float32),
                    "gt_landmarks": gt_pts.astype(np.float32),
                    "pr_angles": pred_cobb.astype(np.float32),
                    "gt_angles": gt_cobb.astype(np.float32),
                },
            )

    with open(os.path.join(args.output_dir, "angles.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "gt_pt", "gt_mt", "gt_tl", "pr_pt", "pr_mt", "pr_tl"])
        writer.writerows(angle_rows)

    with open(os.path.join(args.output_dir, "predictions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "vertebra_index", "point_name", "pred_x", "pred_y", "gt_x", "gt_y"])
        writer.writerows(pred_rows)

    with open(os.path.join(args.output_dir, "point_errors.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "vertebra", "distance_error"])
        writer.writerows(vertebra_distance_errors)

    summary = {
        "phase": "eval",
        "split": args.split,
        "images": len(angle_rows),
        "point_dist_px": None if not point_px_vals else float(np.mean(point_px_vals)),
        "cobb_mae": None if not cobb_mae_vals else float(np.mean(cobb_mae_vals)),
        "cobb_smape": None if not cobb_smape_vals else float(np.mean(cobb_smape_vals)),
        "pt_mae": None if not pt_mae_vals else float(np.mean(pt_mae_vals)),
        "mt_mae": None if not mt_mae_vals else float(np.mean(mt_mae_vals)),
        "tl_mae": None if not tl_mae_vals else float(np.mean(tl_mae_vals)),
        "pt_smape": None if not pt_smape_vals else float(np.mean(pt_smape_vals)),
        "mt_smape": None if not mt_smape_vals else float(np.mean(mt_smape_vals)),
        "tl_smape": None if not tl_smape_vals else float(np.mean(tl_smape_vals)),
        "checkpoint": args.checkpoint,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    if args.phase == "predict":
        summary = run_predict(args)
    else:
        summary = run_eval(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
