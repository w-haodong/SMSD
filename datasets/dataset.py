# -*- coding: utf-8 -*-

import os
import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

from operation import transform


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

TRAIN_SCALE_RANGE = (0.65, 1.00)
LETTERBOX_DOWNSAMPLE_SHARPEN = 0.12
TRAIN_PERCENTILE_RANGE = (0.5, 99.5)
EVAL_PERCENTILE_RANGE = (0.5, 99.5)
USE_TRAIN_CLAHE = True
TRAIN_CLAHE_CLIP_LIMIT = 2.0
TRAIN_CLAHE_TILE_GRID = 8
EVAL_GAMMA = 1.0
INPUT_NORM_MODE = 'zscore_clip'
INPUT_ZSCORE_CLIP = 3.0
EVAL_PREPROCESS_PERCENTILE_ENV = 'SCOLIOSIS_EVAL_PREPROCESS_PERCENTILE'
EVAL_PREPROCESS_CLAHE_ENV = 'SCOLIOSIS_EVAL_PREPROCESS_CLAHE'
EVAL_PREPROCESS_NORMALIZE_ENV = 'SCOLIOSIS_EVAL_PREPROCESS_NORMALIZE'


def _bool_value(value, default=True):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off', 'disable', 'disabled'}:
        return False
    return bool(default)


def _is_eval_phase(phase):
    return str(phase).strip().lower() in {'val', 'valid', 'validation', 'test', 'testing', 'eval'}


def _eval_switch_enabled(phase, config, attr_name, env_name, default=True):
    if not _is_eval_phase(phase):
        return True
    if config is not None and hasattr(config, attr_name):
        return _bool_value(getattr(config, attr_name), default=default)
    return _bool_value(os.environ.get(env_name), default=default)


def imread_gray_unicode_safe(path):
    path = os.fspath(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def gaussian2D(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1.0
    b1 = float(height + width)
    c1 = float(width * height * (1.0 - min_overlap) / (1.0 + min_overlap))
    sq1 = math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * float(height + width)
    c2 = float((1.0 - min_overlap) * width * height)
    sq2 = math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * float(height + width)
    c3 = float((min_overlap - 1.0) * width * height)
    sq3 = math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))
    r3 = (b3 + sq3) / (2.0 * a3) if abs(a3) > 1e-12 else 0.0

    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1.0):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=max(diameter / 6.0, 1e-6))

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    if left < 0 or right <= 0 or top < 0 or bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_gaussian.size > 0 and masked_heatmap.size > 0:
        np.maximum(masked_heatmap, masked_gaussian * float(k), out=masked_heatmap)
    return heatmap


def draw_center_point(heatmap, center, k=1.0):
    """
    单点监督：
    只把中心点所在 cell 置为 1，不再画高斯圆。
    """
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    if 0 <= x < width and 0 <= y < height:
        heatmap[y, x] = max(float(heatmap[y, x]), float(k))
    return heatmap


def draw_gaussian_1d(line, center, radius, k=1.0):
    center = int(center)
    radius = int(max(0, radius))
    length = int(line.shape[0])
    if not (0 <= center < length):
        return line
    if radius == 0:
        line[center] = max(float(line[center]), float(k))
        return line

    diameter = 2 * radius + 1
    x = np.arange(diameter, dtype=np.float32) - float(radius)
    gaussian = np.exp(-(x * x) / (2.0 * max(float(diameter) / 6.0, 1e-6) ** 2))

    left = min(center, radius)
    right = min(length - center, radius + 1)
    if left < 0 or right <= 0:
        return line

    masked_line = line[center - left:center + right]
    masked_gaussian = gaussian[radius - left:radius + right]
    if masked_line.size > 0 and masked_gaussian.size > 0:
        np.maximum(masked_line, masked_gaussian * float(k), out=masked_line)
    return line


def letterbox_image_gray(img, target_h, target_w, pad_value=0, downsample_sharpen=0.0):
    """
    保持长宽比缩放，再 pad 到目标尺寸。
    返回:
        img_pad: (target_h, target_w)
        scale:   等比例缩放系数
        pad_x:   左侧 pad
        pad_y:   上侧 pad
        new_w:   resize 后宽
        new_h:   resize 后高
    """
    ori_h, ori_w = img.shape[:2]

    scale = min(target_w / float(ori_w), target_h / float(ori_h))
    new_w = int(round(ori_w * scale))
    new_h = int(round(ori_h * scale))

    img_resized = transform.smart_resize(
        img,
        (new_w, new_h),
        down_interpolation=cv2.INTER_AREA,
        up_interpolation=cv2.INTER_LINEAR,
        downsample_sharpen=downsample_sharpen,
    )

    canvas = np.full((target_h, target_w), pad_value, dtype=img.dtype)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, scale, pad_x, pad_y, new_w, new_h


class Dataset(Dataset):
    """
    数据目录结构:
        data_dir/
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/
                images/
                labels/

    标签文件名:
        xxx.jpg.mat

    .mat 变量名:
        p2

    标签点顺序固定:
        [TL, TR, BL, BR]
    """

    def __init__(self, args, phase='train', domain=None):
        super().__init__()
        self.args = args
        self.phase = phase

        self.root_dir = os.path.join(args.data_dir, phase)
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.label_dir = os.path.join(self.root_dir, 'labels')

        self.input_h = int(args.input_h)
        self.input_w = int(args.input_w)
        self.K = int(args.K)
        # 高宽独立热图比率
        self.hm_h_r = int(args.hm_h_r)
        self.hm_w_r = int(args.hm_w_r)

        self.debug_center_collision = bool(getattr(args, 'debug_center_collision', False))
        self.letterbox_downsample_sharpen = float(LETTERBOX_DOWNSAMPLE_SHARPEN)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f'Image dir not found: {self.image_dir}')
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f'Label dir not found: {self.label_dir}')

        self.image_files = []
        for fn in sorted(os.listdir(self.image_dir)):
            if fn.lower().endswith(IMG_EXTS):
                self.image_files.append(fn)

        max_samples = int(getattr(args, 'max_samples', 0) or 0)
        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]

        if len(self.image_files) == 0:
            raise RuntimeError(f'No images found in {self.image_dir}')

        # ---------------- augment ----------------
        self.train_aug = transform.Compose([
            transform.ConvertImgFloat(),
            # Use mild grayscale jitter only; aggressive brightening can
            # saturate already bright bony structures in X-ray images.
            transform.PhotometricDistort(
                contrast_range=(0.92, 1.06),
                brightness_delta=(-10.0, 6.0),
                use_lighting_noise=False,
            ),
            transform.RandomGamma(),
            transform.Equalize(),
            transform.Solarize(),
            transform.Posterize(),
            transform.Sharpness(),
            transform.RandomGaussianBlur(),
            transform.RandomGaussianNoise(),
            transform.RandomChannelPerturb(),
            transform.RandomScaleTranslate(
                scale_range=TRAIN_SCALE_RANGE,
                down_interpolation=cv2.INTER_AREA,
                up_interpolation=cv2.INTER_LINEAR,
                downsample_sharpen=self.letterbox_downsample_sharpen,
            ),
            transform.RandomRotate(angle_range=(-12.0, 12.0), prob=0.5),
            transform.RandomMirror_w(),
        ])

        self.eval_aug = transform.Compose([
            transform.ConvertImgFloat(),
        ])

    def __len__(self):
        return len(self.image_files)

    def _build_sample(self, img_name, img_path):
        prev_input_h = self.input_h
        prev_input_w = self.input_w
        try:
            label_path = os.path.join(self.label_dir, img_name + '.mat')
            if not os.path.exists(label_path):
                fallback = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.mat')
                if os.path.exists(fallback):
                    label_path = fallback
                else:
                    raise FileNotFoundError(f'Label not found for {img_name}: {label_path}')

            img = imread_gray_unicode_safe(img_path)
            if img is None:
                raise RuntimeError(f'Failed to read image: {img_path}')
            img = self._apply_xray_preprocess(img)

            ori_h, ori_w = img.shape[:2]

            pts = self._load_points(label_path)
            if pts.shape[0] % 4 != 0:
                raise ValueError(f'Point count must be multiple of 4, got {pts.shape[0]} in {label_path}')

            num_quads = pts.shape[0] // 4
            if num_quads != self.K:
                raise ValueError(
                    f'K mismatch: args.K={self.K}, but label has {num_quads} quads ({pts.shape[0]} points) in {label_path}'
                )

            img_pad, scale, pad_x, pad_y, new_w, new_h = letterbox_image_gray(
                img,
                self.input_h,
                self.input_w,
                pad_value=0,
                downsample_sharpen=self.letterbox_downsample_sharpen,
            )

            pts_scaled = pts.copy()
            pts_scaled[:, 0] = pts_scaled[:, 0] * scale + pad_x
            pts_scaled[:, 1] = pts_scaled[:, 1] * scale + pad_y
            img_pad = np.stack([img_pad, img_pad, img_pad], axis=-1)

            if self.phase == 'train':
                img_pad, pts_scaled = self.train_aug(img_pad, pts_scaled)
            else:
                img_pad, pts_scaled = self.eval_aug(img_pad, pts_scaled)

            img_pad = np.clip(img_pad, 0, 255).astype(np.uint8)
            pts_scaled = pts_scaled.astype(np.float32)
            pts_scaled[:, 0] = np.clip(pts_scaled[:, 0], 0, self.input_w - 1)
            pts_scaled[:, 1] = np.clip(pts_scaled[:, 1], 0, self.input_h - 1)

            quads_abs = pts_scaled.reshape(self.K, 4, 2).astype(np.float32)
            quads_abs = self._fix_quad_order_after_aug(quads_abs)
            (
                gt_hm,
                gt_ind,
                gt_center_reg,
                gt_corner_reg,
                gt_reg_mask,
                centers_abs,
                collision_count,
            ) = self._build_targets(quads_abs, image_id=img_name)
            gt_centerline_hm, gt_progress_map = self._build_anatomy_targets(quads_abs)

            img_norm = self._normalize_input_image(img_pad)
            img_3c = np.transpose(img_norm, (2, 0, 1))

            return {
                'input_image': torch.from_numpy(img_3c).float(),
                'gt_global_hm': torch.from_numpy(gt_hm).float(),
                'gt_ind': torch.from_numpy(gt_ind).long(),
                'gt_center_reg': torch.from_numpy(gt_center_reg).float(),
                'gt_corner_reg': torch.from_numpy(gt_corner_reg).float(),
                'gt_reg_mask': torch.from_numpy(gt_reg_mask).float(),
                'gt_centerline_hm': torch.from_numpy(gt_centerline_hm).float(),
                'gt_progress_map': torch.from_numpy(gt_progress_map).float(),
                'p_gt': torch.from_numpy(quads_abs.reshape(-1, 2)).float(),
                'gt_centers_abs': torch.from_numpy(centers_abs).float(),
                'center_collision_count': torch.tensor(collision_count, dtype=torch.int64),
                'image_id': img_name,
                'ori_size': torch.tensor([ori_h, ori_w], dtype=torch.float32),
                'input_size': torch.tensor([self.input_h, self.input_w], dtype=torch.float32),
                'scale': torch.tensor([scale], dtype=torch.float32),
                'pad': torch.tensor([pad_x, pad_y], dtype=torch.float32),
                'resized_size': torch.tensor([new_h, new_w], dtype=torch.float32),
            }
        finally:
            self.input_h = int(prev_input_h)
            self.input_w = int(prev_input_w)

    def _load_points(self, mat_path):
        mat = loadmat(mat_path)
        if 'p2' not in mat:
            raise KeyError(f"'p2' not found in {mat_path}")

        pts = np.asarray(mat['p2'], dtype=np.float32)
        pts = np.squeeze(pts)

        if pts.ndim != 2:
            pts = pts.reshape(-1, 2)

        # 兼容 (2, N)
        if pts.shape[0] == 2 and pts.shape[1] != 2:
            pts = pts.T

        if pts.shape[1] != 2:
            pts = pts.reshape(-1, 2)

        return pts.astype(np.float32)

    def _fix_quad_order_after_aug(self, quads_abs):
        """
        只修复“左右翻转后左右顺序反了”的情况。
        不做任何几何重排，不做排序推断。
        原始语义固定是 [TL, TR, BL, BR]。
        """
        quads = quads_abs.copy().astype(np.float32)  # (K,4,2)

        for k in range(quads.shape[0]):
            # top pair: TL(0), TR(1)
            if quads[k, 0, 0] > quads[k, 1, 0]:
                quads[k, [0, 1]] = quads[k, [1, 0]]

            # bottom pair: BL(2), BR(3)
            if quads[k, 2, 0] > quads[k, 3, 0]:
                quads[k, [2, 3]] = quads[k, [3, 2]]

        return quads

    @staticmethod
    def _safe_unit(vec, fallback):
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm

        fallback = np.asarray(fallback, dtype=np.float32)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-6:
            return fallback / fallback_norm

        return np.array([1.0, 0.0], dtype=np.float32)

    def _expand_single_spine_quad(self, quad_feat, top_expand_ratio=0.0, bottom_expand_ratio=0.0):
        quad = np.asarray(quad_feat, dtype=np.float32).reshape(4, 2)
        tl, tr, bl, br = quad

        top_vec = tr - tl
        bottom_vec = br - bl
        left_vec = bl - tl
        right_vec = br - tr

        top_width = max(float(np.linalg.norm(top_vec)), 1e-6)
        bottom_width = max(float(np.linalg.norm(bottom_vec)), 1e-6)
        left_height = max(float(np.linalg.norm(left_vec)), 1e-6)
        right_height = max(float(np.linalg.norm(right_vec)), 1e-6)

        top_dir = self._safe_unit(top_vec, bottom_vec)
        bottom_dir = self._safe_unit(bottom_vec, top_vec)
        left_down_dir = self._safe_unit(left_vec, right_vec if np.linalg.norm(right_vec) > 1e-6 else [0.0, 1.0])
        right_down_dir = self._safe_unit(right_vec, left_vec if np.linalg.norm(left_vec) > 1e-6 else [0.0, 1.0])

        lateral_expand_ratio = float(getattr(self.args, 'spine_mask_lateral_expand_ratio', 0.20))

        tl_new = tl - lateral_expand_ratio * top_width * top_dir
        tr_new = tr + lateral_expand_ratio * top_width * top_dir
        bl_new = bl - lateral_expand_ratio * bottom_width * bottom_dir
        br_new = br + lateral_expand_ratio * bottom_width * bottom_dir

        if float(top_expand_ratio) > 0.0:
            tl_new = tl_new - float(top_expand_ratio) * left_height * left_down_dir
            tr_new = tr_new - float(top_expand_ratio) * right_height * right_down_dir

        if float(bottom_expand_ratio) > 0.0:
            bl_new = bl_new + float(bottom_expand_ratio) * left_height * left_down_dir
            br_new = br_new + float(bottom_expand_ratio) * right_height * right_down_dir

        return np.stack([tl_new, tr_new, bl_new, br_new], axis=0).astype(np.float32)

    def _apply_xray_preprocess(self, img_gray):
        img_gray = np.asarray(img_gray)
        if img_gray.ndim != 2:
            raise ValueError(f'Expected grayscale image, got shape {tuple(img_gray.shape)}')

        img_proc = img_gray.astype(np.float32)

        if _eval_switch_enabled(
            self.phase,
            self.args,
            'eval_preprocess_percentile',
            EVAL_PREPROCESS_PERCENTILE_ENV,
            default=True,
        ):
            low = float(TRAIN_PERCENTILE_RANGE[0])
            high = float(TRAIN_PERCENTILE_RANGE[1])
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
            'eval_preprocess_clahe',
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
            'eval_preprocess_normalize',
            EVAL_PREPROCESS_NORMALIZE_ENV,
            default=True,
        ):
            return img

        img = img / 255.0

        norm_mode = str(INPUT_NORM_MODE).lower()

        if norm_mode == 'minus_half':
            return img - 0.5

        if norm_mode == 'zscore_clip':
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True)
            std = np.maximum(std, 1e-6)
            img = (img - mean) / std
            z_clip = max(float(INPUT_ZSCORE_CLIP), 1e-3)
            img = np.clip(img, -z_clip, z_clip) / (2.0 * z_clip)
            return img

        raise ValueError(f'Unsupported input_norm_mode: {norm_mode}')

    def _compute_center_disk_params(self, quad_abs, center_abs):
        quad_feat = quad_abs.copy().astype(np.float32)
        quad_feat[:, 0] /= float(self.hm_w_r)
        quad_feat[:, 1] /= float(self.hm_h_r)

        center_feat = np.array([
            center_abs[0] / float(self.hm_w_r),
            center_abs[1] / float(self.hm_h_r),
        ], dtype=np.float32)

        width_feat = (
            abs(quad_feat[1, 0] - quad_feat[0, 0]) +
            abs(quad_feat[3, 0] - quad_feat[2, 0])
        ) * 0.5
        height_feat = (
            abs(quad_feat[2, 1] - quad_feat[0, 1]) +
            abs(quad_feat[3, 1] - quad_feat[1, 1])
        ) * 0.5

        rx = int(np.clip(
            round(float(getattr(self.args, 'center_disk_frac_x', 0.18)) * float(width_feat)),
            int(getattr(self.args, 'min_center_disk_r', 1)),
            int(getattr(self.args, 'max_center_disk_r', 4)),
        ))
        ry = int(np.clip(
            round(float(getattr(self.args, 'center_disk_frac_y', 0.18)) * float(height_feat)),
            int(getattr(self.args, 'min_center_disk_r', 1)),
            int(getattr(self.args, 'max_center_disk_r', 4)),
        ))

        return center_feat, quad_feat, rx, ry

    @staticmethod
    def _paint_dense_circle(mask, cx, cy, radius, value=1.0):
        H, W = mask.shape[-2:]
        x0 = max(int(cx - radius), 0)
        x1 = min(int(cx + radius), W - 1)
        y0 = max(int(cy - radius), 0)
        y1 = min(int(cy + radius), H - 1)
        rr2 = float(radius * radius)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= rr2:
                    mask[yy, xx] = max(float(mask[yy, xx]), float(value))

    def _build_anatomy_targets(self, quads_abs):
        Hf = int(math.ceil(self.input_h / float(self.hm_h_r)))
        Wf = int(math.ceil(self.input_w / float(self.hm_w_r)))

        gt_centerline = np.zeros((1, Hf, Wf), dtype=np.float32)
        gt_progress = np.zeros((1, Hf, Wf), dtype=np.float32)
        assign_dist = np.full((Hf, Wf), np.inf, dtype=np.float32)

        centers_abs = quads_abs.mean(axis=1).astype(np.float32)
        centers_feat = centers_abs.copy()
        centers_feat[:, 0] /= float(self.hm_w_r)
        centers_feat[:, 1] /= float(self.hm_h_r)
        progress_vals = np.linspace(0.0, 1.0, centers_feat.shape[0], dtype=np.float32) if centers_feat.shape[0] > 1 else np.zeros((1,), dtype=np.float32)

        for k in range(quads_abs.shape[0]):
            center_feat, _, rx, ry = self._compute_center_disk_params(quads_abs[k], centers_abs[k])
            radius = max(1, int(round(0.6 * max(rx, ry))))
            cx = float(center_feat[0])
            cy = float(center_feat[1])
            cx_int = int(np.clip(round(cx), 0, Wf - 1))
            cy_int = int(np.clip(round(cy), 0, Hf - 1))
            self._paint_dense_circle(gt_centerline[0], cx_int, cy_int, radius, value=1.0)

            x0 = max(cx_int - radius, 0)
            x1 = min(cx_int + radius, Wf - 1)
            y0 = max(cy_int - radius, 0)
            y1 = min(cy_int + radius, Hf - 1)
            rr2 = float(radius * radius)
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    dist2 = float((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy))
                    if dist2 > rr2:
                        continue
                    if dist2 >= assign_dist[yy, xx]:
                        continue
                    assign_dist[yy, xx] = dist2
                    gt_progress[0, yy, xx] = progress_vals[k]

        for k in range(centers_feat.shape[0] - 1):
            c0 = centers_feat[k]
            c1 = centers_feat[k + 1]
            seg = c1 - c0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            steps = max(int(math.ceil(seg_len * 2.0)), 2)
            radius = max(1, int(round(0.18 * seg_len)))
            for step in range(steps + 1):
                t = float(step) / float(steps)
                px = (1.0 - t) * c0[0] + t * c1[0]
                py = (1.0 - t) * c0[1] + t * c1[1]
                prog = (1.0 - t) * progress_vals[k] + t * progress_vals[k + 1]
                px_int = int(np.clip(round(px), 0, Wf - 1))
                py_int = int(np.clip(round(py), 0, Hf - 1))
                self._paint_dense_circle(gt_centerline[0], px_int, py_int, radius, value=1.0)

                x0 = max(px_int - radius, 0)
                x1 = min(px_int + radius, Wf - 1)
                y0 = max(py_int - radius, 0)
                y1 = min(py_int + radius, Hf - 1)
                rr2 = float(radius * radius)
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        dist2 = float((xx - px) * (xx - px) + (yy - py) * (yy - py))
                        if dist2 > rr2:
                            continue
                        if dist2 >= assign_dist[yy, xx]:
                            continue
                        assign_dist[yy, xx] = dist2
                        gt_progress[0, yy, xx] = prog

        return gt_centerline, gt_progress

    def _build_slot_targets(self, quads_abs, centers_abs):
        slot_centers = np.zeros((self.K, 2), dtype=np.float32)
        slot_sizes = np.zeros((self.K, 2), dtype=np.float32)
        slot_quads = np.zeros((self.K, 4, 2), dtype=np.float32)
        slot_valid = np.ones((self.K,), dtype=np.float32)

        denom_x = max(float(self.input_w - 1), 1.0)
        denom_y = max(float(self.input_h - 1), 1.0)
        denom_xy = np.array([denom_x, denom_y], dtype=np.float32)

        for k in range(min(self.K, quads_abs.shape[0])):
            quad_abs = quads_abs[k]
            center_abs = centers_abs[k]

            top_w = float(np.linalg.norm(quad_abs[1] - quad_abs[0]))
            bottom_w = float(np.linalg.norm(quad_abs[3] - quad_abs[2]))
            left_h = float(np.linalg.norm(quad_abs[2] - quad_abs[0]))
            right_h = float(np.linalg.norm(quad_abs[3] - quad_abs[1]))

            slot_centers[k, 0] = float(center_abs[0] / denom_x)
            slot_centers[k, 1] = float(center_abs[1] / denom_y)
            slot_sizes[k, 0] = float(((top_w + bottom_w) * 0.5) / denom_x)
            slot_sizes[k, 1] = float(((left_h + right_h) * 0.5) / denom_y)
            slot_quads[k] = quad_abs / denom_xy[None, :]

        return slot_centers, slot_sizes, slot_quads, slot_valid

    def _build_targets(self, quads_abs, image_id=None):
        """
        quads_abs: (K, 4, 2)
        点顺序固定是 [TL, TR, BL, BR]

        改成：
        - gt_hm 只做单点监督
        - x 按 hm_w_r 缩放
        - y 按 hm_h_r 缩放
        - center_reg 学亚像素偏移
        - corner_reg 在 anisotropic heatmap 坐标系下监督
        """
        Hf = int(math.ceil(self.input_h / float(self.hm_h_r)))
        Wf = int(math.ceil(self.input_w / float(self.hm_w_r)))

        gt_hm = np.zeros((1, Hf, Wf), dtype=np.float32)
        gt_ind = np.zeros((self.K,), dtype=np.int64)
        gt_center_reg = np.zeros((self.K, 2), dtype=np.float32)
        gt_corner_reg = np.zeros((self.K, 8), dtype=np.float32)
        gt_reg_mask = np.zeros((self.K,), dtype=np.float32)

        centers_abs = quads_abs.mean(axis=1)  # (K, 2)
        occupied = {}
        collision_count = 0
        for k in range(min(self.K, quads_abs.shape[0])):
            quad_abs = quads_abs[k]      # (4,2), [TL, TR, BL, BR]
            center_abs = centers_abs[k]  # (2,)

            center_feat, quad_feat, rx, ry = self._compute_center_disk_params(quad_abs, center_abs)
            del rx, ry

            cx, cy = float(center_feat[0]), float(center_feat[1])
            cx_int, cy_int = int(cx), int(cy)

            if not (0 <= cx_int < Wf and 0 <= cy_int < Hf):
                continue

            key = (cx_int, cy_int)
            if key in occupied:
                collision_count += 1
            occupied[key] = occupied.get(key, 0) + 1

            # 单点 heatmap
            bbox_w_feat = (
                abs(float(quad_feat[1, 0] - quad_feat[0, 0])) +
                abs(float(quad_feat[3, 0] - quad_feat[2, 0]))
            ) * 0.5
            bbox_h_feat = (
                abs(float(quad_feat[2, 1] - quad_feat[0, 1])) +
                abs(float(quad_feat[3, 1] - quad_feat[1, 1]))
            ) * 0.5
            radius = gaussian_radius(
                (
                    max(1, int(math.ceil(bbox_h_feat))),
                    max(1, int(math.ceil(bbox_w_feat))),
                )
            )
            radius = max(0, int(radius))
            draw_umich_gaussian(gt_hm[0], (cx_int, cy_int), radius=radius, k=1.0)

            gt_ind[k] = int(cy_int * Wf + cx_int)
            gt_center_reg[k, 0] = center_feat[0] - float(cx_int)
            gt_center_reg[k, 1] = center_feat[1] - float(cy_int)

            corner_offsets = (center_feat[None, :] - quad_feat).reshape(-1)
            gt_corner_reg[k] = corner_offsets.astype(np.float32)
            gt_reg_mask[k] = 1.0

        if self.debug_center_collision and collision_count > 0:
            print(
                f"[center-collision] image={image_id}, "
                f"collision_count={collision_count}, occupied_cells={len(occupied)}, K={quads_abs.shape[0]}"
            )

        return (
            gt_hm,
            gt_ind,
            gt_center_reg,
            gt_corner_reg,
            gt_reg_mask,
            centers_abs,
            collision_count,
        )

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        return self._build_sample(
            img_name=img_name,
            img_path=img_path,
        )
