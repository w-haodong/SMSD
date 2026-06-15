# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "as_cobb_triplet",
    "circular_angle_error_deg",
    "cobb_angle_calc",
    "cobb_angle_triplet_from_pts",
    "compute_cobb_error_summary",
    "compute_total_cobb_smape_percent",
    "draw_angle_and_extend_lines",
    "is_S",
    "single_angle_smape_percent",
]

def is_S(mid_p_v):
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num - 2):
        term1 = (mid_p_v[i, 1] - mid_p_v[num - 1, 1]) / (mid_p_v[0, 1] - mid_p_v[num - 1, 1])
        term2 = (mid_p_v[i, 0] - mid_p_v[num - 1, 0]) / (mid_p_v[0, 0] - mid_p_v[num - 1, 0])
        ll.append(term1 - term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]
    ll_pair = np.matmul(ll, np.transpose(ll))
    if ll_pair.shape[0] == 0:
        a = 0.0
        b = 0.0
    else:
        a = float(np.sum(ll_pair))
        b = float(np.sum(np.abs(ll_pair)))
    return abs(a - b) >= 1e-4


def _visual_style(image):
    h, w = image.shape[:2]
    base = max(1.0, float(min(h, w)))
    line_thickness = max(2, int(round(base / 260.0)))
    point_radius = max(2, int(round(base / 220.0)))
    font_scale = float(np.clip((base / 1100.0) * 2.0, 0.70, 1.10))
    font_thickness = max(2, int(round(base / 260.0)))
    return point_radius, line_thickness, font_scale, font_thickness


def _as_int_point(pt):
    pt = np.asarray(pt, dtype=np.float32).reshape(2)
    pt = np.nan_to_num(pt, nan=0.0, posinf=1e6, neginf=-1e6)
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _draw_clipped_line(image, pt1, pt2, color, thickness, line_type=cv2.LINE_AA):
    h, w = image.shape[:2]
    p1 = _as_int_point(pt1)
    p2 = _as_int_point(pt2)
    try:
        ok, q1, q2 = cv2.clipLine((0, 0, int(w), int(h)), p1, p2)
        if ok:
            cv2.line(image, q1, q2, color=color, thickness=int(thickness), lineType=line_type)
    except Exception:
        q1 = (int(np.clip(p1[0], 0, w - 1)), int(np.clip(p1[1], 0, h - 1)))
        q2 = (int(np.clip(p2[0], 0, w - 1)), int(np.clip(p2[1], 0, h - 1)))
        if q1 != q2:
            cv2.line(image, q1, q2, color=color, thickness=int(thickness), lineType=line_type)


def _put_text_inside(image, text, xy, font_scale, thickness, color):
    h, w = image.shape[:2]
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), int(thickness))
    text_w, text_h = text_size
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    margin = max(6, int(round(min(h, w) / 80.0)))
    x = int(np.clip(x, margin, max(margin, w - text_w - margin)))
    y = int(np.clip(y, text_h + margin, max(text_h + margin, h - baseline - margin)))
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_angle_and_extend_lines(image, mid_p, pos1, pos2, cobb_angle1, extend_ratio=2, offset_x=80):
    if int(pos1) == int(pos2):
        return

    line1_start = mid_p[pos1 * 2]
    line1_end = mid_p[pos1 * 2 + 1]
    line2_start = mid_p[pos2 * 2]
    line2_end = mid_p[pos2 * 2 + 1]

    line1_mid_y = (line1_start[1] + line1_end[1]) / 2
    line2_mid_y = (line2_start[1] + line2_end[1]) / 2

    if line1_mid_y < line2_mid_y:
        top_line_start = line1_start
        top_line_end = line1_end
        bottom_line_start = line2_start
        bottom_line_end = line2_end
    else:
        top_line_start = line2_start
        top_line_end = line2_end
        bottom_line_start = line1_start
        bottom_line_end = line1_end

    if abs(top_line_start[1] - bottom_line_start[1]) < abs(top_line_end[1] - bottom_line_end[1]):
        angle_midpoint = (top_line_start + bottom_line_start) / 2
        display_position = angle_midpoint.copy()
        display_position[0] = min(top_line_start[0], bottom_line_start[0]) - offset_x
        if image is not None:
            _, line_thickness, _, _ = _visual_style(image)
            angle_line_thickness = max(line_thickness + 1, int(round(min(image.shape[:2]) / 170.0)))
            distance_top = top_line_start - top_line_end
            distance_bottom = bottom_line_start - bottom_line_end
            _draw_clipped_line(
                image,
                top_line_start + extend_ratio * distance_top,
                top_line_start,
                color=(0, 255, 0),
                thickness=angle_line_thickness,
            )
            _draw_clipped_line(
                image,
                bottom_line_start + extend_ratio * distance_bottom,
                bottom_line_start,
                color=(0, 255, 0),
                thickness=angle_line_thickness,
            )
    else:
        angle_midpoint = (top_line_end + bottom_line_end) / 2
        display_position = angle_midpoint.copy()
        display_position[0] = max(top_line_end[0], bottom_line_end[0])
        if image is not None:
            _, line_thickness, _, _ = _visual_style(image)
            angle_line_thickness = max(line_thickness + 1, int(round(min(image.shape[:2]) / 170.0)))
            distance_top = top_line_start - top_line_end
            distance_bottom = bottom_line_start - bottom_line_end
            _draw_clipped_line(
                image,
                top_line_end,
                top_line_end - extend_ratio * distance_top,
                color=(0, 255, 0),
                thickness=angle_line_thickness,
            )
            _draw_clipped_line(
                image,
                bottom_line_end,
                bottom_line_end - extend_ratio * distance_bottom,
                color=(0, 255, 0),
                thickness=angle_line_thickness,
            )

    if image is not None:
        _, _, font_scale, font_thickness = _visual_style(image)
        _put_text_inside(
            image,
            "{:.2f}".format(cobb_angle1),
            display_position,
            font_scale=font_scale,
            thickness=font_thickness,
            color=(0, 0, 255),
        )


def cobb_angle_calc(pts, image=None, is_train=True):
    offset_h = 80
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if pts.shape[0] < 8 or pts.shape[0] % 4 != 0:
        raise ValueError(f"Cobb angle calculation needs 4-point vertebra groups, got {pts.shape[0]} points.")
    tr_ag = 0

    y_values = pts[:, 1]
    h_top = float(np.min(y_values))
    h_t = float(np.max(y_values) - np.min(y_values))

    num_pts = pts.shape[0]
    vnum = num_pts // 4 - 1

    mid_p_v = (pts[0::2, :] + pts[1::2, :]) / 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i, :] + pts[i + 2, :]) / 2
        pt2 = (pts[i + 1, :] + pts[i + 3, :]) / 2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)
    if image is not None:
        point_radius, _, _, _ = _visual_style(image)
        for pt in mid_p:
            cv2.circle(image, _as_int_point(pt), point_radius, (0, 255, 255), -1, cv2.LINE_AA)

    vec_m = mid_p[1::2, :] - mid_p[0::2, :]
    dot_v = np.matmul(vec_m, np.transpose(vec_m))
    mod_v = np.sqrt(np.sum(vec_m ** 2, axis=1))[:, np.newaxis]
    mod_v = np.matmul(mod_v, np.transpose(mod_v))
    cosine_angles = np.clip(dot_v / (mod_v + 1e-6), a_min=0.0, a_max=1.0)
    angles = np.arccos(cosine_angles)

    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = int(np.argmax(maxt))
    pos1_pos2 = int(pos1[pos2])
    cobb_angle1 = float(np.amax(maxt) / np.pi * 180.0)

    draw_angle_and_extend_lines(
        image=image,
        mid_p=mid_p,
        pos1=pos2,
        pos2=pos1_pos2,
        cobb_angle1=cobb_angle1,
        extend_ratio=2,
        offset_x=offset_h,
    )

    flag_s = is_S(mid_p_v)
    if not flag_s:
        cobb_angle2 = float(angles[0, pos2] / np.pi * 180.0)
        cobb_angle3 = float(angles[vnum, pos1_pos2] / np.pi * 180.0)

        draw_angle_and_extend_lines(
            image=image,
            mid_p=mid_p,
            pos1=0,
            pos2=pos2,
            cobb_angle1=cobb_angle2,
            extend_ratio=2,
            offset_x=offset_h,
        )
        draw_angle_and_extend_lines(
            image=image,
            mid_p=mid_p,
            pos1=vnum,
            pos2=pos1_pos2,
            cobb_angle1=cobb_angle3,
            extend_ratio=2,
            offset_x=offset_h,
        )

        cba_pt = round(cobb_angle2, 2)
        cba_mt = round(cobb_angle1, 2)
        cba_tl = round(cobb_angle3, 2)
        pt_pair = (0, pos2)
        mt_pair = (pos2, pos1_pos2)
        tl_pair = (vnum, pos1_pos2)
    else:
        if (mid_p_v[pos2 * 2, 1] - h_top + mid_p_v[pos1_pos2 * 2, 1] - h_top) < h_t:
            angle2 = angles[pos2, :(pos2 + 1)]
            cobb_angle2 = float(np.max(angle2) / np.pi * 180.0)
            pos1_1 = int(np.argmax(angle2))

            angle3 = angles[pos1_pos2, pos1_pos2:(vnum + 1)]
            cobb_angle3 = float(np.max(angle3) / np.pi * 180.0)
            pos1_2 = int(np.argmax(angle3))
            pos1_2 = pos1_2 + pos1_pos2 - 1

            draw_angle_and_extend_lines(
                image=image,
                mid_p=mid_p,
                pos1=pos1_1,
                pos2=pos2,
                cobb_angle1=cobb_angle2,
                extend_ratio=2,
                offset_x=offset_h,
            )
            draw_angle_and_extend_lines(
                image=image,
                mid_p=mid_p,
                pos1=pos1_2,
                pos2=pos1_pos2,
                cobb_angle1=cobb_angle3,
                extend_ratio=2,
                offset_x=offset_h,
            )

            cba_pt = round(cobb_angle2, 2)
            cba_mt = round(cobb_angle1, 2)
            cba_tl = round(cobb_angle3, 2)
            pt_pair = (pos1_1, pos2)
            mt_pair = (pos2, pos1_pos2)
            tl_pair = (pos1_2, pos1_pos2)
        else:
            angle2 = angles[pos2, :(pos2 + 1)]
            cobb_angle2 = float(np.max(angle2) / np.pi * 180.0)
            pos1_1 = int(np.argmax(angle2))

            angle3 = angles[pos1_1, :(pos1_1 + 1)]
            cobb_angle3 = float(np.max(angle3) / np.pi * 180.0)
            pos1_2 = int(np.argmax(angle3))

            draw_angle_and_extend_lines(
                image=image,
                mid_p=mid_p,
                pos1=pos1_1,
                pos2=pos2,
                cobb_angle1=cobb_angle2,
                extend_ratio=2,
                offset_x=offset_h,
            )
            draw_angle_and_extend_lines(
                image=image,
                mid_p=mid_p,
                pos1=pos1_1,
                pos2=pos1_2,
                cobb_angle1=cobb_angle3,
                extend_ratio=2,
                offset_x=offset_h,
            )

            cba_pt = round(cobb_angle3, 2)
            cba_mt = round(cobb_angle2, 2)
            cba_tl = round(cobb_angle1, 2)
            pt_pair = (pos1_1, pos1_2)
            mt_pair = (pos1_1, pos2)
            tl_pair = (pos2, pos1_pos2)

    if cba_pt <= tr_ag:
        cba_pt = 0
    if cba_mt <= tr_ag:
        cba_mt = 0
    if cba_tl <= tr_ag:
        cba_tl = 0

    pos_list = [
        min(pt_pair) + 1,
        min(mt_pair) + 1,
        max(mt_pair) + 1,
        max(tl_pair) + 1,
    ]
    if is_train:
        return cba_pt, cba_mt, cba_tl
    return cba_pt, cba_mt, cba_tl, pos_list


def cobb_angle_triplet_from_pts(pts_abs):
    pts = np.asarray(pts_abs, np.float32).reshape(-1, 2)
    if pts.shape[0] < 8 or pts.shape[0] % 4 != 0:
        return None
    cba_pt, cba_mt, cba_tl = cobb_angle_calc(pts, image=None, is_train=True)
    return np.asarray([cba_pt, cba_mt, cba_tl], dtype=np.float32)


def as_cobb_triplet(pts_abs):
    return cobb_angle_triplet_from_pts(pts_abs)


def circular_angle_error_deg(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = pred - gt
    return np.degrees(np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff))))


def single_angle_smape_percent(pred, gt, eps=1e-5):
    pred = float(pred)
    gt = float(gt)
    denom = pred + gt
    if abs(denom) < float(eps):
        denom = float(eps)
    return abs(float(circular_angle_error_deg(np.asarray([pred]), np.asarray([gt]))[0])) / denom * 100.0


def compute_total_cobb_smape_percent(pred, gt, eps=1e-5):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.size == 0 or gt.size == 0:
        return 0.0

    pred = pred.reshape(-1, 3)
    gt = gt.reshape(-1, 3)
    out_abs = np.abs(circular_angle_error_deg(pred, gt))
    term1 = np.sum(out_abs, axis=1)
    term2 = np.sum(pred + gt, axis=1)
    term2[np.abs(term2) < float(eps)] = float(eps)
    return float(np.mean(term1 / term2 * 100.0))


def compute_cobb_error_summary(pred, gt):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 3)
    abs_diff = np.abs(circular_angle_error_deg(pred, gt))
    return {
        "smape": compute_total_cobb_smape_percent(pred, gt),
        "cmae": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "pt": float(np.mean(abs_diff[:, 0])) if abs_diff.size else 0.0,
        "mt": float(np.mean(abs_diff[:, 1])) if abs_diff.size else 0.0,
        "lt": float(np.mean(abs_diff[:, 2])) if abs_diff.size else 0.0,
    }
