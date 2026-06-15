# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class ConvLNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=groups,
            bias=False,
        )
        self.norm = LayerNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvLNAct(channels, channels, k=3, s=1, p=1, groups=channels),
            ConvLNAct(channels, channels, k=1, s=1, p=0, act=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, bias_init: float | None = None):
        super().__init__()
        self.head = nn.Sequential(
            ConvLNAct(in_channels, hidden_channels, k=3, s=1, p=1),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )
        if bias_init is not None:
            nn.init.constant_(self.head[-1].bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ScaleProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvLNAct(in_channels, out_channels, k=1, s=1, p=0),
            ResidualDWBlock(out_channels),
            ConvLNAct(out_channels, out_channels, k=3, s=1, p=1, groups=out_channels),
            ConvLNAct(out_channels, out_channels, k=1, s=1, p=0),
            ResidualDWBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _x_coords(width: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, width, device=ref.device, dtype=ref.dtype).view(1, 1, 1, width)


def _interior_readout_mask(ref: torch.Tensor, margin_ratio: float = 0.025, min_margin: int = 2) -> torch.Tensor:
    _, _, height, width = ref.shape
    if width <= 2:
        return ref.new_ones((ref.size(0), 1, height, width))
    margin = max(int(min_margin), int(round(width * float(margin_ratio))))
    margin = min(margin, max(1, width // 8))
    x = torch.arange(width, device=ref.device, dtype=ref.dtype)
    dist_to_edge = torch.minimum(x, (width - 1) - x)
    x_mask = (dist_to_edge / float(margin)).clamp(0.0, 1.0).view(1, 1, 1, width)
    return x_mask.expand(ref.size(0), 1, height, width)


def render_axis_gaussian(
    center: torch.Tensor,
    width: torch.Tensor,
    amplitude: torch.Tensor,
    out_w: int,
    min_sigma: float = 0.015,
) -> torch.Tensor:
    x = _x_coords(out_w, center)
    sigma = width.clamp(min=min_sigma)
    return amplitude * torch.exp(-0.5 * ((x - center) / sigma) ** 2)


def _logit_from_prob(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def _soft_threshold_gate(gate: torch.Tensor, threshold: float, softness: float = 0.08) -> torch.Tensor:
    gate = gate.clamp(0.0, 1.0)
    denom = max(float(softness), 1e-4)
    return gate * torch.sigmoid((gate - float(threshold)) / denom)


def _straight_through_topk_mask(
    scores: torch.Tensor,
    keep_ratio: float,
    min_keep: int = 8,
    pool_kernel: Tuple[int, int] | None = None,
) -> torch.Tensor:
    if scores.ndim != 4:
        raise ValueError(f"topk mask expects 4D scores, got {tuple(scores.shape)}")
    bsz = scores.size(0)
    flat = scores.view(bsz, -1)
    num_items = flat.size(1)
    num_keep = max(int(min_keep), int(round(num_items * float(keep_ratio))))
    num_keep = max(1, min(num_keep, num_items))
    topk_idx = torch.topk(flat, k=num_keep, dim=1).indices
    hard = flat.new_zeros(flat.shape)
    hard.scatter_(1, topk_idx, 1.0)
    hard = hard.view_as(scores)
    if pool_kernel is not None:
        kh, kw = int(pool_kernel[0]), int(pool_kernel[1])
        hard = F.max_pool2d(hard, kernel_size=(kh, kw), stride=1, padding=((kh - 1) // 2, (kw - 1) // 2))
        hard = hard.clamp(0.0, 1.0)
    return (hard - scores).detach() + scores


def _soft_contiguous_row_support(
    scores: torch.Tensor,
    keep_ratio: float,
    min_keep: int = 8,
    smooth_kernel: int = 9,
    sharpness: float = 10.0,
) -> torch.Tensor:
    if scores.ndim != 4:
        raise ValueError(f"row support expects 4D scores, got {tuple(scores.shape)}")
    _, _, height, _ = scores.shape
    row_ratio = max(float(keep_ratio), float(min_keep) / max(1.0, float(height)))
    row_ratio = min(0.95, max(0.08, row_ratio))

    support_score = scores.clamp(0.0, 1.0)
    if smooth_kernel > 1:
        pad = (int(smooth_kernel) - 1) // 2
        support_score = F.avg_pool2d(
            support_score,
            kernel_size=(int(smooth_kernel), 1),
            stride=1,
            padding=(pad, 0),
            count_include_pad=False,
        )

    y = torch.linspace(-1.0, 1.0, height, device=scores.device, dtype=scores.dtype).view(1, 1, height, 1)
    weights = support_score / support_score.sum(dim=2, keepdim=True).clamp(min=1e-6)
    center = (weights * y).sum(dim=2, keepdim=True)
    variance = (weights * (y - center).square()).sum(dim=2, keepdim=True)
    span = variance.sqrt() * 1.35 + row_ratio * 0.14
    span = span.clamp(min=max(0.08, row_ratio * 0.28), max=min(0.82, row_ratio * 0.78 + 0.08))

    interval = torch.sigmoid((y - (center - span)) * float(sharpness))
    interval = interval * torch.sigmoid(((center + span) - y) * float(sharpness))
    support_floor = 0.08 + 0.92 * support_score
    return (support_floor * interval).clamp(0.0, 1.0)


def _suppress_short_row_islands(
    row_mask: torch.Tensor,
    max_bridge_gap: int = 10,
    min_component_len: int = 4,
) -> torch.Tensor:
    if row_mask.ndim != 4:
        raise ValueError(f"row island suppression expects 4D mask, got {tuple(row_mask.shape)}")
    row_mask = row_mask.clamp(0.0, 1.0)
    hard = (row_mask > 0.5).to(dtype=row_mask.dtype)
    cleaned = torch.zeros_like(hard)
    height = int(row_mask.size(2))
    keep_gap = max(2, min(int(max_bridge_gap), max(3, height // 16)))
    near_gap = max(2, keep_gap // 2)
    min_len = max(1, int(min_component_len))

    for b in range(hard.size(0)):
        rows = hard[b, 0, :, 0]
        idx = torch.nonzero(rows > 0.5, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if idx.numel() == 1:
            cleaned[b, 0, idx[0], 0] = 1.0
            continue

        gaps = idx[1:] - idx[:-1]
        breaks = torch.nonzero(gaps > 1, as_tuple=False).flatten()
        starts = torch.cat([idx[:1], idx[breaks + 1]]) if breaks.numel() > 0 else idx[:1]
        ends = torch.cat([idx[breaks], idx[-1:]]) if breaks.numel() > 0 else idx[-1:]
        lengths = ends - starts + 1
        best = int(torch.argmax(lengths).item())
        kept_components = [
            (int(starts[best].item()), int(ends[best].item()), int(lengths[best].item()))
        ]

        pending = [
            (int(starts[i].item()), int(ends[i].item()), int(lengths[i].item()))
            for i in range(int(starts.numel()))
            if i != best
        ]
        changed = True
        while changed and pending:
            changed = False
            kept = []
            for start, end, length in pending:
                gap = min(
                    max(0, kept_start - end - 1) if end < kept_start else max(0, start - kept_end - 1)
                    for kept_start, kept_end, _ in kept_components
                )
                if (gap <= keep_gap and length >= min_len) or gap <= near_gap:
                    kept_components.append((start, end, length))
                    changed = True
                else:
                    kept.append((start, end, length))
            pending = kept

        for start, end, _ in kept_components:
            cleaned[b, 0, start:end + 1, 0] = 1.0

    return (cleaned - row_mask).detach() + row_mask


class SemanticP5Bias(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre = nn.Sequential(
            ResidualDWBlock(channels),
            ConvLNAct(channels, channels, k=1, s=1, p=0),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p5_vec = F.adaptive_avg_pool2d(self.pre(p5), output_size=1)
        gate = self.gate(p5_vec)
        p4_mod = p4 * (0.70 + 0.60 * gate)
        gate_map = gate.mean(dim=1, keepdim=True).expand(-1, 1, p4.size(-2), p4.size(-1))
        return p4_mod, gate_map


class ExplicitAxisRepresentation(nn.Module):
    def __init__(self, channels: int, min_width: float = 0.035, max_width: float = 0.36, keep_ratio: float = 0.60):
        super().__init__()
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.keep_ratio = float(min(0.95, max(0.20, keep_ratio)))

        self.pre = nn.Sequential(
            ConvLNAct(channels, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ConvLNAct(channels, channels, k=(9, 1), s=1, p=(4, 0), groups=channels),
            ConvLNAct(channels, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
        )
        self.center_score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.row_score_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid(),
        )
        self.row_norm = nn.LayerNorm(channels)
        self.row_attn = SparsePolicyAttention(dim=channels, num_heads=4, attn_drop=0.1, proj_drop=0.1)
        self.row_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.center_delta_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
            nn.Tanh(),
        )
        self.width_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid(),
        )
        self.visible_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
            nn.Sigmoid(),
        )
        self.axis_out = nn.Sequential(
            ConvLNAct(channels * 2 + 3, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )

    def forward(self, p4: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, _, _, width = p4.shape
        axis_seed = self.pre(p4)
        center_score = self.center_score(axis_seed)
        center_attn = torch.softmax(center_score, dim=-1)

        x = _x_coords(width, p4)
        axis_center = (center_attn * x).sum(dim=-1, keepdim=True)

        row_tokens = (axis_seed * center_attn).sum(dim=-1).transpose(1, 2).contiguous()
        row_score_raw = self.row_score_head(row_tokens)
        row_score_raw_map = row_score_raw.transpose(1, 2).unsqueeze(-1)
        row_score_smooth = F.avg_pool2d(
            row_score_raw_map,
            kernel_size=(11, 1),
            stride=1,
            padding=(5, 0),
            count_include_pad=False,
        )
        row_span_prior = _soft_contiguous_row_support(
            row_score_smooth,
            keep_ratio=max(self.keep_ratio, 0.62),
            min_keep=max(18, p4.size(-2) // 5),
            smooth_kernel=15,
            sharpness=9.0,
        )
        row_score_map = ((0.60 * row_score_smooth + 0.40 * row_score_raw_map) * (0.20 + 0.80 * row_span_prior)).clamp(0.0, 1.0)
        row_hard = _straight_through_topk_mask(
            row_score_map,
            keep_ratio=self.keep_ratio,
            min_keep=max(12, p4.size(-2) // 7),
            pool_kernel=(5, 1),
        )
        row_hard_raw = row_hard
        row_hard = _suppress_short_row_islands(
            row_hard_raw,
            max_bridge_gap=max(6, p4.size(-2) // 32),
            min_component_len=max(3, p4.size(-2) // 96),
        )
        row_soft_pre = _soft_contiguous_row_support(
            row_score_map,
            keep_ratio=max(self.keep_ratio, 0.78),
            min_keep=max(24, p4.size(-2) // 4),
            smooth_kernel=17,
            sharpness=6.0,
        )
        row_support_gate = torch.maximum(row_hard, 0.55 * row_soft_pre).clamp(0.0, 1.0)
        row_policy = (0.12 + 0.88 * row_support_gate).squeeze(-1).transpose(1, 2).contiguous()
        row_attn_out, row_attn_map = self.row_attn(self.row_norm(row_tokens), row_policy, return_attn=True)
        row_context = row_tokens + row_attn_out
        row_context = row_context + self.row_ffn(row_context)

        center_delta = self.center_delta_head(row_context).transpose(1, 2).unsqueeze(-1)
        axis_center = (axis_center + 0.08 * center_delta).clamp(-1.0, 1.0)
        width_prob = self.width_head(row_context).transpose(1, 2).unsqueeze(-1)
        raw_axis_width = self.min_width + (self.max_width - self.min_width) * width_prob
        visible_prob = self.visible_head(row_context).transpose(1, 2).unsqueeze(-1)
        row_soft = _soft_contiguous_row_support(
            row_score_map,
            keep_ratio=max(self.keep_ratio, 0.72),
            min_keep=max(18, p4.size(-2) // 5),
            smooth_kernel=17,
            sharpness=7.0,
        )
        row_soft = torch.maximum(torch.maximum(row_soft, row_span_prior), row_support_gate)
        visible_smooth = F.avg_pool2d(
            visible_prob,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            count_include_pad=False,
        )
        axis_visible = torch.sigmoid((visible_smooth - 0.52) * 14.0).clamp(0.0, 1.0)
        height_map = axis_visible.expand(-1, -1, p4.size(-2), width)
        center_width = (raw_axis_width * 0.16 + 0.010).clamp(min=0.010, max=0.042)
        corridor_width = (raw_axis_width * 0.26 + 0.016).clamp(min=0.026, max=0.070)

        centerline_prob = render_axis_gaussian(
            axis_center,
            center_width,
            axis_visible,
            out_w=width,
            min_sigma=0.012,
        )
        band_prob = render_axis_gaussian(
            axis_center,
            corridor_width,
            axis_visible,
            out_w=width,
            min_sigma=0.018,
        )
        sparse_support_map = render_axis_gaussian(
            axis_center,
            corridor_width,
            row_support_gate * axis_visible,
            out_w=width,
            min_sigma=0.014,
        )
        sparse_score_map = render_axis_gaussian(
            axis_center,
            corridor_width,
            row_score_map * axis_visible,
            out_w=width,
            min_sigma=0.014,
        )
        row_attention_profile = row_attn_map.mean(dim=1).unsqueeze(1).unsqueeze(-1)
        row_attention_map = render_axis_gaussian(
            axis_center,
            corridor_width,
            row_attention_profile * axis_visible,
            out_w=width,
            min_sigma=0.014,
        )
        confidence_map = height_map.clamp(0.0, 1.0)
        row_context_map = row_context.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, width)
        axis_feat = self.axis_out(torch.cat([axis_seed, row_context_map, centerline_prob, band_prob, confidence_map], dim=1))

        state = {
            "center": axis_center,
            "width": corridor_width,
            "raw_width": raw_axis_width,
            "center_width": center_width,
            "visible": axis_visible,
            "visible_prob": visible_prob,
            "row_validity": visible_prob,
            "height_map": height_map,
            "centerline_prob": centerline_prob,
            "band_prob": band_prob,
            "confidence_map": confidence_map,
            "center_attn": center_attn,
            "row_support": row_support_gate,
            "row_support_raw": row_hard_raw,
            "row_soft_support": row_soft,
            "row_span_prior": row_span_prior,
            "row_sparse_map": sparse_support_map,
            "row_score_map": sparse_score_map,
            "row_score_raw_map": render_axis_gaussian(
                axis_center,
                corridor_width,
                row_score_raw_map * axis_visible,
                out_w=width,
                min_sigma=0.014,
            ),
            "row_score": row_score_map,
            "row_attention": row_attn_map,
            "row_attention_map": row_attention_map,
        }
        return axis_feat, state


class AxisCanonicalStripSampler(nn.Module):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = int(max(num_samples, 5))

    def forward(
        self,
        feat: torch.Tensor,
        center: torch.Tensor,
        width: torch.Tensor,
        width_scale: float = 1.0,
    ) -> torch.Tensor:
        _, _, height, _ = feat.shape
        offsets = torch.linspace(
            -1.0,
            1.0,
            self.num_samples,
            device=feat.device,
            dtype=feat.dtype,
        ).view(1, 1, 1, self.num_samples)
        x_grid = center + offsets * width * float(width_scale)
        y_grid = torch.linspace(
            -1.0,
            1.0,
            height,
            device=feat.device,
            dtype=feat.dtype,
        ).view(1, height, 1).expand(feat.size(0), height, self.num_samples)
        grid = torch.stack([x_grid.squeeze(1), y_grid], dim=-1)
        return F.grid_sample(
            feat,
            grid.clamp(-1.0, 1.0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


class SparsePolicyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.1, proj_drop: float = 0.1):
        super().__init__()
        self.num_heads = int(max(1, num_heads))
        head_dim = max(8, dim // self.num_heads)
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def softmax_with_policy(attn: torch.Tensor, policy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        bsz, num_tokens, _ = policy.size()
        _, _, n, _ = attn.size()
        if n != num_tokens:
            raise ValueError(f"Policy/token mismatch: attn has {n}, policy has {num_tokens}")
        attn_policy = policy.view(bsz, 1, 1, num_tokens)
        eye = torch.eye(num_tokens, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, num_tokens, num_tokens)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / num_tokens) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x: torch.Tensor, policy: torch.Tensor | None = None, return_attn: bool = False):
        bsz, num_tokens, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for t in qkv]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attn.mean(dim=1)
        return out


class TokenPolicyPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.local_linear = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.GELU(),
        )
        self.global_linear = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(tokens)
        bsz, num_tokens, dim = x.shape
        local_x = self.local_linear(x[:, :, : dim // 2])
        denom = policy.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_x = (x[:, :, dim // 2 :] * policy).sum(dim=1, keepdim=True) / denom
        global_x = global_x + self.global_linear(context)
        ax = torch.cat([local_x, global_x.expand(bsz, num_tokens, dim // 2)], dim=-1)
        return self.out(ax)


class LocalCandidateSparseAttention(nn.Module):
    def __init__(self, channels: int, keep_ratio: float = 0.05, min_keep: int = 96):
        super().__init__()
        self.keep_ratio = float(min(0.25, max(0.005, keep_ratio)))
        self.min_keep = int(max(16, min_keep))
        self.score_head = nn.Sequential(
            ConvLNAct(channels, channels, k=3, s=1, p=1, groups=channels),
            ConvLNAct(channels, channels, k=1, s=1, p=0),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.token_norm = nn.LayerNorm(channels)
        self.token_attn = SparsePolicyAttention(dim=channels, num_heads=4, attn_drop=0.1, proj_drop=0.1)
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.out = nn.Sequential(
            ConvLNAct(channels, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
        )

    def forward(self, feat: torch.Tensor, support: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = feat.shape
        support = support.clamp(0.0, 1.0)
        score = self.score_head(feat) * (0.05 + 0.95 * support)
        score = F.avg_pool2d(score, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        num_positions = height * width
        num_keep = min(num_positions, max(self.min_keep, int(round(num_positions * self.keep_ratio))))
        keep_mask = _straight_through_topk_mask(
            score,
            keep_ratio=self.keep_ratio,
            min_keep=num_keep,
            pool_kernel=(1, 1),
        )

        flat_score = score.flatten(2).squeeze(1)
        topk_idx = torch.topk(flat_score, k=num_keep, dim=1).indices
        flat_feat = feat.flatten(2).transpose(1, 2).contiguous()
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, channels)
        tokens = torch.gather(flat_feat, dim=1, index=gather_idx)

        attn_out, attn_map = self.token_attn(self.token_norm(tokens), None, return_attn=True)
        tokens = tokens + attn_out
        tokens = tokens + self.token_ffn(tokens)

        score_norm = score / score.flatten(2).amax(dim=2).view(bsz, 1, 1, 1).clamp_min(1e-6)
        selected_gate = (keep_mask * score_norm * support).clamp(0.0, 1.0)
        selected_gate = _soft_threshold_gate(selected_gate, threshold=0.045, softness=0.020)

        update_flat = torch.zeros_like(flat_feat)
        update_flat.scatter_(1, gather_idx, tokens)
        update = update_flat.transpose(1, 2).reshape(bsz, channels, height, width)
        update = self.out(update) * selected_gate
        token_attn_score = attn_map.mean(dim=1)
        attn_flat = torch.zeros(bsz, num_positions, 1, dtype=feat.dtype, device=feat.device)
        attn_flat.scatter_(1, topk_idx.unsqueeze(-1), token_attn_score.unsqueeze(-1))
        attn_response = attn_flat.transpose(1, 2).reshape(bsz, 1, height, width)
        return update, selected_gate, score, attn_response, attn_map.unsqueeze(1)


class CanonicalStripContext(nn.Module):
    def __init__(self, channels: int, num_samples: int = 17, keep_ratio: float = 0.7, wide_scale: float = 1.7):
        super().__init__()
        self.keep_ratio = float(min(0.95, max(0.25, keep_ratio)))
        self.wide_scale = float(max(1.2, wide_scale))
        self.strip_sampler = AxisCanonicalStripSampler(num_samples=num_samples)

        self.guide_proj = nn.Sequential(
            ConvLNAct(3, channels, k=1, s=1, p=0),
            ConvLNAct(channels, channels, k=1, s=1, p=0),
        )
        self.pre = nn.Sequential(
            ConvLNAct(channels * 3, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )
        self.strip_score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.token_norm = nn.LayerNorm(channels)
        self.token_attn = SparsePolicyAttention(dim=channels, num_heads=4, attn_drop=0.1, proj_drop=0.1)
        self.token_ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.policy_predictor = TokenPolicyPredictor(channels)
        self.axial_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
        )
        self.out = nn.Sequential(
            ConvLNAct(channels * 3, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )

    def _eval_policy(self, logits: torch.Tensor) -> torch.Tensor:
        keep_prob = torch.softmax(logits, dim=-1)[..., 0:1]
        bsz, num_tokens, _ = keep_prob.shape
        num_keep = max(8, int(round(num_tokens * self.keep_ratio)))
        topk_idx = torch.topk(keep_prob.squeeze(-1), k=min(num_keep, num_tokens), dim=1).indices
        policy = keep_prob.new_zeros(bsz, num_tokens, 1)
        policy.scatter_(1, topk_idx.unsqueeze(-1), 1.0)
        return policy

    def _encode_tokens(
        self,
        strip_feat: torch.Tensor,
        strip_band: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        strip_score = self.strip_score(strip_feat)
        strip_attn = torch.softmax(strip_score + strip_band, dim=-1)
        tokens = (strip_feat * strip_attn).sum(dim=-1).transpose(1, 2).contiguous()

        prev_policy = torch.ones(tokens.size(0), tokens.size(1), 1, device=tokens.device, dtype=tokens.dtype)
        policy_logits = self.policy_predictor(tokens, context, prev_policy)
        keep_prob = torch.softmax(policy_logits, dim=-1)[..., 0:1]
        keep_map = keep_prob.transpose(1, 2).unsqueeze(-1)
        keep_policy = _straight_through_topk_mask(
            keep_map,
            keep_ratio=self.keep_ratio,
            min_keep=max(8, int(round(tokens.size(1) * self.keep_ratio))),
            pool_kernel=(5, 1),
        ).squeeze(-1).transpose(1, 2)
        keep_policy = keep_policy * prev_policy
        keep_policy = keep_policy + (1.0 - keep_policy) * 0.035

        attn_out, attn_map = self.token_attn(self.token_norm(tokens), keep_policy, return_attn=True)
        mixed_tokens = tokens + attn_out
        mixed_tokens = mixed_tokens + self.token_ffn(mixed_tokens)
        axial_logits = self.axial_head(mixed_tokens).transpose(1, 2).unsqueeze(-1)
        attn_response = attn_map.mean(dim=1).unsqueeze(-1)
        return mixed_tokens, axial_logits, strip_attn, keep_policy, keep_prob, attn_response, attn_map

    @staticmethod
    def _render_row_tokens(tokens: torch.Tensor, center: torch.Tensor, width: torch.Tensor, out_w: int) -> torch.Tensor:
        amplitude = tokens.transpose(1, 2).unsqueeze(-1)
        spread = render_axis_gaussian(center, width * 0.95 + 0.020, amplitude.new_ones(center.shape), out_w, min_sigma=0.02)
        return amplitude * spread

    @staticmethod
    def _render_strip_weights(
        weights: torch.Tensor,
        center: torch.Tensor,
        width: torch.Tensor,
        out_w: int,
        width_scale: float,
    ) -> torch.Tensor:
        bsz, _, height, num_samples = weights.shape
        offsets = torch.linspace(
            -1.0,
            1.0,
            num_samples,
            device=weights.device,
            dtype=weights.dtype,
        ).view(1, 1, 1, num_samples)
        sample_x = center.unsqueeze(-1) + offsets * width.unsqueeze(-1) * float(width_scale)
        x = _x_coords(out_w, weights).unsqueeze(-1)
        sigma = (width.unsqueeze(-1) * float(width_scale) / max(2.0, float(num_samples))).clamp(min=0.012)
        mix = torch.exp(-0.5 * ((x - sample_x) / sigma) ** 2) * weights.unsqueeze(-2)
        return mix.sum(dim=-1)

    def forward(
        self,
        p3: torch.Tensor,
        axis_feat: torch.Tensor,
        guide_struct: torch.Tensor,
        axis_center: torch.Tensor,
        axis_width: torch.Tensor,
        axis_visible: torch.Tensor,
        p4_sparse_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        band = guide_struct[:, 1:2].clamp(0.0, 1.0)
        guide_center = guide_struct[:, 0:1].clamp(0.0, 1.0)
        vis_map = axis_visible.expand(-1, 1, -1, p3.size(-1))
        if p4_sparse_prior is None:
            p4_prior_map = guide_center
        else:
            p4_prior_map = p4_sparse_prior.float().clamp(0.0, 1.0)
            if p4_prior_map.shape[-2:] != p3.shape[-2:]:
                p4_prior_map = F.interpolate(p4_prior_map, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            p4_prior_map = p4_prior_map.to(dtype=p3.dtype)
        p4_prior_map = F.avg_pool2d(
            p4_prior_map,
            kernel_size=(5, 3),
            stride=1,
            padding=(2, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * vis_map
        p4_prior_row = p4_prior_map.amax(dim=-1).transpose(1, 2).clamp(0.0, 1.0)

        structure_gate = (0.36 * guide_center + 0.24 * band + 0.40 * p4_prior_map).clamp(0.0, 1.0) * vis_map
        structure_gate = F.avg_pool2d(
            structure_gate,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0)
        background_gate = ((1.0 - structure_gate) * vis_map).clamp(0.0, 1.0)
        guide_feat = self.guide_proj(guide_struct)
        pre_fused = self.pre(torch.cat([p3, axis_feat, guide_feat], dim=1))
        smooth_p3 = F.avg_pool2d(
            p3,
            kernel_size=(3, 5),
            stride=1,
            padding=(1, 2),
            count_include_pad=False,
        )
        fused = pre_fused * (0.78 + 0.22 * structure_gate) + smooth_p3 * (0.04 + 0.08 * background_gate)
        context_weight = 0.35 + 0.65 * structure_gate
        context = (axis_feat * context_weight).sum(dim=(-2, -1), keepdim=False) / context_weight.sum(dim=(-2, -1), keepdim=False).clamp_min(1e-6)
        context = context.unsqueeze(1)

        narrow_strip = self.strip_sampler(fused, axis_center, axis_width, width_scale=1.0)
        wide_strip = self.strip_sampler(fused, axis_center, axis_width, width_scale=self.wide_scale)
        narrow_band = self.strip_sampler(band, axis_center, axis_width, width_scale=1.0)
        wide_band = self.strip_sampler(band, axis_center, axis_width, width_scale=self.wide_scale)
        narrow_p4_prior = self.strip_sampler(p4_prior_map, axis_center, axis_width, width_scale=1.0)
        wide_p4_prior = self.strip_sampler(p4_prior_map, axis_center, axis_width, width_scale=self.wide_scale)
        narrow_policy = (0.62 * narrow_band + 0.38 * narrow_p4_prior).clamp(0.0, 1.0)
        wide_policy = (0.66 * wide_band + 0.34 * wide_p4_prior).clamp(0.0, 1.0)

        narrow_tokens, narrow_axial_logits, narrow_attn, narrow_keep, narrow_keep_prob, narrow_token_attn, narrow_attn_matrix = self._encode_tokens(narrow_strip, narrow_policy, context)
        wide_tokens, _, wide_attn, wide_keep, _, _, _ = self._encode_tokens(wide_strip, wide_policy, context)

        vis_row = axis_visible.squeeze(-1).transpose(1, 2)
        narrow_keep = narrow_keep * vis_row
        wide_keep = wide_keep * vis_row
        local_keep = 0.60 * narrow_keep + 0.40 * wide_keep
        local_keep = F.avg_pool1d(local_keep.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
        local_keep = (0.72 * local_keep + 0.28 * p4_prior_row).clamp(0.0, 1.0) * vis_row
        sparse_weight = (0.20 + 0.56 * local_keep + 0.24 * p4_prior_row).clamp(0.0, 1.0)
        keep_support = (0.50 * local_keep + 0.35 * p4_prior_row + 0.15 * vis_row).clamp(0.0, 1.0) * vis_row
        keep_support_raw = keep_support.transpose(1, 2).unsqueeze(-1)
        keep_support_clean = _suppress_short_row_islands(
            keep_support_raw,
            max_bridge_gap=max(6, p3.size(-2) // 32),
            min_component_len=max(3, p3.size(-2) // 96),
        )
        keep_support = keep_support_clean.squeeze(-1).transpose(1, 2).clamp(0.0, 1.0) * vis_row
        sparse_weight = sparse_weight * keep_support

        fused_tokens = narrow_tokens + 0.25 * (wide_tokens - narrow_tokens)
        sparse_tokens = fused_tokens * sparse_weight * vis_row
        token_map = self._render_row_tokens(sparse_tokens, axis_center, axis_width, out_w=fused.size(-1))
        keep_scalar = keep_support.transpose(1, 2).unsqueeze(-1)
        keep_support_map = render_axis_gaussian(
            axis_center,
            (axis_width * 0.90 + 0.012).clamp(min=0.018, max=0.075),
            keep_scalar,
            fused.size(-1),
            min_sigma=0.015,
        )
        keep_support_map = F.avg_pool2d(
            keep_support_map,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * vis_map
        token_map = token_map * (0.65 + 0.35 * structure_gate) * (0.35 + 0.65 * keep_support_map)
        chain_update = self.out(torch.cat([fused, token_map, axis_feat], dim=1))
        chain_gate = keep_support_map * (0.60 + 0.40 * structure_gate)
        chain_selected_gate = chain_gate.clamp(0.0, 1.0).pow(1.4)
        chain_selected_readout = F.max_pool2d(
            chain_selected_gate,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
        ).clamp(0.0, 1.0)
        chain_clean_gate = torch.maximum(chain_selected_gate, 0.50 * chain_selected_readout).clamp(0.0, 1.0)
        chain_residual_gate = _soft_threshold_gate(chain_clean_gate, threshold=0.035, softness=0.018)
        chain_candidate = fused + 0.45 * chain_update * (0.68 + 0.32 * structure_gate)
        chain_feat = p3 + (chain_candidate - p3) * chain_residual_gate
        chain_feat = chain_feat * chain_clean_gate
        token_score_raw = render_axis_gaussian(
            axis_center,
            axis_width * 0.92 + 0.014,
            (0.72 * narrow_keep_prob + 0.28 * p4_prior_row).transpose(1, 2).unsqueeze(-1) * axis_visible,
            fused.size(-1),
            min_sigma=0.014,
        )
        token_score_effective = (token_score_raw * keep_support_map).clamp(0.0, 1.0)
        debug = {
            "structure_gate": structure_gate.detach(),
            "p4_sparse_prior": p4_prior_map.detach(),
            "chain_selected_gate": chain_selected_gate.detach(),
            "chain_clean_gate": chain_clean_gate.detach(),
            "chain_residual_gate": chain_residual_gate.detach(),
            "chain_candidate": chain_candidate.detach(),
            "keep_support_raw": keep_support_raw.detach(),
            "narrow_strip_attn": (self._render_strip_weights(narrow_attn, axis_center, axis_width, fused.size(-1), 1.0) * axis_visible).detach(),
            "wide_strip_attn": (self._render_strip_weights(wide_attn, axis_center, axis_width, fused.size(-1), self.wide_scale) * axis_visible).detach(),
            "token_keep": chain_clean_gate.detach(),
            "token_score": token_score_effective.detach(),
            "token_score_raw": token_score_raw.detach(),
            "token_attention": render_axis_gaussian(
                axis_center,
                axis_width * 0.92 + 0.014,
                narrow_token_attn.transpose(1, 2).unsqueeze(-1) * axis_visible,
                fused.size(-1),
                min_sigma=0.014,
            ).detach(),
            "token_attention_matrix": narrow_attn_matrix.unsqueeze(1).detach(),
            "axial_response": render_axis_gaussian(
                axis_center,
                axis_width * 0.80 + 0.012,
                torch.sigmoid(narrow_axial_logits) * axis_visible,
                fused.size(-1),
                min_sigma=0.012,
            ).detach(),
        }
        return chain_feat, chain_clean_gate, narrow_axial_logits, debug


class CenterlineConditionedHeatmapRenderer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.render_refine = nn.Sequential(
            ConvLNAct(channels * 2 + 4, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )
        self.delta_head = PredictionHead(channels, 1, hidden_channels)
        self.delta_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(
        self,
        axis_feat: torch.Tensor,
        chain_feat: torch.Tensor,
        guide_struct: torch.Tensor,
        axial_logits: torch.Tensor,
        axis_center: torch.Tensor,
        axis_width: torch.Tensor,
        axis_visible: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        axial_prob = torch.sigmoid(axial_logits) * axis_visible
        rendered_prob = render_axis_gaussian(
            axis_center,
            (axis_width * 0.72 + 0.012).clamp(min=0.018, max=0.070),
            axial_prob,
            axis_feat.size(-1),
            min_sigma=0.02,
        )
        render_logits = _logit_from_prob(rendered_prob)
        render_feat = self.render_refine(torch.cat([axis_feat, chain_feat, guide_struct, rendered_prob], dim=1))
        render_envelope = (0.78 * guide_struct[:, 0:1] + 0.22 * guide_struct[:, 1:2].square()) * guide_struct[:, 2:3]
        delta = torch.tanh(self.delta_head(render_feat)) * render_envelope.clamp(0.0, 1.0)
        base_hm_logits = render_logits + torch.tanh(self.delta_scale) * delta
        debug = {
            "rendered_hm": rendered_prob.detach(),
            "render_delta": delta.detach(),
        }
        return base_hm_logits, render_feat, debug


class PeakOffsetRefiner(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        sparse_keep_ratio: float = 0.02,
    ):
        super().__init__()
        self.sparse_keep_ratio = float(min(0.25, max(0.005, sparse_keep_ratio)))
        self.guide_proj = nn.Sequential(
            ConvLNAct(4, channels, k=1, s=1, p=0),
            ConvLNAct(channels, channels, k=1, s=1, p=0),
        )
        self.refine_seed = nn.Sequential(
            ConvLNAct(channels * 3 + 4, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
        )
        self.local_sparse_attn = LocalCandidateSparseAttention(channels, keep_ratio=self.sparse_keep_ratio, min_keep=96)
        self.peak_gate = nn.Sequential(
            ConvLNAct(channels * 3 + 4, channels, k=1, s=1, p=0),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvLNAct(channels * 3, channels, k=1, s=1, p=0),
            ResidualDWBlock(channels),
            ResidualDWBlock(channels),
        )
        self.delta_hm_head = PredictionHead(channels, 1, hidden_channels)
        self.center_hm_head = PredictionHead(channels, 1, hidden_channels, bias_init=-2.19)
        self.p2_support_head = PredictionHead(channels, 1, hidden_channels, bias_init=-2.19)
        self.p2_direct_hm_head = PredictionHead(channels, 1, hidden_channels, bias_init=-2.19)
        self.center_reg_head = PredictionHead(channels, 2, hidden_channels)
        self.corner_reg_head = PredictionHead(channels, 8, hidden_channels)
        self.p2_direct_center_reg_head = PredictionHead(channels, 2, hidden_channels)
        self.p2_direct_corner_reg_head = PredictionHead(channels, 8, hidden_channels)
        self.delta_scale = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))

    def forward(
        self,
        p2: torch.Tensor,
        center_feat: torch.Tensor,
        guide_struct: torch.Tensor,
        base_hm_logits: torch.Tensor,
        p4_sparse_prior: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        base_prob = torch.sigmoid(base_hm_logits)
        guide_input = torch.cat([guide_struct, base_prob], dim=1)
        guide_feat = self.guide_proj(guide_input)
        del p4_sparse_prior
        refine_seed = self.refine_seed(torch.cat([p2, center_feat, guide_feat, guide_input], dim=1))
        p2_support_logits = self.p2_support_head(refine_seed)
        p2_heatmap_support = torch.sigmoid(p2_support_logits).clamp(0.0, 1.0)
        p2_gate = p2_heatmap_support
        sparse_update, sparse_keep, sparse_score, sparse_attn_response, sparse_attn_matrix = self.local_sparse_attn(refine_seed, p2_heatmap_support)
        sparse_keep_gate = sparse_keep.clamp(0.0, 1.0)
        p2_refine_gate = torch.maximum(sparse_keep_gate, p2_heatmap_support).clamp(0.0, 1.0)
        p2_guided = refine_seed + sparse_update * p2_refine_gate

        peak_gate = self.peak_gate(torch.cat([p2_guided, center_feat, guide_feat, guide_input], dim=1))
        peak_gate = peak_gate * p2_refine_gate
        refine_update = self.refine(torch.cat([p2_guided, center_feat * peak_gate, guide_feat], dim=1))
        refined_candidate = refine_seed + refine_update * p2_refine_gate
        refined = refine_seed + (refined_candidate - refine_seed) * p2_refine_gate
        dense_refined = refined

        p2_selected_readout = F.max_pool2d(
            sparse_keep_gate,
            kernel_size=3,
            stride=1,
            padding=1,
        ).clamp(0.0, 1.0)
        p2_selected_readout = torch.maximum(sparse_keep_gate, p2_selected_readout).clamp(0.0, 1.0)
        interior_mask = _interior_readout_mask(refined)
        p2_effective_support = torch.maximum(p2_selected_readout, p2_heatmap_support).clamp(0.0, 1.0)
        refined_clean_gate = (p2_effective_support * interior_mask).clamp(0.0, 1.0)

        refined = dense_refined * refined_clean_gate

        p2_direct_hm = self.p2_direct_hm_head(refine_seed)
        p2_direct_prob_raw = torch.sigmoid(p2_direct_hm)
        p2_direct_prob = (p2_direct_prob_raw * p2_heatmap_support * interior_mask).clamp(0.0, 1.0)
        p2_residual_prob = p2_direct_prob
        hm_support_gate = torch.maximum(sparse_keep_gate, p2_effective_support).clamp(0.0, 1.0)
        hm_support_gate = (hm_support_gate * interior_mask).clamp(0.0, 1.0)
        hm_read_gate = hm_support_gate
        hm_feat = refined * hm_read_gate
        center_hm = self.center_hm_head(hm_feat)
        sparse_prob = torch.sigmoid(center_hm)
        final_prob = torch.maximum(sparse_prob, p2_residual_prob).clamp(1e-4, 1.0 - 1e-4)
        final_hm = _logit_from_prob(final_prob)

        center_offsets_sparse = self.center_reg_head(refined)
        corner_offsets_sparse = self.corner_reg_head(refined)
        center_offsets_direct = self.p2_direct_center_reg_head(refine_seed)
        corner_offsets_direct = self.p2_direct_corner_reg_head(refine_seed)
        direct_decode_gate = (p2_direct_prob / (sparse_prob + p2_direct_prob).clamp_min(1e-4)).clamp(0.0, 1.0)
        center_offsets = center_offsets_sparse * (1.0 - direct_decode_gate) + center_offsets_direct * direct_decode_gate
        corner_offsets = corner_offsets_sparse * (1.0 - direct_decode_gate) + corner_offsets_direct * direct_decode_gate
        return (
            refined,
            peak_gate,
            p2_gate,
            p2_heatmap_support,
            hm_support_gate,
            hm_read_gate,
            hm_feat,
            sparse_keep,
            p2_effective_support,
            sparse_score,
            sparse_attn_response,
            sparse_attn_matrix,
            sparse_update,
            final_hm,
            center_offsets,
            corner_offsets,
            center_hm,
            p2_support_logits,
            p2_direct_hm,
            p2_residual_prob,
            p2_direct_prob,
            direct_decode_gate,
            center_offsets_direct,
            corner_offsets_direct,
        )


class StructuredSpineDecoder(nn.Module):
    def __init__(self, args, in_channels_list: Iterable[int]):
        super().__init__()
        self.args = args
        self.in_channels_list = list(int(v) for v in in_channels_list)
        if len(self.in_channels_list) != 4:
            raise ValueError("StructuredSpineDecoder expects four HRNet scales")

        self.decoder_dim = int(getattr(args, "decoder_dim", 128))
        self.head_dim = int(getattr(args, "head_dim", 128))

        self.scale_projectors = nn.ModuleList([ScaleProjector(in_ch, self.decoder_dim) for in_ch in self.in_channels_list])
        self.semantic_p5 = SemanticP5Bias(self.decoder_dim)
        self.axis_module = ExplicitAxisRepresentation(
            self.decoder_dim,
            min_width=float(getattr(args, "decoder_axis_min_width_ratio", 0.035)),
            max_width=float(getattr(args, "decoder_axis_max_width_ratio", 0.36)),
            keep_ratio=float(getattr(args, "decoder_p4_row_keep_ratio", 0.60)),
        )
        self.strip_context = CanonicalStripContext(
            self.decoder_dim,
            num_samples=int(getattr(args, "decoder_strip_samples", 17)),
            keep_ratio=float(getattr(args, "decoder_strip_keep_ratio", 0.70)),
            wide_scale=float(getattr(args, "decoder_strip_wide_context_scale", 1.70)),
        )
        self.p4_bridge_fuse = nn.Sequential(
            ConvLNAct(self.decoder_dim * 2 + 3, self.decoder_dim, k=1, s=1, p=0),
            ResidualDWBlock(self.decoder_dim),
            ResidualDWBlock(self.decoder_dim),
        )
        self.p3_seed_fuse = nn.Sequential(
            ConvLNAct(self.decoder_dim * 2 + 3, self.decoder_dim, k=1, s=1, p=0),
            ResidualDWBlock(self.decoder_dim),
            ResidualDWBlock(self.decoder_dim),
        )
        self.p2_seed_fuse = nn.Sequential(
            ConvLNAct(self.decoder_dim * 2 + 3, self.decoder_dim, k=1, s=1, p=0),
            ResidualDWBlock(self.decoder_dim),
            ResidualDWBlock(self.decoder_dim),
        )
        self.p2_seed_mix_gate = nn.Sequential(
            ConvLNAct(self.decoder_dim * 2 + 3, self.decoder_dim, k=1, s=1, p=0),
            nn.Conv2d(self.decoder_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.center_fuse = nn.Sequential(
            ConvLNAct(self.decoder_dim * 3 + 3, self.decoder_dim, k=1, s=1, p=0),
            ResidualDWBlock(self.decoder_dim),
            ResidualDWBlock(self.decoder_dim),
        )
        self.heatmap_renderer = CenterlineConditionedHeatmapRenderer(self.decoder_dim, self.head_dim)
        self.base_hm_head = PredictionHead(self.decoder_dim, 1, self.head_dim, bias_init=-2.19)
        self.base_hm_res_scale = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.peak_refiner = PeakOffsetRefiner(
            self.decoder_dim,
            self.head_dim,
            sparse_keep_ratio=float(getattr(args, "decoder_p2_refine_keep_ratio", 0.02)),
        )

    @staticmethod
    def _resize_to(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

    def _target_hw(self, backbone_features: List[torch.Tensor], batch=None) -> Tuple[int, int]:
        if batch is not None and isinstance(batch, dict) and "gt_global_hm" in batch:
            return tuple(int(v) for v in batch["gt_global_hm"].shape[-2:])
        return tuple(int(v) for v in backbone_features[0].shape[-2:])

    def forward(
        self,
        backbone_features: List[torch.Tensor],
        batch=None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if len(backbone_features) != 4:
            raise ValueError(f"Expected 4 backbone features, got {len(backbone_features)}")

        target_hw = self._target_hw(backbone_features, batch=batch)
        projected_native = [proj(feat) for proj, feat in zip(self.scale_projectors, backbone_features)]
        p2_native, p3_native, p4_native, p5_native = projected_native

        p2 = self._resize_to(p2_native, target_hw)
        p3 = self._resize_to(p3_native, target_hw)
        p4 = self._resize_to(p4_native, target_hw)

        p4_mod, p5_sem_gate = self.semantic_p5(p4, p5_native)
        axis_feat_raw, axis_state = self.axis_module(p4_mod)
        raw_guide_centerline = axis_state["centerline_prob"]
        raw_guide_band = axis_state["band_prob"]
        guide_confidence = axis_state["confidence_map"]
        p4_sparse_prior = axis_state["row_sparse_map"].clamp(0.0, 1.0)
        sparse_feedback_gate = F.avg_pool2d(
            p4_sparse_prior,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * guide_confidence
        centerline_clean_gate = sparse_feedback_gate.clamp(0.0, 1.0)
        band_clean_gate = sparse_feedback_gate.clamp(0.0, 1.0)
        guide_centerline = (raw_guide_centerline * centerline_clean_gate).clamp(0.0, 1.0)
        guide_band = (raw_guide_band * band_clean_gate).clamp(0.0, 1.0)
        axis_feat = axis_feat_raw * sparse_feedback_gate
        guide_struct = torch.cat([guide_centerline, guide_band, guide_confidence], dim=1)
        guide_valid = guide_struct[:, 2:3]
        pred_centerline_map = _logit_from_prob(guide_centerline)
        # P4 residual writes must follow the selected sparse support, not the wider
        # centerline/band prior. Otherwise unselected cranial/tail responses leak into res.
        p4_bridge_gate = p4_sparse_prior.clamp(0.0, 1.0).pow(1.5) * guide_valid
        p4_bridge_raw = self.p4_bridge_fuse(torch.cat([p4_mod, axis_feat, guide_struct], dim=1))
        p4_bridge_update = (p4_bridge_raw - p4_mod) * p4_bridge_gate
        axis_update = (axis_feat - p4_mod) * p4_bridge_gate
        p4_sparse_gain = 0.65 * p4_bridge_update + 0.20 * axis_update
        p4_bridge = p4_mod + p4_sparse_gain
        p4_bridge_clean_gate = F.max_pool2d(
            p4_bridge_gate,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
        ).clamp(0.0, 1.0)
        p4_bridge = p4_bridge * p4_bridge_clean_gate
        p4_sparse_gain = p4_sparse_gain * p4_bridge_clean_gate

        p3_seed = self.p3_seed_fuse(torch.cat([p3, p4_bridge, guide_struct], dim=1))
        p3_seed = p3_seed * (0.25 + 0.75 * guide_band) + p4_bridge * (0.35 + 0.65 * guide_centerline) * (0.20 + 0.80 * guide_valid)

        chain_feat, chain_gate_p3, chain_axial_logits, chain_debug = self.strip_context(
            p3_seed,
            p4_bridge,
            guide_struct,
            axis_state["center"],
            axis_state["width"],
            axis_state["visible"],
            p4_sparse_prior,
        )
        structural_base_logits, render_feat, heatmap_debug = self.heatmap_renderer(
            axis_feat,
            chain_feat,
            guide_struct,
            chain_axial_logits,
            axis_state["center"],
            axis_state["width"],
            axis_state["visible"],
        )
        center_feat = self.center_fuse(torch.cat([p4_bridge, axis_feat, chain_feat, guide_struct], dim=1))
        center_feat = center_feat + 0.20 * render_feat * (0.15 + 0.85 * guide_band)
        p4_centerline_prior = F.avg_pool2d(
            p4_sparse_prior.clamp(0.0, 1.0),
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * guide_valid
        p3_centerline_prior = F.avg_pool2d(
            chain_gate_p3.clamp(0.0, 1.0),
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * guide_valid
        centerline_selection_prior = (0.55 * p4_centerline_prior + 0.45 * p3_centerline_prior).clamp(0.0, 1.0)
        heatmap_selection_gate = (0.35 + 0.65 * centerline_selection_prior).clamp(0.0, 1.0)
        center_decay = ((0.78 * guide_centerline + 0.22 * guide_band.square()) * guide_valid * heatmap_selection_gate).clamp(0.0, 1.0)
        center_prior_prob = center_decay.clamp(1e-4, 1.0 - 1e-4)
        residual_gate = center_decay.clamp(0.0, 1.0)
        local_peak_logits = self.base_hm_head(center_feat) * residual_gate
        base_hm_residual = local_peak_logits
        range_suppress_logits = torch.log(center_decay.clamp_min(1e-4))
        base_hm_logits = local_peak_logits + 0.12 * structural_base_logits + 0.60 * range_suppress_logits
        p2_seed_raw = self.p2_seed_fuse(torch.cat([p2, p4_bridge, guide_struct], dim=1))
        p2_seed_mix = self.p2_seed_mix_gate(torch.cat([p2_seed_raw, p4_bridge, guide_struct], dim=1))
        p2_seed = p2_seed_raw * (1.0 - p2_seed_mix) + p4_bridge * p2_seed_mix
        (
            refined_feat,
            peak_gate_p2,
            p2_spine_gate,
            p2_heatmap_support,
            p2_hm_support_gate,
            p2_hm_read_gate,
            p2_hm_feat,
            p2_sparse_support,
            p2_effective_support,
            p2_sparse_score,
            p2_sparse_attention,
            p2_sparse_attention_matrix,
            p2_sparse_update,
            pred_global_hm,
            pred_center_offsets,
            pred_corner_offsets,
            pred_p2_center_hm,
            pred_p2_support_hm,
            pred_p2_direct_hm,
            p2_residual_prob,
            p2_direct_prob,
            p2_direct_decode_gate,
            p2_direct_center_offsets,
            p2_direct_corner_offsets,
        ) = self.peak_refiner(
            p2_seed,
            center_feat,
            guide_struct,
            base_hm_logits,
            p4_sparse_prior,
        )
        p2_heatmap_prior = F.avg_pool2d(
            p2_effective_support.clamp(0.0, 1.0),
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            count_include_pad=False,
        ).clamp(0.0, 1.0) * guide_valid
        final_selection_prior = p2_heatmap_prior.clamp(0.0, 1.0)
        final_delta_gate = final_selection_prior
        outputs: Dict[str, torch.Tensor] = {
            "pred_global_hm": pred_global_hm,
            "pred_base_hm": base_hm_logits,
            "pred_p2_center_hm": pred_p2_center_hm,
            "pred_p2_support_hm": pred_p2_support_hm,
            "pred_p2_direct_hm": pred_p2_direct_hm,
            "pred_p2_heatmap_support": p2_heatmap_support,
            "pred_p2_sparse_support": p2_sparse_support,
            "pred_p2_effective_support": p2_effective_support,
            "pred_p2_residual_prob": p2_residual_prob,
            "pred_p2_direct_prob": p2_direct_prob,
            "pred_p2_direct_decode_gate": p2_direct_decode_gate,
            "pred_p2_direct_center_offsets": p2_direct_center_offsets,
            "pred_p2_direct_corner_offsets": p2_direct_corner_offsets,
            "pred_p2_sparse_score": p2_sparse_score,
            "pred_center_offsets": pred_center_offsets,
            "pred_corner_offsets": pred_corner_offsets,
            "pred_centerline_map": pred_centerline_map,
            "pred_axis_visible": axis_state["row_validity"],
            "pred_band_map": guide_band,
            "pred_p4_row_visible": axis_state["visible"],
            "pred_centerline_row_visible": guide_centerline.amax(dim=-1, keepdim=True),
            "pred_base_row_visible": torch.sigmoid(base_hm_logits).amax(dim=-1, keepdim=True),
            "pred_global_row_visible": torch.sigmoid(pred_global_hm).amax(dim=-1, keepdim=True),
            "pred_p3_row_visible": chain_gate_p3.amax(dim=-1, keepdim=True),
            "pred_p2_row_visible": p2_effective_support.amax(dim=-1, keepdim=True),
            "raw_pred_global_hm": pred_global_hm,
            "decoder_feat_map": refined_feat,
        }

        if return_intermediates:
            feature_stages: Dict[str, torch.Tensor] = {
                "scale2_feat": p2.detach(),
                "scale3_feat": p3.detach(),
                "scale4_feat": p4.detach(),
                "scale5_feat": self._resize_to(p5_native, target_hw).detach(),
                "p5_sem_gate": p5_sem_gate.detach(),
                "guide_centerline": guide_centerline.detach(),
                "raw_guide_centerline": raw_guide_centerline.detach(),
                "guide_band": guide_band.detach(),
                "raw_guide_band": raw_guide_band.detach(),
                "guide_height_mask": axis_state["height_map"].detach(),
                "guide_confidence": guide_confidence.detach(),
                "sparse_feedback_gate": sparse_feedback_gate.detach(),
                "centerline_clean_gate": centerline_clean_gate.detach(),
                "band_clean_gate": band_clean_gate.detach(),
                "axis_center_attn": axis_state["center_attn"].detach(),
                "p4_sparse_score": axis_state["row_score_map"].detach(),
                "p4_sparse_raw_score": axis_state["row_score_raw_map"].detach(),
                "p4_span_prior": axis_state["row_span_prior"].expand(-1, -1, target_hw[0], target_hw[1]).detach(),
                "p4_raw_row_support": axis_state["row_support_raw"].expand(-1, -1, target_hw[0], target_hw[1]).detach(),
                "axis_row_support_p4": axis_state["row_sparse_map"].detach(),
                "p4_sparse_support": axis_state["row_sparse_map"].detach(),
                "p4_soft_spinal_support": axis_state["row_soft_support"].expand(-1, -1, target_hw[0], target_hw[1]).detach(),
                "axis_core_p4": axis_feat.detach(),
                "axis_core_p4_raw": axis_feat_raw.detach(),
                "p4_sparse_attention": axis_state["row_attention_map"].detach(),
                "p4_sparse_attention_matrix": axis_state["row_attention"].unsqueeze(1).detach(),
                "p4_spinal_support": guide_confidence.detach(),
                "p4_bridge": p4_bridge.detach(),
                "p4_bridge_raw": p4_bridge_raw.detach(),
                "p4_bridge_gate": p4_bridge_gate.detach(),
                "p4_bridge_clean_gate": p4_bridge_clean_gate.detach(),
                "p4_sparse_gain": p4_sparse_gain.detach(),
                "p4_raw_sparse_gain": (p4_bridge_raw - p4_mod).detach(),
                "axis_gate_p4": render_axis_gaussian(
                    axis_state["center"],
                    axis_state["width"] * 1.05 + 0.015,
                    axis_state["visible"],
                    target_hw[1],
                    min_sigma=0.015,
                ).detach(),
                "p4_sparse_prior_for_p3p2": p4_sparse_prior.detach(),
                "chain_structure_gate_p3": chain_debug["structure_gate"],
                "p3_p4_sparse_prior": chain_debug["p4_sparse_prior"],
                "p3_selected_gate": chain_debug["chain_selected_gate"],
                "p3_residual_gate": chain_debug["chain_residual_gate"],
                "p3_chain_candidate": chain_debug["chain_candidate"],
                "chain_gate_p3": chain_gate_p3.detach(),
                "p3_sparse_score": chain_debug["token_score"],
                "p3_sparse_raw_score": chain_debug["token_score_raw"],
                "p3_keep_support_raw": chain_debug["keep_support_raw"].expand(-1, -1, target_hw[0], target_hw[1]).detach(),
                "p3_sparse_support": chain_debug["token_keep"],
                "chain_strip_attn_p3": chain_debug["narrow_strip_attn"],
                "chain_wide_strip_attn_p3": chain_debug["wide_strip_attn"],
                "chain_support_p3": chain_debug["token_keep"],
                "p3_sparse_attention": chain_debug["token_attention"],
                "p3_sparse_attention_matrix": chain_debug["token_attention_matrix"],
                "chain_axial_response_p3": chain_debug["axial_response"],
                "p3_seed": p3_seed.detach(),
                "p3_sparse_gain": (chain_feat - p3_seed).detach(),
                "p2_seed_mix_gate": p2_seed_mix.detach(),
                "rendered_hm_prob": heatmap_debug["rendered_hm"],
                "centerline_hm_prior": center_prior_prob.detach(),
                "p4_centerline_prior": p4_centerline_prior.detach(),
                "p3_centerline_prior": p3_centerline_prior.detach(),
                "centerline_selection_prior": centerline_selection_prior.detach(),
                "heatmap_selection_gate": heatmap_selection_gate.detach(),
                "p2_heatmap_support": p2_heatmap_support.detach(),
                "p2_support_hm_prob": torch.sigmoid(pred_p2_support_hm).detach(),
                "p2_direct_hm_prob": torch.sigmoid(pred_p2_direct_hm).detach(),
                "p2_residual_prob": p2_residual_prob.detach(),
                "p2_direct_hm_fused_prob": p2_direct_prob.detach(),
                "p2_direct_decode_gate": p2_direct_decode_gate.detach(),
                "p2_hm_support_gate": p2_hm_support_gate.detach(),
                "p2_hm_read_gate": p2_hm_read_gate.detach(),
                "p2_heatmap_prior": p2_heatmap_prior.detach(),
                "final_selection_prior": final_selection_prior.detach(),
                "final_delta_gate": final_delta_gate.detach(),
                "centerline_range_suppress": range_suppress_logits.detach(),
                "render_hm_delta": heatmap_debug["render_delta"],
                "base_hm_residual": base_hm_residual.detach(),
                "peak_gate_p2": peak_gate_p2.detach(),
                "p2_spine_gate": p2_spine_gate.detach(),
                "p2_sparse_score": p2_sparse_score.detach(),
                "p2_refine_support": p2_effective_support.detach(),
                "p2_effective_support": p2_effective_support.detach(),
                "p2_sparse_support": p2_sparse_support.detach(),
                "p2_sparse_attention": p2_sparse_attention.detach(),
                "p2_sparse_attention_matrix": p2_sparse_attention_matrix.detach(),
                "p2_sparse_update": p2_sparse_update.detach(),
                "p2_seed": p2_seed.detach(),
                "p2_center_hm_prob": torch.sigmoid(pred_p2_center_hm).detach(),
                "decoder_stage_p4": p4_bridge.detach(),
                "decoder_stage_p3": chain_feat.detach(),
                "center_feat": center_feat.detach(),
                "decoder_stage_p2_raw": refined_feat.detach(),
                "decoder_stage_p2": refined_feat.detach(),
                "decoder_stage_p2_hm": p2_hm_feat.detach(),
                "decoder_stage_p2_update": (refined_feat - p2_seed).detach(),
                "base_hm": base_hm_logits.detach(),
                "base_hm_prob": torch.sigmoid(base_hm_logits).detach(),
                "delta_hm": (pred_global_hm - base_hm_logits).detach(),
                "final_hm": pred_global_hm.detach(),
                "final_hm_prob": torch.sigmoid(pred_global_hm).detach(),
            }
            outputs["feature_stages"] = feature_stages

        return outputs


DFFormerDecoder = StructuredSpineDecoder
