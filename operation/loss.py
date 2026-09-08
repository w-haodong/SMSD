# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HeatmapFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, neg_power=4.0, epsilon=1e-6):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.neg_power = float(neg_power)
        self.eps = float(epsilon)

    def forward(self, pred_logits, gt):
        pred_logits = pred_logits.float()
        gt = gt.float()

        p = torch.clamp(torch.sigmoid(pred_logits), min=self.eps, max=1.0 - self.eps)
        pos = gt.eq(1).float()
        neg = gt.lt(1).float()
        neg_w = torch.pow(1 - gt, self.neg_power)

        pos_loss = -self.alpha * torch.log(p) * torch.pow(1 - p, self.gamma) * pos
        neg_loss = -(1 - self.alpha) * torch.log(1 - p) * torch.pow(p, self.gamma) * neg_w * neg
        return (pos_loss.sum() + neg_loss.sum()) / (pos.sum() + self.eps)


class _SparseRegL1Loss(nn.Module):
    def __init__(self, beta=1.0, eps=1e-6):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)

    @staticmethod
    def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        dim = feat.size(2)
        ind = ind.long().unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        return feat.gather(1, ind)

    def forward(self, pred, ind, mask, target):
        pred = pred.float()
        ind = ind.long()
        mask = mask.float()
        target = target.float()

        if pred.ndim != 4:
            raise ValueError(f"[SparseReg] pred must be (B,C,H,W), got {tuple(pred.shape)}")
        pred_gather = pred.permute(0, 2, 3, 1).contiguous()
        pred_gather = pred_gather.view(pred_gather.size(0), -1, pred_gather.size(3))
        pred_gather = self._gather_feat(pred_gather, ind)

        if mask.ndim == 2:
            mask = mask.unsqueeze(2)
        mask = mask.expand_as(pred_gather)
        loss = F.smooth_l1_loss(pred_gather * mask, target * mask, reduction="sum", beta=self.beta)
        return loss / (mask.sum().clamp_min(self.eps))


def _derive_axis_visible_target(gt_centerline_hm: torch.Tensor) -> torch.Tensor:
    centerline = gt_centerline_hm.float()
    row_peak = centerline.amax(dim=-1, keepdim=True)
    return row_peak.gt(0.03).float()


def _prob_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred = pred.float().clamp(min=1e-4, max=1.0 - 1e-4)
    target = target.float()
    logits = torch.log(pred / (1.0 - pred))
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        loss = loss * weight.float()
        return loss.sum() / weight.float().sum().clamp_min(1.0)
    return loss.mean()


def _visible_recall_weight(
    gt_visible: torch.Tensor,
    pos_weight: float = 2.2,
    neg_weight: float = 0.8,
    upper_boost: float = 1.4,
) -> torch.Tensor:
    target = gt_visible.float()
    pos = target.gt(0.5).float()
    bsz, channels, height, width = target.shape
    del bsz, channels, width

    y = torch.linspace(0.0, 1.0, height, device=target.device, dtype=target.dtype).view(1, 1, height, 1)
    y = y.expand_as(target)
    high = torch.full_like(target, 2.0)
    low = torch.full_like(target, -1.0)
    y_min = torch.where(pos.bool(), y, high).amin(dim=2, keepdim=True)
    y_max = torch.where(pos.bool(), y, low).amax(dim=2, keepdim=True)
    has_pos = pos.sum(dim=2, keepdim=True).gt(0.5)

    span = (y_max - y_min).clamp_min(1.0 / max(height - 1, 1))
    rel_y = ((y - y_min) / span).clamp(0.0, 1.0)
    upper_visible = (1.0 - rel_y).square() * pos
    weight = neg_weight * (1.0 - pos) + pos_weight * pos + upper_boost * upper_visible
    return torch.where(has_pos.expand_as(weight), weight, torch.ones_like(weight))


def _match_row_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    return F.interpolate(pred.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)


class SpineLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.lambda_hm = float(getattr(args, "lambda_hm", 1.0))
        self.lambda_center_reg = float(getattr(args, "lambda_center_reg", 1.0))
        self.lambda_corner_reg = float(getattr(args, "lambda_corner_reg", 0.1))
        self.lambda_centerline = float(getattr(args, "lambda_centerline", 0.0))
        self.lambda_axis_visible = float(getattr(args, "lambda_axis_visible", 0.0))
        self.lambda_row_coverage = float(getattr(args, "lambda_row_coverage", 0.0))
        self.lambda_hm_row_recall = float(getattr(args, "lambda_hm_row_recall", 0.0))
        self.lambda_base_hm = float(getattr(args, "lambda_base_hm", 0.0))
        self.lambda_p2_support_hm = float(getattr(args, "lambda_p2_support_hm", 0.0))
        self.lambda_p2_hm = float(getattr(args, "lambda_p2_hm", 0.0))
        self.lambda_p2_direct_reg = float(getattr(args, "lambda_p2_direct_reg", 0.0))

        self.hm_focal = _HeatmapFocalLoss()
        self.reg_l1 = _SparseRegL1Loss(beta=1.0)

    def forward(self, outputs, batch, epoch=None):
        del epoch

        l_hm = self.hm_focal(outputs["pred_global_hm"], batch["gt_global_hm"])
        if self.lambda_base_hm > 0.0 and "pred_base_hm" in outputs:
            l_base_hm = self.hm_focal(outputs["pred_base_hm"], batch["gt_global_hm"])
        else:
            l_base_hm = outputs["pred_global_hm"].new_tensor(0.0)
        if self.lambda_p2_hm > 0.0 and "pred_p2_direct_hm" in outputs:
            l_p2_hm = self.hm_focal(outputs["pred_p2_direct_hm"], batch["gt_global_hm"])
        else:
            l_p2_hm = outputs["pred_global_hm"].new_tensor(0.0)
        if self.lambda_p2_support_hm > 0.0 and "pred_p2_support_hm" in outputs:
            l_p2_support_hm = self.hm_focal(outputs["pred_p2_support_hm"], batch["gt_global_hm"])
        else:
            l_p2_support_hm = outputs["pred_global_hm"].new_tensor(0.0)
        l_ctr_reg = self.reg_l1(
            outputs["pred_center_offsets"],
            batch["gt_ind"],
            batch["gt_reg_mask"],
            batch["gt_center_reg"].to(outputs["pred_center_offsets"].dtype),
        )
        l_crn_reg = self.reg_l1(
            outputs["pred_corner_offsets"],
            batch["gt_ind"],
            batch["gt_reg_mask"],
            batch["gt_corner_reg"].to(outputs["pred_corner_offsets"].dtype),
        )
        if self.lambda_p2_direct_reg > 0.0 and "pred_p2_direct_center_offsets" in outputs:
            l_p2_direct_ctr_reg = self.reg_l1(
                outputs["pred_p2_direct_center_offsets"],
                batch["gt_ind"],
                batch["gt_reg_mask"],
                batch["gt_center_reg"].to(outputs["pred_p2_direct_center_offsets"].dtype),
            )
        else:
            l_p2_direct_ctr_reg = outputs["pred_global_hm"].new_tensor(0.0)
        if self.lambda_p2_direct_reg > 0.0 and "pred_p2_direct_corner_offsets" in outputs:
            l_p2_direct_crn_reg = self.reg_l1(
                outputs["pred_p2_direct_corner_offsets"],
                batch["gt_ind"],
                batch["gt_reg_mask"],
                batch["gt_corner_reg"].to(outputs["pred_p2_direct_corner_offsets"].dtype),
            )
        else:
            l_p2_direct_crn_reg = outputs["pred_global_hm"].new_tensor(0.0)

        if self.lambda_centerline > 0.0 and "pred_centerline_map" in outputs and "gt_centerline_hm" in batch:
            l_centerline = self.hm_focal(outputs["pred_centerline_map"], batch["gt_centerline_hm"])
        else:
            l_centerline = outputs["pred_global_hm"].new_tensor(0.0)

        if "gt_centerline_hm" in batch:
            gt_visible = _derive_axis_visible_target(batch["gt_centerline_hm"])
        else:
            gt_visible = None

        if self.lambda_axis_visible > 0.0 and gt_visible is not None and "pred_axis_visible" in outputs:
            pred_visible = _match_row_shape(outputs["pred_axis_visible"], gt_visible)
            visible_weight = _visible_recall_weight(gt_visible, pos_weight=2.0, neg_weight=1.0, upper_boost=1.2)
            l_axis_visible = _prob_bce(pred_visible, gt_visible, visible_weight)
        else:
            l_axis_visible = outputs["pred_global_hm"].new_tensor(0.0)

        row_terms = []
        if self.lambda_row_coverage > 0.0 and gt_visible is not None:
            row_weight = _visible_recall_weight(gt_visible)
            for key in ("pred_p4_row_visible", "pred_centerline_row_visible", "pred_p3_row_visible", "pred_p2_row_visible"):
                if key in outputs:
                    row_pred = _match_row_shape(outputs[key], gt_visible)
                    row_terms.append(_prob_bce(row_pred, gt_visible, row_weight))
        if row_terms:
            l_row_coverage = torch.stack(row_terms).mean()
        else:
            l_row_coverage = outputs["pred_global_hm"].new_tensor(0.0)

        hm_row_terms = []
        if self.lambda_hm_row_recall > 0.0 and gt_visible is not None:
            hm_row_weight = _visible_recall_weight(gt_visible, pos_weight=2.4, neg_weight=0.6, upper_boost=1.8)
            for key in ("pred_base_row_visible", "pred_global_row_visible"):
                if key in outputs:
                    row_pred = _match_row_shape(outputs[key], gt_visible)
                    hm_row_terms.append(_prob_bce(row_pred, gt_visible, hm_row_weight))
        if hm_row_terms:
            l_hm_row_recall = torch.stack(hm_row_terms).mean()
        else:
            l_hm_row_recall = outputs["pred_global_hm"].new_tensor(0.0)

        total = (
            self.lambda_hm * l_hm
            + self.lambda_base_hm * l_base_hm
            + self.lambda_p2_support_hm * l_p2_support_hm
            + self.lambda_p2_hm * l_p2_hm
            + self.lambda_center_reg * l_ctr_reg
            + self.lambda_corner_reg * l_crn_reg
            + self.lambda_p2_direct_reg * (
                self.lambda_center_reg * l_p2_direct_ctr_reg
                + self.lambda_corner_reg * l_p2_direct_crn_reg
            )
            + self.lambda_centerline * l_centerline
            + self.lambda_axis_visible * l_axis_visible
            + self.lambda_row_coverage * l_row_coverage
            + self.lambda_hm_row_recall * l_hm_row_recall
        )

        stats = {
            "hm_loss": float((self.lambda_hm * l_hm).detach().item()),
            "base_hm_loss": float((self.lambda_base_hm * l_base_hm).detach().item()),
            "p2_support_hm_loss": float((self.lambda_p2_support_hm * l_p2_support_hm).detach().item()),
            "p2_hm_loss": float((self.lambda_p2_hm * l_p2_hm).detach().item()),
            "center_reg_loss": float((self.lambda_center_reg * l_ctr_reg).detach().item()),
            "corner_reg_loss": float((self.lambda_corner_reg * l_crn_reg).detach().item()),
            "p2_direct_center_reg_loss": float(
                (self.lambda_p2_direct_reg * self.lambda_center_reg * l_p2_direct_ctr_reg).detach().item()
            ),
            "p2_direct_corner_reg_loss": float(
                (self.lambda_p2_direct_reg * self.lambda_corner_reg * l_p2_direct_crn_reg).detach().item()
            ),
            "centerline_loss": float((self.lambda_centerline * l_centerline).detach().item()),
            "axis_visible_loss": float((self.lambda_axis_visible * l_axis_visible).detach().item()),
            "row_coverage_loss": float((self.lambda_row_coverage * l_row_coverage).detach().item()),
            "hm_row_recall_loss": float((self.lambda_hm_row_recall * l_hm_row_recall).detach().item()),
            "centerline_raw": float(l_centerline.detach().item()),
            "total_loss": float(total.detach().item()),
        }
        return total, stats


SAIC_Loss = SpineLoss

