# -*- coding: utf-8 -*-
# ============================================================
# File: operation/decode.py
#   DecDecoder: single global center heatmap + corner offset decoder
#   Outputs are still in heatmap coordinates.
# ============================================================

import torch
import torch.nn.functional as F


class DecDecoder(object):
    def __init__(self, K=17, candidate_topk=32, conf_thresh=0.0, seg_thr=0.0):
        self.K = int(K)
        self.candidate_topk = int(max(candidate_topk, self.K))
        self.conf_thresh = float(conf_thresh)
        self.seg_thr = float(seg_thr)

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(
            heat,
            (kernel, kernel),
            stride=1,
            padding=(kernel - 1) // 2,
        )
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        Kc = min(self.candidate_topk, height * width)

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), Kc)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), Kc)
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, Kc)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, Kc)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, Kc)

        return topk_score, topk_inds, topk_ys, topk_xs

    @torch.no_grad()
    def decode_peaks(
        self,
        heat,
        seg_prob=None,
        topk=10,
        conf_thresh=None,
        seg_thr=None,
        nms_kernel=3,
        return_inds=False,
    ):
        assert heat.dim() == 4 and heat.size(1) == 1, f"heat must be [B,1,H,W], got {tuple(heat.shape)}"
        B, _, H, W = heat.shape
        device = heat.device

        topk = int(max(1, topk))
        Kc = int(max(self.candidate_topk, topk))

        _conf = float(self.conf_thresh if conf_thresh is None else conf_thresh)
        _segthr = float(self.seg_thr if seg_thr is None else seg_thr)

        seg_aligned = None
        if seg_prob is not None:
            if seg_prob.dim() == 4:
                seg = seg_prob
            elif seg_prob.dim() == 3:
                seg = seg_prob.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected seg_prob shape: {seg_prob.shape}")
            if seg.shape[-2:] != (H, W):
                seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
            seg_aligned = seg

        heat_nms = self._nms(heat, kernel=int(nms_kernel))

        Kc_eff = min(Kc, H * W)
        cand_scores, cand_inds = torch.topk(heat_nms.view(B, -1), Kc_eff)
        cand_ys = (cand_inds // W).int().float()
        cand_xs = (cand_inds % W).int().float()

        xs_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        ys_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        sc_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        ind_out = torch.zeros((B, topk), device=device, dtype=torch.long)

        for b in range(B):
            s = cand_scores[b]
            ys = cand_ys[b]
            xs = cand_xs[b]
            ind = cand_inds[b]

            valid = torch.ones_like(s, dtype=torch.bool, device=device)
            if _conf > 0:
                valid &= (s >= _conf)
            if (seg_aligned is not None) and (_segthr > 0):
                segb = seg_aligned[b, 0]
                yi = ys.long().clamp(0, H - 1)
                xi = xs.long().clamp(0, W - 1)
                valid &= (segb[yi, xi] >= _segthr)

            if not valid.any():
                valid = torch.ones_like(valid)

            eff = s.clone()
            eff[~valid] = -1e6

            take = min(topk, Kc_eff)
            sc, idx = torch.topk(eff, take)

            if take < topk:
                pad = topk - take
                idx = torch.cat([idx, idx[-1:].repeat(pad)], dim=0)
                sc = torch.cat([sc, sc[-1:].repeat(pad)], dim=0)

            xs_out[b] = xs[idx]
            ys_out[b] = ys[idx]
            sc_out[b] = sc
            ind_out[b] = ind[idx]

        if return_inds:
            return xs_out, ys_out, sc_out, ind_out
        return xs_out, ys_out, sc_out

    @torch.no_grad()
    def decode_peak1(self, heat, seg_prob=None, conf_thresh=None, seg_thr=None, nms_kernel=3):
        xs, ys, sc = self.decode_peaks(
            heat=heat,
            seg_prob=seg_prob,
            topk=1,
            conf_thresh=conf_thresh,
            seg_thr=seg_thr,
            nms_kernel=nms_kernel,
            return_inds=False,
        )
        return xs[:, 0], ys[:, 0], sc[:, 0]

    def ctdet_decode(self, heat, wh, reg, seg_prob=None, topk=None):
        batch, _, height, width = heat.size()
        device = heat.device
        num_keep = int(self.K if topk is None else max(1, topk))

        seg_aligned = None
        if seg_prob is not None:
            if seg_prob.dim() == 4:
                seg = seg_prob
            elif seg_prob.dim() == 3:
                seg = seg_prob.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected seg_prob shape: {seg_prob.shape}")

            if seg.shape[-2:] != (height, width):
                seg = F.interpolate(seg, size=(height, width), mode="bilinear", align_corners=False)
            seg_aligned = seg

        heat_nms = self._nms(heat)
        cand_scores, cand_inds, cand_ys, cand_xs = self._topk(heat_nms)

        Kc = cand_scores.size(1)

        final_inds = torch.zeros((batch, num_keep), dtype=torch.long, device=device)
        final_scores = torch.zeros((batch, num_keep), dtype=torch.float32, device=device)
        final_ys = torch.zeros((batch, num_keep), dtype=torch.float32, device=device)
        final_xs = torch.zeros((batch, num_keep), dtype=torch.float32, device=device)

        for b in range(batch):
            scores_b = cand_scores[b]
            ys_b = cand_ys[b]
            xs_b = cand_xs[b]

            valid_mask = torch.ones_like(scores_b, dtype=torch.bool, device=device)

            if self.conf_thresh > 0.0:
                valid_mask &= (scores_b >= self.conf_thresh)

            if seg_aligned is not None and self.seg_thr > 0.0:
                seg_b = seg_aligned[b, 0]
                ys_int = ys_b.long().clamp(0, height - 1)
                xs_int = xs_b.long().clamp(0, width - 1)
                seg_vals = seg_b[ys_int, xs_int]
                valid_mask &= (seg_vals >= self.seg_thr)

            if not valid_mask.any():
                valid_mask = torch.ones_like(valid_mask, dtype=torch.bool, device=device)

            eff_scores = scores_b.clone()
            eff_scores[~valid_mask] = -1e6

            topk_eff = min(num_keep, Kc)
            top_scores_b, idx_local = torch.topk(eff_scores, topk_eff)

            if topk_eff < num_keep:
                pad = num_keep - topk_eff
                idx_local = torch.cat([idx_local, idx_local[-1:].repeat(pad)], dim=0)
                top_scores_b = torch.cat([top_scores_b, top_scores_b[-1:].repeat(pad)], dim=0)

            chosen_inds = cand_inds[b][idx_local]
            chosen_ys = ys_b[idx_local]
            chosen_xs = xs_b[idx_local]

            final_inds[b] = chosen_inds
            final_scores[b] = top_scores_b
            final_ys[b] = chosen_ys
            final_xs[b] = chosen_xs

        scores = final_scores.view(batch, num_keep, 1)

        reg_feat = self._tranpose_and_gather_feat(reg, final_inds)
        reg_feat = reg_feat.view(batch, num_keep, 2)

        xs = final_xs.view(batch, num_keep, 1) + reg_feat[:, :, 0:1]
        ys = final_ys.view(batch, num_keep, 1) + reg_feat[:, :, 1:2]

        wh_feat = self._tranpose_and_gather_feat(wh, final_inds)
        wh_feat = wh_feat.view(batch, num_keep, 8)

        tl_x = xs - wh_feat[:, :, 0:1]
        tl_y = ys - wh_feat[:, :, 1:2]
        tr_x = xs - wh_feat[:, :, 2:3]
        tr_y = ys - wh_feat[:, :, 3:4]
        bl_x = xs - wh_feat[:, :, 4:5]
        bl_y = ys - wh_feat[:, :, 5:6]
        br_x = xs - wh_feat[:, :, 6:7]
        br_y = ys - wh_feat[:, :, 7:8]

        pts = torch.cat(
            [
                xs,
                ys,
                tl_x,
                tl_y,
                tr_x,
                tr_y,
                br_x,
                br_y,
                bl_x,
                bl_y,
                scores,
            ],
            dim=2,
        )

        if pts.shape[0] == 1:
            pts = pts.squeeze(0)

        return pts.data.cpu().numpy()
