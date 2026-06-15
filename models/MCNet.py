# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .DecNet import DFFormerDecoder
from .HRNetBackbone import HRNetBackbone


class mc_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = HRNetBackbone(
            args=args,
            pretrained_path=getattr(args, "hrnet_pretrained", ""),
        )
        self.decoder = DFFormerDecoder(
            args=args,
            in_channels_list=self.backbone.out_channels,
        )

    def forward_features(self, input_image: torch.Tensor):
        return self.backbone(input_image)

    def forward(self, batch, **kwargs):
        input_image = batch["input_image"]
        return_intermediates = bool(kwargs.get("return_intermediates", False))
        features = self.forward_features(input_image)
        outputs = self.decoder(features, batch=batch, return_intermediates=return_intermediates)
        outputs["num_views"] = torch.ones(
            input_image.size(0),
            device=input_image.device,
            dtype=torch.int64,
        )
        return outputs
