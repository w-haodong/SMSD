# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_inchannels = list(num_inchannels)
        self.fuse_method = fuse_method
        self.num_branches = int(num_branches)
        self.multi_scale_output = multi_scale_output

        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self,
        num_branches: int,
        num_blocks: Sequence[int],
        num_inchannels: Sequence[int],
        num_channels: Sequence[int],
    ):
        if num_branches != len(num_blocks):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != NUM_BLOCKS({len(num_blocks)})")
        if num_branches != len(num_channels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != NUM_CHANNELS({len(num_channels)})")
        if num_branches != len(num_inchannels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(num_inchannels)})")

    def _make_one_branch(
        self,
        branch_index: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        output_branches = num_branches if self.multi_scale_output else 1

        for i in range(output_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            outchannels = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        outchannels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            outchannels = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        outchannels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return list(self.num_inchannels)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            if i == 0:
                y = x[0]
            else:
                y = self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNetBackbone(nn.Module):
    def __init__(self, args=None, pretrained_path: str | None = None):
        super().__init__()
        self.args = args
        self.variant = str(getattr(args, "hrnet_variant", "w32")).lower()
        self.out_channels = [int(v) for v in getattr(args, "hrnet_stage_channels", [32, 64, 128, 256])]
        stage_modules = [int(v) for v in getattr(args, "hrnet_stage_modules", [1, 4, 3])]
        stage_blocks_cfg = getattr(
            args,
            "hrnet_stage_blocks",
            {
                "stage1": [4],
                "stage2": [4, 4],
                "stage3": [4, 4, 4],
                "stage4": [4, 4, 4, 4],
            },
        )

        if len(self.out_channels) != 4:
            raise ValueError(f"Expected four HRNet stage channels, got {self.out_channels}")
        if len(stage_modules) != 3:
            raise ValueError(f"Expected three HRNet stage module counts, got {stage_modules}")

        stage2_channels = list(self.out_channels[:2])
        stage3_channels = list(self.out_channels[:3])
        stage4_channels = list(self.out_channels[:4])
        stage1_planes = max(32, int(self.out_channels[0] * 2))
        stage1_blocks = int(stage_blocks_cfg.get("stage1", [4])[0])
        stage2_blocks = [int(v) for v in stage_blocks_cfg.get("stage2", [4, 4])]
        stage3_blocks = [int(v) for v in stage_blocks_cfg.get("stage3", [4, 4, 4])]
        stage4_blocks = [int(v) for v in stage_blocks_cfg.get("stage4", [4, 4, 4, 4])]

        self.inplanes = 64
        self.pretrained_info: Dict[str, Any] | None = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, stage1_planes, stage1_blocks)
        stage1_out_channel = stage1_planes * Bottleneck.expansion

        self.transition1 = self._make_transition_layer([stage1_out_channel], stage2_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=stage_modules[0],
            num_branches=2,
            num_blocks=stage2_blocks,
            num_channels=stage2_channels,
            block=BasicBlock,
            fuse_method="SUM",
            num_inchannels=stage2_channels,
        )

        self.transition2 = self._make_transition_layer(pre_stage_channels, stage3_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=stage_modules[1],
            num_branches=3,
            num_blocks=stage3_blocks,
            num_channels=stage3_channels,
            block=BasicBlock,
            fuse_method="SUM",
            num_inchannels=stage3_channels,
        )

        self.transition3 = self._make_transition_layer(pre_stage_channels, stage4_channels)
        self.stage4, _ = self._make_stage(
            num_modules=stage_modules[2],
            num_branches=4,
            num_blocks=stage4_blocks,
            num_channels=stage4_channels,
            block=BasicBlock,
            fuse_method="SUM",
            num_inchannels=stage4_channels,
        )

        self._init_weights()
        if pretrained_path:
            print(f"[HRNet] Loading pretrained weights for {self.variant}: {pretrained_path}", flush=True)
            self.pretrained_info = self.load_pretrained(pretrained_path)
            print(
                "[HRNet] Pretrained loaded: "
                f"matched={self.pretrained_info['matched']}/{self.pretrained_info['total']}, "
                f"missing={len(self.pretrained_info['missing'])}, "
                f"ignored={len(self.pretrained_info['ignored'])}, "
                f"shape_mismatch={len(self.pretrained_info['shape_mismatch'])}",
                flush=True,
            )
        else:
            print(f"[HRNet] Pretrained weights disabled or not provided for {self.variant}.", flush=True)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(
        self,
        num_channels_pre_layer: Sequence[int],
        num_channels_cur_layer: Sequence[int],
    ) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                in_channels = num_channels_pre_layer[-1]
                for j in range(i + 1 - num_branches_pre):
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                    in_channels = out_channels
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules: int,
        num_branches: int,
        num_blocks: Sequence[int],
        num_channels: Sequence[int],
        block: type[nn.Module],
        fuse_method: str,
        num_inchannels: Sequence[int],
        multi_scale_output: bool = True,
    ) -> tuple[nn.Sequential, List[int]]:
        modules = []
        num_inchannels = list(num_inchannels)
        expanded_channels = [int(c) * block.expansion for c in num_channels]

        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i < num_modules - 1
            module = HighResolutionModule(
                num_branches=num_branches,
                block=block,
                num_blocks=num_blocks,
                num_inchannels=num_inchannels,
                num_channels=expanded_channels,
                fuse_method=fuse_method,
                multi_scale_output=reset_multi_scale_output,
            )
            modules.append(module)
            num_inchannels = module.get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _normalize_state_key(key: str) -> str:
        prefixes = ("module.backbone.", "backbone.", "module.")
        for prefix in prefixes:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def load_pretrained(self, checkpoint_path: str) -> Dict[str, Any]:
        if not os.path.isfile(checkpoint_path):
            print(f"[HRNet] Pretrained checkpoint missing: {checkpoint_path}", flush=True)
            raise FileNotFoundError(f"HRNet pretrained checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = self.state_dict()

        filtered_state: Dict[str, torch.Tensor] = {}
        ignored = []
        shape_mismatch = []

        for raw_key, tensor in state_dict.items():
            key = self._normalize_state_key(raw_key)
            if key not in model_state:
                ignored.append(raw_key)
                continue
            if model_state[key].shape != tensor.shape:
                shape_mismatch.append((raw_key, tuple(tensor.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = tensor

        if not filtered_state:
            raise ValueError(f"No compatible HRNet backbone weights found in checkpoint: {checkpoint_path}")

        missing = sorted(set(model_state.keys()) - set(filtered_state.keys()))
        self.load_state_dict(filtered_state, strict=False)
        return {
            "path": checkpoint_path,
            "variant": self.variant,
            "matched": len(filtered_state),
            "total": len(model_state),
            "missing": missing,
            "ignored": ignored,
            "shape_mismatch": shape_mismatch,
        }

    def _forward_transition(self, transition: nn.ModuleList, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, layer in enumerate(transition):
            if i < len(inputs):
                outputs.append(layer(inputs[i]))
            else:
                outputs.append(layer(inputs[-1]))
        return outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x_list = self._forward_transition(self.transition1, [x])
        y_list = self.stage2(x_list)

        x_list = self._forward_transition(self.transition2, y_list)
        y_list = self.stage3(x_list)

        x_list = self._forward_transition(self.transition3, y_list)
        y_list = self.stage4(x_list)

        return y_list
