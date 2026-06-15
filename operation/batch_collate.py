# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from torch.utils.data.dataloader import default_collate


def spine_collater(batch: list[Any]):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)
