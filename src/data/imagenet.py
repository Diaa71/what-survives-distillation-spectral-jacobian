"""ImageNet-64 validation loader."""

from __future__ import annotations

from pathlib import Path

import torch


def load_imagenet64_test(
    path: Path | str = "data/imagenet64/cached_data_imagenet64.pt",
) -> torch.Tensor:
    """Load the ImageNet-64 test split from cached .pt file.

    Parameters
    ----------
    path : Path or str

    Returns
    -------
    images : torch.Tensor  [N, 3, 64, 64], float32, [-1, 1]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        key = next(
            (k for k in data if "test" in k.lower()),
            next(iter(data)),
        )
        images = data[key]
    else:
        images = data
    return images.float()
