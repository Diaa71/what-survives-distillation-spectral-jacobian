"""Loaders for Kadkhodaie reference UNets (C^alpha and CelebA checkpoints).

Upstream: https://github.com/LabForComputationalVision/memorization_generalization

The UNet architecture is fixed (64 channels, RF=90, 3 blocks) across all alpha
values and all N values for CelebA. Only the checkpoint weights differ.

Denoising convention: Kadkhodaie networks predict noise, not the clean image.
The denoiser is f(y) = y - model(y).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


DEFAULT_KADKHODAIE_ROOT = Path("checkpoints/kadkhodaie")


def _unet_hparams() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    defaults = [
        ("kernel_size", 3), ("padding", 1), ("num_kernels", 64),
        ("RF", 90), ("num_channels", 1), ("bias", False),
        ("num_enc_conv", 2), ("num_mid_conv", 2), ("num_dec_conv", 2),
        ("pool_window", 2), ("num_blocks", 3),
    ]
    for k, v in defaults:
        parser.add_argument(f"--{k}", default=v)
    return parser.parse_args("")


def load_kadkhodaie_unet(ckpt_path: Path | str, device: str = "cpu") -> nn.Module:
    """Load a Kadkhodaie UNet from a checkpoint file.

    Requires network.py from the upstream repo to be importable (add
    reference_repos/memorization_generalization/code/ to sys.path).
    """
    from network import UNet

    args = _unet_hparams()
    model = UNet(args).to(device).eval().float()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(cleaned)
    return model


def make_denoiser(model: nn.Module, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return f(y) = y - model(y), the denoiser whose Jacobian we analyse."""
    def denoise_fn(y: torch.Tensor) -> torch.Tensor:
        y = y.to(device=device, dtype=torch.float32)
        return y - model(y)
    return denoise_fn


def get_c_alpha_ckpt(alpha: int, root: Path | str = DEFAULT_KADKHODAIE_ROOT) -> Path:
    """Path to the N=10^5 C^alpha checkpoint: {root}/C_alpha{alpha}/model.pt."""
    return Path(root) / f"C_alpha{alpha}" / "model.pt"


def get_celeba_ckpt(n: int, root: Path | str = DEFAULT_KADKHODAIE_ROOT) -> Path:
    """Path to the CelebA checkpoint trained on N images: {root}/celeba_{n}/model.pt."""
    return Path(root) / f"celeba_{n}" / "model.pt"
