"""Generation and noise injection for Kadkhodaie C^alpha synthetic images."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def generate_c_alpha_batch(
    alpha: int, n: int = 200, seed: int = 0, im_size: int = 80,
) -> torch.Tensor:
    """Return a batch of C^alpha 80x80 grayscale images in [0, 1].

    Parameters
    ----------
    alpha : int  (1..5 for released checkpoints)
    n : int
    seed : int
    im_size : int

    Returns
    -------
    imgs : torch.Tensor  [n, 1, im_size, im_size], float32, [0, 1]
    """
    from synthetic_data_generators import make_C_alpha_images

    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs = make_C_alpha_images(
        alpha=alpha, beta=alpha, separable=False, im_size=im_size,
        num_samples=n, constant_background=False, factor=(2, 2),
        antialiasing=0, wavelet="db2", mode="reflect",
    )
    return imgs.float()


def add_noise(
    clean: torch.Tensor, sigma: float, device: str = "cuda",
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise of std sigma (in [0, 1] scale). No clamping.

    Parameters
    ----------
    clean : torch.Tensor  [B, C, H, W] in [0, 1]
    sigma : float
    device : str
    seed : int | None

    Returns
    -------
    noisy, noise : torch.Tensor
    """
    from dataloader_func import add_noise_torch

    if seed is not None:
        torch.manual_seed(seed)
    noisy, noise = add_noise_torch(clean, sigma * 255, device)
    return noisy, noise
