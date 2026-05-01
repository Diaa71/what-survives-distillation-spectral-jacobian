#!/usr/bin/env python3
"""Regenerate sample_grid.png with correct EDM Heun sampler and diverse classes.

Fixes two bugs in the original:
1. Heun correction averaged denoiser outputs D instead of tangent vectors d=(x-D)/σ
2. All samples used class_label=0 (tench) instead of diverse classes

Implements Karras et al. 2022, Algorithm 2 (deterministic sampler).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import load_edm_teacher, load_cd_student, make_denoise_fn


def to_np(t):
    return np.array(t.detach().cpu().float().tolist(), dtype=np.float32)


def edm_heun_sample(model_dict, n_samples, device, seed=42,
                     sigma_max=80.0, sigma_min=0.002, n_steps=50, rho=7):
    """EDM deterministic sampler (Karras et al. 2022, Algorithm 2).

    ODE: dx/dσ = (x - D(x,σ)) / σ, integrated with Heun's method.
    """
    step_indices = torch.arange(n_steps, device=device, dtype=torch.float64)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device, dtype=torch.float64)])

    torch.manual_seed(seed)
    x = torch.randn(n_samples, 3, 64, 64, device=device, dtype=torch.float32) * sigma_max

    with torch.no_grad():
        for i in range(n_steps):
            sigma_cur = float(t_steps[i])
            sigma_next = float(t_steps[i + 1])

            D_cur = make_denoise_fn(model_dict, sigma=sigma_cur)(x)
            d_cur = (x - D_cur) / sigma_cur

            x_next = x + d_cur * (sigma_next - sigma_cur)

            if sigma_next > 0 and i < n_steps - 1:
                D_next = make_denoise_fn(model_dict, sigma=sigma_next)(x_next)
                d_next = (x_next - D_next) / sigma_next
                x_next = x + 0.5 * (d_cur + d_next) * (sigma_next - sigma_cur)

            x = x_next

    return x


CLASS_IDS = [334, 270, 435, 609, 536, 147, 947, 980]
CLASS_NAMES = [
    "porcupine", "grey wolf", "bathtub", "jeep",
    "dock", "grey whale", "mushroom", "volcano",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = [
    ("Teacher (EDM)", "teacher"),
    ("CD-L2", "cd_l2"),
    ("CD-LPIPS", "cd_lpips"),
    ("CT", "ct"),
]

N_CLASSES = len(CLASS_IDS)
fig, axes = plt.subplots(N_CLASSES, len(MODEL_CONFIGS),
                         figsize=(3.5 * len(MODEL_CONFIGS), 3.5 * N_CLASSES))

for col, (label, model_name) in enumerate(MODEL_CONFIGS):
    print(f"Generating samples for {label}...")

    for row, (cls_id, cls_name) in enumerate(zip(CLASS_IDS, CLASS_NAMES)):
        if model_name == "teacher":
            model = load_edm_teacher(device=device, class_label=cls_id)
        else:
            model = load_cd_student(model_name, device=device, class_label=cls_id)

        if model_name == "teacher":
            samples = edm_heun_sample(model, 1, device, seed=42 + row)
        else:
            denoise_fn = make_denoise_fn(model, sigma=80.0)
            torch.manual_seed(42 + row)
            z = torch.randn(1, 3, 64, 64, device=device) * 80.0
            with torch.no_grad():
                samples = denoise_fn(z)

        img = to_np(samples.clamp(-1, 1))[0]
        img = (img + 1) / 2
        img = np.clip(img.transpose(1, 2, 0), 0, 1)

        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(label, fontsize=14, fontweight="bold")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  {cls_name} (class {cls_id}) done")

plt.suptitle("Sample sanity check", fontsize=16, y=1.005)
plt.tight_layout()
out_path = "results/experiments/verification/sample_grid.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved to {out_path}")
