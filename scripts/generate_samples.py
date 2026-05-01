#!/usr/bin/env python3
"""Generate 16 images from each model (teacher + 3 students) using seeds 0-15."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.models import load_edm_teacher, load_cd_student

DEVICE = "cuda"
SEEDS = list(range(16))
CLASS_LABEL = 1  # goldfish
C, H, W = 3, 64, 64
FIG_DIR = Path("results/experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_MAX = 80.0
SIGMA_MIN = 0.002
RHO = 7.0
N_STEPS = 50


def karras_schedule(n_steps, sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN, rho=RHO):
    ramp = np.linspace(0, 1, n_steps + 1)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return sigmas


@torch.no_grad()
def euler_sample_teacher(model_dict, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(1, C, H, W, generator=gen, device=DEVICE) * SIGMA_MAX
    sigmas = karras_schedule(N_STEPS)
    for i in range(N_STEPS):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        denoised = denoise_no_grad(model_dict, x, sigma)
        d = (x - denoised) / sigma
        x = x + (sigma_next - sigma) * d
    return x.clamp(-1, 1).float().cpu()


def denoise_no_grad(model_dict, x, sigma):
    from src.models.edm import _class_labels_kwarg, _sigma_to_t, SIGMA_DATA
    unet = model_dict["unet"]
    scheduler = model_dict["scheduler"]
    model_dtype = next(unet.parameters()).dtype

    sigma_data = SIGMA_DATA
    sigma_min = 0.002
    c_skip = sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2)**0.5
    c_in = 1.0 / (sigma**2 + sigma_data**2)**0.5

    t = torch.tensor([_sigma_to_t(scheduler, sigma)], dtype=torch.float32, device=DEVICE)
    scaled = (x * c_in).to(model_dtype)
    kwargs = _class_labels_kwarg(model_dict, DEVICE)
    out = unet(scaled, t.to(model_dtype), return_dict=False, **kwargs)[0]
    return c_skip * x + c_out * out.to(torch.float32)


@torch.no_grad()
def single_step_student(model_dict, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(1, C, H, W, generator=gen, device=DEVICE) * SIGMA_MAX
    denoised = denoise_no_grad(model_dict, x, SIGMA_MAX)
    return denoised.clamp(-1, 1).float().cpu()


def to_display(img_tensor):
    img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    return np.clip((img + 1) / 2, 0, 1)


def save_grid(images, path, title):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(to_display(images[i]))
        ax.set_title(f"seed {i}", fontsize=7)
        ax.axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


print("=" * 60)
print("SAMPLE GENERATION: 16 seeds × 4 models")
print("=" * 60)
print(f"Class label: {CLASS_LABEL} (goldfish)")
print(f"Teacher: {N_STEPS}-step Euler, Students: single-step")
print()

all_grids = {}

# --- Teacher ---
print("Loading teacher...")
teacher = load_edm_teacher(DEVICE, class_label=CLASS_LABEL)
print("Generating teacher samples (50-step Euler)...")
teacher_imgs = []
for s in SEEDS:
    img = euler_sample_teacher(teacher, s)
    teacher_imgs.append(img)
    print(f"  seed {s} done")
save_grid(teacher_imgs, FIG_DIR / "samples_teacher.png", "Teacher (EDM) — 50-step Euler")
all_grids["teacher"] = teacher_imgs
del teacher
torch.cuda.empty_cache()

# --- Students ---
for variant in ["cd_l2", "cd_lpips", "ct"]:
    from src.models.edm import MODEL_LABELS
    label = MODEL_LABELS[variant]
    print(f"\nLoading {variant}...")
    student = load_cd_student(variant, DEVICE, class_label=CLASS_LABEL)
    print(f"Generating {variant} samples (single-step)...")
    student_imgs = []
    for s in SEEDS:
        img = single_step_student(student, s)
        student_imgs.append(img)
        print(f"  seed {s} done")
    save_grid(student_imgs, FIG_DIR / f"samples_{variant}.png", f"{label} — single-step")
    all_grids[variant] = student_imgs
    del student
    torch.cuda.empty_cache()

# --- Combined comparison ---
print("\nSaving combined comparison...")
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
model_names = ["teacher", "cd_l2", "cd_lpips", "ct"]
titles = ["Teacher (EDM)\n50-step Euler", "CD-L2\nsingle-step",
          "CD-LPIPS\nsingle-step", "CT\nsingle-step"]

for col, (name, title) in enumerate(zip(model_names, titles)):
    for row in range(4):
        seed_idx = col + row * 4  # show seeds 0-3 in row 0, 4-7 in row 1, etc.
        axes[row, col].imshow(to_display(all_grids[name][seed_idx]))
        if row == 0:
            axes[row, col].set_title(title, fontsize=11, fontweight="bold")
        axes[row, col].set_ylabel(f"seed {seed_idx}", fontsize=8) if col == 0 else None
        axes[row, col].axis("off")

plt.suptitle("ImageNet-64 Samples (class 1: goldfish)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(FIG_DIR / "samples_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {FIG_DIR / 'samples_comparison.png'}")

print("\n*** DONE ***")
