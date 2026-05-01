#!/usr/bin/env python3
"""Reproduce CelebA memorisation-to-transition progression for N=1,10,100,1000.

N=10^5 checkpoint is NOT publicly released and is NOT included.

Outputs
-------
figures/reproduction/celeba_N{1,10,100,1000}_eigvec_grid.pdf
figures/reproduction/celeba_N_progression.pdf
results/reproduction/phase_d_results.json
results/reproduction/phase_d_provenance.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference_repos" / "memorization_generalization" / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import add_noise
from src.jacobian import halko_sym_eig
from src.eigenvalues import lambda_0, r_eff
from src.models import get_celeba_ckpt, load_kadkhodaie_unet, make_denoiser
from src.utils import write_provenance
from src.utils.viz import plot_eigvec_grid


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_VALUES = [1, 10, 100, 1000]
SIGMA    = 0.15
IMG_IDX  = 0
SEED_EIG = 42
K_HALKO  = 30
P_OVER   = 10
K_VIZ    = 20
K_COMB   = 8

FIG_DIR = REPO / "figures" / "reproduction"
RES_DIR = REPO / "results" / "reproduction"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce CelebA Figure 3 for N=1,10,100,1000.",
    )
    return p.parse_args()


def _load_celeba_test(repo: Path) -> torch.Tensor:
    path = repo / "data" / "celeba_80x80" / "test_images.pt"
    images = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(images, dict):
        key = next((k for k in images if "test" in k.lower()), next(iter(images)))
        images = images[key]
    images = images.float()
    if images.max() > 5.0:
        images = images / 255.0
    return images


def main() -> None:
    _parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    print(f"Phase D — device={DEVICE}")

    imgs = _load_celeba_test(REPO)
    clean = imgs[IMG_IDX:IMG_IDX + 1].to(DEVICE)

    all_results: dict[int, dict] = {}
    all_evecs: dict[int, np.ndarray] = {}

    for N in N_VALUES:
        print(f"\n=== N={N} ===")
        ckpt = get_celeba_ckpt(N, root=REPO / "checkpoints" / "kadkhodaie")
        model = load_kadkhodaie_unet(ckpt, device=DEVICE)
        denoise_fn = make_denoiser(model, DEVICE)

        noisy, _ = add_noise(clean, SIGMA, device=DEVICE, seed=SEED_EIG)
        eigs, evecs = halko_sym_eig(
            denoise_fn, noisy, K=K_HALKO, p=P_OVER,
            seed=SEED_EIG, device=DEVICE,
        )
        print(f"  lambda_0={eigs[0]:.4f}  r_eff={r_eff(eigs):.1f}")

        fig_ev = plot_eigvec_grid(
            evecs[:K_VIZ], eigs[:K_VIZ], 80, 80, rows=4, cols=5,
            title=f"CelebA N={N} eigenvectors (test img {IMG_IDX}, sigma={SIGMA})",
        )
        fig_ev.savefig(FIG_DIR / f"celeba_N{N}_eigvec_grid.pdf", bbox_inches="tight")
        plt.close(fig_ev)

        all_results[N] = {
            "N": N, "sigma": SIGMA, "lambda0": lambda_0(eigs),
            "r_eff": r_eff(eigs), "eigs": eigs.tolist(),
        }
        all_evecs[N] = evecs

    # Combined N-progression grid
    print("\nBuilding celeba_N_progression.pdf ...")
    fig, axs = plt.subplots(len(N_VALUES), K_COMB, figsize=(24, 13))
    fig.suptitle(f"CelebA eigenvectors: N progression (sigma={SIGMA})", fontsize=12, y=0.997)
    for row_i, N in enumerate(N_VALUES):
        evecs = all_evecs[N]
        eigs = np.array(all_results[N]["eigs"])
        for col in range(K_COMB):
            ax = axs[row_i, col]
            ev = evecs[col].reshape(80, 80)
            vmax = float(np.abs(ev).max())
            ax.imshow(ev, cmap="RdBu", vmin=-vmax, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            if col == 0:
                ax.set_ylabel(f"N={N}\nlambda_0={eigs[0]:.3f}",
                              fontsize=9, rotation=0, labelpad=42, va="center")
            ax.set_title(f"lambda={eigs[col]:.3f}", fontsize=7)
            ax.axis("off")
    plt.tight_layout(rect=[0.06, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "celeba_N_progression.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)

    with open(RES_DIR / "phase_d_results.json", "w") as f:
        json.dump({
            "experiment": "phase_d_celeba_memorization",
            "note": "N=10^5 checkpoint not publicly released",
            "per_N": {str(n): v for n, v in all_results.items()},
        }, f, indent=2)

    write_provenance(RES_DIR / "phase_d_provenance.json",
                     seed=SEED_EIG, runtime_s=time.time() - t_total,
                     extras={"N_values": N_VALUES, "sigma": SIGMA})

    print(f"\nPhase D complete in {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
