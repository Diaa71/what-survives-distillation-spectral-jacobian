#!/usr/bin/env python3
"""Reproduce Kadkhodaie Figure 4 across all 5 regularity values alpha=1..5.

For each alpha: generate reference image, add noise at calibrated sigma,
run Halko K=30 eigendecomposition, produce eigenvector grids.

Outputs
-------
figures/reproduction/alpha{1..5}_eigvec_grid.pdf
figures/reproduction/all_alpha_grid.pdf
results/reproduction/phase_c_results.json
results/reproduction/phase_c_provenance.json
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

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import generate_c_alpha_batch, add_noise
from src.jacobian import halko_sym_eig
from src.eigenvalues import lambda_0, r_eff
from src.models import load_kadkhodaie_unet, make_denoiser, get_c_alpha_ckpt
from src.utils import write_provenance
from src.utils.viz import plot_eigvec_grid


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS   = [1, 2, 3, 4, 5]
SIGMAS   = {1: 0.32, 2: 0.30, 3: 0.25, 4: 0.22, 5: 0.15}
N_IMG    = 9
SEED_IMG = 0
SEED_EIG = 42
K_HALKO  = 30
P_OVER   = 10
K_VIZ    = 20

FIG_DIR = REPO / "figures" / "reproduction"
RES_DIR = REPO / "results" / "reproduction"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Kadkhodaie Figure 4 across alpha=1..5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return p.parse_args()


def main() -> None:
    _parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    all_results: dict[int, dict] = {}
    all_evecs: dict[int, np.ndarray] = {}

    print(f"Phase C — device={DEVICE}")

    for alpha in ALPHAS:
        sigma = SIGMAS[alpha]
        print(f"\n=== alpha={alpha}, sigma={sigma} ===")

        ckpt = get_c_alpha_ckpt(alpha, root=REPO / "checkpoints" / "kadkhodaie")
        model = load_kadkhodaie_unet(ckpt, device=DEVICE)
        denoise_fn = make_denoiser(model, DEVICE)

        batch = generate_c_alpha_batch(alpha=alpha, n=200, seed=SEED_IMG)
        clean = batch[N_IMG:N_IMG + 1].to(DEVICE)
        noisy, _ = add_noise(clean, sigma, device=DEVICE, seed=SEED_EIG)

        eigs, evecs = halko_sym_eig(
            denoise_fn, noisy, K=K_HALKO, p=P_OVER,
            seed=SEED_EIG, device=DEVICE,
        )
        print(f"  lambda_0={eigs[0]:.4f}  lambda_19={eigs[19]:.4f}")

        fig_ev = plot_eigvec_grid(
            evecs[:K_VIZ], eigs[:K_VIZ], 80, 80, rows=4, cols=5,
            title=f"C^alpha alpha={alpha} eigenvectors (image n={N_IMG}, sigma={sigma})",
        )
        fig_ev.savefig(FIG_DIR / f"alpha{alpha}_eigvec_grid.pdf", bbox_inches="tight")
        plt.close(fig_ev)

        all_results[alpha] = {
            "alpha": alpha, "sigma": sigma,
            "lambda0": lambda_0(eigs), "r_eff": r_eff(eigs),
            "eigs": eigs.tolist(),
        }
        all_evecs[alpha] = evecs

    # Combined 5-row x 20-column grid
    print("\nBuilding all_alpha_grid.pdf ...")
    fig, axs = plt.subplots(5, K_VIZ, figsize=(48, 14))
    fig.suptitle("C^alpha denoiser Jacobian eigenvectors: all alpha values", fontsize=13, y=0.997)
    for row_i, alpha in enumerate(ALPHAS):
        evecs = all_evecs[alpha]
        eigs = np.array(all_results[alpha]["eigs"])
        for col in range(K_VIZ):
            ax = axs[row_i, col]
            ev = evecs[col].reshape(80, 80)
            vmax = float(np.abs(ev).max())
            ax.imshow(ev, cmap="RdBu", vmin=-vmax, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            if col == 0:
                ax.set_ylabel(f"alpha={alpha}\nsigma={SIGMAS[alpha]:.2f}",
                              fontsize=9, rotation=0, labelpad=40, va="center")
            ax.set_title(f"lambda={eigs[col]:.3f}", fontsize=6)
            ax.axis("off")
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "all_alpha_grid.pdf", bbox_inches="tight", dpi=120)
    plt.close(fig)

    with open(RES_DIR / "phase_c_results.json", "w") as f:
        json.dump({
            "experiment": "phase_c_calpha_reproduction",
            "per_alpha": {str(a): v for a, v in all_results.items()},
        }, f, indent=2)

    write_provenance(RES_DIR / "phase_c_provenance.json",
                     seed=SEED_EIG, runtime_s=time.time() - t_total,
                     extras={"alphas": ALPHAS, "sigmas": SIGMAS})

    print(f"\nPhase C complete in {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
