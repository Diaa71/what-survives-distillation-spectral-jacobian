"""Visualisation helpers: eigenvector grids, p_k curves, M heatmaps, spectra."""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_eigvec_grid(
    evecs: np.ndarray, eigs: Sequence[float], H: int, W: int,
    rows: int = 2, cols: int = 4, title: str = "Eigenvectors", cmap: str = "RdBu",
) -> plt.Figure:
    """RdBu grid of eigenvectors with symmetric colormap per panel."""
    n_panels = rows * cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.75))
    if rows * cols == 1:
        axs = np.array([[axs]])
    elif rows == 1 or cols == 1:
        axs = axs.reshape(rows, cols)

    fig.suptitle(title, fontsize=11, y=0.995)
    for k in range(n_panels):
        r, c = divmod(k, cols)
        ax = axs[r, c]
        ev = evecs[k].reshape(H, W)
        vmax = float(np.abs(ev).max())
        im = ax.imshow(ev, cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"k={k+1}, λ={eigs[k]:.3f}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_pk_curves(
    records: list[dict], K: int, H_random: float, title: str = "p_k curves",
) -> plt.Figure:
    """Per-eigenvector harmonicity curves for multiple models on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = np.arange(1, K + 1)
    for r in records:
        ax.plot(ks, r["p_k"][:K], "o-", color=r.get("color", None), lw=1.8, ms=4,
                label=f"{r['model']}  λ₀={r['lambda0']:.2f}  H={r['H_dct']:.3f}")
    ax.axhline(H_random, color="k", ls="--", lw=1, label=f"random = {H_random:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvector rank k")
    ax.set_ylabel("p_k = ||Φ^T v_k||²")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_eigenvalue_spectra(
    records: list[dict], title: str = "Eigenvalue spectra",
) -> plt.Figure:
    """Log-log eigenvalue decay curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in records:
        eigs = np.array(r["eigenvalues"])
        ks = np.arange(1, len(eigs) + 1)
        ax.plot(ks, eigs, "o-", color=r.get("color", None), lw=1.5, ms=3,
                label=f"{r['model']}  r_eff={r.get('r_eff', 0):.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank k")
    ax.set_ylabel("λ_k")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_basis_map_heatmap(
    M: np.ndarray, q_teacher: np.ndarray, tau: float = 0.02,
    title: str = "Basis map M",
) -> plt.Figure:
    """M heatmap with harmonic teacher directions highlighted on x-axis."""
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("teacher direction j")
    ax.set_ylabel("student direction i")
    ax.set_title(title)

    harmonic_idx = np.where(q_teacher > tau)[0]
    for j in harmonic_idx:
        ax.axvline(j, color="lime", lw=0.8, alpha=0.5)

    fig.tight_layout()
    return fig


def plot_sv_spectrum(
    sigma_vals: np.ndarray, title: str = "Singular values of M",
) -> plt.Figure:
    """Bar chart of M's singular values."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(sigma_vals) + 1), sigma_vals, color="steelblue")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ_i")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sample_row(
    images: list[np.ndarray], labels: list[str], title: str = "Samples",
) -> plt.Figure:
    """Row of sample images (e.g., same noise through four models)."""
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=11)
    for ax, img, label in zip(axs, images, labels):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        ax.imshow(img.clip(0, 1), cmap="gray" if img.ndim == 2 else None)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig
