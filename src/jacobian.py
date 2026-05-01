"""Halko-Martinsson-Tropp randomised symmetric eigendecomposition.

Computes the top-K eigenpairs of J_sym(y) = 1/2 (J(y) + J(y)^T) where J is the
Jacobian of a denoiser at y. Matvec-only; never forms J explicitly.



"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# JVP / VJP / symmetric matvec
# ---------------------------------------------------------------------------

def jvp_denoiser(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Forward-mode Jacobian-vector product: J_f(y) v.

    Parameters
    ----------
    denoise_fn : callable
        Denoiser f(y). Accepts [B, C, H, W] tensor, returns same shape.
    y : torch.Tensor  [B, C, H, W]
    v : torch.Tensor  [d] or same shape as y

    Returns
    -------
    jv : torch.Tensor  [d]
    """
    y_img = y.detach().clone().requires_grad_(False)
    v_img = v.reshape(y.shape)
    try:
        _, jv = torch.autograd.functional.jvp(
            denoise_fn, (y_img,), (v_img,), strict=False, create_graph=False,
        )
        return jv.reshape(-1)
    except Exception:
        return vjp_denoiser(denoise_fn, y, v)


def vjp_denoiser(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Reverse-mode vector-Jacobian product: J_f(y)^T v.

    Parameters
    ----------
    denoise_fn : callable
    y : torch.Tensor  [B, C, H, W]
    v : torch.Tensor  [d] or same shape as y

    Returns
    -------
    jtv : torch.Tensor  [d]
    """
    y_in = y.detach().clone().reshape(-1).requires_grad_(True)
    y_img = y_in.reshape(y.shape)
    out = denoise_fn(y_img)
    out.reshape(-1).backward(gradient=v.reshape(-1).detach())
    return y_in.grad.clone()


def sym_matvec(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Jacobian matvec: 1/2 (J + J^T) v."""
    jv = jvp_denoiser(denoise_fn, y, v)
    jtv = vjp_denoiser(denoise_fn, y, v)
    return 0.5 * (jv + jtv)


def asymmetry(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    seed: int = 0,
) -> float:
    """Jacobian asymmetry: ||J - J^T||_F / ||J + J^T||_F, estimated with one probe.

    """
    torch.manual_seed(seed)
    v = torch.randn(y.numel(), device=y.device, dtype=torch.float32)
    v = v / (v.norm() + 1e-12)
    jv = jvp_denoiser(denoise_fn, y, v)
    jtv = vjp_denoiser(denoise_fn, y, v)
    return float((jv - jtv).norm() / (jv + jtv).norm().clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Halko symmetric eigendecomposition
# ---------------------------------------------------------------------------

def halko_sym_eig(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    K: int = 30,
    p: int = 10,
    seed: int = 42,
    device: str = "cpu",
    show_progress: bool = True,
    tikhonov_eps_rel: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K eigenpairs of J_sym at y via randomised two-pass algorithm.

    Parameters
    ----------
    denoise_fn : callable
        Single-argument denoiser f(y) -> clean estimate.
    y : torch.Tensor  [B, C, H, W]
    K : int
        Number of eigenpairs to return.
    p : int
        Oversampling (sketch dimension l = K + p).
    seed : int
    device : str
    show_progress : bool
    tikhonov_eps_rel : float
        Relative Tikhonov regularisation on the small matrix B before eigh.
        Set to 1e-6 for float16 EDM models; 0 for float32 Kadkhodaie models.

    Returns
    -------
    eigenvalues : np.ndarray  [K], float32, descending, clamped >= 0
    eigenvectors : np.ndarray  [K, d], float32, unit rows
    """
    d = y.numel()
    ell = K + p

    torch.manual_seed(seed)
    np.random.seed(seed)

    def matvec(v_flat: torch.Tensor) -> torch.Tensor:
        out = sym_matvec(denoise_fn, y, v_flat)
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # Pass 1: Y = A * Omega
    Omega = torch.randn(d, ell, device=device, dtype=torch.float32)
    Y = torch.zeros(d, ell, device=device, dtype=torch.float32)
    iterator = range(ell)
    if show_progress:
        iterator = tqdm(iterator, desc="Halko pass1", leave=False)
    for j in iterator:
        Y[:, j] = matvec(Omega[:, j])

    Q, _ = torch.linalg.qr(Y.to(torch.float64))
    Q = Q[:, :ell].to(torch.float32)

    # Pass 2: JQ = A * Q
    JQ = torch.zeros(d, ell, device=device, dtype=torch.float32)
    iterator = range(ell)
    if show_progress:
        iterator = tqdm(iterator, desc="Halko pass2", leave=False)
    for j in iterator:
        JQ[:, j] = matvec(Q[:, j])

    B = (Q.T @ JQ).to(torch.float64)
    B = 0.5 * (B + B.T)

    if tikhonov_eps_rel > 0:
        eps_abs = float(tikhonov_eps_rel) * float(B.abs().max())
        B = B + eps_abs * torch.eye(B.shape[0], dtype=torch.float64, device=B.device)

    try:
        evals_B, evecs_B = torch.linalg.eigh(B)
    except Exception:
        from scipy.linalg import eigh as scipy_eigh
        ev, evec = scipy_eigh(B.cpu().numpy())
        evals_B = torch.from_numpy(ev).to(B.device)
        evecs_B = torch.from_numpy(evec).to(B.device)

    idx = evals_B.argsort(descending=True)
    evals_B = evals_B[idx][:K]
    evecs_B = evecs_B[:, idx][:, :K]

    evecs_full = (Q.to(torch.float64) @ evecs_B).to(torch.float32)
    norms = evecs_full.norm(dim=0, keepdim=True).clamp(min=1e-8)
    evecs_full = evecs_full / norms

    eigenvalues = evals_B.float().clamp(min=0.0)

    eigs_np = np.array(eigenvalues.cpu().tolist(), dtype=np.float32)
    evecs_np = np.array(evecs_full.T.cpu().float().tolist(), dtype=np.float32)

    return eigs_np, evecs_np
