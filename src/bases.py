"""Harmonic reference basis Phi_dct and harmonicity measurement H(V, Phi).



Core quantity: H(V, Phi) = (1/K) ||V Phi||_F^2, measuring alignment of a
K-dimensional eigenbasis V with the bottom-K graph-Laplacian eigenvectors Phi.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# 8-neighbour grid graph
# ---------------------------------------------------------------------------

_NEIGHBOUR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def _edge_list(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Undirected 8-neighbour edge list for an H x W grid."""
    srcs, dsts = [], []
    for dy, dx in _NEIGHBOUR_OFFSETS:
        if dy < 0 or (dy == 0 and dx < 0):
            continue
        i0 = max(0, -dy)
        i1 = min(H, H - dy)
        j0 = max(0, -dx)
        j1 = min(W, W - dx)
        I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        srcs.append((I * W + J).ravel())
        dsts.append(((I + dy) * W + (J + dx)).ravel())
    return np.concatenate(srcs), np.concatenate(dsts)


def _sym_norm_laplacian(
    H: int, W: int, weights: np.ndarray, src: np.ndarray, dst: np.ndarray,
) -> sp.csr_matrix:
    """L_sym = I - D^{-1/2} W D^{-1/2} for an 8-neighbour grid graph."""
    N = H * W
    rows_sym = np.concatenate([src, dst])
    cols_sym = np.concatenate([dst, src])
    vals_sym = np.concatenate([weights, weights])
    W_sparse = sp.coo_matrix((vals_sym, (rows_sym, cols_sym)), shape=(N, N)).tocsr()
    degrees = np.asarray(W_sparse.sum(axis=1)).ravel()
    d_inv_sq = 1.0 / np.sqrt(degrees + 1e-12)
    D_sqrt = sp.diags(d_inv_sq)
    L = sp.eye(N) - D_sqrt @ W_sparse @ D_sqrt
    return 0.5 * (L + L.T)


# ---------------------------------------------------------------------------
# Phi_dct: bottom-K eigenvectors of the unweighted grid Laplacian
# ---------------------------------------------------------------------------

def unweighted_graph_laplacian(H: int, W: int) -> sp.csr_matrix:
    """Symmetric-normalised Laplacian of the 8-neighbour grid with uniform weights.

    """
    src, dst = _edge_list(H, W)
    weights = np.ones(src.shape[0], dtype=np.float64)
    return _sym_norm_laplacian(H, W, weights, src, dst)


def bottom_k_eigvecs(
    L: sp.csr_matrix, k: int, *, sigma_shift: float = -1e-5,
) -> np.ndarray:
    """Bottom-k eigenvectors of a symmetric PSD sparse matrix L.

    Uses shift-invert mode at sigma=sigma_shift for reliable bottom-end spectra.

    Returns
    -------
    Phi : np.ndarray  (n, k), columns = eigenvectors, ascending eigenvalue order
    """
    vals, vecs = spla.eigsh(L, k=k, sigma=sigma_shift, which="LM")
    order = np.argsort(vals)
    Phi = vecs[:, order].astype(np.float32)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True).clip(min=1e-12)
    return Phi


def build_phi_dct(H: int = 64, W: int = 64, K: int = 30) -> np.ndarray:
    """Build Phi_dct: bottom-K eigenvectors of the unweighted grid Laplacian.

    Returns
    -------
    Phi : np.ndarray  (H*W, K)
    """
    L = unweighted_graph_laplacian(H, W)
    return bottom_k_eigvecs(L, k=K)


# ---------------------------------------------------------------------------
# Phi_edge: bottom-K eigenvectors of the Canny-edge-weighted grid Laplacian
# ---------------------------------------------------------------------------

def build_phi_edge(
    image_gs: np.ndarray, K: int = 30, canny_sigma: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Build Φ_edge(x): bottom-K eigenvectors of a Canny-edge-weighted Laplacian.

    For each edge (i,j) in the 8-neighbour grid, weight w_ij = 0 if exactly one
    of {i,j} is a Canny edge pixel (cuts connections across contours), else 1.

    Parameters
    ----------
    image_gs : np.ndarray (H, W) — grayscale image in [0, 1]
    K : int — number of bottom eigenvectors
    canny_sigma : float — Gaussian sigma for Canny edge detector

    Returns
    -------
    Phi : np.ndarray (H*W, K) — unit-norm columns, ascending eigenvalue order
    edge_fraction : float — fraction of pixels flagged as edges
    """
    from skimage.feature import canny as canny_edge

    H, W = image_gs.shape
    edge_map = canny_edge(image_gs.astype(np.float64), sigma=canny_sigma)
    edge_fraction = float(edge_map.sum()) / (H * W)

    edge_flat = edge_map.ravel()
    src, dst = _edge_list(H, W)
    is_edge_src = edge_flat[src]
    is_edge_dst = edge_flat[dst]
    weights = np.ones(src.shape[0], dtype=np.float64)
    weights[is_edge_src ^ is_edge_dst] = 0.0

    L = _sym_norm_laplacian(H, W, weights, src, dst)
    Phi = bottom_k_eigvecs(L, k=K)
    return Phi, edge_fraction


# ---------------------------------------------------------------------------
# Harmonicity H(V, Phi) and per-eigenvector projection p_k
# ---------------------------------------------------------------------------

def harmonicity(V: np.ndarray, Phi: np.ndarray) -> float:
    """H(V, Phi) = (1/K) ||V Phi||_F^2.

    See docs/02_formalism.md Section 4.

    Parameters
    ----------
    V : np.ndarray  (K, d) -- rows = unit-norm eigenvectors (grayscale-collapsed)
    Phi : np.ndarray  (d, K) -- columns = unit-norm harmonic reference

    Returns
    -------
    H : float in [0, 1]
    """
    K = V.shape[0]
    M = V @ Phi
    return float(np.sum(M * M) / K)


def projection_coefficients(V: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Per-eigenvector harmonicity p_k = ||Phi^T v_k||^2.

    See docs/02_formalism.md Section 4.

    Returns
    -------
    p : np.ndarray  (K,) -- p[k] = sum_j (V[k] . Phi[:, j])^2
    """
    M = V @ Phi
    return (M ** 2).sum(axis=1).astype(np.float64)


def random_baseline_h(
    d: int, K: int, Phi: np.ndarray, n_draws: int = 50, seed: int = 0,
) -> float:
    """Mean H(R, Phi) over n_draws of K random unit vectors in R^d.

    Expected analytic value: K / d.
    """
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_draws):
        R = rng.standard_normal((K, d)).astype(np.float32)
        R = R / np.linalg.norm(R, axis=1, keepdims=True).clip(min=1e-12)
        acc += harmonicity(R, Phi)
    return acc / n_draws


# ---------------------------------------------------------------------------
# Grayscale collapse: (K, 3, H, W) -> (K, H*W)
# ---------------------------------------------------------------------------

def grayscale_collapse(V: np.ndarray, C: int = 3, H: int = 64, W: int = 64):
    """Collapse RGB eigenvectors to grayscale by averaging channels.

    Parameters
    ----------
    V : np.ndarray  (K, C*H*W)

    Returns
    -------
    V_gs : np.ndarray  (K, H*W) -- re-normalised rows
    row_norms : list[float] -- pre-normalisation norms (< 0.1 flags cancellation)
    """
    K = V.shape[0]
    V_rgb = V.reshape(K, C, H, W)
    V_gray = V_rgb.mean(axis=1).reshape(K, H * W)
    row_norms = []
    for k in range(K):
        n = float(np.linalg.norm(V_gray[k]))
        row_norms.append(n)
        V_gray[k] /= max(n, 1e-12)
    return V_gray.astype(np.float32), row_norms


# ---------------------------------------------------------------------------
# DRR: directional derivative ratio at edges
# ---------------------------------------------------------------------------

def compute_drr(
    eigvec_2d: np.ndarray,
    edge_map: np.ndarray,
    gx_image: np.ndarray,
    gy_image: np.ndarray,
) -> float:
    """Directional derivative ratio of an eigenvector at image edges.

    At each edge pixel, decomposes the eigenvector's spatial gradient into
    components normal and tangential to the image edge, then returns
    DRR = mean(g_t^2) / mean(g_n^2).

    DRR < 1 means the eigenvector varies more across edges than along them
    (geometry-adaptive). DRR ~ 1 means no directional preference.

    Parameters
    ----------
    eigvec_2d : (H, W) — one grayscale-collapsed eigenvector
    edge_map : (H, W) bool — Canny edge mask
    gx_image, gy_image : (H, W) — Sobel gradients of the clean image

    Returns
    -------
    DRR : float, or NaN if fewer than 50 edge pixels or denominator ~ 0
    """
    if edge_map.sum() < 50:
        return float("nan")

    gy_ev = np.zeros_like(eigvec_2d)
    gx_ev = np.zeros_like(eigvec_2d)
    gy_ev[1:, :] = eigvec_2d[1:, :] - eigvec_2d[:-1, :]
    gx_ev[:, 1:] = eigvec_2d[:, 1:] - eigvec_2d[:, :-1]

    theta = np.arctan2(gy_image, gx_image)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    mask = edge_map
    g_n = gx_ev[mask] * cos_t[mask] + gy_ev[mask] * sin_t[mask]
    g_t = -gx_ev[mask] * sin_t[mask] + gy_ev[mask] * cos_t[mask]

    mean_gn2 = np.mean(g_n ** 2)
    if mean_gn2 < 1e-12:
        return float("nan")
    return float(np.mean(g_t ** 2) / mean_gn2)


# ---------------------------------------------------------------------------
# Self-test at import time
# ---------------------------------------------------------------------------

def _trivial_test() -> None:
    """V = Phi^T should give H = 1."""
    L = unweighted_graph_laplacian(16, 16)
    Phi = bottom_k_eigvecs(L, k=8)
    V = Phi.T
    h_val = harmonicity(V, Phi)
    if not (0.98 <= h_val <= 1.02):
        raise AssertionError(
            f"Trivial test failed: H(Phi.T, Phi) = {h_val}, expected ~ 1."
        )


_trivial_test()
