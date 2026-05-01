"""Basis map M = V_S^T V_T and its harmonic selectivity analysis.



The basis map M connects the teacher and student eigenbases. Its SVD, subspace
overlap O, Procrustes residual rho, and harmonic selectivity (F_harm, L_harm)
characterise how distillation transforms the spectral structure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .bases import projection_coefficients


def basis_map(V_S: np.ndarray, V_T: np.ndarray) -> np.ndarray:
    """M = V_S^T V_T in R^{K x K}.

    Parameters
    ----------
    V_S : np.ndarray  (K, d) -- student eigenvectors (grayscale-collapsed, unit rows)
    V_T : np.ndarray  (K, d) -- teacher eigenvectors

    Returns
    -------
    M : np.ndarray  (K, K) -- M_{ij} = <v_S^i, v_T^j>
    """
    return (V_S @ V_T.T).astype(np.float64)


def basis_map_svd(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of M = U Sigma W^T.

    Returns
    -------
    U : np.ndarray  (K, K)
    sigma : np.ndarray  (K,) -- singular values, descending
    Wt : np.ndarray  (K, K)
    """
    U, sigma, Wt = np.linalg.svd(M, full_matrices=False)
    return U, sigma, Wt


def subspace_overlap(M: np.ndarray) -> float:
    """O = (1/K) ||M||_F^2 = (1/K) sum sigma_i^2.


    """
    K = M.shape[0]
    return float(np.sum(M * M) / K)


def procrustes_residual(M: np.ndarray) -> float:
    """rho = ||M - U W^T||_F / ||M||_F.


    """
    U, _, Wt = np.linalg.svd(M, full_matrices=False)
    R = U @ Wt
    return float(np.linalg.norm(M - R) / max(np.linalg.norm(M), 1e-12))


def harmonic_selectivity(
    M: np.ndarray,
    V_T_gs: np.ndarray,
    Phi: np.ndarray,
    tau: float = 0.02,
) -> Tuple[float, np.ndarray]:
    """F_harm(tau): fraction of M's energy from harmonic teacher directions.


    Parameters
    ----------
    M : np.ndarray  (K, K) -- basis map
    V_T_gs : np.ndarray  (K, d) -- teacher eigenvectors (grayscale)
    Phi : np.ndarray  (d, K_phi) -- harmonic reference
    tau : float -- threshold for "harmonic" teacher direction

    Returns
    -------
    F_harm : float -- ||M[:, harmonic]||_F^2 / ||M||_F^2
    q : np.ndarray  (K,) -- per-teacher-direction harmonicity q_j
    """
    q = projection_coefficients(V_T_gs, Phi)
    harmonic_mask = q > tau

    total_energy = float(np.sum(M * M))
    if total_energy < 1e-20:
        return 0.0, q

    harmonic_energy = float(np.sum(M[:, harmonic_mask] ** 2))
    F_harm = harmonic_energy / total_energy
    return F_harm, q


def harmonic_landing(
    M: np.ndarray,
    V_S_gs: np.ndarray,
    V_T_gs: np.ndarray,
    Phi: np.ndarray,
    tau: float = 0.02,
) -> np.ndarray:
    """L_harm(j): harmonicity of the student-space image of harmonic teacher direction j.

    For each harmonic teacher direction j (q_j > tau), reconstructs
    v_landed = V_S^T M[:, j] / ||M[:, j]|| in pixel space, then measures
    its harmonicity as ||Phi^T v_landed||^2.

    Returns
    -------
    L_harm : np.ndarray  (n_harmonic,) -- one entry per harmonic teacher direction
    """
    q = projection_coefficients(V_T_gs, Phi)
    harmonic_idx = np.where(q > tau)[0]

    L_vals = []
    for j in harmonic_idx:
        m_j = M[:, j]
        norm_mj = float(np.linalg.norm(m_j))
        if norm_mj < 1e-12:
            L_vals.append(0.0)
            continue
        v_landed = (V_S_gs.T @ m_j) / norm_mj
        v_landed = v_landed / max(float(np.linalg.norm(v_landed)), 1e-12)
        L_vals.append(float(np.sum((Phi.T @ v_landed) ** 2)))

    return np.array(L_vals, dtype=np.float64)
