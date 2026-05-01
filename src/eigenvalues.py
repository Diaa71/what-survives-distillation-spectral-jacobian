"""Eigenvalue-spectrum characterisation: r_eff, spectral decay, lambda_0.


"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats as scipy_stats


def lambda_0(eigs: np.ndarray) -> float:
    """Top eigenvalue. Degeneracy gate: lambda_0 < 0.5 => Jacobian is near-null."""
    return float(eigs[0])


def r_eff(eigs: np.ndarray) -> float:
    """Effective rank = (sum lambda_k)^2 / sum lambda_k^2.


    """
    s1 = float(eigs.sum())
    s2 = float((eigs ** 2).sum())
    return s1 * s1 / s2 if s2 > 0 else 0.0


def alpha_fit(eigs: np.ndarray) -> Tuple[float, float]:
    """Power-law decay exponent: log lambda_k = -alpha log k + const.

    Returns (alpha, r_squared).
    """
    K = len(eigs)
    ks = np.arange(1, K + 1, dtype=float)
    mask = eigs > 0
    if mask.sum() < 3:
        return float("nan"), float("nan")
    slope, _, r, _, _ = scipy_stats.linregress(
        np.log(ks[mask]), np.log(eigs[mask])
    )
    return float(-slope), float(r * r)
