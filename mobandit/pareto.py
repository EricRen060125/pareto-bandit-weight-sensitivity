"""Pareto dominance.

`pareto_front` is called once per round by the Pareto bandits, so it is the
hot spot of the whole experiment. If numba is installed, a compiled version is
used; otherwise a vectorized NumPy version gives the same result, just slower.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False


def _pareto_front_numpy(P):
    # j dominates i  <=>  P[j] >= P[i] on every objective and > on at least one
    ge_all = (P[:, None, :] >= P[None, :, :]).all(axis=2)   # ge_all[j, i]
    gt_any = (P[:, None, :] > P[None, :, :]).any(axis=2)
    dominated = (ge_all & gt_any).any(axis=0)
    return ~dominated


if HAS_NUMBA:
    @njit(cache=True)
    def _pareto_front_numba(P):
        n, d = P.shape
        nd = np.ones(n, dtype=np.bool_)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ge_all = True
                gt_any = False
                for k in range(d):
                    if P[j, k] < P[i, k]:
                        ge_all = False
                        break
                    if P[j, k] > P[i, k]:
                        gt_any = True
                if ge_all and gt_any:
                    nd[i] = False
                    break
        return nd


def pareto_front(points):
    """Boolean mask of the non-dominated rows (all objectives maximized).

    Accepts a NumPy array or a pandas DataFrame of shape (n_points, n_objectives).
    """
    P = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if HAS_NUMBA:
        return _pareto_front_numba(P)
    return _pareto_front_numpy(P)
