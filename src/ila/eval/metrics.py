"""Per-episode metrics added to P8's P3-D1 section 3 set (Phase 8), and the statistics P8 lacks.

P3-D1 §1 names three statistics that are not P8's and are written in ``ila.eval``; each has a
unit test on a case of known value. This module holds the first piece of H2's: the exact 1-D
empirical Wasserstein-1 distance (:func:`wasserstein_1`). The H2 bootstrap around it is
Phase 8's.

``ila.data.audit.wasserstein_1`` computes the same quantity for the demo-set precondition
(P3-D1 §6 H2); ``tests/test_eval_h2_select.py`` checks the two agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["wasserstein_1"]


def wasserstein_1(
    a: Sequence[float] | npt.NDArray[np.float64], b: Sequence[float] | npt.NDArray[np.float64]
) -> float:
    """Exact 1-D Wasserstein-1 distance between two empirical distributions.

    Each sample carries weight ``1/len`` of its own set. The distance is
    ``integral |F_a(x) - F_b(x)| dx`` over the merged support, computed exactly on the merged
    sorted samples (no binning, no interpolation). For H2 the inputs are touched-down
    ``time_to_touchdown_s`` values, seconds model scale (lambda = 1/25; x5 for full scale),
    and the result is in the same units.

    Args:
        a: Samples of the first distribution (finite).
        b: Samples of the second distribution (finite, same units).

    Returns:
        The distance, in the inputs' units; NaN if either set is empty (an empty pool has no
        distribution; P3-D1 §1 counts such cases rather than imputing them).

    Raises:
        ValueError: If any sample is not finite (a missing touchdown must be dropped by the
            caller and counted, never passed in as NaN or inf).
    """
    x = np.sort(np.asarray(a, dtype=np.float64).ravel())
    y = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("wasserstein_1 takes finite samples only")
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    fa = np.searchsorted(x, grid[:-1], side="right") / x.size
    fb = np.searchsorted(y, grid[:-1], side="right") / y.size
    return float(np.sum(np.abs(fa - fb) * np.diff(grid)))
