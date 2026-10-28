"""DART: Gaussian noise on the executed action, clean teacher label (plan §1.6, Laskey et al. 2017).

During a DART collection the teacher flies, and at every control step the action sent to
``env.step`` is ``clip(teacher(obs) + sigma * z, -1, 1)`` with ``z ~ N(0, I_3)``; the stored
``label`` is the clean ``teacher(obs)``. The noise is never put on the label
(il-conventions). The collector rounds the executed action to float32 before ``env.step``
(``ila.data.collect.executable``), so the stored ``action_exec`` is exactly what ran.
Units: normalised action units (x v_max = 1.5 m/s model scale, lambda = 1/25), so
sigma = 0.10 is 0.15 m/s model scale per axis before the env's norm cap.

Per-episode generator. ``z`` comes from ``numpy.random.default_rng(SeedSequence([seed,
H("dart"), demo_index, episode_seed]))`` and is drawn once per control step, in order. It does
not depend on sigma, so the three sigma sets share their standard-normal draws step by step
(common random numbers; their trajectories still diverge once the noise moves the drone). It
depends on nothing else, so a re-flown episode reproduces its noise exactly. ``H`` is the
SHA-256 label entropy of ``ila.data.episodes`` (never ``hash()``).
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ila.data.episodes import label_entropy

__all__ = ["DART_LABEL", "DartNoise", "dart_entropy", "perturb"]

#: The label mixed into every DART generator's entropy.
DART_LABEL: Final[str] = "dart"


def dart_entropy(seed: int, demo_index: int, episode_seed: int) -> list[int]:
    """Return the entropy words of one episode's DART generator.

    Args:
        seed: The DART seed (``configs/data/demos.yaml`` ``collections.dart.seed``).
        demo_index: The episode's global index in its stream.
        episode_seed: The env's episode seed.

    Returns:
        ``[seed, H("dart"), demo_index, episode_seed]``.
    """
    return [int(seed), label_entropy(DART_LABEL), int(demo_index), int(episode_seed)]


def perturb(
    action: NDArray[np.float64], sigma: float, z: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the executed DART action ``clip(action + sigma * z, -1, 1)``.

    Args:
        action: The clean teacher action, ``(3,)`` normalised.
        sigma: Noise std, normalised units (> 0).
        z: Standard-normal draw, ``(3,)``.

    Returns:
        ``(3,)`` float64 in ``[-1, 1]``.
    """
    return np.clip(np.asarray(action, np.float64) + float(sigma) * np.asarray(z), -1.0, 1.0)


class DartNoise:
    """One episode's noise stream: call :meth:`executed` once per control step, in order.

    Attributes:
        sigma: Noise std, normalised units.
        steps: Draws taken so far.
    """

    def __init__(self, sigma: float, seed: int, demo_index: int, episode_seed: int) -> None:
        """Seed the episode's generator.

        Raises:
            ValueError: If ``sigma`` is not positive and finite.
        """
        if not (np.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"DART sigma must be positive, got {sigma}")
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(
            np.random.SeedSequence(dart_entropy(seed, demo_index, episode_seed))
        )
        self.steps = 0

    def executed(self, clean: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the executed action for the next step from the clean teacher action."""
        z = self._rng.standard_normal(3)
        self.steps += 1
        return perturb(clean, self.sigma, z)
