"""Copycat diagnostic: the student's d action / d last_action (Phase 5; promoted from Phase 4).

Phase 4's ``training-run-diagnostician`` measured how much a vision student's action follows
its own ``last_action`` proprio input (``results/phase4/diagnostics/scripts/diag2.py``,
``bcm_jtrend.py``; P4-D5 S4 table: ``bc`` 0.70-0.91 against the teacher's 0.07-0.18). This
module is that measurement as library code, by autograd instead of central differences:

* **Quantity.** ``J[i, j] = mean over states of d a_i / d last_action_j``, where ``a`` is the
  student's output (``tanh``, before the env's norm cap) and ``last_action`` is the newest row
  (step ``t``) of the proprio window; the older ``To - 1`` rows and the images are held fixed.
  The derivative is taken w.r.t. the **raw** proprio input, i.e. through the encoder's fixed
  standardisation. Both are normalised world-frame velocity setpoints (x v_max = 1.5 m/s model
  scale, lambda = 1/25), so ``J`` is in **normalised action per normalised action**
  (dimensionless). The copycat signature is a diagonal far above the teacher's.
* **Path.** The checkpoint's **EMA** weights, eval mode (centre crop), fp32, on the CPU: the
  weights and path that fly (P3-D1 §4).
* **States.** (a) :func:`heldout_states`: P4-D5's set -- ``ppo_2000`` stream indices 1950-1999,
  every second episode (25 episodes, unfiltered; all 50 are ``success``), 512 frames drawn by
  ``numpy.random.default_rng(0)``. (b) :func:`rollout_states`: a seeded sample of one DAgger
  iteration's own visited states (``default_rng([COPYCAT_SEED, run_seed, iteration])``, 512
  frames). DAgger rollouts break the demos' ``last_action[t] = label[t-1]`` identity, which is
  why (b) is reported beside (a).

Phase 4's values came from central differences with ``eps = 0.05``; for a smooth network they
agree with the autograd values to O(eps^2) (``tests/test_train_dagger.py`` checks a tiny net).
The ``last_action`` columns are located by field name from the observation contract
(:func:`proprio_field_slice`); nothing here selects observation fields from an observation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ila.data.store import DemoStore
from ila.train.dataset import BCWindowDataset, FrameTable
from ila.train.sampling import choose_episodes

__all__ = [
    "COPYCAT_BATCH",
    "COPYCAT_FIELD",
    "COPYCAT_SAMPLE",
    "COPYCAT_SEED",
    "HELDOUT_EVERY",
    "copycat_record",
    "heldout_states",
    "last_action_jacobian",
    "last_action_jacobian_fd",
    "proprio_field_slice",
    "rollout_states",
    "sample_indices",
]

#: The proprio field whose influence is measured.
COPYCAT_FIELD: Final[str] = "last_action"

#: States per measurement (P4-D5).
COPYCAT_SAMPLE: Final[int] = 512

#: Sample seed (P4-D5's ``default_rng(0)``).
COPYCAT_SEED: Final[int] = 0

#: Every second held-out episode (P4-D5: 25 of ``ppo_2000`` 1950-1999).
HELDOUT_EVERY: Final[int] = 2

#: Samples per autograd batch (memory only; the result does not depend on it).
COPYCAT_BATCH: Final[int] = 64

#: A network ``(images uint8[B, To, 96, 96, 3], proprio float32[B, To, D]) -> float32[B, 3]``.
type ActionNet = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def proprio_field_slice(field: str = COPYCAT_FIELD, variant: str = "default") -> slice:
    """Return the columns of ``field`` in the student proprio vector (by contract field name).

    Args:
        field: A student proprio field (``last_action``).
        variant: Observation-contract variant.

    Returns:
        The slice into the proprio's last axis (``last_action``: 9:12 for the default 13-d
        contract).

    Raises:
        ValueError: If ``field`` is not in the variant's student proprio.
    """
    from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

    contract = load_contract(variant=variant)
    offset = 0
    for name in contract.student_proprio:
        size = contract.layout[name][1]
        if name == field:
            return slice(offset, offset + size)
        offset += size
    raise ValueError(f"{field!r} is not a student proprio field of {variant!r}")


def last_action_jacobian(
    net: ActionNet,
    images: torch.Tensor,
    proprio: torch.Tensor,
    columns: slice,
    *,
    batch: int = COPYCAT_BATCH,
) -> NDArray[np.float64]:
    """Mean autograd Jacobian of the action w.r.t. the newest proprio row's ``columns``.

    Samples are independent through the network (eval mode, GroupNorm, no batch statistics),
    so the gradient of ``sum over the batch of a_i`` w.r.t. the inputs is each sample's own
    ``d a_i / d x``.

    Args:
        net: The network, already in eval mode, fp32, on the CPU.
        images: ``uint8[N, To, 96, 96, 3]``.
        proprio: ``float32[N, To, D]`` (raw, as the network takes it).
        columns: The measured field's columns (:func:`proprio_field_slice`).
        batch: Samples per backward pass.

    Returns:
        ``float64[A, F]``: ``J[i, j]`` = mean over the N samples of ``d a_i / d x_j``, with
        ``x`` = ``proprio[:, -1, columns]`` (normalised action per normalised action for
        ``last_action``).
    """
    n = int(proprio.shape[0])
    total: NDArray[np.float64] | None = None
    for start in range(0, n, batch):
        im = images[start : start + batch]
        x = proprio[start : start + batch].detach().clone().float().requires_grad_(True)
        with torch.enable_grad():
            out = net(im, x).float()
            rows = []
            for i in range(out.shape[-1]):
                (g,) = torch.autograd.grad(out[:, i].sum(), x, retain_graph=i < out.shape[-1] - 1)
                rows.append(g[:, -1, columns].sum(dim=0))
        part = torch.stack(rows).double().numpy()
        total = part if total is None else total + part
    assert total is not None
    return total / n


@torch.no_grad()
def last_action_jacobian_fd(
    net: ActionNet,
    images: torch.Tensor,
    proprio: torch.Tensor,
    columns: slice,
    eps: float = 0.05,
) -> NDArray[np.float64]:
    """Phase 4's estimator: central differences of ``±eps`` (raw units) on the same columns.

    Kept to reconcile with P4-D5's committed values; :func:`last_action_jacobian` is the one
    reported from Phase 5 on.

    Returns:
        ``float64[A, F]`` as :func:`last_action_jacobian`.
    """
    cols = range(columns.start, columns.stop)
    out = np.zeros((3, len(cols)))
    for k, c in enumerate(cols):
        hi, lo = proprio.clone(), proprio.clone()
        hi[:, -1, c] += eps
        lo[:, -1, c] -= eps
        d = (net(images, hi).float() - net(images, lo).float()) / (2 * eps)
        out[:, k] = d.double().mean(dim=0).numpy()
    return out


def sample_indices(n: int, seed: int | Sequence[int], k: int = COPYCAT_SAMPLE) -> NDArray[np.int64]:
    """Draw ``min(k, n)`` distinct sample indices with ``default_rng(seed)`` (P4-D5's draw)."""
    rng = np.random.default_rng(list(seed) if isinstance(seed, Sequence) else seed)
    return rng.choice(n, min(k, n), replace=False).astype(np.int64)


def heldout_states(
    data_root: Path, *, start: int = 1950, n: int = 50, every: int = HELDOUT_EVERY
) -> BCWindowDataset:
    """P4-D5's held-out copycat states: ``ppo_2000`` ``start .. start+n-1``, every ``every``-th.

    Read through the student path only (images, allowlisted proprio, labels).
    """
    with DemoStore(data_root / "demos" / "ppo_2000.h5") as store:
        names = list(choose_episodes(store, n, False, start=start).names)[::every]
        table = FrameTable.from_store(store, names)
    return BCWindowDataset(table, _n_obs_steps())


def rollout_states(store_path: Path) -> BCWindowDataset:
    """Every visited state of one DAgger iteration's store (student path only)."""
    with DemoStore(store_path) as store:
        table = FrameTable.from_store(store, store.episodes())
    return BCWindowDataset(table, _n_obs_steps())


def _n_obs_steps() -> int:
    from ila.obs.split import load_contract  # lazy: the contract loads P8's field layout

    return load_contract().n_obs_steps


def copycat_record(
    ckpt: Path, states: BCWindowDataset, indices: NDArray[np.int64]
) -> dict[str, Any]:
    """The copycat measurement of one checkpoint's EMA student on chosen states.

    Args:
        ckpt: A training checkpoint (its ``ema`` weights are measured).
        states: The state set.
        indices: Sample indices into ``states``.

    Returns:
        ``{"diag": [x, y, z], "J": 3x3, "trace_over_3", "n", "units", "weights", "path"}``.
    """
    from ila.policies.registry import load_student  # lazy: builds the network

    student = load_student(ckpt, "cpu", weights="ema")
    net: Any = student.net
    net.eval()
    batch = states.batch(np.asarray(indices, dtype=np.int64))
    images = torch.from_numpy(np.ascontiguousarray(batch["images"]))
    proprio = torch.from_numpy(np.ascontiguousarray(batch["proprio"]))
    cols = proprio_field_slice(COPYCAT_FIELD, states.table.variant)
    jac = last_action_jacobian(net, images, proprio, cols)
    return {
        "diag": [float(v) for v in np.diag(jac)],
        "J": jac.tolist(),
        "trace_over_3": float(np.trace(jac) / 3.0),
        "n": len(indices),
        "units": "normalised action per normalised action (d a / d last_action[t], autograd)",
        "weights": "ema",
        "path": "cpu fp32 eval mode (centre crop)",
    }
