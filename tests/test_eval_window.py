"""The runner's student window is ``chunking.obs_window_steps`` (the one index authority).

Every frame and observation is stamped with its control step, so the steps the runner put in
each ``StudentInput`` can be read back and compared with ``obs_window_steps(t, To)``, for the
committed ``To`` and for other ``To`` values. No PyBullet needed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ila.eval.runner import StudentWindow
from ila.obs.split import ObsContract, load_contract
from ila.policies.dp import chunking
from ila.render.camera import Frame


def _stamped(step: int, contract: ObsContract) -> tuple[np.ndarray, Frame]:
    obs = np.full(contract.obs_dim, float(step), dtype=np.float32)
    rgb = np.full((8, 8, 3), step % 256, dtype=np.uint8)
    return obs, Frame(rgb=rgb)


@pytest.mark.parametrize("to", [None, 1, 3, 5])
def test_window_equals_obs_window_steps(to: int | None) -> None:
    contract = load_contract()
    if to is not None:
        contract = replace(contract, n_obs_steps=to)
    n_obs = contract.n_obs_steps
    window = StudentWindow(contract)
    for episode in range(2):  # clear() between episodes: no step leaks across
        window.clear()
        for t in range(40):
            obs, frame = _stamped(t + 1000 * episode, contract)
            si = window.push(t, obs, frame)
            expected = [s + 1000 * episode for s in chunking.obs_window_steps(t, n_obs)]
            assert si.images.shape[0] == si.proprio.shape[0] == n_obs
            assert [int(v) for v in si.proprio[:, 0]] == expected
            assert [int(v) for v in si.images[:, 0, 0, 0]] == [s % 256 for s in expected]
            assert si.proprio_fields == contract.student_proprio


def test_committed_to_is_the_contract_yaml() -> None:
    assert load_contract().n_obs_steps == 2


def test_window_rejects_gaps_and_missing_frames() -> None:
    contract = load_contract()
    window = StudentWindow(contract)
    obs, frame = _stamped(0, contract)
    window.push(0, obs, frame)
    with pytest.raises(ValueError, match="pushed 2"):
        window.push(2, obs, frame)
    with pytest.raises(ValueError, match="frames"):
        window.push(1, obs, None)
