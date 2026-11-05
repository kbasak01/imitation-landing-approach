"""Diagnostics state sets under an ablation's observation contract (CPU, synthetic stores).

:func:`ila.train.diagnostics.heldout_states` and :func:`~ila.train.diagnostics.rollout_states`
read the student path under the measured checkpoint's ``obs_variant``: 13-d proprio for
``default`` (byte-identical to the call without the keyword, which committed Phase 4/6
diagnostics depend on), 10-d for ``minus_last_action``, 14-d for ``plus_clock``. The copycat
record refuses states read under another contract than its checkpoint's, and
:func:`ila.train.dp_diagnostics.run_diagnostics` measures a ``dp_minus_last_action`` checkpoint
(1 CPU training step) on 10-d held-out windows instead of failing on 13-d ones.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ila.data.store import DemoStore
from ila.obs.split import VARIANTS, load_contract
from ila.policies import registry
from ila.train import diagnostics, loop
from ila.train import dp_diagnostics as dd
from ila.train import manifest as mf
from ila.train.config import TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset
from test_data_store import synthetic_record, write_store

#: Proprio width per contract variant.
WIDTH = {"default": 13, "minus_last_action": 10, "plus_clock": 14}

N_EPISODES = 6
RUN_SEED = 2


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    records = [synthetic_record(j, 4 + j % 3, seed=300 + j) for j in range(N_EPISODES)]
    write_store(root / "demos" / "ppo_2000.h5", records)
    return root


def _expected_proprio(store_path: Path, names: list[str], variant: str) -> np.ndarray:
    with DemoStore(store_path) as store:
        return np.concatenate([store.student_inputs(n, variant).proprio for n in names])


def _same_table(a: BCWindowDataset, b: BCWindowDataset) -> None:
    assert a.table.names == b.table.names and a.table.variant == b.table.variant
    assert a.n_obs_steps == b.n_obs_steps
    for key in ("images", "proprio", "label", "episode_len"):
        assert getattr(a.table, key).tobytes() == getattr(b.table, key).tobytes(), key


def test_variants_cover_every_width() -> None:
    assert set(WIDTH) == set(VARIANTS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_heldout_states_read_the_variant_contract(variant: str, data_root: Path) -> None:
    held = diagnostics.heldout_states(data_root, start=0, n=4, every=2, obs_variant=variant)
    table = held.table
    assert table.variant == variant and table.n_episodes == 2
    assert table.proprio.shape[1] == WIDTH[variant]
    assert held.n_obs_steps == load_contract(variant=variant).n_obs_steps
    names = list(table.names)
    expected = _expected_proprio(data_root / "demos" / "ppo_2000.h5", names, variant)
    assert table.proprio.tobytes() == expected.tobytes()  # the store's student path, unchanged
    batch = held.batch(np.arange(len(held), dtype=np.int64))
    assert batch["proprio"].shape == (len(held), held.n_obs_steps, WIDTH[variant])
    assert batch["images"].shape[:2] == (len(held), held.n_obs_steps)


def test_heldout_states_default_is_unchanged(data_root: Path) -> None:
    kw: dict[str, Any] = {"start": 0, "n": 4, "every": 2}
    _same_table(
        diagnostics.heldout_states(data_root, **kw),
        diagnostics.heldout_states(data_root, **kw, obs_variant="default"),
    )


def test_minus_last_action_drops_exactly_last_action(data_root: Path) -> None:
    kw: dict[str, Any] = {"start": 0, "n": 4, "every": 1}
    full = diagnostics.heldout_states(data_root, **kw).table.proprio
    minus = diagnostics.heldout_states(data_root, **kw, obs_variant="minus_last_action")
    cols = diagnostics.proprio_field_slice("last_action")
    np.testing.assert_array_equal(
        minus.table.proprio, np.delete(full, np.arange(cols.start, cols.stop), axis=1)
    )
    with pytest.raises(ValueError, match="last_action"):  # copycat is undefined without it
        diagnostics.proprio_field_slice("last_action", minus.table.variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_rollout_states_read_the_variant_contract(variant: str, data_root: Path) -> None:
    path = data_root / "demos" / "ppo_2000.h5"
    states = diagnostics.rollout_states(path, obs_variant=variant)
    assert states.table.variant == variant and states.table.n_episodes == N_EPISODES
    assert states.table.proprio.shape[1] == WIDTH[variant]
    if variant == "default":
        _same_table(states, diagnostics.rollout_states(path))


def test_copycat_record_refuses_states_of_another_contract(data_root: Path, tmp_path: Path) -> None:
    ckpt = tmp_path / "clock.pt"
    torch.save({"format": registry.CHECKPOINT_FORMAT, "obs_variant": "plus_clock"}, ckpt)
    held = diagnostics.heldout_states(data_root, start=0, n=2, every=1)
    with pytest.raises(ValueError, match="plus_clock"):
        diagnostics.copycat_record(ckpt, held, np.arange(2, dtype=np.int64))


def _cfg(arm: str) -> TrainConfig:
    cfg = TrainConfig.load(registry.REGISTRY[arm].config)
    return dataclasses.replace(cfg, batch_size=2, precision="fp32")


def test_dp_diagnostics_measure_a_variant_checkpoint(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``dp_minus_last_action``: held-out windows are 10-d; the record is written (no flight)."""
    torch.set_num_threads(1)
    arm = "dp_minus_last_action"
    run = tmp_path / "runs" / arm / f"s{RUN_SEED}"
    spec = loop.RunSpec(
        cfg=_cfg(arm),
        run_seed=RUN_SEED,
        run_dir=run,
        data_root=data_root,
        device="cpu",
        purpose="test",
        first_n_override=4,
        steps_override=1,
        checkpoint_every_override=1,
        heldout=None,
        log_every=1,
        prefetch=0,
        hash_store=False,
    )
    assert loop.train_run(spec, loop.build_stages(spec)).state == "done"
    ckpt = run / "checkpoints" / "s0_step000001.pt"
    assert registry.read_checkpoint(ckpt)["obs_variant"] == "minus_last_action"
    selection = {
        "arm": arm,
        "run_seed": RUN_SEED,
        "selected": {
            "candidate": {"file": str(ckpt), "sha256": sha256_file(ckpt), "global_step": 1},
            "rows_file": "unused.csv",
        },
    }
    mf.write_json_atomic(run / "selection.json", selection)

    real = diagnostics.heldout_states
    seen: list[str] = []

    def small(root: Path, **kw: Any) -> BCWindowDataset:
        seen.append(kw["obs_variant"])  # run_diagnostics passes the checkpoint's contract
        return real(root, start=0, n=2, every=1, **kw)

    monkeypatch.setattr(diagnostics, "heldout_states", small)
    rec = dd.run_diagnostics(run, data_root=data_root, closed_loop=False, out_root=tmp_path / "o")
    assert seen == ["minus_last_action"]
    held = rec["heldout"]
    assert held["episodes"] == 2 and held["spread_states"] == held["frames"] > 0
    assert "select" not in rec and (tmp_path / "o" / arm / f"s{RUN_SEED}.json").is_file()
