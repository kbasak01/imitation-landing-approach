"""Model specs record ``results/readme/model_specs.json`` (``ila.policies.specs``; Phase 11).

CPU only. The committed student parameter counts must equal freshly built networks (the training
loop's builder, no checkpoint); ``--check`` must re-derive the committed file byte for byte on a
machine without checkpoints (CI is simulated by hiding every checkpoint and
``configs/paths.local.yaml``), carrying the committed cross-checks over; a stale cross-check is
never carried. The real cross-checks (pinned seed-0 checkpoints, P8's PPO ``final/``) are marked
``checkpoints`` (workstation only).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ila import p8_adapter
from ila.policies import specs
from ila.policies.encoder import count_parameters
from ila.policies.registry import REGISTRY, build_net

COMMITTED: dict[str, Any] = json.loads(specs.OUTPUT.read_text(encoding="utf-8"))


def _script() -> ModuleType:
    path = specs.REPO_ROOT / "scripts" / "model_specs.py"
    spec = importlib.util.spec_from_file_location("model_specs_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_student_config_is_recorded() -> None:
    assert set(COMMITTED["students"]) == set(REGISTRY)
    for arm, rec in COMMITTED["students"].items():
        assert rec["tag"] == REGISTRY[arm].tag
        assert rec["config"]["path"] == REGISTRY[arm].config.relative_to(specs.REPO_ROOT).as_posix()
    assert COMMITTED["students"]["R1_lerobot_dp"]["built_here"] is False
    assert COMMITTED["students"]["R1_lerobot_dp"]["parameters"] is None
    assert set(COMMITTED["p8_controllers"]) == {"ppo", *specs.CONTROLLERS}


@pytest.mark.parametrize("arm", sorted(a for a in REGISTRY if REGISTRY[a].has_ila_network))
def test_committed_parameter_counts_equal_a_fresh_build(arm: str) -> None:
    net = build_net(arm, 1)  # another init seed: counts do not depend on it
    rec = COMMITTED["students"][arm]["parameters"]
    assert rec["total"] == count_parameters(net) == rec["trainable"]
    for name, child in net.named_children():
        assert rec["by_module"][name] == count_parameters(child)
    top = [k for k in rec["by_module"] if "." not in k]
    assert sum(rec["by_module"][k] for k in top) == rec["total"]


def test_shared_encoder_is_one_architecture() -> None:
    vision = {
        a: r for a, r in COMMITTED["students"].items() if r["built_here"] and r["kind"] != "state"
    }
    default = {a: r for a, r in vision.items() if r["obs_variant"] == "default"}
    counts = {r["parameters"]["by_module"]["encoder"] for r in default.values()}
    assert len(counts) == 1
    trunks = {r["parameters"]["by_module"]["encoder.trunk"] for r in vision.values()}
    assert len(trunks) == 1


def test_ppo_parameter_counts_are_the_sb3_layout() -> None:
    ppo = COMMITTED["p8_controllers"]["ppo"]
    width = ppo["policy"]["net_arch_pi"]
    assert width == ppo["policy"]["net_arch_vf"] == [512, 512]
    obs, act = ppo["observation"]["dim"], ppo["action"]["dim"]
    assert (obs, act) == (25, 3)
    mlp = obs * 512 + 512 + 512 * 512 + 512
    assert ppo["parameters"]["actor"] == mlp + 512 * act + act + act
    assert ppo["parameters"]["critic"] == mlp + 512 + 1


def _hide_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CI: no ``artifacts/runs`` checkpoints and no ``configs/paths.local.yaml``."""
    from ila.eval import arms as ev_arms

    def no_paths(*_: object) -> dict[str, Path]:
        raise FileNotFoundError("configs/paths.local.yaml (hidden by the test)")

    monkeypatch.setattr(ev_arms, "_local", lambda _recorded: tmp_path / "absent.pt")
    monkeypatch.setattr(p8_adapter, "load_paths", no_paths)


def test_check_passes_without_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _hide_checkpoints(monkeypatch, tmp_path)
    assert _script().main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "model_specs --check: OK" in out
    built = [a for a in REGISTRY if REGISTRY[a].has_ila_network]
    for what in (*built, "ppo"):
        assert f"{what}: checkpoint absent here; committed cross-check carried over" in out


def test_check_fails_on_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "model_specs.json"
    doc = json.loads(specs.OUTPUT.read_text(encoding="utf-8"))
    doc["students"]["bc"]["parameters"]["total"] += 1
    path.write_text(specs.dumps(doc), encoding="utf-8")
    result = specs.check(path, checkpoints=False)
    assert not result.ok
    assert any("DIFFERS" in line for line in result.notes)


def test_a_stale_crosscheck_is_never_carried() -> None:
    rec = COMMITTED["students"]["bc"]["checkpoint_crosscheck"]
    total = COMMITTED["students"]["bc"]["parameters"]["total"]
    sha = COMMITTED["students"]["bc"]["config"]["sha256"]
    notes: list[str] = []
    kept = specs._carry(
        "bc", None, rec, total=total, config_sha256=sha, attempted=True, notes=notes
    )
    assert kept == rec
    with pytest.raises(AssertionError, match="parameters"):
        specs._carry(
            "bc", None, rec, total=total + 1, config_sha256=sha, attempted=True, notes=notes
        )
    with pytest.raises(AssertionError, match="another config"):
        specs._carry(
            "bc", None, rec, total=total, config_sha256="0" * 64, attempted=True, notes=notes
        )


@pytest.mark.checkpoints
def test_crosschecks_run_against_the_pinned_checkpoints() -> None:
    result = specs.check(checkpoints=True)
    assert result.ok, "\n".join(result.notes)
    built = [a for a in REGISTRY if REGISTRY[a].has_ila_network]
    for what in (*built, "ppo"):
        assert f"{what}: checkpoint cross-check run, match" in result.notes
    for arm in built:
        cc = COMMITTED["students"][arm]["checkpoint_crosscheck"]
        assert cc["match"] is True and cc["weights_checked"][0] == "ema"
    assert COMMITTED["p8_controllers"]["ppo"]["checkpoint_crosscheck"]["match"] is True
