"""H1-H5 scorers (``ila.eval.hypotheses``) on synthetic rows with known verdicts (P3-D1 §6).

Every student row here is synthetic: nothing is flown and no student row of a frozen list exists
or is read. Two tests read committed files that P3-D1 §4 allows to be read: P8's committed
``ppo`` / ``gated`` rows with the frozen ``id`` SS4 keys (H2's reference construction, nothing
flown) and this project's committed ``select`` selection records (the best end-to-end student).
Times are seconds, model scale (lambda = 1/25).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ila import p8_adapter
from ila.eval import hypotheses as hy
from ila.eval import runner

N = 200
CORE = hy.CORE_SEEDS
ABL = hy.ABLATION_SEEDS
SHA = runner.render_cfg_sha256("full", None)

Row = dict[str, Any]


def _rows(
    arm: str,
    ss: str,
    seeds: Sequence[int],
    k: int | Callable[[int], int],
    *,
    td: Callable[[int, int], float] | None = None,
    ta: int | None = None,
    kk: int | None = None,
    ensemble: bool | None = None,
    latency: int = 0,
    render: str = SHA,
    ckpt: str = "c",
    n: int = N,
    tunnelled: int | None = None,
    regime: str = "id",
) -> list[Row]:
    """Rows where seed s succeeds on episodes 0..k(s)-1 and times out on the rest.

    With ``tunnelled`` set, rows carry P8's ``tunnelled`` column, True on episodes
    0..tunnelled-1 of every seed.
    """
    out = []
    for s in seeds:
        ks = k(s) if callable(k) else k
        for i in range(n):
            ok = i < ks
            t = td(s, i) if td is not None else (1.5 if ok else math.nan)
            extra = {} if tunnelled is None else {"tunnelled": str(i < tunnelled)}
            out.append(
                {
                    **extra,
                    "regime": regime,
                    "ss": ss,
                    "index": str(i),
                    "episode_seed": str(5000 + i),
                    "outcome": "success" if ok else ("timeout" if math.isnan(t) else "bounce"),
                    "time_to_touchdown_s": "" if math.isnan(t) else repr(t),
                    "arm": arm,
                    "student_seed": str(s),
                    "Ta": "" if ta is None else str(ta),
                    "K": "" if kk is None else str(kk),
                    "ensemble": "" if ensemble is None else str(ensemble),
                    "latency_steps": str(latency),
                    "render_cfg_sha256": render,
                    "ckpt_sha256": f"{ckpt}{s}",
                }
            )
    return out


def _m(arm: str, ss: str, k: int, **kw: Any) -> hy.ArmMatrix:
    return hy.student_matrix(_rows(arm, ss, CORE, k, **kw), arm=arm, cell=("id", ss))


def _p8(method: str, ss: str, k: int, td: Callable[[int], float] | None = None) -> list[Row]:
    return [
        {
            "regime": "id",
            "ss": ss,
            "index": str(i),
            "episode_seed": str(5000 + i),
            "method": method,
            "run_seed": "0",
            "outcome": "success" if i < k else "hard_landing",
            "time_to_touchdown_s": repr(td(i) if td else 1.5),
        }
        for i in range(N)
    ]


FRAMES = {"bc_dagger": [130_000, 131_000, 129_500, 132_000, 130_500], "bc_matched": [123_000] * 5}


def _h1(
    dagger5: int,
    matched5: int,
    dagger6: int = 150,
    matched6: int = 150,
    dart5: int = 150,
    bc5: int = 150,
    dart6: int = 150,
    bc6: int = 150,
) -> tuple[hy.HypothesisResult, ...]:
    return hy.score_h1(
        dagger_ss5=_m("bc_dagger", "SS5", dagger5),
        matched_ss5=_m("bc_matched", "SS5", matched5),
        dagger_ss6=_m("bc_dagger", "SS6", dagger6),
        matched_ss6=_m("bc_matched", "SS6", matched6),
        dart_ss5=_m("bc_dart", "SS5", dart5),
        bc_ss5=_m("bc", "SS5", bc5),
        dart_ss6=_m("bc_dart", "SS6", dart6),
        bc_ss6=_m("bc", "SS6", bc6),
        labelled_frames=FRAMES,
    )


def _beside(r: hy.HypothesisResult | hy.UnscoredContrast) -> dict[str, str]:
    return dict(r.beside)


# --------------------------------------------------------------------------- the rule


@pytest.mark.parametrize(
    ("est", "lo", "hi", "verdict", "fragment"),
    [
        (0.08, 0.02, 0.14, "supported", "supported"),
        (0.05, 0.01, 0.09, "supported", "supported"),
        (0.05 - 1e-12, 0.01, 0.09, "supported", "supported"),
        (0.04, 0.01, 0.07, "inconclusive", "below +5 points"),
        (0.04, -0.01, 0.09, "not supported", "CI includes 0"),
        (0.08, 0.0, 0.16, "not supported", "CI includes 0"),
        (-0.08, -0.14, -0.02, "not supported", "opposite direction"),
        (math.nan, math.nan, math.nan, "not supported", "undefined"),
    ],
)
def test_magnitude_rule(est: float, lo: float, hi: float, verdict: str, fragment: str) -> None:
    v, note = hy.magnitude_rule(est, lo, hi, 0.05, unit="+5 points")
    assert v == verdict and fragment in note


def test_flip_floor_is_p3d1_verbatim() -> None:
    text = (p8_adapter.REPO_ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
    assert hy.FLIP_FLOOR in re.sub(r"\s+", " ", text)
    assert hy.FLIP_FLOOR.startswith("6.6 % of `id` SS6")


# --------------------------------------------------------------------------- H1


def test_h1_supported_with_everything_beside() -> None:
    h1, ss6, d5, d6 = _h1(190, 170)
    assert (h1.hypothesis, h1.cell, h1.verdict, h1.verdict_note) == (
        "H1",
        "id SS5",
        "supported",
        "supported",
    )
    assert h1.estimate == pytest.approx(0.10) and h1.ci_lo > 0
    assert h1.n_seeds == (("bc_dagger", 5), ("bc_matched", 5)) and h1.n_episodes == N
    b = _beside(h1)
    assert "post hoc, not scored: paired_iqm_difference" in b
    assert b["labelled frames"].startswith("bc_dagger: 653,000 (per seed 130,000")
    assert "SS6 flip floor" not in b
    assert _beside(ss6)["SS6 flip floor"] == hy.FLIP_FLOOR
    assert _beside(d6)["SS6 flip floor"] == hy.FLIP_FLOOR
    assert hy.NO_MULTIPLICITY in h1.caveats
    assert any("conservative" in c for c in h1.caveats)
    assert (d5.hypothesis, d5.contrast) == ("H1-DART-SS5", "success(bc_dart) - success(bc)")
    assert any("N = 1 000" in c for c in d5.caveats)


def test_h1_exactly_five_points_is_supported() -> None:
    h1 = _h1(180, 170)[0]
    assert h1.estimate == pytest.approx(0.05, abs=1e-12)
    assert h1.verdict == "supported"


def test_h1_inconclusive() -> None:
    h1 = _h1(178, 170)[0]
    assert h1.ci_lo > 0 and h1.verdict == "inconclusive"
    assert h1.verdict_note.startswith("inconclusive")


def test_h1_not_supported_ci_includes_zero() -> None:
    h1 = _h1(171, 170)[0]
    assert h1.ci_lo <= 0 <= h1.ci_hi
    assert (h1.verdict, h1.verdict_note) == ("not supported", "not supported (CI includes 0)")


def test_h1_opposite_direction() -> None:
    h1, ss6, _, _ = _h1(150, 180, dagger6=100, matched6=150)
    assert h1.ci_hi < 0 and h1.verdict == "not supported"
    assert "opposite direction" in h1.verdict_note
    assert "opposite direction" in ss6.verdict_note


def test_h1_ceiling_limited_with_ss6_beside() -> None:
    h1, ss6, _, _ = _h1(197, 196, dagger6=170, matched6=150)
    assert h1.verdict == "not supported" and h1.ceiling_limited
    assert h1.verdict_note == (
        "not supported (ceiling-limited: the largest attainable gain is below +5 points)"
    )
    beside = _beside(h1)["H1-SS6 beside it"]
    assert beside.endswith(": supported") and beside.startswith("+10.0 points")
    assert ss6.verdict == "supported" and not ss6.ceiling_limited
    # 196/200 per seed -> 980/1000 -> largest attainable gain 2.0 points (computed).
    assert any("largest attainable gain is 2.0 points" in c for c in h1.caveats)


def test_h1_inconclusive_under_the_ceiling_keeps_its_verdict() -> None:
    """S1: an inconclusive verdict under the ceiling keeps its verdict.

    The computed ceiling caveat and the SS6 secondary are attached (P8-D1 item 3).
    """
    h1, ss6, _, _ = _h1(200, 192, dagger6=170, matched6=150)
    assert h1.ci_lo > 0 and h1.estimate == pytest.approx(0.04)
    assert h1.verdict == "inconclusive" and h1.ceiling_limited
    assert h1.verdict_note.startswith("inconclusive (CI excludes 0")
    cav = [c for c in h1.caveats if c.startswith("Ceiling-limited")]
    assert cav == [
        "Ceiling-limited: bc_matched is at 96.0 % (960/1000) in id SS5, so the largest "
        "attainable gain is 4.0 points, below +5; 'supported' was unattainable. The rule's "
        "verdict is kept (P3-D1 §6 ceilings; P8-D1 item 3)."
    ]
    beside = _beside(h1)["H1-SS6 beside it"]
    assert beside.startswith(f"{100.0 * ss6.estimate:+.1f} points") and beside.endswith(
        ": supported"
    )
    # No ceiling: no caveat and nothing attached.
    h1b = _h1(178, 170)[0]
    assert not h1b.ceiling_limited and "H1-SS6 beside it" not in _beside(h1b)
    assert not any(c.startswith("Ceiling-limited") for c in h1b.caveats)


def test_h1_tunnel_removed_difference_beside() -> None:
    """M10: success minus P8-tunnelled successes, descriptive, computed from the rows."""
    a = hy.student_matrix(
        _rows("bc_dagger", "SS6", CORE, 190, tunnelled=3), arm="bc_dagger", cell=("id", "SS6")
    )
    b = hy.student_matrix(
        _rows("bc_matched", "SS6", CORE, 170, tunnelled=1), arm="bc_matched", cell=("id", "SS6")
    )
    assert a.tunnelled is not None and int(a.tunnelled.sum()) == 15
    res = hy.score_h1(
        dagger_ss5=_m("bc_dagger", "SS5", 150),
        matched_ss5=_m("bc_matched", "SS5", 150),
        dagger_ss6=a,
        matched_ss6=b,
        dart_ss5=_m("bc_dart", "SS5", 150),
        bc_ss5=_m("bc", "SS5", 150),
        dart_ss6=_m("bc_dart", "SS6", 150),
        bc_ss6=_m("bc", "SS6", 150),
        labelled_frames=FRAMES,
    )[1]
    text = _beside(res)["descriptive, not scored: success minus tunnelled successes"]
    # (950 - 15) - (850 - 5) = 90 of 1000 -> +9.0 points; the verdict is unchanged.
    assert text.startswith(
        "bc_dagger 935/1000 (tunnelled successes removed: 15); bc_matched 845/1000 (tunnelled "
        "successes removed: 5); difference +9.0 points"
    )
    assert res.estimate == pytest.approx(0.10)
    # Rows without the column print nothing.
    assert "descriptive, not scored: success minus tunnelled successes" not in _beside(
        _h1(190, 170)[0]
    )


def test_h1_ceiling_does_not_hide_an_opposite_effect() -> None:
    h1 = _h1(180, 199)[0]
    assert "opposite direction" in h1.verdict_note
    assert any("ceiling-limited" in c.lower() for c in h1.caveats)


def test_h1_dart_secondaries_ceiling_uses_bc() -> None:
    _, _, d5, d6 = _h1(190, 170, dart5=197, bc5=197, dart6=180, bc6=150)
    assert "ceiling-limited" in d5.verdict_note
    assert _beside(d5)["H1-DART-SS6 beside it"].endswith("supported")
    assert d6.verdict == "supported"


def test_h1_refuses_wrong_cells_and_missing_frames() -> None:
    a, b = _m("bc_dagger", "SS5", 190), _m("bc_matched", "SS5", 170)
    with pytest.raises(hy.RowError):
        hy.score_h1(
            dagger_ss5=a,
            matched_ss5=b,
            dagger_ss6=a,  # SS5 rows offered as SS6
            matched_ss6=b,
            dart_ss5=_m("bc_dart", "SS5", 150),
            bc_ss5=_m("bc", "SS5", 150),
            dart_ss6=_m("bc_dart", "SS6", 150),
            bc_ss6=_m("bc", "SS6", 150),
            labelled_frames=FRAMES,
        )
    with pytest.raises(hy.RowError):
        hy.score_h1(
            dagger_ss5=a,
            matched_ss5=b,
            dagger_ss6=_m("bc_dagger", "SS6", 150),
            matched_ss6=_m("bc_matched", "SS6", 150),
            dart_ss5=_m("bc_dart", "SS5", 150),
            bc_ss5=_m("bc", "SS5", 150),
            dart_ss6=_m("bc_dart", "SS6", 150),
            bc_ss6=_m("bc", "SS6", 150),
            labelled_frames={"bc_dagger": [1]},
        )


def test_dart_vs_matched_is_unscored_two_sided() -> None:
    u = hy.dart_vs_matched(_m("bc_dart", "SS5", 190), _m("bc_matched", "SS5", 170))
    assert u.reading.startswith("separates: bc_dart above bc_matched")
    assert any("Unscored" in c for c in u.caveats)


# --------------------------------------------------------------------------- row guards


def test_rows_no_seed_dropped_and_tags_never_mixed() -> None:
    rows = _rows("bc", "SS5", (0, 1, 2, 3), 150)
    with pytest.raises(hy.RowError, match="no seed is dropped"):
        hy.student_matrix(rows, arm="bc", cell=("id", "SS5"))
    rows = _rows("bc", "SS5", CORE, 150) + _rows("bc", "SS5", CORE, 10, latency=3)
    assert hy.student_matrix(rows, arm="bc", cell=("id", "SS5")).rate == 0.75  # latency rows out
    shifted = runner.render_cfg_sha256("full", "some_variant")
    rows = _rows("bc", "SS5", CORE, 150) + _rows("bc", "SS5", (0,), 10, render=shifted)
    with pytest.raises(hy.RowError, match="non-core render"):
        hy.student_matrix(rows, arm="bc", cell=("id", "SS5"))
    rows = _rows("bc", "SS5", CORE, 150)
    with pytest.raises(hy.RowError, match="twice"):
        hy.student_matrix(rows + rows[:1], arm="bc", cell=("id", "SS5"))
    short = _rows("bc", "SS5", (0, 1, 2, 3), 150) + _rows("bc", "SS5", (4,), 150, n=N - 1)
    with pytest.raises(hy.RowError, match="episodes differ"):
        hy.student_matrix(short, arm="bc", cell=("id", "SS5"))
    other = _rows("bc_matched", "SS5", CORE, 150, n=N - 1)
    with pytest.raises(hy.RowError, match="identical episodes"):
        hy._same_episodes(
            _m("bc", "SS5", 150), hy.student_matrix(other, arm="bc_matched", cell=("id", "SS5"))
        )


# --------------------------------------------------------------------------- H2


def _listed(n: int = N) -> list[p8_adapter.ListedEpisode]:
    return [
        p8_adapter.ListedEpisode(
            regime="id",
            ss="SS4",
            index=i,
            pad="aft",
            vessel="frigate",
            heading_deg=135.0,
            speed_kn=6.0,
            realization_seed=i % 7,
            episode_seed=5000 + i,
            t0_model_s=10.0,
            init_x_m=0.0,
            init_y_m=0.0,
            init_z_m=1.7,
            in_training_distribution=False,
        )
        for i in range(n)
    ]


PPO_T, GATED_T = 1.5, 3.4


@pytest.fixture(scope="module")
def ref() -> hy.H2Reference:
    return hy.build_h2_reference(
        _listed(),
        _p8("ppo", "SS4", N, lambda _i: PPO_T),
        _p8("gated", "SS4", N, lambda _i: GATED_T),
    )


def _mix(arm: str, td: Callable[[int, int], float]) -> hy.ArmMatrix:
    ta, kk, ens = (8, 10, False) if arm == "dp_mix" else (None, None, None)
    rows = _rows(arm, "SS4", ABL, N, td=td, ta=ta, kk=kk, ensemble=ens)
    return hy.student_matrix(
        rows, arm=arm, cell=("id", "SS4"), seeds=ABL, ta=ta, k=kk, ensemble=ens
    )


def _ref_t(ref: hy.H2Reference) -> Callable[[int, int], float]:
    return lambda _s, i: float(ref.touchdown_s[i])


def test_h2_reference_construction(ref: hy.H2Reference) -> None:
    assert Counter(ref.assigned) == {"ppo": 100, "gated": 100}
    t = dict(zip(ref.assigned, ref.touchdown_s, strict=False))
    assert t == {"ppo": PPO_T, "gated": GATED_T}
    with pytest.raises(hy.RowError, match="no committed"):
        hy.build_h2_reference(_listed(), _p8("ppo", "SS4", N)[:-1], _p8("gated", "SS4", N)[:-1])


def test_h2_supported(ref: hy.H2Reference) -> None:
    r = hy.score_h2(_mix("bc_mix", lambda _s, _i: 2.45), _mix("dp_mix", _ref_t(ref)), ref)
    assert r.estimate == pytest.approx(0.95) and (r.ci_lo, r.ci_hi) == pytest.approx((0.95, 0.95))
    assert (r.hypothesis, r.cell, r.verdict) == ("H2", "id SS4", "supported")
    assert r.n_seeds == (("bc_mix", 3), ("dp_mix", 3))
    b = _beside(r)
    assert b["fraction without touchdown"] == "bc_mix 0.000; dp_mix 0.000; reference 0.000"
    assert b["undefined replicates (empty pool, excluded)"] == "0 of 10000"
    assert "100 ppo" in b["reference"] and "100 gated" in b["reference"]
    assert any("not shown to come from multimodality" in c for c in r.caveats)
    assert any("select W1" in c for c in r.caveats)
    assert any("1 : 2 by frames" in c for c in r.caveats)


def test_h2_inconclusive(ref: hy.H2Reference) -> None:
    rt = _ref_t(ref)
    r = hy.score_h2(_mix("bc_mix", lambda s, i: rt(s, i) + 0.02), _mix("dp_mix", rt), ref)
    assert 0 < r.estimate < 0.033 and r.ci_lo > 0
    assert r.verdict == "inconclusive" and "0.033 s" in r.verdict_note


def test_h2_not_supported_and_opposite(ref: hy.H2Reference) -> None:
    rt = _ref_t(ref)
    same = _mix("bc_mix", lambda s, i: rt(s, i) + 0.1)
    r = hy.score_h2(same, _mix("dp_mix", lambda s, i: rt(s, i) + 0.1), ref)
    assert (r.verdict, r.verdict_note) == ("not supported", "not supported (CI includes 0)")
    r = hy.score_h2(_mix("bc_mix", rt), _mix("dp_mix", lambda s, i: rt(s, i) + 0.5), ref)
    assert r.verdict == "not supported" and "opposite direction" in r.verdict_note


def test_h2_empty_pool_replicates_counted(ref: hy.H2Reference) -> None:
    rt = _ref_t(ref)
    bc = _mix("bc_mix", lambda _s, i: 2.45 if i == 0 else math.nan)  # one touchdown per seed
    r = hy.score_h2(bc, _mix("dp_mix", rt), ref)
    undefined = int(_beside(r)["undefined replicates (empty pool, excluded)"].split()[0])
    assert 0.32 < undefined / 10_000 < 0.42  # (199/200)^200 = 0.367 expected
    assert _beside(r)["fraction without touchdown"].startswith("bc_mix 0.995")
    assert math.isfinite(r.estimate)


def test_h2_arm_without_touchdown_is_not_supported(ref: hy.H2Reference) -> None:
    r = hy.score_h2(_mix("bc_mix", lambda _s, _i: math.nan), _mix("dp_mix", _ref_t(ref)), ref)
    assert r.verdict == "not supported" and "D undefined" in r.verdict_note


def test_h2_needs_three_seeds(ref: hy.H2Reference) -> None:
    rows = _rows("bc_mix", "SS4", CORE, N)
    with pytest.raises(hy.RowError):
        hy.student_matrix(rows, arm="bc_mix", cell=("id", "SS4"), seeds=ABL)
    with pytest.raises(hy.RowError):
        hy.score_h2(
            hy.student_matrix(rows, arm="bc_mix", cell=("id", "SS4")),
            _mix("dp_mix", _ref_t(ref)),
            ref,
        )


def test_h2_reference_from_committed_p8_rows() -> None:
    listed = [e for e in p8_adapter.read_list(p8_adapter.list_path("id")) if e.ss == "SS4"]
    reference = hy.build_h2_reference(
        listed,
        hy.p8_committed_rows("ppo", ("id", "SS4")),
        hy.p8_committed_rows("gated", ("id", "SS4")),
    )
    assert len(reference.episodes) == 200
    assert Counter(reference.assigned) == {"ppo": 100, "gated": 100}
    assert reference.assigned == tuple(
        p for _, p in sorted(zip((e.index for e in listed), _assign(listed), strict=True))
    )


def _assign(listed: list[p8_adapter.ListedEpisode]) -> list[str]:
    from ila.data import mix

    return mix.assign_rows(20261023, listed)


# --------------------------------------------------------------------------- H3


def _dp(ta: int, k: int, ckpt: str = "c", ss: str = "SS6", regime: str = "id") -> hy.ArmMatrix:
    rows = _rows("dp", ss, CORE, k, ta=ta, kk=10, ensemble=False, ckpt=ckpt, regime=regime)
    rows += _rows(
        "dp", ss, CORE, 0, ta=ta, kk=10, ensemble=True, ckpt=ckpt, regime=regime
    )  # filtered out
    return hy.student_matrix(rows, arm="dp", cell=(regime, ss), ta=ta, k=10, ensemble=False)


def test_h3_supported_with_flip_floor() -> None:
    r = hy.score_h3(_dp(1, 180), _dp(8, 160))
    assert (r.hypothesis, r.cell, r.verdict) == ("H3", "id SS6", "supported")
    assert r.estimate == pytest.approx(0.10) and r.n_seeds == (("dp", 5),)
    assert _beside(r)["SS6 flip floor"] == hy.FLIP_FLOOR
    assert any("selected at Ta = 8" in c for c in r.caveats)
    assert any("switch modes" in c for c in r.caveats)
    assert any("dp_minus_last_action" in c for c in r.caveats)
    assert any(c.startswith(hy.H3_DECK_MOTION) for c in r.caveats)  # P6-D5, always stated


def test_h3_deck_motion_caveat_is_computed() -> None:
    """B1: P6-D5's caveat with the per-sea-state gains and the static-deck rate, from rows."""
    r = hy.score_h3(
        _dp(1, 180),
        _dp(8, 160),
        other_ss=((_dp(1, 190, ss="SS5"), _dp(8, 150, ss="SS5")), (_dp(1, 195, ss="SS4"),
                  _dp(8, 145, ss="SS4"))),
        static_ta8=_dp(8, 82, ss="static", regime="static"),
    )  # fmt: skip
    assert r.verdict == "supported" and r.estimate == pytest.approx(0.10)
    assert (
        "P6-D5: a Ta contrast cannot attribute the loss to deck motion. On the frozen lists the "
        "gain is +25.0 / +20.0 / +10.0 points at SS4 / SS5 / SS6 (Ta = 1 minus Ta = 8, mean "
        "over seeds x episodes, same checkpoints; comparing the gains across sea states is an "
        "unpaired reading), and `dp` at Ta = 8 succeeds on 41.0 % (410/1000) of the "
        "static-deck episodes."
    ) in r.caveats
    with pytest.raises(hy.RowError, match="same checkpoints"):
        hy.score_h3(
            _dp(1, 180), _dp(8, 160), static_ta8=_dp(8, 82, "x", ss="static", regime="static")
        )
    with pytest.raises(hy.RowError, match="other than SS6"):
        hy.score_h3(_dp(1, 180), _dp(8, 160), other_ss=((_dp(1, 180), _dp(8, 160)),))


def test_h3_inconclusive_not_supported_opposite_ceiling() -> None:
    assert hy.score_h3(_dp(1, 168), _dp(8, 160)).verdict == "inconclusive"
    r = hy.score_h3(_dp(1, 161), _dp(8, 160))
    assert r.verdict_note == "not supported (CI includes 0)"
    r = hy.score_h3(_dp(1, 120), _dp(8, 160))
    assert "opposite direction" in r.verdict_note
    r = hy.score_h3(_dp(1, 198), _dp(8, 197))
    assert r.verdict_note == (
        "not supported (ceiling-limited: the largest attainable gain is below +5 points)"
    )
    assert r.ceiling_limited
    assert any("dp at Ta = 8 is at 98.5 % (985/1000)" in c for c in r.caveats)
    assert any("largest attainable gain is 1.5 points" in c for c in r.caveats)


def test_h3_same_checkpoints_and_settings_only() -> None:
    with pytest.raises(hy.RowError, match="same checkpoints"):
        hy.score_h3(_dp(1, 180), _dp(8, 160, ckpt="other"))
    with pytest.raises(hy.RowError, match="Ta=1"):
        hy.score_h3(_dp(2, 180), _dp(8, 160))


# --------------------------------------------------------------------------- H4


def _h4(teacher: int, bc_state: int, bc: int) -> hy.HypothesisResult:
    t = hy.p8_matrix(_p8("ppo", "SS6", teacher), method="ppo", cell=("id", "SS6"))
    return hy.score_h4(_m("bc_state", "SS6", bc_state, render=""), _m("bc", "SS6", bc), t)


def test_h4_supported() -> None:
    r = _h4(197, 190, 120)
    assert r.estimate == pytest.approx(0.35 - 0.035) and r.verdict == "supported"
    b = _beside(r)
    assert b["perception gap P"].startswith("+35.0 points")
    assert b["imitation gap I"].startswith("+3.5 points")
    # Teacher at 197/200: I cannot fall below -0.015 (stated beside I and as a caveat).
    assert b["imitation gap I"].endswith("ceiling-limited: I >= -1.5 points (teacher at 98.5 %)")
    assert b["SS6 flip floor"] == hy.FLIP_FLOOR
    assert r.n_seeds == (("bc_state", 5), ("bc", 5), ("teacher ppo s0", 1))
    for fragment in ("time_fraction", "fp32", "teacher seed 0", "seed-dependent", "-0.015"):
        assert any(fragment in c for c in r.caveats), fragment
    # ceiling_limited is P3-D1 §6's reference-arm ceiling only; H4's I-floor is a note.
    assert not r.ceiling_limited
    assert len(r.notes) == 1 and "I >= -1.5 points" in r.notes[0]
    assert "ceiling_limited is False" in r.notes[0]


def test_h4_imitation_gap_larger() -> None:
    r = _h4(197, 120, 110)
    assert r.ci_hi < 0 and r.verdict == "not supported"
    assert r.verdict_note == "not supported (imitation gap larger: CI wholly below 0)"


def test_h4_not_supported_ci_includes_zero() -> None:
    r = _h4(197, 160, 123)
    assert r.estimate == pytest.approx(0.0, abs=1e-12)
    assert r.verdict_note == "not supported (CI includes 0)"


def test_h4_p_prime_and_modular_vs_best() -> None:
    clock = hy.student_matrix(
        _rows("bc_plus_clock", "SS6", ABL, 150), arm="bc_plus_clock", cell=("id", "SS6"), seeds=ABL
    )
    pp = hy.h4_p_prime(_m("bc_state", "SS6", 190, render=""), clock)
    assert pp.n_seeds == (("bc_state", 5), ("bc_plus_clock", 3))
    assert pp.reading.startswith("separates: bc_state above bc_plus_clock")
    assert _beside(pp)["SS6 flip floor"] == hy.FLIP_FLOOR
    mv = hy.modular_vs_best(_m("modular", "SS6", 150), _m("bc_dart", "SS6", 152), "reason x")
    assert mv.reading == "does not separate (CI includes 0)"
    assert _beside(mv)["best end-to-end"] == "bc_dart: reason x"


# --------------------------------------------------------------------------- best end-to-end


def _score(arm: str, succ: Sequence[int]) -> hy.SelectScore:
    return hy.SelectScore(arm, CORE, tuple(succ), (150,) * 5, ((Path("x"),),) * 5)


def test_best_end_to_end_rule() -> None:
    base = {a: _score(a, [140] * 5) for a in hy.END_TO_END_ARMS}
    unique = dict(base, bc_dagger=_score("bc_dagger", [141, 140, 140, 140, 140]))
    assert hy.best_end_to_end(unique, p95=lambda _s: 1 / 0)[0] == "bc_dagger"
    tied = dict(base, dp=_score("dp", [141, 140, 140, 140, 140]))
    tied["bc_dart"] = _score("bc_dart", [140, 140, 140, 140, 141])
    speeds = {"dp": 0.40, "bc_dart": 0.38}
    arm, reason = hy.best_end_to_end(tied, p95=lambda s: speeds[s.arm])
    assert arm == "bc_dart" and "lower pooled select p95" in reason
    speeds = {"dp": 0.38, "bc_dart": 0.38}
    arm, reason = hy.best_end_to_end(tied, p95=lambda s: speeds[s.arm])
    assert arm == "bc_dart" and "first in P3-D1 §6 order" in reason


def test_best_end_to_end_from_committed_select_scores() -> None:
    import csv

    scores = hy.load_select_scores()
    root = p8_adapter.REPO_ROOT / "results"
    summary: dict[str, int] = {}
    for phase in ("phase4", "phase5", "phase6"):
        with (root / phase / "runs_summary.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["purpose"] == "train" and r["arm"] in hy.END_TO_END_ARMS:
                    summary[r["arm"]] = summary.get(r["arm"], 0) + int(r["select_success"])
    assert {a: sum(s.success) for a, s in scores.items()} == summary
    assert all(s.n == (150,) * 5 for s in scores.values())
    arm, reason = hy.best_end_to_end(scores)
    assert arm in hy.END_TO_END_ARMS and reason
    assert np.isfinite(float(scores[arm].mean_success))


# --------------------------------------------------------------------------- H5


def _lat(
    arm: str, ss: str, k: int, d: int, *, ckpt: str = "c", n: int = N, sync_k: int | None = None
) -> tuple[hy.ArmMatrix, hy.ArmMatrix]:
    """(latency arm at d, synchronous) of one arm: seed s succeeds on episodes 0..k-1."""
    inf: dict[str, Any] = {"ta": 8, "kk": 10, "ensemble": False} if arm == "dp" else {}
    sel: dict[str, Any] = {"ta": 8, "k": 10, "ensemble": False} if arm == "dp" else {}
    rows = _rows(arm, ss, CORE, k, latency=d, ckpt=ckpt, n=n, **inf)
    rows += _rows(arm, ss, CORE, 150 if sync_k is None else sync_k, n=n, **inf)
    cell = ("id", ss)
    return (
        hy.student_matrix(rows, arm=arm, cell=cell, latency_steps=d, **sel),
        hy.student_matrix(rows, arm=arm, cell=cell, **sel),
    )


@pytest.mark.parametrize(
    ("est", "lo", "hi", "verdict", "fragment"),
    [
        (-0.01, -0.025, 0.005, "supported", "CI lower bound >= -3 points)"),
        (0.04, 0.01, 0.07, "supported", "wholly above 0, which P3-D1 §6 reports as non-inferior"),
        (-0.01, -0.02, -0.005, "supported", "wholly below 0, so the delay costs success"),
        (-0.01, -0.03, 0.01, "supported", "non-inferior"),  # the bound itself
        (-0.01, -0.03 - 5e-10, 0.01, "supported", "non-inferior"),  # float rounding only
        (-0.02, -0.0301, 0.0, "not supported", "below -3 points"),
        (-0.10, -0.14, -0.06, "not supported", "below -3 points"),  # no opposite-direction note
        (math.nan, -0.01, 0.01, "not supported", "undefined"),
    ],
)
def test_noninferiority_rule(est: float, lo: float, hi: float, verdict: str, fragment: str) -> None:
    v, note = hy.noninferiority_rule(est, lo, hi)
    assert v == verdict and fragment in note
    assert "opposite direction" not in note  # P3-D1 §6: no two-sided reversal for H5(b)


def test_h5b_supported_ci_includes_zero_with_everything_beside() -> None:
    lat, sync = _lat("dp", "SS5", 149, 3)
    static = hy.student_matrix(
        _rows("dp", "static", CORE, 82, ta=8, kk=10, ensemble=False, regime="static"),
        arm="dp", cell=("static", "static"), ta=8, k=10, ensemble=False,
    )  # fmt: skip
    ss6 = hy.student_matrix(
        _rows("dp", "SS6", CORE, 70, ta=8, kk=10, ensemble=False),
        arm="dp", cell=("id", "SS6"), ta=8, k=10, ensemble=False,
    )  # fmt: skip
    rec = {"d": 3, "provider": "ort_cuda_fp32", "row": "dp_e2e", "p99_ms": 80.2, "fallback": False}
    r = hy.score_h5b(lat, sync, delay_steps=3, delay_record=rec, sync_ss6=ss6, sync_static=static)
    assert (r.hypothesis, r.cell, r.verdict) == ("H5(b)", "id SS5", "supported")
    assert r.verdict_note == "supported (non-inferior: CI lower bound >= -3 points)"
    assert r.estimate == pytest.approx(-0.005) and r.ci_lo >= -0.03 and r.ci_lo < 0 <= r.ci_hi
    assert r.magnitude == hy.H5B_MARGIN == -0.03 and r.n_seeds == (("dp", 5),)
    assert r.n_episodes == N and not r.ceiling_limited
    b = _beside(r)
    assert b["delay"].startswith("d = 3 control steps (100.0 ms model scale, 500 ms full scale)")
    assert "provider ort_cuda_fp32" in b["delay"] and "p99 80.2 ms" in b["delay"]
    assert b["success (mean over seeds x episodes)"] == (
        "latency arm 74.5 % (745/1000); synchronous 75.0 % (750/1000)"
    )
    assert b["outcome classes, seed-episodes"] == (
        "latency arm: crash 0, off_pad 0, hard_landing 0, bounce 0, success 745, timeout 255 "
        "(of 1000); synchronous: crash 0, off_pad 0, hard_landing 0, bounce 0, success 750, "
        "timeout 250 (of 1000)"
    )
    assert "SS6 flip floor" not in b  # SS5
    assert any("No two-sided reading for H5(b)" in c for c in r.caveats)
    assert any("seed indices jointly" in c for c in r.caveats)
    assert any("PyBullet does not advance" in c for c in r.caveats)
    assert any("no embedded target" in c for c in r.caveats)
    assert (
        "Synchronous core dp at Ta = 8 succeeds on 75.0 % (750/1000) at id SS5; 35.0 % (350/1000) "
        "at id SS6; 41.0 % (410/1000) at static (mean over seeds x episodes). H5(b) compares this "
        "dp, distilled from P8 ppo s0, with itself, so a low baseline"
    ) in " ".join(r.caveats)
    assert r.caveats[-1] == hy.NO_MULTIPLICITY


def test_h5b_supported_ci_wholly_above_zero_is_non_inferior_not_a_gain() -> None:
    lat, sync = _lat("dp", "SS5", 160, 2)
    r = hy.score_h5b(lat, sync, delay_steps=2)
    assert r.verdict == "supported" and r.ci_lo > 0.0
    assert "reports as non-inferior, not as a gain" in r.verdict_note


def test_h5b_supported_ci_wholly_below_zero_within_the_margin() -> None:
    lat, sync = _lat("dp", "SS5", 1480, 2, n=2000, sync_k=1500)
    r = hy.score_h5b(lat, sync, delay_steps=2)
    assert r.estimate == pytest.approx(-0.01) and r.ci_hi < 0.0 and r.ci_lo >= -0.03
    assert r.verdict == "supported" and "wholly below 0" in r.verdict_note


def test_h5b_not_supported() -> None:
    lat, sync = _lat("dp", "SS5", 130, 4)
    r = hy.score_h5b(lat, sync, delay_steps=4)
    assert r.verdict == "not supported" and r.ci_lo < -0.03
    assert r.verdict_note == "not supported (CI lower bound below -3 points)"
    assert r.estimate == pytest.approx(-0.10)


def test_h5b_same_checkpoint_same_settings_only() -> None:
    lat, sync = _lat("dp", "SS5", 149, 3)
    with pytest.raises(hy.RowError, match="same checkpoints"):
        hy.score_h5b(_lat("dp", "SS5", 149, 3, ckpt="other")[0], sync, delay_steps=3)
    with pytest.raises(hy.RowError, match="expected 2 / 0"):
        hy.score_h5b(lat, sync, delay_steps=2)
    with pytest.raises(hy.RowError, match="delay.json d"):
        hy.score_h5b(lat, sync, delay_steps=3, delay_record={"d": 4})
    with pytest.raises(hy.RowError, match="id SS5"):
        hy.score_h5b(*_lat("dp", "SS6", 149, 3), delay_steps=3)
    with pytest.raises(hy.RowError, match="H5\\(b\\) needs dp"):
        hy.score_h5b(*_lat("bc_dagger", "SS5", 149, 3), delay_steps=3)
    ens = hy.student_matrix(
        _rows("dp", "SS5", CORE, 150, ta=8, kk=10, ensemble=True),
        arm="dp", cell=("id", "SS5"), ta=8, k=10, ensemble=True,
    )  # fmt: skip
    with pytest.raises(hy.RowError, match="no ensembling"):
        hy.score_h5b(lat, ens, delay_steps=3)
    # A latency row never enters the synchronous matrix, and vice versa.
    assert sync.latency_steps == 0 and lat.latency_steps == 3
    assert sync.rate == 0.75 and lat.rate == 0.745


def test_latency_context_is_unscored_two_sided() -> None:
    lat, sync = _lat("bc_dagger", "SS6", 120, 2)
    u = hy.latency_context(lat, sync, delay_steps=2)
    assert u.name == "H5(b) context: bc_dagger latency arm - synchronous (SS6, unscored)"
    assert u.reading == "separates: bc_dagger d=2 below bc_dagger synchronous (CI wholly below 0)"
    assert u.n_seeds == (("bc_dagger d=2", 5), ("bc_dagger synchronous", 5))
    assert _beside(u)["SS6 flip floor"] == hy.FLIP_FLOOR
    assert any(c.startswith("Unscored context") for c in u.caveats)
    assert any("success 600" in c and "success 750" in c for c in u.caveats)
    u4 = hy.latency_context(*_lat("dp", "SS4", 150, 3), delay_steps=3)
    assert u4.reading == "does not separate (CI includes 0)" and "SS6 flip floor" not in dict(
        u4.beside
    )
    with pytest.raises(hy.RowError, match="is H5\\(b\\), scored"):
        hy.latency_context(*_lat("dp", "SS5", 150, 3), delay_steps=3)


def _write_h5a(root: Path, verdict: str, *, csv_verdict: str | None = None) -> Path:
    import eval_support as es

    return es.write_latency_results(root, verdict=verdict, csv_verdict=csv_verdict)


def test_h5a_is_read_as_written(tmp_path: Path) -> None:
    rec = hy.read_h5a(tmp_path / "absent")
    assert rec["status"].startswith("not scored: results/latency/h5a.csv")
    rec = hy.read_h5a(_write_h5a(tmp_path / "a", "supported"))
    assert (rec["status"], rec["verdict"], rec["verdict_note"]) == (
        "scored",
        "supported",
        "supported",
    )
    assert rec["estimate"] == 21.5  # the one scored provider passing both parities, timed
    assert rec["magnitude"] == hy.H5A_THRESHOLD_MS == 33.3
    assert rec["units"].startswith("ms wall clock") and "RTX A4000" in rec["units"]
    assert [p["provider"] for p in rec["providers"]] == [
        "ort_cuda_fp32",
        "ort_trt_fp16",
        "ort_trt_fp32",
    ]
    assert set(rec["sources"]) == {"results/latency/h5a.csv", "results/latency/delay.json"}
    beside = {b["label"]: b["text"] for b in rec["beside"]}
    assert beside["delay d, dp"].startswith("d = 3 control steps; provider ort_cuda_fp32")
    assert "ort_trt_fp32 (unscored)" in beside  # reported, never counted
    assert any("no embedded target" in c for c in rec["caveats"])
    rec = hy.read_h5a(_write_h5a(tmp_path / "b", "not supported"))
    assert (rec["status"], rec["verdict"]) == ("scored", "not supported")
    scorer_text = "not scored - ort_cpu_fp32: closed-loop parity not flown"
    rec = hy.read_h5a(_write_h5a(tmp_path / "c", scorer_text))
    assert rec["status"] == scorer_text and rec["verdict"] is None
    with pytest.raises(hy.RowError, match="h5a_verdict"):
        hy.read_h5a(_write_h5a(tmp_path / "d", "supported", csv_verdict="not supported"))


def test_h5a_reads_the_committed_deploy_scorer_output() -> None:
    """results/latency as make bench-score wrote it (Phase 9), read as written."""
    root = p8_adapter.REPO_ROOT / "results" / "latency"
    if not (root / "h5a.csv").is_file():
        pytest.skip("results/latency/h5a.csv not committed on this branch")
    import json

    rec = hy.read_h5a(root)
    delay = json.loads((root / "delay.json").read_text(encoding="utf-8"))
    assert rec["status"] == "scored" and rec["verdict"] == delay["h5a_verdict"]
    assert rec["delays"] == delay["arms"]
    assert [p["provider"] for p in rec["providers"]]
