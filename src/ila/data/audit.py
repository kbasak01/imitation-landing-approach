"""Demo-store audits run before a dataset is called done (plan Phase 3 item 3).

Random steps. Every random choice is :func:`sample_steps`: ``n`` distinct stored
``(episode, step)`` pairs, uniform over all frames, from
``default_rng(SeedSequence([seed, H(purpose), total frames]))`` with ``H`` the SHA-256 label
entropy and ``purpose`` ``"replay"`` or ``"relabel"``; the project seed is 20261023.

* :func:`replay_audit` (open loop): for the episodes of 20 random steps, reset the listed
  ``(realization, episode seed)`` through the runner and execute the **stored** ``action_exec``;
  at each sampled step the re-rendered frame and the observation must be byte-identical to
  ``images[t]`` / ``teacher_obs[t]``. This checks that ``action_exec[t]`` is what produced
  ``obs[t+1]`` / ``image[t+1]``. **Binding for every store**: every executed action is
  float32 (P8 teachers return float32; DART's noised action is rounded to float32 before
  ``env.step``, ``ila.data.collect.executable``), so the replay must be exact.
* :func:`refly_audit` (closed loop, an extra check): the same episodes re-flown by the same
  recorder (same teacher, same T_mix assignment, same DART noise); every stored array must be
  byte-identical. ``ila/data`` may not touch PyBullet state (P1-D2
  item 5), so a "saved state" is re-created by re-simulation, never restored.
* :func:`relabel_audit`: 1 000 random steps; each episode's stored teacher is ``reset`` with
  the episode seed and called on ``teacher_obs[0 .. t]`` in order (a stateful teacher is never
  queried on an isolated state); ``label[t] == teacher(teacher_obs[t])`` exactly.
* :func:`attr_audit`: disjointness from the stored attributes alone (not manifests): every
  realization key built from an episode's attrs is in P8's ``train`` pool, in no ``tune``
  realization and in none of the five frozen lists; no SS6, no 90 deg; ``(key, episode seed)``
  unique; and each episode equals the split row its ``demo_index`` claims.
* :func:`store_summary`: counts, outcomes per sea state, teacher success per sea state with
  Wilson 95 % intervals (P8's ``wilson_interval``; analytic, no bootstrap), kept fraction,
  frames, episode lengths, clear-colour statistics, and touchdown-time quantiles from the
  collection's runner rows (``time_to_touchdown_s``, P8's contact time, s model scale).

Units: actions normalised; times s model scale (lambda = 1/25: 1 model s = 5 full-scale s).
"""

from __future__ import annotations

import csv
import functools
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ila import p8_adapter
from ila.data import collect
from ila.data import episodes as data_episodes
from ila.data.store import STORE_KEYS, DemoStore, EpisodeMeta
from ila.eval import runner
from ila.obs.split import ObsContract

__all__ = [
    "AUDIT_SEED",
    "QUANTILES",
    "attr_audit",
    "load_runner_rows",
    "refly_audit",
    "relabel_audit",
    "replay_audit",
    "sample_steps",
    "store_summary",
    "touchdown_stats",
    "wasserstein_1",
]

#: The project seed for every random step choice (configs/data/demos.yaml generator_seed).
AUDIT_SEED: Final[int] = 20261023

#: Quantiles reported for touchdown times (P8's ``quantile_linear``).
QUANTILES: Final[tuple[float, ...]] = (0.10, 0.50, 0.90)


def sample_steps(
    store: DemoStore, n: int, seed: int = AUDIT_SEED, purpose: str = "replay"
) -> list[tuple[str, int]]:
    """Return ``n`` distinct uniformly random stored ``(episode, step)`` pairs.

    Args:
        store: The store.
        n: Pairs wanted (fewer if the store holds fewer frames).
        seed: Generator seed.
        purpose: Mixed into the entropy (``"replay"``, ``"relabel"``).

    Returns:
        Pairs sorted by episode then step.
    """
    names = store.episodes()
    lengths = np.asarray([store.meta(e).steps for e in names], dtype=np.int64)
    total = int(lengths.sum())
    entropy = [int(seed), data_episodes.label_entropy(purpose), total]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    flat = np.sort(rng.choice(total, size=min(n, total), replace=False))
    edges = np.concatenate([[0], np.cumsum(lengths)])
    out: list[tuple[str, int]] = []
    for f in flat:
        e = int(np.searchsorted(edges, f, side="right") - 1)
        out.append((names[e], int(f - edges[e])))
    return out


def _collection(store: DemoStore) -> collect.Collection:
    name = str(store.file_attrs["collection"])
    return collect.load_collections()[name]


# --------------------------------------------------------------------------- re-flight


class _Replay:
    """Runner actor executing stored actions open loop; records what it is shown."""

    needs_frames: bool = True
    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(self, actions: Mapping[int, NDArray[np.float32]]) -> None:
        self.actions = dict(actions)
        self.seen: dict[int, list[tuple[NDArray[np.float32], NDArray[np.uint8]]]] = {}
        self.overrun: dict[int, int] = {}
        self._seed = -1

    def begin(self, listed: p8_adapter.ListedEpisode, context: object) -> None:
        del context
        self._seed = int(listed.episode_seed)
        self.seen[self._seed] = []
        self.overrun[self._seed] = 0

    def act(self, step: int, obs: NDArray[Any], frame: Any) -> NDArray[np.float64]:
        self.seen[self._seed].append(
            (np.array(obs, dtype=np.float32, copy=True), np.array(frame.rgb, copy=True))
        )
        stored = self.actions[self._seed]
        if step >= stored.shape[0]:  # replay outlived the stored episode: hover command
            self.overrun[self._seed] += 1
            return np.zeros(3, dtype=np.float64)
        return np.asarray(stored[step], dtype=np.float64)


def replay_audit(path: Path, n_states: int = 20, seed: int = AUDIT_SEED) -> dict[str, Any]:
    """Open-loop replay of stored ``action_exec`` (module docstring).

    Args:
        path: The store.
        n_states: Random ``(episode, step)`` pairs.
        seed: Sampling seed.

    Returns:
        Counts of identical frames and observations at the sampled steps, per-episode first
        divergence, and ``passed``.
    """
    with DemoStore(path) as store:
        collection = _collection(store)
        rows = data_episodes.read_split(collection.split)
        picks = sample_steps(store, n_states, seed, "replay")
        names = sorted({e for e, _ in picks})
        metas = {e: store.meta(e) for e in names}
        stored = {e: store.audit_arrays(e) for e in names}
    replay = _Replay({metas[e].episode_seed: stored[e]["action_exec"] for e in names})
    task = runner.VisionTask(
        episodes=[rows[metas[e].demo_index] for e in names],
        # The spec only carries the non-privileged flags; its build is never called.
        spec=p8_adapter.callable_spec(
            "audit:replay", functools.partial(p8_adapter.teacher, "ppo", 0), run_seed=0
        ),
        cfgs=p8_adapter.load_eval_configs(),
        tags=runner.RowTags(arm="audit:replay"),
        render="full",
    )

    def make(c: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[runner.Actor, object]:
        del c, contract
        return replay, replay

    out_rows = runner.fly_chunk(task, make)
    frames_ok = obs_ok = 0
    divergence: dict[str, Any] = {}
    for e, row in zip(names, out_rows, strict=True):
        seen = replay.seen[metas[e].episode_seed]
        n = min(len(seen), metas[e].steps)
        first = next(
            (
                t
                for t in range(n)
                if seen[t][0].tobytes() != stored[e]["teacher_obs"][t].tobytes()
                or seen[t][1].tobytes() != stored[e]["images"][t].tobytes()
            ),
            None,
        )
        if first is not None or len(seen) != metas[e].steps:
            divergence[e] = {
                "first_differing_step": first,
                "steps_stored": metas[e].steps,
                "steps_replayed": len(seen),
                "outcome_stored": metas[e].outcome,
                "outcome_replayed": str(row["outcome"]),
            }
    max_obs = 0.0
    max_px = 0
    for e, t in picks:
        seen = replay.seen[metas[e].episode_seed]
        if t >= len(seen):
            continue
        obs, rgb = seen[t]
        same_obs = obs.tobytes() == stored[e]["teacher_obs"][t].tobytes()
        same_rgb = rgb.tobytes() == stored[e]["images"][t].tobytes()
        obs_ok += int(same_obs)
        frames_ok += int(same_rgb)
        max_obs = max(max_obs, float(np.max(np.abs(obs - stored[e]["teacher_obs"][t]))))
        max_px = max(max_px, int(np.count_nonzero(np.any(rgb != stored[e]["images"][t], -1))))
    exact = frames_ok == obs_ok == len(picks) and not divergence
    return {
        "n_states": len(picks),
        "n_episodes": len(names),
        "frames_identical": frames_ok,
        "teacher_obs_identical": obs_ok,
        "max_abs_obs_diff_at_sampled": max_obs,
        "max_differing_pixels_at_sampled": max_px,
        "episodes_diverging": divergence,
        "passed": exact,
    }


def refly_audit(path: Path, n_states: int = 20, seed: int = AUDIT_SEED) -> dict[str, Any]:
    """Closed-loop re-flight of the episodes of ``n_states`` random steps (module docstring).

    Args:
        path: The store.
        n_states: Random ``(episode, step)`` pairs (the same pairs as :func:`replay_audit`).
        seed: Sampling seed.

    Returns:
        ``{"n_states", "n_episodes", "frames_identical", "episodes_identical",
        "mismatches": [...], "passed"}``.
    """
    with DemoStore(path) as store:
        collection = _collection(store)
        rows = data_episodes.read_split(collection.split)
        picks = sample_steps(store, n_states, seed, "replay")
        names = sorted({e for e, _ in picks})
        metas = {e: store.meta(e) for e in names}
        stored = {e: store.audit_arrays(e) for e in names}
    cfgs = p8_adapter.load_eval_configs()
    prov = collect.provenance()
    policies = collect.collection_policies(collection, rows)
    episodes = tuple(
        (metas[e].demo_index, rows[metas[e].demo_index], policies[metas[e].demo_index])
        for e in names
    )
    records, _ = collect.collect_chunk(collect.CollectTask(collection, episodes, cfgs, prov))
    fresh = {f"ep_{r.meta.demo_index:05d}": r for r in records}
    mismatches: list[str] = []
    for e in names:
        rec = fresh[e]
        if rec.meta.episode_teacher != metas[e].episode_teacher:
            mismatches.append(f"{e}: teacher {rec.meta.episode_teacher} != stored")
            continue
        if rec.meta.outcome != metas[e].outcome or rec.meta.steps != metas[e].steps:
            mismatches.append(f"{e}: outcome/steps {rec.meta.outcome}/{rec.meta.steps}")
            continue
        for k in STORE_KEYS:
            a, b = stored[e][k], rec.arrays[k]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                mismatches.append(f"{e}: {k} differs")
    frames_ok = sum(
        fresh[e].arrays["images"].shape[0] > t
        and stored[e]["images"][t].tobytes() == fresh[e].arrays["images"][t].tobytes()
        for e, t in picks
    )
    return {
        "n_states": len(picks),
        "n_episodes": len(names),
        "frames_identical": int(frames_ok),
        "episodes_identical": len(names) - len({m.split(":")[0] for m in mismatches}),
        "mismatches": mismatches,
        "passed": not mismatches and frames_ok == len(picks),
    }


def relabel_audit(path: Path, n_steps: int = 1000, seed: int = AUDIT_SEED) -> dict[str, Any]:
    """Re-label ``n_steps`` random stored steps with the episode's teacher, shadowed in order.

    Args:
        path: The store.
        n_steps: Random steps.
        seed: Sampling seed.

    Returns:
        ``{"n_steps", "n_episodes", "identical", "max_abs_diff", "by_teacher", "passed"}``.
    """
    cfgs = p8_adapter.load_eval_configs()
    teachers: dict[str, p8_adapter.EvalPolicy] = {}
    identical = 0
    max_abs = 0.0
    by_teacher: Counter[str] = Counter()
    with DemoStore(path) as store:
        picks = sample_steps(store, n_steps, seed, "relabel")
        wanted: dict[str, list[int]] = {}
        for e, t in picks:
            wanted.setdefault(e, []).append(t)
        for e, steps in wanted.items():
            meta = store.meta(e)
            name = meta.episode_teacher
            if name not in teachers:
                teachers[name] = p8_adapter.teacher(name, 0, cfgs)
            teacher = teachers[name]
            obs = store.privileged_observations(e)
            labels = store.labels(e)
            teacher.reset(meta.episode_seed, None)
            fresh = np.stack(
                [
                    np.asarray(teacher.act(obs[t]), dtype=np.float64).astype(np.float32)
                    for t in range(max(steps) + 1)
                ]
            )
            for t in steps:
                same = fresh[t].tobytes() == labels[t].tobytes()
                identical += int(same)
                by_teacher[name] += 1
                max_abs = max(max_abs, float(np.max(np.abs(fresh[t] - labels[t]))))
    return {
        "n_steps": len(picks),
        "n_episodes": len(wanted),
        "identical": identical,
        "max_abs_diff": max_abs,
        "by_teacher": dict(sorted(by_teacher.items())),
        "passed": identical == len(picks),
    }


# --------------------------------------------------------------------------- disjointness


def _meta_key(m: EpisodeMeta) -> tuple[Any, ...]:
    return tuple(
        p8_adapter.realization_key(m.ss, m.heading_deg, m.speed_kn, m.vessel, m.realization_seed)
    )


def attr_audit(path: Path) -> dict[str, Any]:
    """Disjointness and split identity from the stored attributes alone (module docstring).

    Returns:
        Counts of violations per check and ``passed``.
    """
    with DemoStore(path) as store:
        collection = _collection(store)
        metas = [store.meta(e) for e in store.episodes()]
    rows = data_episodes.read_split(collection.split)
    train, tune = data_episodes.pool_keys(data_episodes.p8_episode_api())
    frozen = data_episodes.frozen_keys()
    keys = [_meta_key(m) for m in metas]
    key_attr_mismatch = sum(
        json.loads(m.realization_key) != list(k) for m, k in zip(metas, keys, strict=True)
    )
    identity = []
    for m, k in zip(metas, keys, strict=True):
        r = rows[m.demo_index]
        claimed = (
            m.regime, m.ss, m.index, m.pad, m.vessel, m.heading_deg, m.speed_kn,
            m.realization_seed, m.episode_seed, m.t0_model_s, k,
        )  # fmt: skip
        listed = (
            r.regime, r.ss, r.index, r.pad, r.vessel, r.heading_deg, r.speed_kn,
            r.realization_seed, r.episode_seed, r.t0_model_s, tuple(r.key),
        )  # fmt: skip
        if claimed != listed or m.stream != r.regime:
            identity.append(f"ep_{m.demo_index:05d}")
    checks = {
        "outside_train_pool": sum(k not in train for k in keys),
        "in_tune_pool": sum(k in tune for k in keys),
        **{f"in_frozen_{name}": sum(k in f for k in keys) for name, f in frozen.items()},
        "ss6": sum(m.ss in data_episodes.FORBIDDEN_SEA_STATES for m in metas),
        "heading_90": sum(
            any(abs(m.heading_deg - h) < 1e-9 for h in data_episodes.FORBIDDEN_HEADINGS_DEG)
            for m in metas
        ),
        "repeated_key_seed": len(metas)
        - len({(k, m.episode_seed) for k, m in zip(keys, metas, strict=True)}),
        "realization_key_attr_mismatch": key_attr_mismatch,
        "split_identity_mismatch": len(identity),
        "demo_index_not_prefix": int(
            sorted(m.demo_index for m in metas) != list(range(len(metas)))
        ),
    }
    return {
        "n_episodes": len(metas),
        "split_file": f"splits/{collection.split}.parquet",
        "frozen_lists_checked": sorted(frozen),
        "violations": checks,
        "identity_mismatches": identity[:20],
        "passed": not any(checks.values()),
    }


def split_audit(path: Path) -> dict[str, Any]:
    """Backward-compatible alias of :func:`attr_audit`."""
    return attr_audit(path)


# --------------------------------------------------------------------------- statistics


def load_runner_rows(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    """Read a collection's ``episodes.csv`` keyed by ``(ss, cell index, episode seed)``."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["ss"], int(r["index"]), int(r["episode_seed"])): r for r in csv.DictReader(fh)}


def _q(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(p8_adapter.stats.quantile_linear(arr, q)) if arr.size else float("nan")


def wasserstein_1(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact 1-D Wasserstein-1 distance between two empirical distributions (same units).

    ``integral |F_a(x) - F_b(x)| dx`` over the merged support.
    """
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    if not x.size or not y.size:
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    fa = np.searchsorted(x, grid[:-1], side="right") / x.size
    fb = np.searchsorted(y, grid[:-1], side="right") / y.size
    return float(np.sum(np.abs(fa - fb) * np.diff(grid)))


def touchdown_stats(times: Sequence[float]) -> dict[str, Any]:
    """Quantiles of touchdown times, s model scale; NaN entries count as no touchdown."""
    arr = np.asarray(times, dtype=np.float64)
    touched = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "n_touchdown": int(touched.size),
        "n_without_touchdown": int(arr.size - touched.size),
        **{f"p{int(round(q * 100))}_s": _q(touched.tolist(), q) for q in QUANTILES},
        "mean_s": float(touched.mean()) if touched.size else float("nan"),
    }


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = p8_adapter.stats.wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}


def _outcome_block(metas: Sequence[EpisodeMeta]) -> dict[str, Any]:
    by_ss: dict[str, Counter[str]] = defaultdict(Counter)
    for m in metas:
        by_ss[m.ss][m.outcome] += 1
    total = Counter(m.outcome for m in metas)
    return {
        "outcomes": dict(sorted(total.items())),
        "outcomes_by_ss": {ss: dict(sorted(c.items())) for ss, c in sorted(by_ss.items())},
        "success": _rate(total.get("success", 0), len(metas)),
        "success_by_ss": {
            ss: _rate(c.get("success", 0), sum(c.values())) for ss, c in sorted(by_ss.items())
        },
    }


def store_summary(path: Path, runner_rows: Path | None) -> dict[str, Any]:
    """Counts, outcomes, success with Wilson 95 %, lengths, clear colour, touchdown times.

    Args:
        path: The store.
        runner_rows: The collection's ``episodes.csv`` (P8 row columns), or ``None``.

    Returns:
        The per-store statistics block.
    """
    with DemoStore(path) as store:
        attrs = dict(store.file_attrs)
        metas = [store.meta(e) for e in store.episodes()]
    by = {
        "by_ss": Counter(m.ss for m in metas),
        "by_heading_deg": Counter(f"{m.heading_deg:g}" for m in metas),
        "by_speed_kn": Counter(f"{m.speed_kn:g}" for m in metas),
    }
    steps = np.asarray([m.steps for m in metas], dtype=np.float64)
    clear = np.concatenate([m.clear_frac for m in metas]).astype(np.float64)
    kept = [m for m in metas if m.success]
    out: dict[str, Any] = {
        "collection": attrs["collection"],
        "teacher": attrs["teacher"],
        "dart_sigma": attrs["dart_sigma"],
        "ila_commit": attrs["ila_commit"],
        "p8_commit": attrs["p8_commit"],
        "camera_cfg_sha256": attrs["camera_cfg_sha256"],
        "split_file": attrs["split_file"],
        "split_file_sha256": attrs["split_file_sha256"],
        "episodes": len(metas),
        "frames": int(steps.sum()),
        **{k: dict(sorted(v.items())) for k, v in by.items()},
        **_outcome_block(metas),
        "kept_fraction_success_filter": len(kept) / len(metas),
        "kept_episodes": len(kept),
        "kept_frames": int(sum(m.steps for m in kept)),
        "episode_length_steps": {
            "mean": float(steps.mean()),
            "p50": _q(steps.tolist(), 0.5),
            "max": int(steps.max()),
        },
        "clear_colour": {
            "frac_max": float(clear.max()),
            "frac_p99": _q(clear.tolist(), 0.99),
            "frames_gt_0": int(np.count_nonzero(clear)),
            "frames": int(clear.size),
        },
        "episode_teachers": dict(sorted(Counter(m.episode_teacher for m in metas).items())),
        "file_bytes": path.stat().st_size,
        "file_sha256": p8_adapter.sha256_file(path),
    }
    if attrs["teacher"] == "T_mix":
        out["by_episode_teacher"] = {
            name: {
                "episodes": sum(m.episode_teacher == name for m in metas),
                **_outcome_block([m for m in metas if m.episode_teacher == name]),
            }
            for name in sorted({m.episode_teacher for m in metas})
        }
    if runner_rows is not None:
        rows = load_runner_rows(runner_rows)
        joined = [rows.get((m.ss, m.index, m.episode_seed)) for m in metas]
        bad = sum(
            r is None or r["outcome"] != m.outcome or int(r["steps"]) != m.steps
            for r, m in zip(joined, metas, strict=True)
        )
        out["runner_rows_mismatch"] = bad
        times = {
            m.demo_index: float(r["time_to_touchdown_s"]) if r and r["time_to_touchdown_s"] else
            float("nan")
            for m, r in zip(metas, joined, strict=True)
        }  # fmt: skip
        out["time_to_touchdown_s"] = {"all": touchdown_stats(list(times.values()))}
        if attrs["teacher"] == "T_mix":
            split = {
                name: [times[m.demo_index] for m in metas if m.episode_teacher == name]
                for name in ("ppo", "gated")
            }
            out["time_to_touchdown_s"].update({k: touchdown_stats(v) for k, v in split.items()})
            out["time_to_touchdown_s"]["w1_ppo_vs_gated_s"] = wasserstein_1(
                [t for t in split["ppo"] if np.isfinite(t)],
                [t for t in split["gated"] if np.isfinite(t)],
            )
        if attrs["collection"] == "ppo_2000":
            first = [times[j] for j in sorted(times) if j < 1000]
            out["time_to_touchdown_s"]["first_1000"] = touchdown_stats(first)
            out["episode_length_steps"]["mean_first_1000"] = float(
                np.mean([m.steps for m in metas if m.demo_index < 1000])
            )
    return out
