"""``bc_dagger`` stage data: roll out, audit, aggregate, then train (Phase 5; P3-D1 §3).

``bc_dagger`` runs on ``bc_matched``'s iteration driver (:func:`ila.train.loop.build_stages` and
:func:`ila.train.loop.train_run`: warm start from the same-seed ``bc`` run, normaliser inherited,
AdamW reset and a 10 000-step cosine per iteration; P4-D5 item 1). The only arm-specific part is
what stage ``i``'s lazy ``data`` callable does when the stage starts (:func:`stage_data`):

1. **Rollout student** (:func:`rollout_student`): the EMA weights at the last step of iteration
   ``i - 1``. Iteration 1 flies the same-seed ``bc`` run's ``s0_step030000.pt`` (``--init-from``,
   its last step); iteration ``i > 1`` flies this run's ``s<i-1>_step010000.pt``, which the
   driver always writes (a checkpoint is saved at the last step of every stage). The file's
   SHA-256 must equal the one the run manifest recorded when it was written. No list is read to
   choose it. ``ila.data.dagger.run_iteration`` flies its **EMA** weights on the CPU fp32
   single-sample path.
2. **Rollouts** (:func:`ila.data.dagger.run_iteration`): iteration ``i``'s 200 ``train``
   episodes (``demos`` stream indices ``1000 + 200 (i - 1) ..``), beta per step, every visited
   state labelled by the teacher. Idempotent: an iteration already flown by the same checkpoint
   is reused, one flown by another checkpoint raises. That is what makes resume work, mid-stage
   and between stages. Iterations ``1 .. i - 1`` are read back with
   :func:`ila.data.dagger.load_rollout` (re-hashed) and must match the SHA-256s this run's
   manifest recorded when they were flown, and each must have been flown by the checkpoint that
   step 1 names for it.
3. **Audit** (:func:`ila.data.dagger.audit_iteration`, 1 000 re-labelled states and the
   attribute, coin and executed-action checks). A stage whose audit is not ``clean`` is never
   trained: :class:`DaggerAuditError`.
4. **Aggregate**: ``ppo_2000`` stream indices 0-999, success-filtered (iteration 0's demos,
   exactly as ``bc`` read them), plus the rollout stores of iterations ``1 .. i``, **unfiltered**
   (P3-D1 §2), concatenated with :meth:`ila.train.dataset.FrameTable.concat` and sampled
   uniformly over frames by the driver. Labelled episodes ``= 1000 + 200 i`` (asserted here and
   by :func:`ila.data.dagger.aggregate_manifest`; 2 000 after iteration 5, asserted again by the
   driver). Labelled frames are recorded beside them.
5. **Record**: rollout summary, aggregate manifest and audit go into the run manifest
   (``data.<i>.dagger``) and into ``<run_dir>/dagger/iter<i>.json``.

Held-out loss is unchanged (``ppo_2000`` 1950-1999). Those stored teacher episodes are never in
``bc_dagger``'s training data (overlap 0), but iteration 5 flies the same 50 starting conditions
(realization and episode seed) with the student, so their first frames coincide; the count is
logged as ``heldout_shared_starting_conditions``.

Units: actions normalised (x v_max = 1.5 m/s model scale); steps are optimiser steps; control
steps are 1/30 s model scale (lambda = 1/25).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ila.data.store import DemoStore
from ila.train import loop
from ila.train import manifest as mf
from ila.train.config import TrainConfig, sha256_file
from ila.train.dataset import BCWindowDataset, FrameTable
from ila.train.sampling import EpisodeChoice, choose_episodes

if TYPE_CHECKING:
    from ila.data.dagger import RolloutResult

__all__ = ["AUDIT_STEPS", "DaggerAuditError", "record_path", "rollout_student", "stage_data"]

#: Visited states re-labelled per iteration (plan Phase 5 item 2).
AUDIT_STEPS: Final[int] = 1000


class DaggerAuditError(RuntimeError):
    """An iteration's rollout audit is not clean; its stage is never trained."""


def record_path(run_dir: Path, iteration: int) -> Path:
    """Return ``<run_dir>/dagger/iter<iteration>.json`` (the stage's DAgger record)."""
    return run_dir / "dagger" / f"iter{int(iteration)}.json"


def _ckpt_name(stage: int, stage_step: int) -> str:
    return f"s{stage}_step{stage_step:06d}.pt"


def _run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    return mf.read_json(path) if path.is_file() else {}


def rollout_student(spec: loop.RunSpec, iteration: int, stage_steps: int) -> tuple[Path, str]:
    """Return the checkpoint iteration ``iteration`` flies and its SHA-256 (module docstring).

    Args:
        spec: The ``bc_dagger`` run.
        iteration: 1 .. n.
        stage_steps: Steps per iteration (10 000; the smoke's override).

    Returns:
        ``(path, sha256)``. Iteration 1: ``spec.init_from`` (the ``bc`` run's last step).
        Iteration ``i > 1``: ``<run_dir>/checkpoints/s<i-1>_step<stage_steps>.pt``.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If its SHA-256 differs from the run manifest's record of it (or of
            ``init_from``), or the manifest has no record of a checkpoint ``i > 1``.
    """
    man = _run_manifest(spec.run_dir)
    if iteration == 1:
        if spec.init_from is None:
            raise RuntimeError("bc_dagger iteration 1 flies the same-seed bc run (--init-from)")
        path = spec.init_from
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = sha256_file(path)
        recorded = man.get("init_from_sha256")
        if recorded is not None and recorded != sha:
            raise RuntimeError(f"{path}: SHA-256 differs from the run manifest's init_from")
        return path, sha
    name = _ckpt_name(iteration - 1, stage_steps)
    path = spec.run_dir / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError(f"{path}: iteration {iteration - 1}'s last step was never saved")
    sha = sha256_file(path)
    entries = [c for c in man.get("checkpoints", []) if c["file"] == name and c["kind"] == "full"]
    if not entries or entries[-1]["sha256"] != sha:
        raise RuntimeError(f"{path}: not the checkpoint the run manifest recorded")
    return path, sha


def _check_rollout(
    r: RolloutResult,
    cfg: TrainConfig,
    iteration: int,
    student_sha: str,
    n_expected: int,
    recorded: Mapping[str, Any] | None,
) -> None:
    """Refuse a rollout that is not iteration ``iteration`` of this run as recorded."""
    assert cfg.dagger is not None
    problems = []
    if r.iteration != iteration:
        problems.append(f"iteration {r.iteration}")
    if r.beta != cfg.dagger.beta_for(iteration):
        problems.append(f"beta {r.beta} != {cfg.dagger.beta_for(iteration)}")
    if r.student_ckpt_sha256 != student_sha:
        problems.append(f"flown by {r.student_ckpt_sha256[:12]}..., expected {student_sha[:12]}...")
    if r.labelled_episodes != n_expected:
        problems.append(f"{r.labelled_episodes} episodes, expected {n_expected}")
    if recorded is not None and recorded.get("store_sha256") != r.store_sha256:
        problems.append("store SHA-256 differs from the run manifest's record")
    if problems:
        raise RuntimeError(f"DAgger iteration {iteration}: {'; '.join(problems)}")


def _recorded_rollout(man: Mapping[str, Any], iteration: int) -> Mapping[str, Any] | None:
    rec = man.get("data", {}).get(str(iteration), {}).get("dagger")
    if rec is None:
        return None
    rollout: Mapping[str, Any] = rec["rollout"]
    return rollout


def _student_sha(spec: loop.RunSpec, iteration: int, stage_steps: int) -> str:
    """SHA-256 of the checkpoint that must have flown ``iteration`` (from the files)."""
    return rollout_student(spec, iteration, stage_steps)[1]


def stage_data(
    spec: loop.RunSpec, iteration: int, stage_steps: int, hash_store: bool
) -> loop.DataBundle:
    """Build ``bc_dagger`` stage ``iteration``'s data (module docstring, steps 1-5).

    Args:
        spec: The run.
        iteration: The stage index ``i`` (1 .. n).
        stage_steps: Steps per iteration (locates iteration ``i - 1``'s last checkpoint).
        hash_store: Hash the demo store for the manifest.

    Returns:
        The aggregate's :class:`ila.train.loop.DataBundle`; its ``extra["dagger"]`` is the
        stage record (also written to :func:`record_path`).

    Raises:
        DaggerAuditError: If iteration ``i``'s audit is not clean.
        RuntimeError: On any provenance or count mismatch.
    """
    from ila.data import dagger  # lazy: it imports the runner (P8 physics)

    cfg = spec.cfg
    it, dg = cfg.iterations, cfg.dagger
    if it is None or dg is None:
        raise ValueError(f"{cfg.arm} is not an on-policy iterated arm")
    first_n = cfg.first_n if spec.first_n_override is None else spec.first_n_override
    n_per = it.add_per_iteration if spec.rollout_limit is None else spec.rollout_limit
    log = _logger(spec)

    # 1-2: the rollout student and iteration i's rollouts (flown, or reused on resume).
    student, student_sha = rollout_student(spec, iteration, stage_steps)
    man = _run_manifest(spec.run_dir)
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    current = dagger.run_iteration(
        run_seed=spec.run_seed,
        iteration=iteration,
        student_ckpt=student,
        data_root=spec.data_root,
        workers=spec.rollout_workers,
        limit=spec.rollout_limit,
        purpose=spec.purpose,
        log=log,
    )
    _check_rollout(current, cfg, iteration, student_sha, n_per, _recorded_rollout(man, iteration))
    rollouts: list[RolloutResult] = []
    for j in range(1, iteration):
        r = dagger.load_rollout(spec.data_root, spec.run_seed, j)
        _check_rollout(
            r, cfg, j, _student_sha(spec, j, stage_steps), n_per, _recorded_rollout(man, j)
        )
        rollouts.append(r)
    rollouts.append(current)
    timing["rollouts"] = time.perf_counter() - t0

    # 3: audit before anything is loaded; an unclean iteration is never trained on.
    t0 = time.perf_counter()
    audit = dagger.audit_iteration(spec.data_root, spec.run_seed, iteration, n_steps=AUDIT_STEPS)
    timing["audit"] = time.perf_counter() - t0
    if audit.get("clean") is not True:
        raise DaggerAuditError(
            f"bc_dagger s{spec.run_seed} iteration {iteration}: rollout audit not clean; "
            f"stage {iteration} is not trained "
            f"(see {dagger.rollout_dir(spec.data_root, spec.run_seed)})"
        )

    # 4: the aggregate -- demos 0..first_n-1 success-filtered, rollouts 1..i unfiltered.
    t0 = time.perf_counter()
    demos = loop.load_stage_data(spec, [(0, first_n)], hash_store=hash_store)
    if not isinstance(demos.train, BCWindowDataset):
        raise ValueError("bc_dagger is a vision BC arm")
    tables = [demos.train.table]
    choices: list[EpisodeChoice] = []
    stores: list[dict[str, Any]] = []
    for r in rollouts:
        with DemoStore(r.store) as store:
            choice = choose_episodes(
                store, n_per, dg.rollouts_success_filter, start=it.added_range(r.iteration).start
            )
            table = FrameTable.from_store(store, choice.names, cfg.obs_variant)
        if choice.kept != r.labelled_episodes or table.n_frames != r.labelled_frames:
            raise RuntimeError(
                f"iteration {r.iteration}: {choice.kept} episodes / {table.n_frames} frames in "
                f"the store, the rollout manifest says {r.labelled_episodes} / {r.labelled_frames}"
            )
        tables.append(table)
        choices.append(choice)
        stores.append(
            {
                "collection": dagger.collection_name(spec.run_seed, r.iteration),
                "path": str(r.store),
                "sha256": r.store_sha256,
                "expected_sha256": r.store_sha256,
                "matches": True,  # re-hashed against its manifest by ila.data.dagger
                "rollout_manifest": str(r.manifest),
            }
        )
    train = loop._wrap_table(cfg.kind, cfg, FrameTable.concat(tables))
    del tables
    labelled = demos.labelled_episodes + sum(c.attempted for c in choices)
    expected = first_n + n_per * iteration
    if labelled != expected:
        raise RuntimeError(f"iteration {iteration}: {labelled} labelled episodes != {expected}")
    strict = spec.purpose == "train" and spec.rollout_limit is None
    aggregate = dagger.aggregate_manifest(
        run_seed=spec.run_seed,
        iteration=iteration,
        data_root=spec.data_root,
        demos=demos.summary(),
        rollouts=rollouts,
        strict=strict,
    )
    if int(aggregate["labelled_episodes"]) != labelled:
        raise RuntimeError(
            f"aggregate manifest counts {aggregate['labelled_episodes']} != {labelled}"
        )

    timing["aggregate"] = time.perf_counter() - t0

    # 5: the record.
    held = spec.heldout
    shared = 0
    if held is not None and held.collection == cfg.collection:
        held_idx = set(held.indices())
        shared = sum(len(held_idx & set(c.demo_indices)) for c in choices)
    record: dict[str, Any] = {
        "iteration": iteration,
        "beta": dg.beta_for(iteration),
        "policy_flown": _policy_flown(dg.beta_for(iteration)),
        "rollout_student": {
            "file": str(student),
            "sha256": student_sha,
            "weights": "ema",
            "inference_path": dg.inference_path,
            "rule": dg.rollout_weights,
            "source": "init_from (bc run, last step)" if iteration == 1 else "this run",
        },
        "rollout": current.summary(),
        "earlier_rollouts": [
            {"iteration": r.iteration, "store_sha256": r.store_sha256, "reused": r.reused}
            for r in rollouts[:-1]
        ],
        "audit": audit,
        "aggregate": aggregate,
        "labelled_episodes": labelled,
        "labelled_frames_rollouts": sum(r.labelled_frames for r in rollouts),
        "aggregate_frames": len(train),
        "heldout_overlap_episodes": demos.heldout_overlap,
        "heldout_shared_starting_conditions": shared,
        "rollout_workers": spec.rollout_workers,
        "rollout_limit": spec.rollout_limit,
        "strict_counts": strict,
        "timing_s": {**timing, "note": "CPU rollouts (reused: ~0), audit, aggregate load"},
    }
    mf.write_json_atomic(record_path(spec.run_dir, iteration), record)
    log(
        f"bc_dagger s{spec.run_seed} iteration {iteration}: {labelled} labelled episodes, "
        f"{len(train)} frames, rollout success {current.outcomes.get('success', 0)}/"
        f"{current.labelled_episodes}, audit clean"
    )
    return loop.DataBundle(
        train=train,
        heldout=demos.heldout,
        choices=[*demos.choices, *choices],
        labelled_episodes=labelled,
        heldout_overlap=demos.heldout_overlap,
        stores=[*demos.stores, *stores],
        extra={"dagger": record},
    )


def _policy_flown(beta: float) -> str:
    """Name the policy an iteration's rollouts measure (beta > 0: the mixture, not the student)."""
    if beta == 0.0:
        return "student EMA"
    return f"beta-mixture: teacher label with p={beta:g} per step, else student EMA"


def _logger(spec: loop.RunSpec) -> Callable[[str], None]:
    log_file = spec.run_dir / "dagger" / "rollouts.log"

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log
