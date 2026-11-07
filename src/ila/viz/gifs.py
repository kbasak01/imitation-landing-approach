"""Phase 8 GIF gallery: episodes hand-picked by a stated rule, re-flown and checked, then drawn.

Simulation only: PyBullet physics, frames rendered by PyBullet's TinyRenderer (96 x 96 px, shown
enlarged); model scale, lambda = 1/25; one frame per 30 Hz control step (33.3 ms model scale).

The rule (:data:`RULE`, also ``results/e08/gifs/RULE.md``) was committed before any GIF was
made. :func:`select_episodes` applies it to committed rows only (P8's e07 ``ppo`` seed 0 rows;
this project's core ``id`` rows of ``bc``, ``bc_dagger`` and ``dp`` seed 0). Each panel is then
re-flown through the matrix's code path (:func:`fly_panel`: :func:`ila.eval.runner.fly_chunk`
on the CPU fp32 single-sample path, one torch thread, render ``full``; students with the
selected checkpoint, EMA weights, diagnostics and the ``ppo`` shadow teacher exactly as
:func:`ila.eval.matrix.fly_job` / :func:`ila.train.select.fly_checkpoint` build them) with a
frame recorder wrapped around the actor. The recorder only copies the frame the actor is given;
it does not touch the environment. :func:`check_panel` asserts the re-flown row equals the
committed row (students: the whole matrix CSV line; the teacher: P8's ``compare_to_reference``
on P8's columns) before :func:`write_gif` runs.
"""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import io
import json
import multiprocessing as mp
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import matrix as ev_matrix
from ila.eval import runner
from ila.obs.split import ObsContract
from ila.policies.registry import read_checkpoint
from ila.train import select as train_select

__all__ = [
    "CATEGORIES",
    "CELLS",
    "GIF_LIMIT_BYTES",
    "PANELS",
    "RULE",
    "Panel",
    "PanelResult",
    "Pick",
    "check_panel",
    "compose",
    "fly_panel",
    "rule_from_md",
    "select_episodes",
    "RowsNotReadyError",
    "committed_lines",
    "committed_panel_rows",
    "fly_panels",
    "run_gallery",
    "student_job",
    "student_panel",
    "write_gif",
]

#: The selection rule, verbatim in ``results/e08/gifs/RULE.md`` (between the markers).
RULE: Final[str] = """\
1. Pool: P8's frozen `id` list, cells SS5 and SS6 (400 episodes), in list order (list
   position = the episode's row number in P8's `id` list: SS3, SS4, SS5, SS6 blocks, index
   ascending within a block).
2. Panels, in this order: the teacher, P8 `ppo` seed 0 (its committed e07 row); `bc` seed 0;
   `bc_dagger` seed 0; `dp` seed 0 at the core inference (Ta = 8, K = 10, no ensembling). Each
   student's row is its committed Phase 8 core row (`results/e08/rows/<arm>/s0/id.csv.gz`).
3. Categories, taken in this order; each picks the episode with the lowest list position among
   the qualifying episodes not already picked by an earlier category:
   (a) all four panels succeed;
   (b) the students split: among `bc`, `bc_dagger`, `dp`, at least one succeeds and at least one
       does not (the teacher's outcome is not considered);
   (c) `dp`'s outcome is `off_pad`;
   (d) the teacher does not succeed, in SS6.
   A category with no qualifying episode is reported empty; nothing is substituted.
4. Outcomes are read only from the committed rows. Every panel is then re-flown on the CPU fp32
   single-sample path through the matrix's code path (students: the selected checkpoint,
   EMA weights, render `full`, the `ppo` shadow teacher; the teacher via
   `p8_adapter.teacher_spec("ppo")`, render `full`). The script asserts each re-flown row equals
   its committed row (students: the whole CSV line, every column; the teacher: P8's
   `compare_to_reference` byte-for-byte on P8's columns) before any GIF is written.
5. Each GIF shows the four panels side by side, one control step (33.3 ms model scale) per
   frame; a shorter episode holds its last frame. Label: "hand-picked by stated rule ·
   simulation (PyBullet TinyRenderer)" and, per panel, "<arm> · id <SS> ep <index> ·
   <outcome>".
6. If the GIFs total more than 10 MB they are not committed: only `manifest.json` (rule,
   picks, SHA-256 of each GIF) is, and the GIFs are release assets.
"""

#: Panel name -> arm (``teacher`` is P8 ``ppo`` seed 0).
PANELS: Final[tuple[str, ...]] = ("teacher", "bc", "bc_dagger", "dp")
STUDENTS: Final[tuple[str, ...]] = PANELS[1:]
STUDENT_SEED: Final[int] = 0
LIST_NAME: Final[str] = "id"
CELLS: Final[tuple[str, ...]] = ("SS5", "SS6")
CATEGORIES: Final[tuple[str, ...]] = (
    "a_all_four_succeed",
    "b_students_split",
    "c_dp_off_pad",
    "d_teacher_ss6_failure",
)
GIF_LIMIT_BYTES: Final[int] = 10 * 1024 * 1024
LABEL: Final[str] = "hand-picked by stated rule · simulation (PyBullet TinyRenderer)"
#: Display scale of the 96 px frames, and GIF frame duration (one control step), ms.
SCALE: Final[int] = 3
FRAME_MS: Final[int] = 33

type Row = Mapping[str, str]


def rule_from_md(text: str) -> str:
    """The rule text between ``<!-- RULE START -->`` and ``<!-- RULE END -->`` in RULE.md."""
    start = text.index("<!-- RULE START -->\n") + len("<!-- RULE START -->\n")
    return text[start : text.index("<!-- RULE END -->")]


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Pick:
    """One picked episode.

    Attributes:
        category: One of :data:`CATEGORIES`.
        ss: Sea-state cell.
        index: Index within the cell.
        episode_seed: P8's episode seed.
        position: Row number in P8's ``id`` list (0-based).
        outcomes: Panel -> committed outcome.
    """

    category: str
    ss: str
    index: int
    episode_seed: int
    position: int
    outcomes: Mapping[str, str]


def _qualifies(category: str, ss: str, o: Mapping[str, str]) -> bool:
    students = [o[p] == "success" for p in STUDENTS]
    if category == "a_all_four_succeed":
        return all(o[p] == "success" for p in PANELS)
    if category == "b_students_split":
        return any(students) and not all(students)
    if category == "c_dp_off_pad":
        return o["dp"] == "off_pad"
    if category == "d_teacher_ss6_failure":
        return ss == "SS6" and o["teacher"] != "success"
    raise ValueError(category)


def select_episodes(
    listed: Sequence[p8_adapter.ListedEpisode],
    rows: Mapping[str, Sequence[Row]],
) -> dict[str, Pick | None]:
    """Apply :data:`RULE` to committed rows.

    Args:
        listed: P8's ``id`` list, in list order (positions are taken from it).
        rows: Panel -> its committed rows (any superset of the ``id`` SS5-SS6 episodes).

    Returns:
        Category -> its :class:`Pick`, or ``None`` when no episode qualifies.

    Raises:
        ValueError: If a panel lacks a row for a pooled episode or a seed differs.
    """
    by: dict[str, dict[tuple[str, int], Row]] = {
        p: {(r["ss"], int(r["index"])): r for r in rows[p] if r["regime"] == LIST_NAME}
        for p in PANELS
    }
    pool = [(pos, e) for pos, e in enumerate(listed) if e.ss in CELLS]
    outcomes: list[tuple[int, p8_adapter.ListedEpisode, dict[str, str]]] = []
    for pos, e in pool:
        o = {}
        for p in PANELS:
            r = by[p].get((e.ss, int(e.index)))
            if r is None or int(r["episode_seed"]) != int(e.episode_seed):
                raise ValueError(f"{p}: no committed row for id {e.ss} {e.index}")
            o[p] = str(r["outcome"])
        outcomes.append((pos, e, o))
    picked: set[int] = set()
    out: dict[str, Pick | None] = {}
    for cat in CATEGORIES:
        hit = next(
            (
                (pos, e, o)
                for pos, e, o in outcomes
                if pos not in picked and _qualifies(cat, e.ss, o)
            ),
            None,
        )
        if hit is None:
            out[cat] = None
            continue
        pos, e, o = hit
        picked.add(pos)
        out[cat] = Pick(cat, e.ss, int(e.index), int(e.episode_seed), pos, o)
    return out


# --------------------------------------------------------------------------- re-flight


@dataclass(frozen=True)
class Panel:
    """One panel to re-fly (picklable).

    Attributes:
        panel: ``teacher`` or a student arm.
        episode: The listed episode.
        checkpoint: Student checkpoint path (``None`` for the teacher).
        sha256: Its SHA-256 (``""`` for the teacher).
        teacher: The student's teacher label (``T_ppo``).
        inference: ``(Ta, K, ensemble)`` for ``dp``, else ``None``.
    """

    panel: str
    episode: p8_adapter.ListedEpisode
    checkpoint: str | None = None
    sha256: str = ""
    teacher: str = "T_ppo"
    inference: tuple[int, int, bool] | None = None


@dataclass
class PanelResult:
    """A re-flown panel: its row and the frames its actor was given (uint8 H x W x 3)."""

    panel: Panel
    row: dict[str, Any]
    frames: list[npt.NDArray[np.uint8]] = field(default_factory=list)


class _Recorder:
    """Wraps an actor; copies each frame it is given, changes nothing else."""

    def __init__(self, inner: Any, sink: list[npt.NDArray[np.uint8]]) -> None:
        self.inner = inner
        self.sink = sink
        self.needs_frames = inner.needs_frames

    def begin(
        self, listed: p8_adapter.ListedEpisode, context: p8_adapter.PrivilegedContext | None
    ) -> None:
        self.inner.begin(listed, context)

    def act(self, step: int, obs: Any, frame: Any) -> npt.NDArray[np.float64]:
        if frame is not None:
            self.sink.append(np.array(frame.rgb, dtype=np.uint8, copy=True))
        out: npt.NDArray[np.float64] = self.inner.act(step, obs, frame)
        return out


def _recording_factory(
    spec: p8_adapter.PolicySpec, sink: list[npt.NDArray[np.uint8]]
) -> Callable[[p8_adapter.EvalConfigs, ObsContract], tuple[Any, object]]:
    """The runner's in-worker actor factory (``_local_actor``) with a :class:`_Recorder`."""

    def make(cfgs: p8_adapter.EvalConfigs, contract: ObsContract) -> tuple[Any, object]:
        policy: Any = spec.build(cfgs)
        if runner.is_modular(policy) or runner.is_student(policy):
            actor: Any = runner.StudentActor(policy, contract)
        else:
            actor = runner.TeacherActor(policy)
        return _Recorder(actor, sink), policy

    return make


def _task(panel: Panel) -> runner.VisionTask:
    cfgs = p8_adapter.load_eval_configs()
    if panel.panel == "teacher":
        spec = p8_adapter.teacher_spec("ppo")
        return runner.VisionTask(
            episodes=[panel.episode], spec=spec, cfgs=cfgs, tags=runner.RowTags(arm="ppo")
        )
    assert panel.checkpoint is not None
    base = functools.partial(train_select.checkpoint_policy, panel.checkpoint)
    if panel.inference is None:
        build: Any = base
        tags = runner.RowTags(
            arm=panel.panel,
            teacher=panel.teacher,
            student_seed=STUDENT_SEED,
            ckpt_sha256=panel.sha256,
        )
    else:
        ta, k, ens = panel.inference
        build = functools.partial(
            base, n_action_steps=ta, num_inference_steps=k, temporal_ensembling=ens
        )
        tags = runner.RowTags(
            arm=panel.panel,
            teacher=panel.teacher,
            student_seed=STUDENT_SEED,
            Ta=ta,
            K=k,
            ensemble=ens,
            ckpt_sha256=panel.sha256,
        )
    spec = p8_adapter.callable_spec(panel.panel, build, run_seed=STUDENT_SEED)
    return runner.VisionTask(
        episodes=[panel.episode],
        spec=spec,
        cfgs=cfgs,
        tags=tags,
        render="full",
        contract_variant=str(read_checkpoint(Path(panel.checkpoint))["obs_variant"]),
        diagnostics=True,
        shadow_teacher=p8_adapter.teacher_spec("ppo"),
    )


def fly_panel(panel: Panel) -> PanelResult:
    """Re-fly one panel in this process (one torch thread) and record its frames."""
    runner.set_single_thread_env()
    runner.init_worker_threads()
    task = _task(panel)
    frames: list[npt.NDArray[np.uint8]] = []
    rows = runner.fly_chunk(task, _recording_factory(task.spec, frames))
    return PanelResult(panel, rows[0], frames)


def fly_panels(panels: Sequence[Panel], *, workers: int = 4) -> list[PanelResult]:
    """Fly panels, at most ``workers`` (<= 4) processes; results in input order."""
    workers = max(1, min(4, workers))
    if workers == 1 or len(panels) <= 1:
        return [fly_panel(p) for p in panels]
    with mp.get_context("spawn").Pool(processes=min(workers, len(panels))) as pool:
        return list(pool.map(fly_panel, panels, chunksize=1))


def student_job(arm: str) -> ev_arms.Job:
    """The core matrix job whose rows hold a student panel (``id``, seed 0)."""
    m = ev_arms.load_matrix()
    for j in m.jobs:
        if (j.tag, j.arm, j.seed, j.list_name) == ("core", arm, STUDENT_SEED, LIST_NAME):
            return j
    raise KeyError(arm)


def check_panel(result: PanelResult, committed: Row | str, job: ev_arms.Job | None) -> None:
    """Assert a re-flown row equals its committed row (rule item 4).

    Args:
        result: The re-flown panel.
        committed: Students: the committed CSV line (text, no newline) of that episode; the
            teacher: unused (P8's committed file is compared through the adapter).
        job: The student's core job (``None`` for the teacher).

    Raises:
        AssertionError: On any difference.
    """
    if result.panel.panel == "teacher":
        cmp = p8_adapter.compare_to_reference(
            runner.p8_rows([result.row]),
            p8_adapter.P8_ROOT / p8_adapter.COMMITTED_EPISODES["e07"],
            pad="aft",
        )
        if not cmp.get("all_identical") or cmp["methods"].get("ppo", {}).get("n_identical") != 1:
            raise AssertionError(f"teacher re-flight differs from P8's committed row: {cmp}")
        return
    assert job is not None
    text = ev_matrix.rows_csv([result.row], job)
    line = text.split("\n")[1]
    if line != committed:
        raise AssertionError(
            f"{result.panel.panel} re-flight differs from its committed row:\n{line}\n{committed}"
        )


def committed_lines(path: Path) -> dict[tuple[str, int], str]:
    """``(ss, index) -> CSV line`` of a committed rows file (``.csv.gz``)."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        text = f.read()
    lines = text.split("\n")
    header = lines[0].split(",")
    i_ss, i_idx = header.index("ss"), header.index("index")
    out = {}
    for ln in lines[1:]:
        if ln:
            cells = ln.split(",")
            out[(cells[i_ss], int(cells[i_idx]))] = ln
    return out


# --------------------------------------------------------------------------- drawing


def _panel_label(arm: str, pick: Pick | None, ss: str, index: int, outcome: str) -> str:
    del pick
    return f"{arm} · {LIST_NAME} {ss} ep {index} · {outcome}"


def compose(results: Sequence[PanelResult], labels: Sequence[str]) -> list[npt.NDArray[np.uint8]]:
    """Side-by-side frames (H x W x 3, uint8) with the label band; short panels hold."""
    from PIL import Image, ImageDraw

    n = max(len(r.frames) for r in results)
    h, w = results[0].frames[0].shape[:2]
    ph, pw = h * SCALE, w * SCALE
    band = 30
    out = []
    for t in range(n):
        canvas = Image.new("RGB", (pw * len(results), ph + band), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 2), LABEL, fill=(0, 0, 0))
        for k, (r, lab) in enumerate(zip(results, labels, strict=True)):
            f = r.frames[min(t, len(r.frames) - 1)]
            img = Image.fromarray(f).resize((pw, ph), Image.Resampling.NEAREST)
            canvas.paste(img, (k * pw, band))
            draw.text((k * pw + 4, 16), lab, fill=(0, 0, 0))
        out.append(np.asarray(canvas, dtype=np.uint8))
    return out


def write_gif(frames: Sequence[npt.NDArray[np.uint8]], path: Path) -> str:
    """Write frames as a looping GIF (one control step per frame); return its SHA-256."""
    from PIL import Image

    imgs = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(
        buf, format="GIF", save_all=True, append_images=imgs[1:], duration=FRAME_MS, loop=0
    )
    data = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- gallery


class RowsNotReadyError(RuntimeError):
    """A panel's committed rows do not exist yet (its matrix job has not finished)."""


def committed_panel_rows(root: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """Committed rows of every panel on ``id`` and their sources (path -> SHA-256).

    Raises:
        RowsNotReadyError: If a student's core ``id`` rows file has no sidecar yet.
    """
    teacher = p8_adapter.committed_rows("e07", "ppo", 0, regime=LIST_NAME)
    rows: dict[str, list[dict[str, str]]] = {"teacher": list(teacher.rows)}
    sources = {teacher.file: teacher.file_sha256}
    for arm in STUDENTS:
        rel = student_job(arm).rel_path
        path, side = root / rel, root / rel.with_name(rel.name.removesuffix(".csv.gz") + ".json")
        if not (path.is_file() and side.is_file()):
            raise RowsNotReadyError(f"{rel.as_posix()}: not flown yet (no sidecar)")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            rows[arm] = list(csv.DictReader(f))
        sources[f"results/e08/{rel.as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, sources


def student_panel(arm: str, episode: p8_adapter.ListedEpisode) -> Panel:
    """A student panel: the arm's selected seed-0 checkpoint (bytes verified), core inference."""
    spec = ev_arms.load_matrix().arm(arm)
    ckpt = ev_arms.resolve_checkpoint(spec, STUDENT_SEED)
    ev_arms.verify_checkpoint_bytes(ckpt)
    inf = None if spec.inference is None else spec.inference.as_tuple()
    return Panel(arm, episode, str(ckpt.path), ckpt.sha256, ckpt.teacher, inf)


def run_gallery(
    root: Path,
    out_dir: Path,
    *,
    release_dir: Path,
    workers: int = 4,
    rule_md: Path | None = None,
) -> dict[str, Any]:
    """Select by :data:`RULE`, re-fly, assert, draw; return the manifest record.

    Args:
        root: ``results/e08`` (committed rows).
        out_dir: Where ``manifest.json`` (and the GIFs, when they total <= 10 MB) go.
        release_dir: Where the GIFs are written first (``artifacts/``; release assets if large).
        workers: Re-flight processes (<= 4).
        rule_md: ``RULE.md`` (default ``out_dir / "RULE.md"``); must state :data:`RULE`.

    Returns:
        The manifest record.

    Raises:
        RowsNotReadyError: If a panel's committed rows do not exist yet.
        AssertionError: If RULE.md differs from :data:`RULE` or a re-flown row differs.
    """
    md = (out_dir / "RULE.md") if rule_md is None else rule_md
    if rule_from_md(md.read_text(encoding="utf-8")) != RULE:
        raise AssertionError(f"{md} does not state ila.viz.gifs.RULE verbatim")
    rows, sources = committed_panel_rows(root)
    listed = ev_arms.list_episodes(LIST_NAME)
    picks = select_episodes(listed, rows)
    panels: list[Panel] = []
    for pick in picks.values():
        if pick is not None:
            e = listed[pick.position]
            panels.append(Panel("teacher", e))
            panels.extend(student_panel(arm, e) for arm in STUDENTS)
    results = fly_panels(panels, workers=workers)
    jobs = {arm: student_job(arm) for arm in STUDENTS}
    lines = {arm: committed_lines(root / jobs[arm].rel_path) for arm in STUDENTS}
    for r in results:  # every assertion passes before any GIF is written
        p = r.panel
        if p.panel == "teacher":
            check_panel(r, "", None)
        else:
            check_panel(r, lines[p.panel][(p.episode.ss, int(p.episode.index))], jobs[p.panel])
    gifs: list[dict[str, Any]] = []
    staging = release_dir / "gifs"
    k = 0
    for cat, pick in picks.items():
        if pick is None:
            continue
        group = results[k : k + len(PANELS)]
        k += len(PANELS)
        labels = [
            _panel_label(
                "ppo s0 (teacher)" if g.panel.panel == "teacher" else f"{g.panel.panel} s0",
                pick,
                pick.ss,
                pick.index,
                pick.outcomes[g.panel.panel],
            )
            for g in group
        ]
        name = f"{cat}__id_{pick.ss}_ep{pick.index}.gif"
        sha = write_gif(compose(group, labels), staging / name)
        gifs.append({"file": name, "sha256": sha, "bytes": (staging / name).stat().st_size})
    total = sum(int(g["bytes"]) for g in gifs)
    committed = total <= GIF_LIMIT_BYTES
    if committed:
        for g in gifs:
            (out_dir / g["file"]).write_bytes((staging / g["file"]).read_bytes())
    record = {
        "label": LABEL,
        "simulation_only": "PyBullet physics; frames rendered by PyBullet TinyRenderer; "
        "model scale lambda = 1/25",
        "rule_sha256": hashlib.sha256(RULE.encode("utf-8")).hexdigest(),
        "picks": {
            cat: None
            if p is None
            else {
                "ss": p.ss,
                "index": p.index,
                "episode_seed": p.episode_seed,
                "list_position": p.position,
                "outcomes": dict(p.outcomes),
            }
            for cat, p in picks.items()
        },
        "reflight": "every panel re-flown on the CPU fp32 single-sample path; every row asserted "
        "equal to its committed row before any GIF was written",
        "committed_row_sources": sources,
        "gifs": gifs,
        "total_bytes": total,
        "gifs_committed": committed,
        "release_location": None if committed else "release assets (not committed)",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return record
