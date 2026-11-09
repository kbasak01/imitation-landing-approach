"""Scoring: committed episode rows -> per-cell summaries, aggregates, contrasts, H1-H5.

Everything here is simulation only (PyBullet physics, PyBullet TinyRenderer frames). Success is
dimensionless in ``[0, 1]`` ("points" are hundredths); speeds m/s, lengths m and times s, all
model scale (lambda = 1/25: x5 for speeds and times, x25 for lengths at full scale); angles
degrees; actions in P8's normalised units.

Inputs (:class:`ScoreInputs`, :func:`default_inputs`)
----------------------------------------------------
* The matrix (:func:`ila.eval.arms.load_matrix`): every expected cell is one
  (tag, arm, seed, list, sea state, ``Ta``, ``K``, ensemble, appearance shift). A flown job's
  file holds its cells; a *shared* cell (``run_info.json`` ``shared_cells``,
  :class:`ila.eval.arms.SharedCell`) takes the named source job's rows of its sea states.
* The rows under ``results/e08/rows/`` (:mod:`ila.eval.matrix`). A file counts only once its
  sidecar ``.json`` exists, and its bytes must have the sidecar's SHA-256. A cell without
  its file (or whose shared source is not flown yet) is **missing** and is reported as such in
  every output; nothing is silently skipped. Every score runs on a partial matrix.
* P8's committed rows (:func:`ila.p8_adapter.committed_rows`): the teacher (``ppo`` seed 0,
  e07), ``pid_feedforward``, ``pid_track_descend``, ``oracle_gated`` and ``gated`` (e01), and the
  ``T_mix`` reference built from them by ``ila.data.mix.assign_rows`` (seed 20261023, per sea
  state), exactly H2's construction (P3-D1 §6 H2).
* H1's labelled-frame counts (:func:`load_labelled_frames`): each run's committed
  ``results/phase{4,5}/runs/<arm>/s<seed>/manifest.json``, ``data[<last stage>].train_frames``
  -- the frames trained on after the success filter (held-out frames excluded); for
  ``bc_dagger`` that is kept demo frames plus every rollout frame, cross-checked against
  ``results/phase5/dagger_iterations.csv`` ``aggregate_train_frames`` at iteration 5.
* H4's best end-to-end student (:func:`ila.eval.hypotheses.best_end_to_end`) from the
  committed ``select`` selection records only.
* Phase 9 (``latency_dir``, default ``results/latency``): H5(a) as ``ila.deploy.latency.score``
  wrote it (``h5a.csv``, ``delay.json``; :func:`ila.eval.hypotheses.read_h5a`, never
  re-derived), and ``delay.json``'s ``d`` per arm, cross-checked against the ``d`` the
  ``latency_arm`` rows were flown with. H5(b) is scored from the ``latency_arm`` rows and the
  core ``dp`` rows (:func:`ila.eval.hypotheses.score_h5b`); the latency arm's other cells
  (``bc_dagger`` SS4-SS6, ``dp`` SS4 and SS6) are unscored context
  (:func:`ila.eval.hypotheses.latency_context`). ``latency_arm`` cells are their own groups
  (tag ``latency_arm``) in every table: never mixed into a core section.

Outputs (deterministic: sorted rows, fixed float format :func:`fmt`, no timestamps)
---------------------------------------------------------------------------------
``summary.csv`` (per cell and seed), ``baselines.csv``, ``aggregate.csv`` (per cell across
seeds), ``contrasts.csv``, ``hypotheses.csv`` + ``hypotheses.json``, ``above_teacher.csv``,
``missing.csv``, ``h2_touchdown_times.csv`` (the episodes behind H2, for its figure),
``bc_dart_s1_4.csv`` (the ``bc_dart`` seeds 1-4 sensitivity, P4-D5 item 2:
:func:`dart_s1_4_rows`; printed beside the 5-seed results, never substituted for them) and
``score_info.json`` (what was read, with SHA-256s).

Statistics are P8's through :mod:`ila.eval.stats` (Wilson 95 % per cell and seed; IQM with a
stratified-bootstrap 95 % CI over seeds, 2 000 replicates; P8 ``paired_bootstrap`` and
``paired_iqm_difference``, 10 000 replicates; P8 ``paired_relative_p95`` over touched-down
episodes only) plus the three P3-D1 §1 exceptions inside the H2-H4 scorers; bootstrap seed
20261023 always. The per-cell reduction is P8's ``cell_metrics`` (via the adapter), so every
summary column has P8's name and meaning. No multiplicity correction is applied.
"""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ila import p8_adapter
from ila.eval import arms as ev_arms
from ila.eval import h2_select
from ila.eval import hypotheses as hy
from ila.eval import stats as st

__all__ = [
    "AGGREGATE_COLUMNS",
    "ABOVE_TEACHER_COLUMNS",
    "BASELINE_COLUMNS",
    "BASELINE_METHODS",
    "CAVEATS_SEE",
    "CONTRAST_COLUMNS",
    "DART_S1_4_COLUMNS",
    "DART_S1_4_FILE",
    "DART_S1_4_LABEL",
    "DART_S1_4_SEEDS",
    "GROUP_COLUMNS",
    "HYPOTHESIS_COLUMNS",
    "KEY_COLUMNS",
    "LOSS_CLASSES",
    "MISSING_COLUMNS",
    "OUTCOMES",
    "OUT_ROOT",
    "SUMMARY_COLUMNS",
    "Baselines",
    "CellData",
    "CellIndex",
    "CellKey",
    "GroupKey",
    "ScoreInputError",
    "ScoreInputs",
    "above_teacher_rows",
    "aggregate_rows",
    "as_bool",
    "as_float",
    "baseline_rows",
    "cell_summary",
    "dart_s1_4_rows",
    "default_inputs",
    "dump_json",
    "expected_cells",
    "fmt",
    "load_baselines",
    "load_cells",
    "load_labelled_frames",
    "outside_training",
    "read_csv",
    "reading",
    "run_score",
    "summary_rows",
    "teacher_rows",
    "tmix_reference",
    "write_csv",
]

type Row = Mapping[str, Any]
type FloatArray = npt.NDArray[np.float64]

#: Committed output root of the matrix and of every scored file.
OUT_ROOT: Final[Path] = p8_adapter.REPO_ROOT / "results" / "e08"

#: P8's outcome classes, in P8's matching order (P3-D1 §1).
OUTCOMES: Final[tuple[str, ...]] = (
    "crash",
    "off_pad",
    "hard_landing",
    "bounce",
    "success",
    "timeout",
)

#: Every class except ``success``: the losses counted per 1 000 seed-episodes.
LOSS_CLASSES: Final[tuple[str, ...]] = tuple(o for o in OUTCOMES if o != "success")

#: P8 baselines and teacher components: method -> (experiment, run seed, role).
BASELINE_METHODS: Final[Mapping[str, tuple[str, int, str]]] = {
    "ppo": ("e07", 0, "teacher T_ppo (P8 ppo seed 0)"),
    "pid_feedforward": ("e01", 0, "baseline; teacher T_pid"),
    "pid_track_descend": ("e01", 0, "baseline"),
    "oracle_gated": ("e01", 0, "baseline: commit-timing oracle (privileged), not a bound"),
    "gated": ("e01", 0, "T_mix component (H2 reference)"),
}

#: The always-printed baselines, in print order (P3-D1 §3).
ALWAYS_PRINTED: Final[tuple[str, ...]] = (
    "ppo",
    "pid_feedforward",
    "pid_track_descend",
    "oracle_gated",
)

#: The synthetic method name of the assembled ``T_mix`` reference in ``baselines.csv``.
TMIX: Final[str] = "T_mix_reference"

#: Teacher label -> P8 method whose committed rows are that teacher (``T_mix``: assembled).
TEACHER_METHOD: Final[Mapping[str, str]] = {
    "T_ppo": "ppo",
    "T_pid": "pid_feedforward",
    "T_mix": TMIX,
}

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "tag",
    "arm",
    "seed",
    "list",
    "ss",
    "Ta",
    "K",
    "ensemble",
    "appearance_shift",
)
GROUP_COLUMNS: Final[tuple[str, ...]] = tuple(c for c in KEY_COLUMNS if c != "seed")

_COUNT_COLUMNS: Final[tuple[str, ...]] = tuple(f"n_{o}" for o in OUTCOMES)
_DIAG_COLUMNS: Final[tuple[str, ...]] = ("teacher_agreement_mean", "inference_calls_mean")


def _unique(*columns: str) -> tuple[str, ...]:
    """Drop repeated column names, keeping the first occurrence.

    ``n_success`` is both one of the six outcome counts and one of P8's
    ``CELL_METRIC_COLUMNS``; the CSV carries it once (same value).
    """
    return tuple(dict.fromkeys(columns))


SUMMARY_COLUMNS: Final[tuple[str, ...]] = _unique(
    *KEY_COLUMNS,
    "status",
    "rows_from",
    "teacher",
    "outside_training",
    *_COUNT_COLUMNS,
    *p8_adapter.CELL_METRIC_COLUMNS,
    *_DIAG_COLUMNS,
)

BASELINE_COLUMNS: Final[tuple[str, ...]] = _unique(
    "method",
    "role",
    "experiment",
    "run_seed",
    "list",
    "ss",
    "outside_training",
    *_COUNT_COLUMNS,
    *p8_adapter.CELL_METRIC_COLUMNS,
)

AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    *GROUP_COLUMNS,
    "status",
    "teacher",
    "outside_training",
    "n_seeds",
    "n_seeds_expected",
    "seeds",
    "n_episodes_per_seed",
    "success_iqm",
    "success_iqm_lo",
    "success_iqm_hi",
    "success_mean",
    "success_seed_min",
    "success_seed_max",
    "success_per_seed",
    *_COUNT_COLUMNS,
    *(f"per_1000_{o}" for o in LOSS_CLASSES),
    "iqm_reps",
    "bootstrap_seed",
)

CONTRAST_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "kind",
    "hypothesis",
    "list",
    "ss",
    "a",
    "b",
    "statistic",
    "estimate",
    "ci_lo",
    "ci_hi",
    "units",
    "n_seeds_a",
    "n_seeds_b",
    "n_episodes",
    "reading",
    "status",
    "flip_floor",
    "note",
    "inputs",
)

#: Where ``hypotheses.csv``'s caveats are (the sentences are long and carry commas and quotes;
#: the CSV points to the record instead of flattening them).
CAVEATS_SEE: Final[str] = "hypotheses.json"

#: ``ceiling_limited`` is P3-D1 §6's reference-arm ceiling only
#: (:attr:`ila.eval.hypotheses.HypothesisResult.ceiling_limited`); ``notes`` carries reading aids
#: such as H4's I-floor; ``caveats_see`` names the file with every caveat verbatim.
HYPOTHESIS_COLUMNS: Final[tuple[str, ...]] = (
    "hypothesis",
    "cell",
    "status",
    "verdict",
    "verdict_note",
    "estimate",
    "ci_lo",
    "ci_hi",
    "units",
    "magnitude",
    "statistic",
    "contrast",
    "n_seeds",
    "n_episodes",
    "flip_floor",
    "inputs",
    "ceiling_limited",
    "notes",
    "caveats_see",
)

ABOVE_TEACHER_COLUMNS: Final[tuple[str, ...]] = (
    *GROUP_COLUMNS,
    "teacher",
    "teacher_source",
    "teacher_k",
    "teacher_n",
    "teacher_rate",
    "status",
    "success_iqm",
    "iqm_margin",
    "iqm_above",
    "seeds_above",
    "max_margin",
    "audit_file",
)

MISSING_COLUMNS: Final[tuple[str, ...]] = (*KEY_COLUMNS, "rows_file", "reason")

_H2_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "seed",
    "index",
    "episode_seed",
    "outcome",
    "touchdown_contact",
    "time_to_touchdown_s",
)

#: The unscored-reading words used throughout (P3-D1 §6, claim-trace skill).
UNPAIRED: Final[str] = "unpaired reading"


class ScoreInputError(ValueError):
    """Committed inputs are inconsistent (a SHA-256, a row count, a tag or a seed differs)."""


# --------------------------------------------------------------------------- formatting


def fmt(value: Any) -> str:
    """Format one CSV cell deterministically.

    ``None`` -> ``""``; bools ``True``/``False``; ints as is; floats with 12 significant
    digits (``nan``/``inf`` spelled out); strings unchanged.
    """
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return "True" if bool(value) else "False"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        x = float(value)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        out = format(x, ".12g")
        return "0" if out == "-0" else out
    return str(value)


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows (LF line ends, every cell through :func:`fmt`); return the text."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in columns])
    text = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a committed CSV as string cells (no type coercion)."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return fmt(value) if not math.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_clean(v) for v in value]
    return value


def dump_json(path: Path, record: Any) -> str:
    """Write JSON with sorted keys, 2-space indent, non-finite floats as strings."""
    text = json.dumps(_json_clean(record), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_float(value: Any) -> float:
    """A row cell as a float (``""``/``None`` -> NaN; ``"nan"`` parses to NaN)."""
    if value is None:
        return math.nan
    if isinstance(value, str):
        text = value.strip()
        return math.nan if text in ("", "None") else float(text)
    return float(value)


def as_bool(value: Any) -> bool:
    """A row cell as a bool (``True``/``true``/``1``; anything else is False)."""
    return str(value).strip() in ("True", "true", "1")


# --------------------------------------------------------------------------- keys


@dataclass(frozen=True, order=False)
class GroupKey:
    """One cell across seeds: (tag, arm, list, sea state, inference, appearance shift).

    Attributes:
        tag: Job tag (:data:`ila.eval.arms.TAGS`).
        arm: Arm name.
        list_name: P8 frozen list.
        ss: Sea-state cell of that list (``SS3``-``SS6`` or ``static``).
        Ta: Executed actions per chunk (diffusion arms), else ``None``.
        K: DDIM steps (diffusion arms), else ``None``.
        ensemble: Temporal ensembling (diffusion arms), else ``None``.
        shift: Appearance-shift variant, or ``None`` for the committed look.
    """

    tag: str
    arm: str
    list_name: str
    ss: str
    Ta: int | None
    K: int | None
    ensemble: bool | None
    shift: str | None

    def record(self) -> dict[str, Any]:
        """The :data:`GROUP_COLUMNS` values."""
        return {
            "tag": self.tag,
            "arm": self.arm,
            "list": self.list_name,
            "ss": self.ss,
            "Ta": self.Ta,
            "K": self.K,
            "ensemble": self.ensemble,
            "appearance_shift": self.shift,
        }

    @property
    def label(self) -> str:
        """Short human label, e.g. ``dp_Ta dp id SS6 Ta1 K10 ens0``."""
        parts = [self.tag, self.arm, self.list_name, self.ss]
        if self.Ta is not None:
            parts.append(f"Ta{self.Ta} K{self.K} ens{int(bool(self.ensemble))}")
        if self.shift is not None:
            parts.append(f"shift-{self.shift}")
        return " ".join(parts)

    @property
    def slug(self) -> str:
        """File-name-safe label (above-teacher audit files)."""
        return "__".join(p.replace(" ", "-") for p in self.label.split(" "))


@dataclass(frozen=True)
class CellKey:
    """One expected cell for one seed.

    Attributes:
        group: The :class:`GroupKey`.
        seed: Training seed of the run.
    """

    group: GroupKey
    seed: int

    def record(self) -> dict[str, Any]:
        """The :data:`KEY_COLUMNS` values."""
        return {**self.group.record(), "seed": self.seed}


@dataclass(frozen=True)
class CellData:
    """One expected cell's rows (or their absence).

    Attributes:
        key: The cell.
        rows_file: Rows file relative to the output root (for a shared cell: its source's).
        shared_reason: Why a shared cell's rows are its source's, or ``None`` if flown.
        rows: The cell's rows in list order (string cells), or ``None`` if missing.
        reason: Why it is missing (``""`` when present).
    """

    key: CellKey
    rows_file: str
    shared_reason: str | None
    rows: tuple[dict[str, str], ...] | None
    reason: str = ""

    @property
    def present(self) -> bool:
        """Whether the rows exist."""
        return self.rows is not None

    @property
    def status(self) -> str:
        """``flown`` / ``shared`` / ``missing``."""
        if self.rows is None:
            return "missing"
        return "flown" if self.shared_reason is None else "shared"


@dataclass(frozen=True)
class CellIndex:
    """Every expected cell, present or missing, in report order.

    Attributes:
        cells: ``CellKey -> CellData``, in report order.
        groups: ``GroupKey -> expected seeds`` (ascending), in report order.
        files: Rows files read: relative path -> ``{"rows_gz_sha256", "n_rows"}``.
        latency: ``latency_arm`` groups -> their ``d`` (control steps); every other group's
            rows carry ``latency_steps`` 0.
    """

    cells: Mapping[CellKey, CellData]
    groups: Mapping[GroupKey, tuple[int, ...]]
    files: Mapping[str, Mapping[str, Any]]
    latency: Mapping[GroupKey, int] = field(default_factory=dict)

    def group_cells(self, g: GroupKey) -> list[CellData]:
        """The group's cells, one per expected seed, ascending seed."""
        return [self.cells[CellKey(g, s)] for s in self.groups[g]]

    def group_rows(self, g: GroupKey) -> list[dict[str, str]] | None:
        """All seeds' rows of a complete group (seed-major), or ``None`` if any is missing."""
        cells = self.group_cells(g)
        if not all(c.present for c in cells):
            return None
        return [r for c in cells for r in (c.rows or ())]

    def find(
        self,
        arm: str,
        list_name: str,
        ss: str,
        *,
        tag: str | None = None,
        inference: tuple[int, int, bool] | None = None,
        shift: str | None = None,
    ) -> GroupKey | None:
        """The one group matching (arm, list, ss, inference, shift[, tag]); ``None`` if none."""
        hits = [
            g
            for g in self.groups
            if g.arm == arm
            and g.list_name == list_name
            and g.ss == ss
            and g.shift == shift
            and (tag is None or g.tag == tag)
            and (inference is None or (g.Ta, g.K, g.ensemble) == inference)
        ]
        if inference is None and tag is None:
            hits = [g for g in hits if g.tag not in ("dp_Ta", "dp_K", ev_arms.LATENCY_TAG)]
        if len(hits) > 1:
            raise ScoreInputError(f"{arm} {list_name} {ss}: {len(hits)} groups match")
        return hits[0] if hits else None


def _inference(inf: ev_arms.Inference | None) -> tuple[int | None, int | None, bool | None]:
    return (None, None, None) if inf is None else (inf.Ta, inf.K, inf.ensemble)


def _ss_rank(ss: str) -> tuple[int, str]:
    return (1, ss) if ss == "static" else (0, ss)


def _group_sort(matrix: ev_arms.Matrix) -> Callable[[GroupKey], tuple[Any, ...]]:
    tag_rank = {t: i for i, t in enumerate(ev_arms.TAGS)}
    arm_rank = {a.arm: i for i, a in enumerate(matrix.arms)}
    list_rank = {n: i for i, n in enumerate(matrix.lists_all)}

    def key(g: GroupKey) -> tuple[Any, ...]:
        return (
            tag_rank[g.tag],
            arm_rank[g.arm],
            list_rank[g.list_name],
            _ss_rank(g.ss),
            -1 if g.Ta is None else g.Ta,
            -1 if g.K is None else g.K,
            -1 if g.ensemble is None else int(g.ensemble),
            "" if g.shift is None else g.shift,
        )

    return key


def expected_cells(matrix: ev_arms.Matrix) -> dict[CellKey, tuple[ev_arms.Job, str | None]]:
    """Every expected cell -> (the job whose rows hold it, shared reason or ``None``)."""
    out: dict[CellKey, tuple[ev_arms.Job, str | None]] = {}
    for j in matrix.jobs:
        ta, k, ens = _inference(j.inference)
        for ss in j.sea_states:
            g = GroupKey(j.tag, j.arm, j.list_name, ss, ta, k, ens, j.appearance_shift)
            key = CellKey(g, j.seed)
            if key in out:
                raise ScoreInputError(f"cell {key} is expected twice")
            out[key] = (j, None)
    for s in matrix.shared:
        ta, k, ens = _inference(s.inference)
        for ss in s.sea_states:
            g = GroupKey(s.tag, s.arm, s.list_name, ss, ta, k, ens, None)
            key = CellKey(g, s.seed)
            if key in out:
                raise ScoreInputError(f"shared cell {key} is also flown")
            out[key] = (s.source, s.reason)
    return out


def _sidecar(root: Path, rel: str) -> Path:
    return root / (rel.removesuffix(".csv.gz") + ".json")


def _read_file(root: Path, rel: str) -> tuple[list[dict[str, str]], dict[str, Any]] | None:
    """Read one finished rows file (sidecar present) and verify its bytes; ``None`` if not done.

    Raises:
        ScoreInputError: If the bytes or row count disagree with the sidecar.
    """
    side = _sidecar(root, rel)
    path = root / rel
    if not side.is_file():
        return None
    rec = json.loads(side.read_text(encoding="utf-8"))
    data = path.read_bytes()
    if _sha256(data) != rec.get("rows_gz_sha256"):
        raise ScoreInputError(f"{rel}: SHA-256 differs from its sidecar")
    text = gzip.decompress(data).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if len(rows) != int(rec.get("n_rows", -1)):
        raise ScoreInputError(f"{rel}: {len(rows)} rows, sidecar says {rec.get('n_rows')}")
    return rows, {"rows_gz_sha256": rec["rows_gz_sha256"], "n_rows": len(rows)}


def _check_cell(
    key: CellKey, job: ev_arms.Job, rows: Sequence[Mapping[str, str]], n_per_cell: int
) -> None:
    where = f"{job.rel_path.as_posix()} {key.group.ss}"
    if len(rows) != n_per_cell:
        raise ScoreInputError(f"{where}: {len(rows)} rows, expected {n_per_cell}")
    ta, k, ens = _inference(job.inference)
    want = {
        "arm": job.arm,
        "student_seed": str(job.seed),
        "regime": job.list_name,
        "tag": job.tag,
        "appearance_shift": job.appearance_shift or "",
        "Ta": "" if ta is None else str(ta),
        "K": "" if k is None else str(k),
        "ensemble": "" if ens is None else str(ens),
        "latency_steps": str(job.latency_steps or 0),
    }
    for col, value in want.items():
        bad = {str(r[col]) for r in rows} - {value}
        if bad:
            raise ScoreInputError(f"{where}: column {col} {sorted(bad)[:3]} != {value!r}")
    idx = [int(r["index"]) for r in rows]
    if len(set(idx)) != len(idx):
        raise ScoreInputError(f"{where}: an episode appears twice")


def load_cells(
    root: Path,
    matrix: ev_arms.Matrix,
    *,
    episodes_per_cell: int = ev_arms.EPISODES_PER_CELL,
) -> CellIndex:
    """Index every expected cell of ``matrix`` against the rows under ``root``.

    Args:
        root: The output root (``results/e08``; rows under ``root/rows``).
        matrix: The expanded matrix.
        episodes_per_cell: Rows every present cell must hold (200 on a frozen list).

    Returns:
        The :class:`CellIndex` (missing cells included, with their reason).

    Raises:
        ScoreInputError: On a SHA-256, row-count, tag, seed or duplicate mismatch.
    """
    expected = expected_cells(matrix)
    files: dict[str, tuple[list[dict[str, str]], dict[str, Any]] | None] = {}
    for job, _ in expected.values():
        rel = job.rel_path.as_posix()
        if rel not in files:
            files[rel] = _read_file(root, rel)
    order = _group_sort(matrix)
    groups: dict[GroupKey, list[int]] = {}
    latency: dict[GroupKey, int] = {}
    for key, (job, _) in expected.items():
        groups.setdefault(key.group, []).append(key.seed)
        d = job.latency_steps
        if d is not None and latency.setdefault(key.group, d) != d:
            raise ScoreInputError(f"{key.group.label}: two delays in one group")
    sorted_groups = {g: tuple(sorted(groups[g])) for g in sorted(groups, key=order)}
    cells: dict[CellKey, CellData] = {}
    for g, seeds in sorted_groups.items():
        for seed in seeds:
            key = CellKey(g, seed)
            job, shared = expected[key]
            rel = job.rel_path.as_posix()
            got = files[rel]
            if got is None:
                reason = (
                    "not yet flown (no sidecar)"
                    if shared is None
                    else f"shared cell; its source {rel} is not yet flown"
                )
                cells[key] = CellData(key, rel, shared, None, reason)
                continue
            rows = tuple(r for r in got[0] if r["ss"] == g.ss)
            _check_cell(key, job, rows, episodes_per_cell)
            cells[key] = CellData(key, rel, shared, rows)
    done = {rel: v[1] for rel, v in sorted(files.items()) if v is not None}
    return CellIndex(cells=cells, groups=sorted_groups, files=done, latency=latency)


# --------------------------------------------------------------------------- baselines


@dataclass(frozen=True)
class Baselines:
    """P8's committed rows of the teacher, the baselines and ``gated`` (read-only).

    Attributes:
        rows: Method -> its committed rows (one run seed), P8's text cells.
        provenance: :meth:`ila.p8_adapter.CommittedRows.provenance` per method.
    """

    rows: Mapping[str, tuple[dict[str, str], ...]]
    provenance: tuple[Mapping[str, Any], ...]

    def cell(self, method: str, list_name: str, ss: str) -> list[dict[str, str]]:
        """One method's committed rows in one cell, ascending ``index``."""
        out = [r for r in self.rows[method] if r["regime"] == list_name and r["ss"] == ss]
        return sorted(out, key=lambda r: int(r["index"]))


def load_baselines() -> Baselines:
    """Read P8's committed rows of :data:`BASELINE_METHODS` (nothing flown)."""
    rows: dict[str, tuple[dict[str, str], ...]] = {}
    prov = []
    for method, (exp, seed, _) in BASELINE_METHODS.items():
        cr = p8_adapter.committed_rows(exp, method, seed)
        rows[method] = cr.rows
        prov.append(cr.provenance())
    return Baselines(rows=rows, provenance=tuple(prov))


def tmix_reference(
    listed: Sequence[p8_adapter.ListedEpisode], baselines: Baselines
) -> list[tuple[str, dict[str, str]]]:
    """The ``T_mix`` reference of one cell: ``(assigned policy, its committed row)`` per episode.

    ``ila.data.mix.assign_rows(20261023, ...)`` per sea state through
    :func:`ila.eval.h2_select.tmix_assignment` (H2's construction, P3-D1 §6 H2); each episode
    takes P8's committed row of its assigned policy (``ppo`` seed 0 from e07, ``gated`` from
    e01). Nothing is flown.

    Args:
        listed: One cell's listed episodes, in list order.
        baselines: The committed rows.

    Returns:
        One ``(policy, row)`` per listed episode, ascending ``index``.

    Raises:
        ScoreInputError: If a committed row is missing or its episode seed differs.
    """
    if not listed:
        return []
    assigned = h2_select.tmix_assignment(listed)
    out = []
    for e, policy in sorted(zip(listed, assigned, strict=True), key=lambda p: p[0].index):
        rows = {int(r["index"]): r for r in baselines.cell(policy, e.regime, e.ss)}
        r = rows.get(int(e.index))
        if r is None or int(r["episode_seed"]) != int(e.episode_seed):
            raise ScoreInputError(f"T_mix: no committed {policy} row for {e.regime} {e.ss}")
        out.append((policy, r))
    return out


def teacher_rows(
    teacher: str,
    list_name: str,
    ss: str,
    baselines: Baselines,
    listed: Callable[[str, str], Sequence[p8_adapter.ListedEpisode]],
) -> list[dict[str, str]]:
    """The committed rows of an arm's own teacher in one cell (``T_mix``: the reference)."""
    if teacher == "T_mix":
        return [r for _, r in tmix_reference(listed(list_name, ss), baselines)]
    return baselines.cell(TEACHER_METHOD[teacher], list_name, ss)


def outside_training(list_name: str, ss: str) -> bool:
    """Whether a cell is shaded as outside the training distribution (SS6 or 90°).

    P3-D1 §2: no SS6 and no 90° heading in any training, tuning or selection; the skill's
    shading rule is "SS6 and 90° cells". ``unseen_heading`` is the 90° list.
    """
    return ss == "SS6" or list_name == "unseen_heading"


# --------------------------------------------------------------------------- per-cell summaries


def _counts(rows: Sequence[Row]) -> dict[str, int]:
    out = dict.fromkeys(OUTCOMES, 0)
    for r in rows:
        out[str(r["outcome"])] += 1
    return {f"n_{o}": n for o, n in out.items()}


def cell_summary(rows: Sequence[Row]) -> dict[str, Any]:
    """P8's ``cell_metrics`` of one cell plus the six counts (P8 order)."""
    m = p8_adapter.cell_metrics(rows)
    return {**_counts(rows), **asdict(m)}


def _diag_means(rows: Sequence[Row]) -> dict[str, float]:
    out = {}
    for col, name in (
        ("teacher_agreement", _DIAG_COLUMNS[0]),
        ("inference_calls", _DIAG_COLUMNS[1]),
    ):
        vals = np.asarray([as_float(r.get(col)) for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[name] = float(finite.mean()) if finite.size else math.nan
    return out


def summary_rows(index: CellIndex) -> list[dict[str, Any]]:
    """One :data:`SUMMARY_COLUMNS` row per expected cell (missing cells: status only)."""
    out = []
    for key, cell in index.cells.items():
        g = key.group
        base = {
            **key.record(),
            "status": cell.status,
            "rows_from": cell.rows_file if cell.shared_reason else "",
            "teacher": ev_arms.expected_teacher(g.arm),
            "outside_training": outside_training(g.list_name, g.ss),
        }
        if cell.rows is not None:
            base.update(cell_summary(cell.rows))
            base.update(_diag_means(cell.rows))
        out.append(base)
    return out


def baseline_rows(
    baselines: Baselines,
    list_cells: Mapping[str, Sequence[str]],
    listed: Callable[[str, str], Sequence[p8_adapter.ListedEpisode]],
    tmix_cells: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """``baselines.csv``: each P8 method per cell, then the ``T_mix`` reference cells."""
    out = []
    for method, (exp, seed, role) in BASELINE_METHODS.items():
        for list_name, cells in list_cells.items():
            for ss in cells:
                rows = baselines.cell(method, list_name, ss)
                if not rows:
                    continue
                out.append(
                    {
                        "method": method,
                        "role": role,
                        "experiment": exp,
                        "run_seed": seed,
                        "list": list_name,
                        "ss": ss,
                        "outside_training": outside_training(list_name, ss),
                        **cell_summary(rows),
                    }
                )
    for list_name, ss in tmix_cells:
        ref = tmix_reference(listed(list_name, ss), baselines)
        if not ref:
            continue
        n_ppo = sum(p == "ppo" for p, _ in ref)
        out.append(
            {
                "method": TMIX,
                "role": f"teacher T_mix: assign_rows(20261023) {n_ppo} ppo s0 (e07) / "
                f"{len(ref) - n_ppo} gated (e01)",
                "experiment": "e07+e01",
                "run_seed": 0,
                "list": list_name,
                "ss": ss,
                "outside_training": outside_training(list_name, ss),
                **cell_summary([r for _, r in ref]),
            }
        )
    return out


# --------------------------------------------------------------------------- aggregate


def _rate(rows: Sequence[Row]) -> float:
    return sum(str(r["outcome"]) == "success" for r in rows) / len(rows)


def aggregate_rows(index: CellIndex) -> list[dict[str, Any]]:
    """One :data:`AGGREGATE_COLUMNS` row per group (IQM only when every seed is present)."""
    stats_cfg = st.matrix_statistics()
    out = []
    for g, seeds in index.groups.items():
        cells = index.group_cells(g)
        present = [c for c in cells if c.rows is not None]
        rec: dict[str, Any] = {
            **g.record(),
            "teacher": ev_arms.expected_teacher(g.arm),
            "outside_training": outside_training(g.list_name, g.ss),
            "n_seeds": len(present),
            "n_seeds_expected": len(seeds),
            "seeds": "|".join(str(c.key.seed) for c in present),
        }
        if len(present) < len(seeds):
            missing = [str(c.key.seed) for c in cells if c.rows is None]
            rec["status"] = f"incomplete: seeds {'|'.join(missing)} missing"
            out.append(rec)
            continue
        rec.update(
            status="complete",
            **_seed_stats(g, present),
            iqm_reps=stats_cfg.iqm_reps,
            bootstrap_seed=stats_cfg.bootstrap_seed,
        )
        out.append(rec)
    return out


def _seed_stats(g: GroupKey, present: Sequence[CellData]) -> dict[str, Any]:
    """IQM [stratified-bootstrap CI] over ``present``'s seeds, rates and outcome counts.

    The one reduction behind ``aggregate.csv`` and the ``bc_dart`` seeds 1-4 sensitivity
    (:func:`dart_s1_4_rows`): P8's IQM over seeds of each seed's success rate (dimensionless),
    2 000 replicates, seed 20261023 (:func:`ila.eval.stats.iqm_ci`).
    """
    rates = [_rate(c.rows or ()) for c in present]
    pooled = [r for c in present for r in (c.rows or ())]
    n_eps = {len(c.rows or ()) for c in present}
    if len(n_eps) != 1:
        raise ScoreInputError(f"{g.label}: seeds have different episode counts")
    est = st.iqm_ci(np.asarray(rates, dtype=np.float64).reshape(-1, 1))
    counts = _counts(pooled)
    total = len(pooled)
    return {
        "n_episodes_per_seed": n_eps.pop(),
        "success_iqm": float(est.point),
        "success_iqm_lo": float(est.lo),
        "success_iqm_hi": float(est.hi),
        "success_mean": float(np.mean(rates)),
        "success_seed_min": min(rates),
        "success_seed_max": max(rates),
        "success_per_seed": "|".join(fmt(x) for x in rates),
        **counts,
        **{f"per_1000_{o}": 1000.0 * counts[f"n_{o}"] / total for o in LOSS_CLASSES},
    }


# --------------------------------------------------------------------------- matrices


@dataclass(frozen=True)
class ScoreInputs:
    """Everything :func:`run_score` reads.

    Attributes:
        matrix: The expanded matrix.
        root: Output root holding ``rows/`` and ``run_info.json``.
        baselines: P8's committed rows.
        listed: ``(list, ss) -> listed episodes`` of that cell, in list order (keys only).
        labelled_frames: H1's labelled training frames per arm, per seed in seed order.
        labelled_frames_sources: What they were read from (path -> SHA-256 / note).
        best_end_to_end: ``(arm, reason)`` from committed ``select`` scores (P3-D1 §6 H4).
        best_sources: What the best end-to-end choice was read from.
        episodes_per_cell: Rows per present cell (200 on a frozen list).
        list_cells: List -> its cells in list order (for ``baselines.csv``).
        extra: Extra ``score_info.json`` keys.
        latency_dir: ``results/latency`` (H5(a)'s ``h5a.csv`` and ``delay.json``), or
            ``None`` (H5(a) not scored; H5(b) without the delay record beside it).
        select_scores: The committed ``select`` scores behind ``best_end_to_end``
            (:func:`ila.eval.hypotheses.load_select_scores`), for the ``bc_dart`` seeds 1-4
            value printed beside H4's best end-to-end rule (P4-D5 item 2); empty: not printed.
    """

    matrix: ev_arms.Matrix
    root: Path
    baselines: Baselines
    listed: Callable[[str, str], Sequence[p8_adapter.ListedEpisode]]
    labelled_frames: Mapping[str, Sequence[int]]
    labelled_frames_sources: Mapping[str, str]
    best_end_to_end: tuple[str, str]
    best_sources: Mapping[str, str]
    list_cells: Mapping[str, Sequence[str]]
    episodes_per_cell: int = ev_arms.EPISODES_PER_CELL
    extra: Mapping[str, Any] = field(default_factory=dict)
    latency_dir: Path | None = None
    select_scores: Mapping[str, hy.SelectScore] = field(default_factory=dict)


def _seeds_of(index: CellIndex, g: GroupKey) -> tuple[int, ...]:
    return index.groups[g]


def _student(index: CellIndex, g: GroupKey | None) -> tuple[hy.ArmMatrix | None, str]:
    """The group's :class:`ila.eval.hypotheses.ArmMatrix`, or ``(None, why missing)``."""
    if g is None:
        return None, "cell not in the matrix"
    rows = index.group_rows(g)
    if rows is None:
        missing = [str(c.key.seed) for c in index.group_cells(g) if not c.present]
        return None, f"{g.label}: seeds {'|'.join(missing)} missing"
    return (
        hy.student_matrix(
            rows,
            arm=g.arm,
            cell=(g.list_name, g.ss),
            seeds=_seeds_of(index, g),
            ta=g.Ta,
            k=g.K,
            ensemble=g.ensemble,
            latency_steps=index.latency.get(g, 0),
        ),
        "",
    )


def _speed_matrix(rows: Sequence[Row], seed_col: str, m: hy.ArmMatrix) -> FloatArray:
    """Closing speeds along the deck normal as ``m``'s seeds x episodes, m/s model scale.

    ``max(0, -rel_vz_normal_m_s)`` where the contact detector fired, else NaN: P8's
    construction (``rld.eval.hypotheses``), never imputed.
    """
    by: dict[tuple[int, int], float] = {}
    for r in rows:
        v = (
            max(0.0, -as_float(r["rel_vz_normal_m_s"]))
            if as_bool(r["touchdown_contact"])
            else math.nan
        )
        by[(int(float(r[seed_col])), int(r["index"]))] = v
    return np.asarray([[by[(s, i)] for i, _ in m.episodes] for s in m.seeds], dtype=np.float64)


@dataclass(frozen=True)
class _Side:
    """One side of a contrast: its matrix, its rows (for speeds) and their seed column."""

    m: hy.ArmMatrix
    rows: Sequence[Row]
    seed_col: str
    label: str = ""

    @property
    def speeds(self) -> FloatArray:
        return _speed_matrix(self.rows, self.seed_col, self.m)


def _student_side(index: CellIndex, g: GroupKey | None) -> tuple[_Side | None, str]:
    m, why = _student(index, g)
    if m is None or g is None:
        return None, why
    return _Side(m, index.group_rows(g) or [], "student_seed", g.label), ""


def _p8_side(rows: Sequence[Row], method: str, cell: tuple[str, str]) -> _Side:
    label = f"P8 committed {method} {cell[0]} {cell[1]}"
    return _Side(hy.p8_matrix(rows, method=method, cell=cell), rows, "run_seed", label)


def _inputs(*sides: _Side) -> str:
    """The input cells of a contrast (``GroupKey.label`` / P8 rows), ``; ``-joined."""
    return "; ".join(s.label for s in sides)


# --------------------------------------------------------------------------- contrasts


def reading(lo: float, hi: float, a: str, b: str, *, up: str = "above", down: str = "below") -> str:
    """Two-sided reading of a 95 % CI (P3-D1 §6).

    Returns:
        Which side of 0 the CI lies on, or that it does not separate.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "undefined"
    if lo > 0.0:
        return f"separates: {a} {up} {b} (CI wholly above 0)"
    if hi < 0.0:
        return f"separates: {a} {down} {b} (CI wholly below 0)"
    return "does not separate (CI includes 0)"


def _flip(ss: str) -> str:
    return p8_adapter.FLIP_FLOOR_SS6.sentence if ss == "SS6" else ""


def _contrast_record(
    name: str,
    kind: str,
    hypothesis: str,
    cell: tuple[str, str],
    a: str,
    b: str,
    **values: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "hypothesis": hypothesis,
        "list": cell[0],
        "ss": cell[1],
        "a": a,
        "b": b,
        "flip_floor": _flip(cell[1]),
        **values,
    }


def _missing_contrast(
    name: str, kind: str, hypothesis: str, cell: tuple[str, str], a: str, b: str, why: str
) -> dict[str, Any]:
    return _contrast_record(name, kind, hypothesis, cell, a, b, status=f"missing: {why}")


_P8_PAIRED: Final[str] = (
    "P8 paired_bootstrap (seeds per arm, shared episodes; 10 000 reps, seed 20261023)"
)
_P8_IQM: Final[str] = (
    "P8 paired_iqm_difference (seeds per arm, shared episodes; 10 000 reps, seed 20261023)"
)
_P8_P95: Final[str] = (
    "P8 paired_relative_p95 r = 1 - p95(a)/p95(b), touched-down episodes only, never imputed "
    "(seeds per arm, shared episodes; 10 000 reps, seed 20261023)"
)
_UNITS_SUCCESS: Final[str] = "success difference, dimensionless (x100 = points)"
_UNITS_R: Final[str] = "dimensionless; positive = a lands softer (lower p95 closing speed)"


def _paired(
    name: str, kind: str, hypothesis: str, a: _Side, b: _Side, *, note: str = ""
) -> dict[str, Any]:
    c = st.paired_contrast(a.m.success, b.m.success)
    return _contrast_record(
        name,
        kind,
        hypothesis,
        a.m.cell,
        a.m.arm,
        b.m.arm,
        statistic=_P8_PAIRED,
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds_a=c.n_seeds_a,
        n_seeds_b=c.n_seeds_b,
        n_episodes=c.n_episodes,
        reading=reading(c.lo, c.hi, a.m.arm, b.m.arm),
        status="computed",
        note=note,
        inputs=_inputs(a, b),
    )


def _post_hoc_iqm(name: str, hypothesis: str, a: _Side, b: _Side) -> dict[str, Any]:
    c = st.paired_iqm_contrast(a.m.success, b.m.success)
    return _contrast_record(
        name,
        "post hoc, not scored",
        hypothesis,
        a.m.cell,
        a.m.arm,
        b.m.arm,
        statistic=_P8_IQM,
        estimate=c.diff,
        ci_lo=c.lo,
        ci_hi=c.hi,
        units=_UNITS_SUCCESS,
        n_seeds_a=c.n_seeds_a,
        n_seeds_b=c.n_seeds_b,
        n_episodes=c.n_episodes,
        reading=reading(c.lo, c.hi, a.m.arm, b.m.arm),
        status="computed",
        note="printed beside the scored contrast as post hoc, not scored (P3-D1 §1)",
        inputs=_inputs(a, b),
    )


def _rel_p95(
    name: str,
    hypothesis: str,
    a: _Side,
    b: _Side,
    *,
    note: str = "",
    names: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Relative p95 closing-speed contrast; ``names`` label the sides when both are one arm."""
    r = st.relative_p95(a.speeds, b.speeds)
    na, nb = names if names is not None else (a.m.arm, b.m.arm)
    return _contrast_record(
        name,
        "descriptive, unscored",
        hypothesis,
        a.m.cell,
        a.m.arm,
        b.m.arm,
        statistic=_P8_P95,
        estimate=r.r,
        ci_lo=r.lo,
        ci_hi=r.hi,
        units=_UNITS_R,
        n_seeds_a=r.n_seeds_a,
        n_seeds_b=r.n_seeds_b,
        n_episodes=r.n_episodes,
        reading=reading(r.lo, r.hi, na, nb, up="softer than", down="harder than"),
        status="computed",
        note=(
            f"p95 closing speed (deck normal, m/s model scale): {na} {fmt(r.p95_a)}, "
            f"{nb} {fmt(r.p95_b)}; touchdowns {r.n_touchdowns_a} / {r.n_touchdowns_b}; "
            f"undefined replicates {r.n_reps_undefined}" + (f"; {note}" if note else "")
        ),
        inputs=_inputs(a, b),
    )


def _from_result(
    res: hy.HypothesisResult, cell: tuple[str, str], a: str, b: str, inputs: str
) -> dict[str, Any]:
    return _contrast_record(
        res.hypothesis,
        "scored",
        res.hypothesis,
        cell,
        a,
        b,
        statistic=res.statistic,
        estimate=res.estimate,
        ci_lo=res.ci_lo,
        ci_hi=res.ci_hi,
        units=res.units,
        n_seeds_a=res.n_seeds[0][1],
        n_seeds_b=res.n_seeds[1][1] if len(res.n_seeds) > 1 else res.n_seeds[0][1],
        n_episodes=res.n_episodes,
        reading=res.verdict_note,
        status="scored",
        note=f"verdict: {res.verdict}",
        inputs=inputs,
    )


def _from_unscored(
    u: hy.UnscoredContrast, kind: str, hypothesis: str, inputs: str
) -> dict[str, Any]:
    regime, ss = u.cell.split(" ")
    beside = "; ".join(f"{k}: {v}" for k, v in u.beside if k != "SS6 flip floor")
    return _contrast_record(
        u.name,
        kind,
        hypothesis,
        (regime, ss),
        u.n_seeds[0][0],
        u.n_seeds[1][0],
        statistic=u.statistic,
        estimate=u.estimate,
        ci_lo=u.ci_lo,
        ci_hi=u.ci_hi,
        units=u.units,
        n_seeds_a=u.n_seeds[0][1],
        n_seeds_b=u.n_seeds[1][1],
        n_episodes=u.n_episodes,
        reading=u.reading,
        status="computed",
        note=" ".join([*u.caveats, beside]).strip(),
        inputs=inputs,
    )


@dataclass
class _Scored:
    """Everything the hypothesis pass produces."""

    contrasts: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)


_ID: Final[str] = "id"


def _g(index: CellIndex, arm: str, ss: str, **kw: Any) -> GroupKey | None:
    return index.find(arm, _ID, ss, **kw)


def _hyp_missing(name: str, cell: str, why: str) -> dict[str, Any]:
    return {
        "hypothesis": name,
        "cell": cell,
        "status": f"not scored: missing ({why})",
        "flip_floor": _flip(cell.split(" ")[1]),
    }


def _hyp_row(res: hy.HypothesisResult, inputs: str) -> dict[str, Any]:
    return {
        "hypothesis": res.hypothesis,
        "cell": res.cell,
        "status": "scored",
        "verdict": res.verdict,
        "verdict_note": res.verdict_note,
        "estimate": res.estimate,
        "ci_lo": res.ci_lo,
        "ci_hi": res.ci_hi,
        "units": res.units,
        "magnitude": res.magnitude,
        "statistic": res.statistic,
        "contrast": res.contrast,
        "n_seeds": "; ".join(f"{a} {n}" for a, n in res.n_seeds),
        "n_episodes": res.n_episodes,
        "flip_floor": _flip(res.cell.split(" ")[1]),
        "inputs": inputs,
        "ceiling_limited": res.ceiling_limited,
        "notes": " ".join(res.notes),
        "caveats_see": CAVEATS_SEE,
    }


def _hyp_record(res: hy.HypothesisResult, inputs: str) -> dict[str, Any]:
    rec = asdict(res)
    rec["status"] = "scored"
    rec["inputs"] = inputs.split("; ")
    rec["beside"] = [{"label": k, "text": v} for k, v in res.beside]
    rec["n_seeds"] = [{"arm": a, "seeds": n} for a, n in res.n_seeds]
    return rec


#: The latency arm's unscored context cells (P3-D1 §7: H5(b) is scored at ``id`` SS5 only).
_LATENCY_CONTEXT: Final[tuple[tuple[str, str], ...]] = (
    ("bc_dagger", "SS4"),
    ("bc_dagger", "SS5"),
    ("bc_dagger", "SS6"),
    ("dp", "SS4"),
    ("dp", "SS6"),
)


def _latency_d(matrix: ev_arms.Matrix, arm: str) -> int | None:
    """The ``d`` the matrix's ``latency_arm`` jobs of ``arm`` fly with (``None``: none)."""
    ds = {
        j.latency_steps
        for j in matrix.jobs
        if j.tag == ev_arms.LATENCY_TAG and j.arm == arm and j.latency_steps is not None
    }
    if len(ds) > 1:
        raise ScoreInputError(f"latency_arm {arm}: more than one d {sorted(ds)}")
    return next(iter(ds)) if ds else None


def _delay_records(inp: ScoreInputs) -> dict[str, Any] | None:
    """``delay.json`` under ``inp.latency_dir`` (``None`` when absent)."""
    if inp.latency_dir is None or not (inp.latency_dir / "delay.json").is_file():
        return None
    rec: dict[str, Any] = json.loads((inp.latency_dir / "delay.json").read_text(encoding="utf-8"))
    return rec


def _check_delays(inp: ScoreInputs) -> dict[str, Any]:
    """The flown ``d`` per arm against ``delay.json`` (``score_info.json``'s ``latency``).

    Raises:
        ScoreInputError: If ``delay.json`` exists and names another ``d`` than the rows'.
    """
    rec = _delay_records(inp)
    flown = {a: _latency_d(inp.matrix, a) for a in ("bc_dagger", "dp")}
    if rec is not None:
        for arm, d in flown.items():
            want = rec.get("arms", {}).get(arm, {}).get("d")
            if d is not None and want != d:
                raise ScoreInputError(f"latency_arm {arm}: flown d {d} != delay.json d {want}")
    out: dict[str, Any] = {"flown_d": flown}
    if rec is not None and inp.latency_dir is not None:
        out["delay_json_sha256"] = _sha256((inp.latency_dir / "delay.json").read_bytes())
    if inp.latency_dir is not None:
        out.update(latency_stage_commits(inp.latency_dir))
    return out


#: Phase 9 tables whose rows carry the commit that produced them (``ila_commit`` column).
_LATENCY_ROW_COMMITS: Final[tuple[str, ...]] = ("latency.csv", "closed_loop.csv")


def latency_stage_commits(latency_dir: Path) -> dict[str, Any]:
    """The ila commits of Phase 9's latency stages (``results/latency/run_info.json``).

    ``run_info.json`` keeps the last invocation of each stage; the per-row ``ila_commit``
    columns of ``latency.csv`` and ``closed_loop.csv`` are added, so a commit that produced
    committed rows is listed even when a later invocation of its stage replaced the stage
    record (batch-1 and batch-24 latency rows come from different commits).

    Returns:
        ``{"stage_commits": [{"ila_commit", "stages", "dirty_outside_results",
        "dirty_inside_results_only"}], "run_info_sha256": ...}`` sorted by commit, or ``{}``
        when ``run_info.json`` is absent.
    """
    path = latency_dir / "run_info.json"
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    stages: dict[str, set[str]] = {}
    dirty_out: dict[str, bool] = {}
    dirty_in: dict[str, bool] = {}
    for name, rec in sorted(json.loads(raw).get("stages", {}).items()):
        vcs = rec.get("git", {})
        commit = str(vcs.get("ila_commit", ""))
        if not commit:
            continue
        stages.setdefault(commit, set()).add(name)
        dirty_out[commit] = dirty_out.get(commit, False) or bool(vcs.get("dirty_outside_results"))
        dirty_in[commit] = dirty_in.get(commit, False) or (
            bool(vcs.get("dirty")) and not bool(vcs.get("dirty_outside_results"))
        )
    for fname in _LATENCY_ROW_COMMITS:
        f = latency_dir / fname
        if not f.is_file():
            continue
        for row in read_csv(f):
            commit = row.get("ila_commit", "")
            if commit:
                stages.setdefault(commit, set()).add(f"{fname} rows")
    # Per-job and per-flight records keep their own tree state, which survives a later
    # invocation replacing the stage record (results-skeptic re-review MINOR 1).
    for rec_path in sorted(
        [*latency_dir.glob("jobs/*.json"), *latency_dir.glob("closed_loop/*/*.json")]
    ):
        vcs = json.loads(rec_path.read_text(encoding="utf-8")).get("git", {})
        commit = str(vcs.get("ila_commit", ""))
        if commit not in stages:
            continue
        dirty_out[commit] = dirty_out.get(commit, False) or bool(vcs.get("dirty_outside_results"))
        dirty_in[commit] = dirty_in.get(commit, False) or (
            bool(vcs.get("dirty")) and not bool(vcs.get("dirty_outside_results"))
        )
    return {
        "stage_commits": [
            {
                "ila_commit": c,
                "stages": sorted(stages[c]),
                "dirty_outside_results": dirty_out.get(c, False),
                "dirty_inside_results_only": dirty_in.get(c, False),
            }
            for c in sorted(stages)
        ],
        "run_info_sha256": _sha256(raw),
    }


def _score_h5a(inp: ScoreInputs, out: _Scored) -> None:
    """H5(a), read from the deploy scorer's committed output (never re-derived)."""
    if inp.latency_dir is None:
        rec: dict[str, Any] = {
            "hypothesis": "H5(a)",
            "cell": "exported dp (K = 10), dp_e2e, batch 1",
            "kind": "latency",
            "status": "not scored: no latency results directory given",
        }
    else:
        rec = hy.read_h5a(inp.latency_dir)
    row = {c: rec.get(c) for c in HYPOTHESIS_COLUMNS if c in rec}
    if rec["status"] == "scored":
        row.update(
            n_seeds="dp 1 (seed 0's selected checkpoint; applies to all five seeds)",
            inputs="; ".join(rec["inputs"]),
            ceiling_limited=False,
            notes="",
            caveats_see=CAVEATS_SEE,
        )
    out.hypotheses.append(row)
    out.records.append(rec)


def _score_h5b(
    index: CellIndex, inp: ScoreInputs, out: _Scored, facts: hy.LatencyFacts | None = None
) -> None:
    """H5(b) at ``id`` SS5 and the latency arm's unscored context cells.

    ``facts`` (Phase 9's results) supply why ``d`` is what it is, the bc_dagger delay reading
    and this ``dp``'s measured closed-loop flip rate beside every ``dp`` contrast.
    """
    flip_note = _dp_flip_note(facts)
    core_inf = (8, 10, False)
    d_dp = _latency_d(inp.matrix, "dp")
    delay = _delay_records(inp)
    lat_side, why_l = (None, "latency_arm dp is not in the matrix")
    if d_dp is not None:
        lat_side, why_l = _student_side(
            index, _g(index, "dp", "SS5", tag=ev_arms.LATENCY_TAG, inference=core_inf)
        )
    sync_side, why_s = _student_side(index, _g(index, "dp", "SS5", tag="core", inference=core_inf))
    if lat_side is None or sync_side is None or d_dp is None:
        why = "; ".join(w for w in (why_l, why_s) if w)
        out.hypotheses.append(_hyp_missing("H5(b)", "id SS5", why))
        out.records.append(_hyp_missing("H5(b)", "id SS5", why))
        out.contrasts.append(
            _missing_contrast(
                "H5(b)", "scored", "H5(b)", (_ID, "SS5"), "dp latency arm", "dp", why
            )  # fmt: skip
        )
    else:
        ss6, _ = _student(index, _g(index, "dp", "SS6", tag="core", inference=core_inf))
        static, _ = _student(index, index.find("dp", "static", "static", tag="core",
                                               inference=core_inf))  # fmt: skip
        res = hy.score_h5b(
            lat_side.m,
            sync_side.m,
            delay_steps=d_dp,
            delay_record=None if delay is None else delay.get("arms", {}).get("dp"),
            sync_ss6=ss6,
            sync_static=static,
            facts=facts,
        )
        inputs = _inputs(lat_side, sync_side)
        out.hypotheses.append(_hyp_row(res, inputs))
        out.records.append(_hyp_record(res, inputs))
        out.contrasts.append(
            _with_note(
                _from_result(res, (_ID, "SS5"), f"dp latency d={d_dp}", "dp synchronous", inputs),
                flip_note,
            )
        )
        out.contrasts.append(
            {
                **_rel_p95(
                    "H5(b) relative p95 closing speed",
                    "H5(b)",
                    lat_side,
                    sync_side,
                    note="same checkpoints; P8's function resamples seeds per side (not jointly)"
                    + (f"; {flip_note}" if flip_note else ""),
                    names=(f"dp latency d={d_dp}", "dp synchronous"),
                ),
                "a": f"dp latency d={d_dp}",
                "b": "dp synchronous",
            }
        )
    for arm, ss in _LATENCY_CONTEXT:
        d = _latency_d(inp.matrix, arm)
        name = f"H5(b) context: {arm} latency arm - synchronous ({ss}, unscored)"
        inf = core_inf if ev_arms.is_diffusion(arm) else None
        if d is None:
            out.contrasts.append(
                _missing_contrast(
                    name,
                    "descriptive, unscored",
                    "H5(b)",
                    (_ID, ss),
                    arm,
                    arm,
                    "latency_arm is not in the matrix",
                )  # fmt: skip
            )
            continue
        sl, w1 = _student_side(index, _g(index, arm, ss, tag=ev_arms.LATENCY_TAG, inference=inf))
        sy, w2 = _student_side(index, _g(index, arm, ss, tag="core", inference=inf))
        if sl is None or sy is None:
            why = "; ".join(w for w in (w1, w2) if w)
            out.contrasts.append(
                _missing_contrast(name, "descriptive, unscored", "H5(b)", (_ID, ss), arm, arm, why)
            )
            continue
        u = hy.latency_context(sl.m, sy.m, delay_steps=d, facts=facts)
        out.contrasts.append(_from_unscored(u, "descriptive, unscored", "H5(b)", _inputs(sl, sy)))
        names = (f"{arm} latency d={d}", f"{arm} synchronous")
        p95_flip = f"; {flip_note}" if flip_note and arm == "dp" else ""
        out.contrasts.append(
            {
                **_rel_p95(
                    f"{name} relative p95 closing speed",
                    "H5(b)",
                    sl,
                    sy,
                    note="same checkpoints; P8's function resamples seeds per side (not jointly)"
                    + p95_flip,
                    names=names,
                ),
                "a": names[0],
                "b": names[1],
            }
        )


def _score_h1(index: CellIndex, inp: ScoreInputs, out: _Scored) -> None:
    pairs = (
        ("H1", "bc_dagger", "bc_matched", "SS5"),
        ("H1-SS6", "bc_dagger", "bc_matched", "SS6"),
        ("H1-DART-SS5", "bc_dart", "bc", "SS5"),
        ("H1-DART-SS6", "bc_dart", "bc", "SS6"),
    )
    sides: dict[tuple[str, str], _Side] = {}
    missing: list[str] = []
    for _, a, b, ss in pairs:
        for arm in (a, b):
            side, why = _student_side(index, _g(index, arm, ss, tag="core"))
            if side is None:
                missing.append(why)
            else:
                sides[(arm, ss)] = side
    for name, a, b, ss in pairs:
        if (a, ss) in sides and (b, ss) in sides:
            sa, sb = sides[(a, ss)], sides[(b, ss)]
            out.contrasts.append(_post_hoc_iqm(f"{name} post hoc IQM", name, sa, sb))
            out.contrasts.append(_rel_p95(f"{name} relative p95 closing speed", name, sa, sb))
    if missing:
        why = "; ".join(sorted(set(missing)))
        for name, a, b, ss in pairs:
            out.hypotheses.append(_hyp_missing(name, f"id {ss}", why))
            out.records.append({**_hyp_missing(name, f"id {ss}", why)})
            out.contrasts.append(_missing_contrast(name, "scored", name, (_ID, ss), a, b, why))
        return
    results = hy.score_h1(
        dagger_ss5=sides[("bc_dagger", "SS5")].m,
        matched_ss5=sides[("bc_matched", "SS5")].m,
        dagger_ss6=sides[("bc_dagger", "SS6")].m,
        matched_ss6=sides[("bc_matched", "SS6")].m,
        dart_ss5=sides[("bc_dart", "SS5")].m,
        bc_ss5=sides[("bc", "SS5")].m,
        dart_ss6=sides[("bc_dart", "SS6")].m,
        bc_ss6=sides[("bc", "SS6")].m,
        labelled_frames=inp.labelled_frames,
    )
    for res, (_, a, b, ss) in zip(results, pairs, strict=True):
        inputs = _inputs(sides[(a, ss)], sides[(b, ss)])
        out.hypotheses.append(_hyp_row(res, inputs))
        out.records.append(_hyp_record(res, inputs))
        out.contrasts.append(_from_result(res, (_ID, ss), a, b, inputs))


def _score_h2(index: CellIndex, inp: ScoreInputs, out: _Scored) -> hy.H2Reference | None:
    g_bc = _g(index, "bc_mix", "SS4", tag="ablation")
    g_dp = _g(index, "dp_mix", "SS4", tag="ablation")
    bc, why_bc = _student(index, g_bc)
    dp, why_dp = _student(index, g_dp)
    listed = inp.listed(_ID, "SS4")
    ref = hy.build_h2_reference(
        listed, inp.baselines.cell("ppo", _ID, "SS4"), inp.baselines.cell("gated", _ID, "SS4")
    )
    if bc is None or dp is None:
        why = "; ".join(w for w in (why_bc, why_dp) if w)
        out.hypotheses.append(_hyp_missing("H2", "id SS4", why))
        out.records.append(_hyp_missing("H2", "id SS4", why))
        out.contrasts.append(
            _missing_contrast("H2", "scored", "H2", (_ID, "SS4"), "bc_mix", "dp_mix", why)
        )
        return ref
    res = hy.score_h2(bc, dp, ref)
    inputs = "; ".join(
        [*(g.label for g in (g_bc, g_dp) if g is not None), "T_mix reference id SS4 (P8 rows)"]
    )
    out.hypotheses.append(_hyp_row(res, inputs))
    out.records.append(_hyp_record(res, inputs))
    out.contrasts.append(_from_result(res, (_ID, "SS4"), "bc_mix", "dp_mix", inputs))
    return ref


def _dp_flip_note(facts: hy.LatencyFacts | None) -> str:
    """Phase 9's measured dp flip rate as a contrast-note sentence ("" before Phase 9)."""
    text = None if facts is None else facts.dp_flip_text()
    return f"{hy.DP_FLIP_LABEL}: {text}." if text else ""


def _with_note(row: dict[str, Any], extra: str) -> dict[str, Any]:
    if extra:
        row["note"] = f"{row['note']}; {extra}" if row.get("note") else extra
    return row


def _score_h3(index: CellIndex, out: _Scored, facts: hy.LatencyFacts | None = None) -> None:
    g1 = _g(index, "dp", "SS6", tag="dp_Ta", inference=(1, 10, False))
    g8 = _g(index, "dp", "SS6", tag="dp_Ta", inference=(8, 10, False))
    s1, why1 = _student_side(index, g1)
    s8, why8 = _student_side(index, g8)
    if s1 is None or s8 is None:
        why = "; ".join(w for w in (why1, why8) if w)
        out.hypotheses.append(_hyp_missing("H3", "id SS6", why))
        out.records.append(_hyp_missing("H3", "id SS6", why))
        out.contrasts.append(
            _missing_contrast("H3", "scored", "H3", (_ID, "SS6"), "dp Ta=1", "dp Ta=8", why)
        )
        return
    # P6-D5's caveat: the same-checkpoint gain at id SS4 and SS5 and dp's static-deck rate at
    # Ta = 8 (numbers for the caveat only; the statistic is the SS6 contrast). The caveat is
    # binding, so a missing pair or static cell is an error, never a shortened caveat.
    other: list[tuple[hy.ArmMatrix, hy.ArmMatrix]] = []
    for ss in ("SS4", "SS5"):
        m1, w1 = _student(index, _g(index, "dp", ss, tag="dp_Ta", inference=(1, 10, False)))
        m8, w8 = _student(index, _g(index, "dp", ss, tag="dp_Ta", inference=(8, 10, False)))
        if m1 is None or m8 is None:
            why = "; ".join(w for w in (w1, w8) if w)
            raise hy.RowError(
                f"H3's P6-D5 caveat needs dp Ta = 1 and Ta = 8 at id {ss} ({why}); the SS6 "
                "contrast is present, so the caveat would be silently shortened"
            )
        other.append((m1, m8))
    static, ws = _student(
        index, index.find("dp", "static", "static", tag="core", inference=(8, 10, False))
    )
    if static is None:
        raise hy.RowError(
            f"H3's P6-D5 caveat needs dp at Ta = 8 on the static list ({ws}); the SS6 contrast "
            "is present, so the caveat would be silently shortened"
        )
    dp_flip = None if facts is None else facts.dp_flip_text()
    res = hy.score_h3(s1.m, s8.m, other_ss=other, static_ta8=static, dp_flip=dp_flip)
    inputs = _inputs(s1, s8)
    flip_note = _dp_flip_note(facts)
    out.hypotheses.append(_hyp_row(res, inputs))
    out.records.append(_hyp_record(res, inputs))
    out.contrasts.append(
        _with_note(_from_result(res, (_ID, "SS6"), "dp Ta=1", "dp Ta=8", inputs), flip_note)
    )
    rel = _rel_p95(
        "H3 relative p95 closing speed",
        "H3",
        s1,
        s8,
        note="same checkpoints; P8's function resamples seeds per side (not jointly)"
        + (f"; {flip_note}" if flip_note else ""),
        names=("dp Ta=1", "dp Ta=8"),
    )
    out.contrasts.append({**rel, "a": "dp Ta=1", "b": "dp Ta=8"})


def _score_h4(index: CellIndex, inp: ScoreInputs, out: _Scored) -> None:
    cell = (_ID, "SS6")
    s_state, why1 = _student_side(index, _g(index, "bc_state", "SS6", tag="core"))
    s_bc, why2 = _student_side(index, _g(index, "bc", "SS6", tag="core"))
    teacher = _p8_side(inp.baselines.cell("ppo", *cell), "ppo", cell)
    if s_state is None or s_bc is None:
        why = "; ".join(w for w in (why1, why2) if w)
        out.hypotheses.append(_hyp_missing("H4", "id SS6", why))
        out.records.append(_hyp_missing("H4", "id SS6", why))
        for name in ("H4", "H4 P", "H4 I"):
            out.contrasts.append(_missing_contrast(name, "scored", "H4", cell, "", "", why))
    else:
        res = hy.score_h4(s_state.m, s_bc.m, teacher.m)
        inputs = _inputs(s_state, s_bc, teacher)
        out.hypotheses.append(_hyp_row(res, inputs))
        out.records.append(_hyp_record(res, inputs))
        out.contrasts.append(_from_result(res, cell, "P", "I", inputs))
        g = st.h4_three_arm_bootstrap(s_state.m.success, s_bc.m.success, teacher.m.success)
        for name, a, b, est, lo, hi in (
            ("H4 P", "bc_state", "bc", g.p, g.p_lo, g.p_hi),
            ("H4 I", "teacher ppo s0", "bc_state", g.i, g.i_lo, g.i_hi),
        ):
            out.contrasts.append(
                _contrast_record(
                    name,
                    "component of the scored H4",
                    "H4",
                    cell,
                    a,
                    b,
                    statistic="three-arm bootstrap (ila.eval.stats.h4_three_arm_bootstrap; "
                    "10 000 reps, seed 20261023)",
                    estimate=est,
                    ci_lo=lo,
                    ci_hi=hi,
                    units=_UNITS_SUCCESS,
                    n_seeds_a=5 if a == "bc_state" else 1,
                    n_seeds_b=5,
                    n_episodes=g.n_episodes,
                    reading=reading(lo, hi, a, b),
                    status="computed",
                    note="P = S(bc_state) - S(bc); I = S(teacher) - S(bc_state); D = P - I",
                    inputs=inputs,
                )
            )
    # P' (unscored, tag +clock) and modular vs best end-to-end (unscored, two-sided).
    s_clock, why3 = _student_side(index, _g(index, "bc_plus_clock", "SS6", tag="ablation"))
    if s_state is None or s_clock is None:
        why = "; ".join(w for w in (why1, why3) if w)
        out.contrasts.append(
            _missing_contrast(
                "H4 P' (unscored, tag +clock)",
                "descriptive, unscored",
                "H4",
                cell,
                "bc_state",
                "bc_plus_clock",
                why,
            )  # fmt: skip
        )
    else:
        out.contrasts.append(
            _from_unscored(
                hy.h4_p_prime(s_state.m, s_clock.m),
                "descriptive, unscored",
                "H4",
                _inputs(s_state, s_clock),
            )
        )
    best, reason = inp.best_end_to_end
    s_mod, why4 = _student_side(index, _g(index, "modular", "SS6", tag="core"))
    s_best, why5 = _student_side(index, _g(index, best, "SS6", tag="core"))
    if s_mod is None or s_best is None:
        why = "; ".join(w for w in (why4, why5) if w)
        out.contrasts.append(
            _missing_contrast(
                "H4 modular vs best end-to-end (unscored)",
                "descriptive, unscored",
                "H4",
                cell,
                "modular",
                best,
                why,
            )  # fmt: skip
        )
    else:
        u = hy.modular_vs_best(s_mod.m, s_best.m, reason)
        out.contrasts.append(
            _from_unscored(u, "descriptive, unscored", "H4", _inputs(s_mod, s_best))
        )


#: Which dp the Phase 9 flip rate was measured on, where a contrast also has dp_mix.
_CORE_DP_ONLY: Final[str] = " (core dp only; not measured for dp_mix)"


def _descriptives(
    index: CellIndex, inp: ScoreInputs, out: _Scored, facts: hy.LatencyFacts | None = None
) -> None:
    """Unscored descriptives; ``facts`` add the measured dp flip rate beside dp's H2 rows."""
    flip_note = _dp_flip_note(facts)
    dp_flip = f"; {flip_note[:-1]}{_CORE_DP_ONLY}." if flip_note else ""
    # bc_dart - bc_matched at id SS5 and SS6 (1 000 vs 2 000 labelled episodes).
    for ss in ("SS5", "SS6"):
        sd, why1 = _student_side(index, _g(index, "bc_dart", ss, tag="core"))
        sm, why2 = _student_side(index, _g(index, "bc_matched", ss, tag="core"))
        name = f"bc_dart - bc_matched ({ss}, unscored)"
        if sd is None or sm is None:
            why = "; ".join(w for w in (why1, why2) if w)
            out.contrasts.append(
                _missing_contrast(
                    name, "descriptive, unscored", "H1", (_ID, ss), "bc_dart", "bc_matched", why
                )  # fmt: skip
            )
            continue
        u = hy.dart_vs_matched(sd.m, sm.m)
        rec = _from_unscored(u, "descriptive, unscored", "H1", _inputs(sd, sm))
        out.contrasts.append({**rec, "name": name})
        out.contrasts.append(_rel_p95(f"{name} relative p95 closing speed", "H1", sd, sm))
    # H2 secondary at id SS4 (unscored, P3-D1 §8): success drop T_ppo -> T_mix, bc vs dp.
    drops: dict[str, float] = {}
    drop_inputs: list[str] = []
    for base, mix in (("bc", "bc_mix"), ("dp", "dp_mix")):
        sb, why1 = _student_side(index, _g(index, base, "SS4", tag="core"))
        sx, why2 = _student_side(index, _g(index, mix, "SS4", tag="ablation"))
        name = f"H2 secondary: success drop {base} - {mix} (unscored)"
        if sb is None or sx is None:
            why = "; ".join(w for w in (why1, why2) if w)
            out.contrasts.append(
                _missing_contrast(name, "descriptive, unscored", "H2", (_ID, "SS4"), base, mix, why)
            )
            continue
        rec = _paired(
            name,
            "descriptive, unscored",
            "H2",
            sb,
            sx,
            note="different training data as well as teacher; 5 vs 3 seeds (P3-D1 §8)"
            + (dp_flip if base == "dp" else ""),
        )
        drops[base] = float(rec["estimate"])
        drop_inputs.append(str(rec["inputs"]))
        out.contrasts.append(rec)
    name = "H2 secondary: drop(bc) - drop(dp) (unscored)"
    if len(drops) == 2:
        out.contrasts.append(
            _contrast_record(
                name,
                "descriptive, unscored",
                "H2",
                (_ID, "SS4"),
                "drop(bc)",
                "drop(dp)",
                statistic="difference of the two paired_bootstrap point estimates; no interval",
                estimate=drops["bc"] - drops["dp"],
                units=_UNITS_SUCCESS,
                reading=f"{UNPAIRED}: point estimate only, no CI (unscored, P3-D1 §8)",
                status="computed",
                note="plan §1.8's directional secondary, unscored by P3-D1 §8" + dp_flip,
                inputs="; ".join(drop_inputs),
            )
        )
    else:
        out.contrasts.append(
            _missing_contrast(
                name,
                "descriptive, unscored",
                "H2",
                (_ID, "SS4"),
                "drop(bc)",
                "drop(dp)",
                "a drop is missing",
            )  # fmt: skip
        )
    _teacher_checks(index, inp, out)


def _teacher_checks(index: CellIndex, inp: ScoreInputs, out: _Scored) -> None:
    """Phase 8 "teacher checks": students vs their own teacher's committed rows."""
    checks: list[tuple[str, str, str, str]] = []
    for arm in ("dp_pid", "bc_pid"):
        for ss in ("SS4", "SS5", "SS6"):
            checks.append((arm, "ablation", _ID, ss))
    for ss in ("SS4", "SS5", "SS6"):
        checks.append(("dp_minus_last_action", "ablation", _ID, ss))
    for g in index.groups:
        if g.arm == "modular" and g.tag == "core":
            checks.append(("modular", "core", g.list_name, g.ss))
    for arm, tag, list_name, ss in checks:
        teacher = ev_arms.expected_teacher(arm)
        method = TEACHER_METHOD[teacher]
        name = f"teacher check: {arm} - {method} ({list_name} {ss})"
        side, why = _student_side(index, index.find(arm, list_name, ss, tag=tag))
        if side is None:
            out.contrasts.append(
                _missing_contrast(
                    name, "teacher check, unscored", "", (list_name, ss), arm, method, why
                )  # fmt: skip
            )
            continue
        t = _p8_side(inp.baselines.cell(method, list_name, ss), method, (list_name, ss))
        rec = _paired(
            name,
            "teacher check, unscored",
            "",
            side,
            t,
            note=f"teacher = P8 committed {method} rows (one row set; Phase 8 before-you-start)",
        )
        out.contrasts.append(rec)
        out.contrasts.append(
            {
                **_rel_p95(f"{name} relative p95 closing speed", "", side, t),
                "kind": "teacher check, unscored",
            }
        )


# --------------------------------------------------------------------------- above teacher


def above_teacher_rows(
    index: CellIndex,
    aggregates: Sequence[Mapping[str, Any]],
    inp: ScoreInputs,
) -> list[dict[str, Any]]:
    """Every group whose IQM or any seed's success exceeds its own teacher's committed rate.

    The teacher is P8 ``ppo`` seed 0 (e07) for ``T_ppo`` arms, ``pid_feedforward`` (e01) for
    ``*_pid`` and the ``T_mix`` reference for ``*_mix``, in the same (list, sea state) cell.
    Exceeding means strictly greater success rate; the margin is the difference (dimensionless).
    """
    agg = dict(zip(index.groups, aggregates, strict=True))
    out = []
    for g in index.groups:
        cells = [c for c in index.group_cells(g) if c.rows is not None]
        if not cells:
            continue
        teacher = ev_arms.expected_teacher(g.arm)
        trows = teacher_rows(teacher, g.list_name, g.ss, inp.baselines, inp.listed)
        if not trows:
            raise ScoreInputError(f"{g.label}: no committed {teacher} rows")
        tk = sum(str(r["outcome"]) == "success" for r in trows)
        t_rate = tk / len(trows)
        seeds_above = []
        margins = []
        for c in cells:
            rate = _rate(c.rows or ())
            if rate > t_rate:
                seeds_above.append(f"s{c.key.seed}:{fmt(rate - t_rate)}")
                margins.append(rate - t_rate)
        a = agg[g]
        raw_iqm = a.get("success_iqm")
        iqm = None if raw_iqm is None else float(raw_iqm)
        iqm_above = iqm is not None and iqm > t_rate
        if iqm is not None and iqm_above:
            margins.append(iqm - t_rate)
        if not seeds_above and not iqm_above:
            continue
        method = TEACHER_METHOD[teacher]
        out.append(
            {
                **g.record(),
                "teacher": teacher,
                "teacher_source": "assign_rows(20261023): ppo s0 (e07) / gated (e01)"
                if teacher == "T_mix"
                else f"P8 {BASELINE_METHODS[method][0]} {method} run_seed 0",
                "teacher_k": tk,
                "teacher_n": len(trows),
                "teacher_rate": t_rate,
                "status": a["status"],
                "success_iqm": iqm,
                "iqm_margin": None if iqm is None else float(iqm) - t_rate,
                "iqm_above": iqm_above,
                "seeds_above": "|".join(seeds_above),
                "max_margin": max(margins),
                "audit_file": f"above_teacher/{g.slug}.json",
            }
        )
    return out


# --------------------------------------------------------------------------- H2 times


def h2_time_rows(index: CellIndex, ref: hy.H2Reference) -> list[dict[str, Any]]:
    """The episodes behind H2 (id SS4): ``bc_mix``/``dp_mix`` per seed and the reference."""
    out: list[dict[str, Any]] = []
    for (idx, eps), policy, outcome, t in zip(
        ref.episodes, ref.assigned, ref.outcomes, ref.touchdown_s, strict=True
    ):
        out.append(
            {
                "group": f"reference:{policy}",
                "seed": 0,
                "index": idx,
                "episode_seed": eps,
                "outcome": outcome,
                "touchdown_contact": math.isfinite(float(t)),
                "time_to_touchdown_s": float(t),
            }
        )
    for arm in ("bc_mix", "dp_mix"):
        g = _g(index, arm, "SS4", tag="ablation")
        if g is None:
            continue
        for c in index.group_cells(g):
            for r in c.rows or ():
                out.append(
                    {
                        "group": arm,
                        "seed": c.key.seed,
                        "index": int(r["index"]),
                        "episode_seed": int(r["episode_seed"]),
                        "outcome": r["outcome"],
                        "touchdown_contact": as_bool(r["touchdown_contact"]),
                        "time_to_touchdown_s": as_float(r["time_to_touchdown_s"]),
                    }
                )
    return out


# --------------------------------------------------------------------------- inputs


def load_labelled_frames(
    results_root: Path = p8_adapter.REPO_ROOT / "results",
    arms: Sequence[str] = ("bc_dagger", "bc_matched", "bc_dart", "bc"),
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """H1's labelled training frames per arm and seed, from committed run manifests.

    Source: ``results/phase{4,5}/runs/<arm>/s<seed>/manifest.json``, ``data[<last stage>]``
    ``train_frames``: frames trained on after the success filter, held-out frames excluded
    (``bc_dagger``: kept demo frames plus every rollout frame). ``bc_dagger`` is cross-checked
    against ``results/phase5/dagger_iterations.csv`` ``aggregate_train_frames`` at iteration 5.

    Returns:
        ``(arm -> per-seed frames in seed order, source path -> SHA-256)``.

    Raises:
        ScoreInputError: If a manifest is missing or the DAgger cross-check fails.
    """
    frames: dict[str, list[int]] = {}
    sources: dict[str, str] = {}
    for arm in arms:
        homes = [results_root / p / "runs" / arm for p in ("phase4", "phase5")]
        homes = [h for h in homes if h.is_dir()]
        if len(homes) != 1:
            raise ScoreInputError(f"{arm}: manifests in {len(homes)} phases")
        per_seed = []
        for s in hy.CORE_SEEDS:
            path = homes[0] / f"s{s}" / "manifest.json"
            if not path.is_file():
                raise ScoreInputError(f"{path}: missing")
            data = json.loads(path.read_text(encoding="utf-8"))["data"]
            stages = sorted(data, key=int) if isinstance(data, Mapping) else None
            last = data[stages[-1]] if stages else data[-1]
            per_seed.append(int(last["train_frames"]))
            sources[path.relative_to(results_root.parent).as_posix()] = p8_adapter.sha256_file(path)
        frames[arm] = per_seed
    it_path = results_root / "phase5" / "dagger_iterations.csv"
    if "bc_dagger" in frames and it_path.is_file():
        last = {
            int(r["seed"]): int(r["aggregate_train_frames"])
            for r in read_csv(it_path)
            if r["iteration"] == "5"
        }
        if [last.get(s) for s in hy.CORE_SEEDS] != frames["bc_dagger"]:
            raise ScoreInputError("bc_dagger train_frames disagree with dagger_iterations.csv")
        sources[it_path.relative_to(results_root.parent).as_posix()] = p8_adapter.sha256_file(
            it_path
        )
    return frames, sources


@functools.cache
def _frozen_listed(list_name: str, ss: str) -> tuple[p8_adapter.ListedEpisode, ...]:
    return tuple(e for e in ev_arms.list_episodes(list_name) if e.ss == ss)


def default_inputs(root: Path = OUT_ROOT) -> ScoreInputs:
    """The committed inputs: the frozen matrix, P8's rows and lists, manifests, selections."""
    matrix = ev_arms.load_matrix()
    frames, frame_sources = load_labelled_frames()
    scores = hy.load_select_scores()
    best = hy.best_end_to_end(scores)
    best_sources = {}
    for arm in hy.END_TO_END_ARMS:
        for s in hy.CORE_SEEDS:
            rel = None
            for phase in ("phase4", "phase5", "phase6"):
                p = p8_adapter.REPO_ROOT / "results" / phase / "runs" / arm / f"s{s}"
                if (p / "selection.json").is_file():
                    rel = p / "selection.json"
            if rel is not None:
                best_sources[rel.relative_to(p8_adapter.REPO_ROOT).as_posix()] = (
                    p8_adapter.sha256_file(rel)
                )
    return ScoreInputs(
        matrix=matrix,
        root=root,
        baselines=load_baselines(),
        listed=_frozen_listed,
        labelled_frames=frames,
        labelled_frames_sources=frame_sources,
        best_end_to_end=best,
        best_sources=best_sources,
        list_cells=ev_arms.frozen_list_cells(),
        latency_dir=p8_adapter.REPO_ROOT / "results" / "latency",
        select_scores=scores,
    )


# --------------------------------------------------------------------------- bc_dart seeds 1-4

#: P4-D5 item 2's label for the ``bc_dart`` seeds 1-4 sensitivity, verbatim
#: (``docs/protocol.md`` P4-D4 correction 2, P4-D5 item 2).
DART_S1_4_LABEL: Final[str] = "post hoc: added after `select`, before any frozen-list read"

#: The seeds of the sensitivity: ``bc_dart`` without seed 0, the sigma = 0.10 tuning trial.
DART_S1_4_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4)

#: The sensitivity's file under the output root.
DART_S1_4_FILE: Final[str] = "bc_dart_s1_4.csv"

#: ``iqm`` rows carry :data:`AGGREGATE_COLUMNS`-named fields (same group key as the 5-seed
#: row); ``contrast`` and ``best_end_to_end`` rows carry ``estimate``/``ci_*``. ``frozen_*``
#: is the 5-seed result the row is printed beside (copied from the same scoring run).
DART_S1_4_COLUMNS: Final[tuple[str, ...]] = (
    "kind",
    "name",
    "label",
    "hypothesis",
    *GROUP_COLUMNS,
    "status",
    "seeds",
    "n_seeds",
    "n_seeds_expected",
    "n_episodes_per_seed",
    "success_iqm",
    "success_iqm_lo",
    "success_iqm_hi",
    "success_seed_min",
    "success_seed_max",
    "success_per_seed",
    *_COUNT_COLUMNS,
    *(f"per_1000_{o}" for o in LOSS_CLASSES),
    "b",
    "seeds_b",
    "n_seeds_b",
    "statistic",
    "estimate",
    "ci_lo",
    "ci_hi",
    "units",
    "n_episodes",
    "reading",
    "frozen_seeds",
    "frozen_estimate",
    "frozen_ci_lo",
    "frozen_ci_hi",
    "frozen_verdict",
    "frozen_source",
    "flip_floor",
    "note",
    "inputs",
    "reps",
    "bootstrap_seed",
)

_DART_NEVER: Final[str] = (
    "Sensitivity only: printed beside the frozen 5-seed result (frozen_*), never substituted "
    "for it; bc_dart seed 0 is the sigma = 0.10 tuning trial, kept as frozen (deviation from "
    "P3-D1 §4, user-approved; docs/protocol.md P4-D4 correction 2, P4-D5 item 2)."
)


def _seeds_text(seeds: Iterable[int]) -> str:
    return "|".join(str(s) for s in seeds)


def _seed_subset(m: hy.ArmMatrix, seeds: Sequence[int]) -> hy.ArmMatrix:
    """``m`` restricted to ``seeds`` (rows of every per-seed array; episodes unchanged)."""
    idx = [m.seeds.index(s) for s in seeds]
    return replace(
        m,
        seeds=tuple(seeds),
        success=m.success[idx],
        touchdown_s=m.touchdown_s[idx],
        outcomes=tuple(m.outcomes[i] for i in idx),
        ckpt_sha256=tuple(m.ckpt_sha256[i] for i in idx),
        tunnelled=None if m.tunnelled is None else m.tunnelled[idx],
    )


def _dart_iqm_rows(
    index: CellIndex, frozen: Mapping[GroupKey, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    stats_cfg = st.matrix_statistics()
    out = []
    for g in index.groups:
        if (g.tag, g.arm) != ("core", "bc_dart"):
            continue
        cells = [c for c in index.group_cells(g) if c.key.seed in DART_S1_4_SEEDS]
        agg = frozen[g]
        rec: dict[str, Any] = {
            "kind": "iqm",
            "name": f"bc_dart {g.list_name} {g.ss} IQM, seeds 1-4",
            "label": DART_S1_4_LABEL,
            **g.record(),
            "seeds": _seeds_text(c.key.seed for c in cells),
            "n_seeds": sum(c.present for c in cells),
            "n_seeds_expected": len(DART_S1_4_SEEDS),
            "statistic": "P8 IQM over seeds, stratified-bootstrap 95 % CI "
            "(ila.eval.stats.iqm_ci; 2 000 reps, seed 20261023)",
            "units": "success, dimensionless",
            "frozen_seeds": agg.get("seeds"),
            "frozen_estimate": agg.get("success_iqm"),
            "frozen_ci_lo": agg.get("success_iqm_lo"),
            "frozen_ci_hi": agg.get("success_iqm_hi"),
            "frozen_source": "aggregate.csv",
            "flip_floor": _flip(g.ss),
            "note": _DART_NEVER,
            "inputs": g.label,
            "reps": stats_cfg.iqm_reps,
            "bootstrap_seed": stats_cfg.bootstrap_seed,
        }
        have = {c.key.seed for c in cells if c.present}
        missing = [str(s) for s in DART_S1_4_SEEDS if s not in have]
        if missing:
            rec["status"] = f"incomplete: seeds {'|'.join(missing)} missing"
        else:
            stats = _seed_stats(g, cells)
            stats.pop("success_mean")
            rec.update(status="complete", **stats)
        out.append(rec)
    return out


def _dart_contrast_rows(index: CellIndex, scored: _Scored) -> list[dict[str, Any]]:
    stats_cfg = st.matrix_statistics()
    hyp = {str(r["hypothesis"]): r for r in scored.hypotheses}
    out = []
    for name, ss in (("H1-DART-SS5", "SS5"), ("H1-DART-SS6", "SS6")):
        a, why_a = _student_side(index, _g(index, "bc_dart", ss, tag="core"))
        b, why_b = _student_side(index, _g(index, "bc", ss, tag="core"))
        h = hyp.get(name, {})
        rec: dict[str, Any] = {
            "kind": "contrast",
            "name": f"{name} seeds 1-4",
            "label": DART_S1_4_LABEL,
            "hypothesis": name,
            "tag": "core",
            "arm": "bc_dart",
            "list": _ID,
            "ss": ss,
            "seeds": _seeds_text(DART_S1_4_SEEDS),
            "n_seeds": len(DART_S1_4_SEEDS),
            "b": "bc",
            "statistic": _P8_PAIRED,
            "units": _UNITS_SUCCESS,
            "frozen_estimate": h.get("estimate"),
            "frozen_ci_lo": h.get("ci_lo"),
            "frozen_ci_hi": h.get("ci_hi"),
            "frozen_verdict": h.get("verdict") or h.get("status"),
            "frozen_source": f"hypotheses.json {name}",
            "flip_floor": _flip(ss),
            "note": "bc keeps its five frozen seeds; only bc_dart seed 0 is left out. No verdict "
            "is given on seeds 1-4: the scored verdict is frozen_verdict. " + _DART_NEVER,
            "reps": stats_cfg.paired_reps,
            "bootstrap_seed": stats_cfg.bootstrap_seed,
        }
        if a is None or b is None:
            rec["status"] = f"missing: {'; '.join(w for w in (why_a, why_b) if w)}"
            out.append(rec)
            continue
        sub = _seed_subset(a.m, DART_S1_4_SEEDS)
        c = st.paired_contrast(sub.success, b.m.success)
        rec.update(
            status="computed",
            seeds_b=_seeds_text(b.m.seeds),
            frozen_seeds=f"bc_dart {_seeds_text(a.m.seeds)}; bc {_seeds_text(b.m.seeds)}",
            n_seeds_b=c.n_seeds_b,
            n_seeds=c.n_seeds_a,
            estimate=c.diff,
            ci_lo=c.lo,
            ci_hi=c.hi,
            n_episodes=c.n_episodes,
            reading=reading(c.lo, c.hi, "bc_dart (seeds 1-4)", "bc"),
            inputs=_inputs(a, b),
        )
        out.append(rec)
    return out


def _pct2(x: Any) -> str:
    return f"{100.0 * float(x):.2f} %"


def _dart_best_row(inp: ScoreInputs) -> list[dict[str, Any]]:
    """H4's best end-to-end rule beside the same rule with ``bc_dart`` on seeds 1-4."""
    scores = inp.select_scores
    if "bc_dart" not in scores:
        return []
    full = scores["bc_dart"]
    idx = [full.seeds.index(s) for s in DART_S1_4_SEEDS]
    sub = hy.SelectScore(
        "bc_dart",
        DART_S1_4_SEEDS,
        tuple(full.success[i] for i in idx),
        tuple(full.n[i] for i in idx),
        tuple(full.rows_files[i] for i in idx),
    )
    arm, why = hy.best_end_to_end({**scores, "bc_dart": sub})
    others = ", ".join(
        f"{a} {_pct2(scores[a].mean_success)}" for a in hy.END_TO_END_ARMS if a != "bc_dart"
    )
    k, n = sum(sub.success), sum(sub.n)
    return [
        {
            "kind": "best_end_to_end",
            "name": "H4 best end-to-end rule, bc_dart seeds 1-4",
            "label": DART_S1_4_LABEL,
            "hypothesis": "H4",
            "arm": "bc_dart",
            "status": "computed",
            "seeds": _seeds_text(DART_S1_4_SEEDS),
            "n_seeds": len(DART_S1_4_SEEDS),
            "success_per_seed": "|".join(
                fmt(s / m) for s, m in zip(sub.success, sub.n, strict=True)
            ),
            "statistic": "mean over seeds of the selected checkpoint's select success "
            "(ila.eval.hypotheses.best_end_to_end, P3-D1 §6 H4)",
            "estimate": float(sub.mean_success),
            "units": "select success, dimensionless (select split: splits/select.parquet)",
            "n_episodes": n // len(DART_S1_4_SEEDS),
            "reading": f"the same rule with bc_dart on seeds 1-4 picks {arm} ({why}); other "
            f"end-to-end arms on their five frozen seeds: {others}",
            "frozen_seeds": _seeds_text(full.seeds),
            "frozen_estimate": float(full.mean_success),
            "frozen_verdict": f"best end-to-end as frozen (five seeds): "
            f"{inp.best_end_to_end[0]} ({inp.best_end_to_end[1]})",
            "frozen_source": "score_info.json best_end_to_end; committed selection.json files",
            "note": f"bc_dart seeds 1-4: {k}/{n} select episodes. H4's best end-to-end rule "
            "uses all five seeds as frozen. " + _DART_NEVER,
            "inputs": "; ".join(sorted(inp.best_sources)),
        }
    ]


def dart_s1_4_rows(
    index: CellIndex,
    aggregates: Sequence[Mapping[str, Any]],
    scored: _Scored,
    inp: ScoreInputs,
) -> list[dict[str, Any]]:
    """The ``bc_dart`` seeds 1-4 sensitivity (P4-D5 item 2): :data:`DART_S1_4_COLUMNS` rows.

    ``bc_dart`` s0 is the sigma = 0.10 tuning trial, kept as frozen. Beside every 5-seed
    ``bc_dart`` result the report prints, labelled :data:`DART_S1_4_LABEL`:

    * ``iqm``: each core ``bc_dart`` cell's IQM [stratified-bootstrap 95 % CI] over seeds 1-4,
      by the reduction ``aggregate.csv`` uses (success, dimensionless);
    * ``contrast``: H1's DART secondaries (``id`` SS5, SS6), success(``bc_dart`` seeds 1-4) -
      success(``bc`` seeds 0-4), P8 ``paired_bootstrap`` (10 000 reps, seed 20261023) as the
      scored contrast; no verdict is given;
    * ``best_end_to_end``: H4's best end-to-end rule with ``bc_dart`` on seeds 1-4 (other arms
      on their five seeds), from the committed ``select`` scores (only if ``inp`` has them).

    Nothing here feeds any other output: verdicts, ``aggregate.csv`` and ``contrasts.csv`` use
    all five seeds as frozen.
    """
    frozen = dict(zip(index.groups, aggregates, strict=True))
    return [
        *_dart_iqm_rows(index, frozen),
        *_dart_contrast_rows(index, scored),
        *_dart_best_row(inp),
    ]


# --------------------------------------------------------------------------- run


def _missing_rows(index: CellIndex) -> list[dict[str, Any]]:
    return [
        {**k.record(), "rows_file": c.rows_file, "reason": c.reason}
        for k, c in index.cells.items()
        if not c.present
    ]


def _tmix_cells(index: CellIndex) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    for g in index.groups:
        if ev_arms.expected_teacher(g.arm) == "T_mix":
            seen[(g.list_name, g.ss)] = None
    return list(seen)


def run_score(inp: ScoreInputs, out_dir: Path | None = None) -> dict[str, Any]:
    """Score the matrix and write every output (module docstring) under ``out_dir``.

    Args:
        inp: :class:`ScoreInputs` (:func:`default_inputs` for the committed tree).
        out_dir: Where to write (default ``inp.root``).

    Returns:
        ``score_info.json``'s record.

    Raises:
        ila.eval.hypotheses.RowError: If H3's ``id`` SS6 contrast is present but a cell its
            binding P6-D5 caveat needs is not (``dp`` Ta = 1 / Ta = 8 at ``id`` SS4 or SS5, or
            ``dp`` Ta = 8 on ``static``). Every other missing cell is listed, not raised.
    """
    out = inp.root if out_dir is None else out_dir
    index = load_cells(inp.root, inp.matrix, episodes_per_cell=inp.episodes_per_cell)
    summary = summary_rows(index)
    write_csv(out / "summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(
        out / "baselines.csv",
        BASELINE_COLUMNS,
        baseline_rows(inp.baselines, inp.list_cells, inp.listed, _tmix_cells(index)),
    )
    aggregates = aggregate_rows(index)
    write_csv(out / "aggregate.csv", AGGREGATE_COLUMNS, aggregates)

    scored = _Scored()
    _score_h1(index, inp, scored)
    ref = _score_h2(index, inp, scored)
    facts = None if inp.latency_dir is None else hy.read_latency_facts(inp.latency_dir)
    _score_h3(index, scored, facts)
    _score_h4(index, inp, scored)
    _descriptives(index, inp, scored, facts)
    latency_info = _check_delays(inp)
    _score_h5a(inp, scored)
    _score_h5b(index, inp, scored, facts)
    write_csv(out / "contrasts.csv", CONTRAST_COLUMNS, scored.contrasts)
    write_csv(out / "hypotheses.csv", HYPOTHESIS_COLUMNS, scored.hypotheses)
    dump_json(
        out / "hypotheses.json",
        {
            "no_multiplicity": hy.NO_MULTIPLICITY,
            "flip_floor_ss6": p8_adapter.FLIP_FLOOR_SS6.sentence,
            "hypotheses": scored.records,
        },
    )
    write_csv(
        out / DART_S1_4_FILE, DART_S1_4_COLUMNS, dart_s1_4_rows(index, aggregates, scored, inp)
    )
    write_csv(
        out / "above_teacher.csv", ABOVE_TEACHER_COLUMNS, above_teacher_rows(index, aggregates, inp)
    )
    missing = _missing_rows(index)
    write_csv(out / "missing.csv", MISSING_COLUMNS, missing)
    if ref is not None:
        write_csv(out / "h2_touchdown_times.csv", _H2_TIME_COLUMNS, h2_time_rows(index, ref))
    stats_cfg = st.matrix_statistics()
    info = {
        "simulation_only": "PyBullet physics, PyBullet TinyRenderer frames; model scale "
        "lambda = 1/25",
        "cells_expected": len(index.cells),
        "cells_present": sum(c.present for c in index.cells.values()),
        "cells_missing": len(missing),
        "complete": not missing,
        "rows_files": {k: dict(v) for k, v in index.files.items()},
        "committed_rows": [dict(p) for p in inp.baselines.provenance],
        "labelled_frames": {k: list(v) for k, v in inp.labelled_frames.items()},
        "labelled_frames_sources": dict(inp.labelled_frames_sources),
        "labelled_frames_definition": "manifest.json data[<last stage>].train_frames: frames "
        "trained on after the success filter, held-out frames excluded",
        "best_end_to_end": {"arm": inp.best_end_to_end[0], "reason": inp.best_end_to_end[1]},
        "best_end_to_end_sources": dict(inp.best_sources),
        "statistics": {
            "bootstrap_seed": stats_cfg.bootstrap_seed,
            "iqm_reps": stats_cfg.iqm_reps,
            "paired_reps": stats_cfg.paired_reps,
            "no_multiplicity": hy.NO_MULTIPLICITY,
        },
        "flip_floor_ss6": p8_adapter.FLIP_FLOOR_SS6.sentence,
        "latency_arm": latency_info,
        **dict(inp.extra),
    }
    dump_json(out / "score_info.json", info)
    return info
