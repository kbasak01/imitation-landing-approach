# GIF gallery selection rule (Phase 8) — fixed before any GIF is made

The GIFs are **hand-picked by this stated rule**. They are simulation only: PyBullet physics,
frames rendered by PyBullet's TinyRenderer at 96 × 96 px (shown enlarged); model scale,
λ = 1/25. They illustrate; they are not evidence, and no rate is read from them.

The text below is `ila.viz.gifs.RULE` verbatim; `tests/test_eval_figures.py` checks this file
equals it.

<!-- RULE START -->
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
<!-- RULE END -->
