"""Shared fixtures for the runner tests: P8's own ``tiny_list`` and trivial test students.

``tiny_list`` is copied verbatim from P8's ``tests/test_eval_runner.py`` (pinned commit): two
static-pad episodes and three **dev-pool train** frigate realizations, resolved through P8's
draw -> real-reset path. It never touches a frozen evaluation list, and it switches hull
geometry between episodes, so a stale ``set_motion`` would show up. Tests may import P8
directly; only ``src/`` is restricted to the adapter.

The students here are not networks: they are deterministic functions of the pixels and the
proprio they are given, so any difference in their inputs shows up in the rows. Units: actions
normalised (x v_max = 1.5 m/s model scale); lengths metres model scale (lambda = 1/25).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from rld.deck.splits import dev_pool
from rld.eval.envs import STATIC_LABEL, EvalConfigs
from rld.eval.episodes import EpisodeDraw, ListedEpisode, resolve_draws

from ila import p8_adapter
from ila.obs.split import load_contract
from ila.render import scene


def build_tiny_list(cfgs: EvalConfigs) -> list[ListedEpisode]:
    """Return P8's five resolved test episodes: two static, three dev-pool-train frigate."""
    train, _ = dev_pool(cfgs.sim)
    picks = [train[0], train[len(train) // 2], train[-1]]
    draws = [
        EpisodeDraw("test", STATIC_LABEL, 0, "aft", STATIC_LABEL, 180.0, 0.0, 0, 11, False),
        *(
            EpisodeDraw(
                "test",
                spec.sea_state,
                i + 1,
                "aft",
                spec.vessel,
                float(spec.heading_deg),
                float(spec.speed_kn),
                int(spec.seed),
                1000 + i,
                True,
            )
            for i, spec in enumerate(picks)
        ),
        EpisodeDraw("test", STATIC_LABEL, 4, "aft", STATIC_LABEL, 180.0, 0.0, 0, 12, False),
    ]
    return resolve_draws(draws, cfgs, workers=1)


def pixel_action(images: NDArray[np.uint8], proprio: NDArray[np.float32]) -> NDArray[np.float64]:
    """A descent whose lateral command depends on the newest frame and proprio row.

    Returns:
        ``(3,)`` normalised action: about -0.3 vertical (0.45 m/s model descent), small
        lateral terms from mean pixel colour and the newest proprio values.
    """
    newest = images[-1].astype(np.float64) / 255.0
    p = proprio[-1].astype(np.float64)
    return np.array(
        [
            0.2 * (float(newest[..., 0].mean()) - 0.3) - 0.05 * float(p[6]),
            0.2 * (float(newest[..., 2].mean()) - 0.3) - 0.05 * float(p[7]),
            -0.3 + 0.05 * float(newest.mean()),
        ],
        dtype=np.float64,
    )


class PixelStudent:
    """A stateless test student: :func:`pixel_action` on the newest frame and proprio."""

    privileged = False
    needs_motion_feed = False

    def reset(self, seed: int, context: Any = None, motion_feed: Any = None) -> None:
        """Refuse a context or a feed; nothing to clear."""
        del seed
        if context is not None or motion_feed is not None:
            raise ValueError("a student is never privileged or fed")

    def act_vision(
        self, image_stack: NDArray[np.uint8], proprio: NDArray[np.float32]
    ) -> NDArray[np.float64]:
        """Return :func:`pixel_action`."""
        return pixel_action(image_stack, proprio)

    def act(self, obs: Any) -> Any:
        """Never called by the runner: a student never sees P8's observation."""
        del obs
        raise TypeError("PixelStudent.act(obs) called: students take act_vision only")


def build_pixel_student(cfgs: EvalConfigs) -> PixelStudent:
    """Picklable factory for :class:`PixelStudent`."""
    del cfgs
    return PixelStudent()


def replay_inputs(
    listed: ListedEpisode, cfgs: EvalConfigs, n_steps: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Re-fly one episode by hand and build the contract input at every step.

    The episode is flown with :func:`pixel_action` on each step's contract input, the way
    :class:`PixelStudent` flies it through the runner, so the inputs a student received there
    can be compared with these byte for byte. Window indices are ``max(0, t-To+1+i)``, written
    out here independently of ``ila.policies.dp.chunking``.
    """
    contract = load_contract()
    to = contract.n_obs_steps
    env = p8_adapter.make_env(cfgs, listed)
    out = []
    try:
        p8_adapter.set_motion(env, cfgs, listed)
        obs, _ = env.reset(seed=listed.episode_seed)
        sc = scene.attach(env)
        frames, observations = [sc.render().rgb], [obs]
        for t in range(n_steps):
            window = [max(0, i) for i in range(t - to + 1, t + 1)]
            si = contract.student_input(
                np.stack([frames[i] for i in window]), np.stack([observations[i] for i in window])
            )
            assert si.proprio_fields == contract.student_proprio
            out.append((si.images, si.proprio))
            obs, _, terminated, truncated, _ = env.step(pixel_action(si.images, si.proprio))
            frames.append(sc.render().rgb)
            observations.append(obs)
            if terminated or truncated:
                break
    finally:
        env.close()
    return out
