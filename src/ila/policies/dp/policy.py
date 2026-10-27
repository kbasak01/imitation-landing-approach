"""Diffusion Policy: DDPM training, DDIM sampling, chunk execution (Phase 6).

Sampling noise comes from a per-episode ``torch.Generator`` seeded by
``(run_seed, episode_seed, step)``.
"""
