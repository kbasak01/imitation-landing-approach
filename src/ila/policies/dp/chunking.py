"""The only place ``To``/``Tp``/``Ta`` indices are computed (Phase 6).

``Ta <= Tp - To + 1``; executed actions start at chunk index ``To - 1``. One control
step is 1/30 s model scale.
"""
