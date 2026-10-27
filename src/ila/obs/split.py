"""The only place observation fields are selected (Phase 1).

Splits P8's 25-d observation, by field name from P8's ``OBS_FIELDS``, into the student
proprio allowlist (``configs/obs/student.yaml``) and the privileged block. Privileged
fields never reach a student input.
"""
