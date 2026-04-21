"""
Discover ego camera frames under ``<sequence_dir>/frames``.

Filenames follow ``<sec>_<frac>.png``: **wall-clock** epoch second plus a fractional part
``frac`` made of digits only. ``frac`` is normalized to **exactly 9 characters** before
interpreting as nanoseconds: if longer than 9 digits, only the **first 9** are kept; if
shorter, **zeros are appended on the right** (e.g. ``5`` → ``500000000`` ns). Sorting is
numeric on ``(sec, nanosec)``, not lexicographic on the stem string.

Callers that need alignment with merged logs should subtract the recording start epoch
(first ``sec`` + ``nanosec`` on the prepared dataframe) from each ``t_sec``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict


class CameraFrameRecord(TypedDict):
    """One decoded frame path aligned to ``t_sec`` (seconds since epoch, float)."""

    path: str
    t_sec: float


_STEM_RE = re.compile(r"^(-?\d+)_(\d+)$")


def _normalize_fraction_nanos(frac_s: str) -> int | None:
    """
    Map digit string ``frac`` to nanoseconds in ``[0, 999_999_999]``.

    Truncate to 9 digits if longer; right-pad with ``'0'`` if shorter than 9.
    Returns ``None`` if ``frac_s`` is empty or not all decimal digits.
    """
    if not frac_s or not frac_s.isdigit():
        return None
    if len(frac_s) > 9:
        frac_s = frac_s[:9]
    elif len(frac_s) < 9:
        frac_s = frac_s + "0" * (9 - len(frac_s))
    return int(frac_s)


def _parse_frame_timestamp(stem: str) -> tuple[int, int] | None:
    """
    Parse ``<sec>_<frac>`` from filename stem (no extension).

    ``sec`` is a signed integer epoch second; ``frac`` is a digit string normalized to 9
    digits (truncate / right-pad) then read as nanoseconds. Rejects stems that do not
    match the stem pattern ``(-?digits)_(digits)`` (one underscore; see ``_STEM_RE``).
    """
    m = _STEM_RE.match(stem)
    if m is None:
        return None
    sec_s, frac_s = m.group(1), m.group(2)
    sec = int(sec_s)
    nanosec = _normalize_fraction_nanos(frac_s)
    if nanosec is None:
        return None
    return sec, nanosec


def discover_frame_timeline(sequence_dir: str | Path) -> list[CameraFrameRecord]:
    """
    List ``*.png`` under ``sequence_dir/frames`` with valid ``<sec>_<frac>`` stems.

    The fractional segment is normalized to 9 digits (see :func:`_normalize_fraction_nanos`).
    Returns records sorted by ``t_sec``. If ``frames`` is missing, not a directory, or
    has no valid PNGs, returns an empty list.
    """
    root = Path(sequence_dir).expanduser().resolve()
    frames_dir = root / "frames"
    if not frames_dir.is_dir():
        return []

    records: list[tuple[int, int, str]] = []
    for p in frames_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        parsed = _parse_frame_timestamp(p.stem)
        if parsed is None:
            continue
        sec, nanosec = parsed
        records.append((sec, nanosec, str(p.resolve())))

    records.sort(key=lambda x: (x[0], x[1]))
    out: list[CameraFrameRecord] = []
    for sec, nanosec, path_str in records:
        t_sec = float(sec) + float(nanosec) * 1e-9
        out.append({"path": path_str, "t_sec": t_sec})
    return out
