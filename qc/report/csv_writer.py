"""Write ``qc_report.csv``."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List

from qc.report.analyze import VideoStats

COLUMNS = [
    "filename",
    "site",
    "task",
    "duration_s",
    "total_frames",
    "frames_both_hands",
    "frames_one_hand",
    "frames_zero_hands",
    "frames_visible_no_pose",
    "occluded_or_absent_pct",
    "pose_recovery_pct",
    "longest_gap_s",
    "mean_confidence",
    "recommended_clip_start_s",
    "recommended_clip_end_s",
]

_FLOAT_PRECISION = {
    "duration_s": 2,
    "occluded_or_absent_pct": 2,
    "pose_recovery_pct": 2,
    "longest_gap_s": 2,
    "mean_confidence": 4,
    "recommended_clip_start_s": 2,
    "recommended_clip_end_s": 2,
}


def write(stats: Iterable[VideoStats], path: Path) -> Path:
    """Write the report atomically so a killed run leaves the old one intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in stats:
            writer.writerow(_format_row(row))

    tmp.replace(path)
    return path


def _format_row(row: VideoStats) -> dict:
    out = {}
    for column in COLUMNS:
        value = getattr(row, column)
        if isinstance(value, float):
            if math.isnan(value):
                # An undefined recovery rate (no hand ever visible) is
                # written blank, not as 0 — those mean different things
                # and a spreadsheet would average a 0 into the portfolio.
                out[column] = ""
                continue
            out[column] = f"{value:.{_FLOAT_PRECISION.get(column, 2)}f}"
        else:
            out[column] = value
    return out
