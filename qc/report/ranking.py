"""Ranked stdout summary — the thing actually used to pick reel clips.

Printed best-first and sized to a terminal, because the whole point is
to glance at it after the analysis pass and decide what to render.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from qc.report.analyze import VideoStats

BAR_WIDTH = 12


def format_table(stats: Sequence[VideoStats]) -> str:
    """Render the ranked table as a string."""
    if not stats:
        return "No videos analysed."

    name_w = max(8, min(34, max(len(s.filename) for s in stats)))
    site_w = max(4, min(18, max(len(s.site) for s in stats)))
    task_w = max(4, min(22, max(len(s.task) for s in stats)))

    header = (
        f"{'#':>3}  {'VIDEO':<{name_w}}  {'SITE':<{site_w}}  {'TASK':<{task_w}}  "
        f"{'RECOVERY':>9}  {'':<{BAR_WIDTH}}  {'OCCLUDED':>8}  {'2-HAND':>7}  "
        f"{'GAP':>6}  {'BEST CLIP':>15}"
    )
    lines = [header, "-" * len(header)]

    for i, s in enumerate(stats, start=1):
        recovery = "     n/a" if math.isnan(s.pose_recovery_pct) else f"{s.pose_recovery_pct:7.1f}%"
        clip = (
            f"{s.recommended_clip_start_s:6.1f}-{s.recommended_clip_end_s:6.1f}s"
            if s.clip else "            n/a"
        )
        two_hand = (
            100.0 * s.frames_both_hands / s.total_frames if s.total_frames else 0.0
        )
        lines.append(
            f"{i:>3}  {_clip(s.filename, name_w):<{name_w}}  "
            f"{_clip(s.site, site_w):<{site_w}}  {_clip(s.task, task_w):<{task_w}}  "
            f"{recovery:>9}  {_bar(s.pose_recovery_pct):<{BAR_WIDTH}}  "
            f"{s.occluded_or_absent_pct:7.1f}%  {two_hand:6.1f}%  "
            f"{s.longest_gap_s:5.1f}s  {clip:>15}"
        )

    lines.append("")
    lines.extend(portfolio_summary(stats))
    return "\n".join(lines)


def portfolio_summary(stats: Sequence[VideoStats]) -> List[str]:
    """Portfolio-wide figures, stated the same way the customer sees them."""
    total_slots = sum(s.total_frames * 2 for s in stats)
    if not total_slots:
        return []

    occluded_slots = sum(
        s.occluded_or_absent_pct / 100.0 * s.total_frames * 2 for s in stats
    )
    visible_slots = total_slots - occluded_slots
    recovered_slots = sum(
        s.frames_both_hands * 2 + s.frames_one_hand for s in stats
    )
    recovery = 100.0 * recovered_slots / visible_slots if visible_slots else float("nan")

    return [
        f"Portfolio: {len(stats)} videos, "
        f"{sum(s.duration_s for s in stats) / 60.0:.1f} min total.",
        f"  Hand visible in {100.0 * visible_slots / total_slots:.1f}% of hand-slots "
        f"({occluded_slots / total_slots * 100.0:.1f}% occluded or out of frame).",
        f"  Pose recovered on {recovery:.1f}% of visible-hand slots.",
    ]


def _bar(pct: float) -> str:
    if math.isnan(pct):
        return "?" * 0
    filled = int(round(BAR_WIDTH * max(0.0, min(100.0, pct)) / 100.0))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
