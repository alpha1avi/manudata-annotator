"""Hitting a target file size in one encoding pass.

The reel gets attached to a customer email, so ``--max-size-mb`` is a hard
ceiling, not a wish. Re-encoding to hunt for the right CRF would mean
re-rendering every frame, which on a rented GPU is the most expensive
possible way to save a few megabytes. So instead of iterating on CRF we
cap the bitrate directly and let quality float: CRF drives quality when
there is headroom, the cap takes over when there is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Leave room for the container overhead and the moov atom.
CONTAINER_HEADROOM = 0.94
# Never drive quality below something an evaluator would find acceptable.
MIN_BITRATE_KBPS = 900
DEFAULT_CRF = 21


@dataclass
class EncodeQuality:
    crf: int
    max_bitrate_kbps: Optional[int]

    @property
    def bufsize_kbps(self) -> Optional[int]:
        return self.max_bitrate_kbps * 2 if self.max_bitrate_kbps else None


def plan_quality(
    duration_s: float,
    max_size_mb: Optional[float],
    crf: int = DEFAULT_CRF,
) -> EncodeQuality:
    """Choose CRF and an optional bitrate ceiling for a clip of *duration_s*."""
    if not max_size_mb or duration_s <= 0:
        return EncodeQuality(crf=crf, max_bitrate_kbps=None)

    budget_bits = max_size_mb * 8 * 1024 * 1024 * CONTAINER_HEADROOM
    kbps = int(budget_bits / duration_s / 1000)

    if kbps < MIN_BITRATE_KBPS:
        logger.warning(
            "A %.1f MB ceiling over %.1fs works out to %d kbps, below the %d kbps "
            "quality floor. Clamping to the floor — the output will exceed the "
            "target. Shorten the clip or raise --max-size-mb.",
            max_size_mb, duration_s, kbps, MIN_BITRATE_KBPS,
        )
        kbps = MIN_BITRATE_KBPS

    return EncodeQuality(crf=crf, max_bitrate_kbps=kbps)
