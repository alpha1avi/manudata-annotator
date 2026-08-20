"""Progress reporting for the two passes that take tens of minutes.

A tqdm bar is the right thing on an attached terminal and the wrong
thing everywhere else this tool actually runs: several workers writing
bars to one terminal interleave into noise, and a bar does not survive
being piped through ``tee`` into a batch log. So when the bar is off,
the same information goes out as periodic log lines instead.

The rule this module exists to enforce: **no long pass is ever silent.**
An operator on a rented instance cannot tell a working run from a wedged
one without output, and the difference between those two is an hour of
paid GPU time.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 30.0


def human_duration(seconds: float) -> str:
    """Format a duration the way an operator reads it, not in raw seconds."""
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class PeriodicProgress:
    """Throughput and ETA as log lines, at most one every *interval_s*.

    *start* is work already completed by an earlier run that this one
    resumed from. It counts towards the percentage but never towards the
    measured rate — otherwise a run resumed at 90% would report a
    throughput it never achieved, and an ETA derived from it.
    """

    def __init__(
        self,
        name: str,
        total: int,
        interval_s: float = DEFAULT_INTERVAL_S,
        start: int = 0,
        unit: str = "frames",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.total = max(1, int(total))
        self.interval_s = interval_s
        self.start = max(0, int(start))
        self.unit = unit
        self.enabled = enabled
        self.started = time.monotonic()
        self._last_log = self.started

    def rate(self, done: int, elapsed: float) -> float:
        """Frames per second over work done by *this* run."""
        fresh = done - self.start
        return fresh / elapsed if elapsed > 0 and fresh > 0 else 0.0

    def update(self, done: int, suffix: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_log < self.interval_s:
            return
        self._last_log = now

        rate = self.rate(done, now - self.started)
        remaining = (self.total - done) / rate if rate > 0 else float("inf")
        logger.info(
            "%s: %d/%d %s (%.1f%%) | %.1f fps | %s left%s",
            self.name, done, self.total, self.unit,
            100.0 * done / self.total, rate, human_duration(remaining), suffix,
        )

    def close(self, done: int, what: str = "done", suffix: str = "") -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started
        resumed = f" (resumed at {self.start})" if self.start else ""
        logger.info(
            "%s: %s — %d %s in %s (%.1f fps)%s%s",
            self.name, what, done, self.unit, human_duration(elapsed),
            self.rate(done, elapsed), resumed, suffix,
        )
