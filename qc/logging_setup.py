"""Logging to stdout and to a file.

The file handler is not optional. When something fails at hour three of
a rented instance, the terminal scrollback is usually gone and the
question is always "what actually happened to video 17" — so every run
leaves a timestamped log next to the outputs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
DATE_FORMAT = "%H:%M:%S"


class _TqdmSafeHandler(logging.StreamHandler):
    """Write through tqdm so log lines do not shred the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            super().emit(record)


def setup(log_dir: Optional[Path] = None, verbose: bool = False) -> Optional[Path]:
    """Configure root logging. Returns the log file path, if any."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = _TqdmSafeHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    if log_dir is None:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"qc-render-{stamp}.log"

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
    )
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", path)
    return path
