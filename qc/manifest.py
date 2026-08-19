"""Site and task labels, read from a manifest CSV.

These labels are burned into every output frame and printed in the
report a customer reads, so they are never guessed. A missing or blank
row is a hard error rather than an "unknown" placeholder: mislabelling
which factory a clip came from is exactly the kind of quiet mistake that
costs credibility when the evaluator notices it and we did not.

``init-manifest`` generates the file with the site column pre-filled
from the containing folder, leaving the task column for a human.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

FIELDNAMES = ["filename", "site", "task", "notes"]
DEFAULT_MANIFEST_NAME = "manifest.csv"


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Label:
    site: str
    task: str
    notes: str = ""


def split_camel(token: str) -> str:
    """``AlpineFootwear01`` -> ``Alpine Footwear 01``."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|\d+|[A-Z]+", token)
    return " ".join(parts) if parts else token


def parse_filename(stem: str) -> Optional[Label]:
    """Read site and task out of a structured capture filename.

    Recognises the delivery convention
    ``Country_City_Site_Task_NNN_NNN``, e.g.::

        India_Faridabad_AlpineFootwear01_ShoeAssembly_003_039
                        ^site            ^task

    Returns ``None`` for anything that does not clearly match, so an
    off-convention name falls back to folder-derived site and a blank
    task rather than silently inventing a label. The trailing numeric
    tokens are required precisely because they are what distinguishes
    this convention from an arbitrary underscore-separated name.
    """
    tokens = [t for t in stem.split("_") if t]
    if len(tokens) < 5:
        return None

    country, city, site, task = tokens[0], tokens[1], tokens[2], tokens[3]
    trailing = tokens[4:]
    if not trailing or not all(t.isdigit() for t in trailing):
        return None
    if not (country.isalpha() and city.replace("-", "").isalpha()):
        return None

    return Label(
        site=split_camel(site),
        task=split_camel(task).lower(),
        notes=f"{split_camel(city)}, {split_camel(country)} · clip {'-'.join(trailing)}",
    )


def prettify_site(raw: str) -> str:
    """Turn a folder name into a presentable site label.

    ``tangerine shoes vl`` -> ``Tangerine Shoes VL``.  Short tokens are
    upper-cased rather than title-cased because they are almost always
    initialisms, and ``Vl`` on a customer-facing frame looks like a typo.
    """
    tokens = [t for t in raw.replace("_", " ").replace("-", " ").split() if t]
    out: List[str] = []
    for token in tokens:
        out.append(token.upper() if len(token) <= 2 else token[:1].upper() + token[1:])
    return " ".join(out) or raw


def load(path: Path) -> Dict[str, Label]:
    """Read a manifest into a filename-keyed mapping (case-insensitive)."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(
            f"No manifest at {path}. Generate one with:\n"
            f"    manudata-qc-render init-manifest <video-dirs...> --out {path}\n"
            "then fill in the task column."
        )

    labels: Dict[str, Label] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManifestError(f"{path} is empty")
        missing = {"filename", "site", "task"} - set(
            (f or "").strip() for f in reader.fieldnames
        )
        if missing:
            raise ManifestError(
                f"{path} is missing required column(s): {sorted(missing)}. "
                f"Expected at least filename, site, task."
            )

        for lineno, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            key = filename.lower()
            if key in labels:
                raise ManifestError(
                    f"{path} line {lineno}: duplicate row for {filename!r}"
                )
            labels[key] = Label(
                site=(row.get("site") or "").strip(),
                task=(row.get("task") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )

    if not labels:
        raise ManifestError(f"{path} has a header but no rows")
    return labels


def resolve(video_path: Path, labels: Dict[str, Label]) -> Label:
    """Look up one video's label, failing loudly when it is unusable."""
    key = Path(video_path).name.lower()
    label = labels.get(key)
    if label is None:
        raise ManifestError(
            f"{Path(video_path).name} has no row in the manifest. Add one, or "
            "re-run init-manifest to pick up newly added videos."
        )
    blank = [name for name, value in (("site", label.site), ("task", label.task))
             if not value]
    if blank:
        raise ManifestError(
            f"{Path(video_path).name}: manifest column(s) {blank} are blank. "
            "These are burned into every frame of the render, so they have to "
            "be filled in before rendering."
        )
    return label


def init(video_paths: Iterable[Path], out_path: Path, root: Optional[Path] = None) -> Path:
    """Write a manifest template, preserving any rows already filled in.

    Re-running after adding videos is safe: existing rows keep whatever
    was typed into them and only genuinely new files are appended.
    """
    out_path = Path(out_path)
    existing: Dict[str, Label] = {}
    if out_path.exists():
        try:
            existing = load(out_path)
            logger.info("Merging into existing manifest at %s", out_path)
        except ManifestError as exc:
            raise ManifestError(
                f"Refusing to overwrite {out_path}, which exists but could not "
                f"be read: {exc}"
            ) from exc

    rows = []
    added = 0
    parsed_rows = 0
    for path in video_paths:
        path = Path(path)
        key = path.name.lower()
        prior = existing.get(key)
        if prior is not None:
            rows.append({
                "filename": path.name, "site": prior.site,
                "task": prior.task, "notes": prior.notes,
            })
            continue
        added += 1
        # A structured capture filename carries both labels; fall back to
        # the folder name for site and leave task for a human otherwise.
        parsed = parse_filename(path.stem)
        if parsed is not None:
            parsed_rows += 1
            rows.append({
                "filename": path.name, "site": parsed.site,
                "task": parsed.task, "notes": parsed.notes,
            })
        else:
            rows.append({
                "filename": path.name,
                "site": prettify_site(path.parent.name),
                "task": "",
                "notes": "",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_path)

    blank_tasks = sum(1 for r in rows if not r["task"])
    logger.info(
        "Wrote %s: %d rows (%d new, %d labelled from the filename convention).",
        out_path, len(rows), added, parsed_rows,
    )
    if blank_tasks:
        logger.warning(
            "%d row(s) still have a blank task and will not render until it is "
            "filled in.", blank_tasks,
        )
    return out_path
