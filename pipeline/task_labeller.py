"""ManuData Annotator — Task Labeller (Stage 4 Integration).

Dual-mode labelling engine:
    - **bootstrap** — every batch goes to a cloud VLM.
    - **annotate** — local fine-tuned model with optional VLM fallback
      for low-confidence or confused-pair predictions.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from config import AnnotatorConfig, ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
from pipeline.local_vision_pipeline import FrameAnnotation, LocalVisionPipeline
from utils.image_utils import frame_to_base64, resize_frame, load_frame

# Lazy VLM imports — these pull in httpx / tenacity which may not be installed
BaseVLMClient = None  # type: ignore[assignment]


def _ensure_vlm_imports():
    """Import VLM client modules on first use."""
    global BaseVLMClient
    if BaseVLMClient is not None:
        return
    from vlm.base_client import BaseVLMClient as _Base
    BaseVLMClient = _Base

logger = logging.getLogger(__name__)


# ── result dataclass ──────────────────────────────────────────────────


@dataclass
class TaskLabel:
    """Structured action label for a batch of frames."""

    action: str
    action_description: str
    objects_involved: List[str]
    grasp_type: str
    hand_used: str
    manipulation_phase: str
    task_hierarchy: Dict[str, str]  # high_level, mid_level, low_level
    confidence: float
    is_idle: bool
    is_transition: bool
    labelling_method: str           # "vlm" | "local_model" | "local_model_vlm_fallback"
    was_fallback: bool
    vlm_backend: Optional[str]
    raw_response: Dict[str, Any]
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


# ── labeller ──────────────────────────────────────────────────────────


class TaskLabeller:
    """Dual-mode task labelling: VLM bootstrap or local model + fallback."""

    def __init__(self, mode: str, config: AnnotatorConfig) -> None:
        """Initialise the labeller.

        Args:
            mode: ``"bootstrap"`` (pure VLM) or ``"annotate"`` (local + fallback).
            config: Pipeline configuration.
        """
        self.mode = mode
        self.config = config
        self.vlm_client = None
        self.local_model = None
        self._class_mapping: Dict[int, str] = {}
        self._confused_pairs: List[List[str]] = []
        self._vision_pipeline: Optional[LocalVisionPipeline] = None

        if mode == "bootstrap":
            self.vlm_client = self._create_vlm_client(config)
        elif mode == "annotate":
            self.local_model = self._load_local_model(config)
            if not getattr(config, "_no_fallback", False):
                self.vlm_client = self._create_vlm_client(config)

    # ── factory helpers ───────────────────────────────────────────────

    @staticmethod
    def _create_vlm_client(config: AnnotatorConfig):
        """Instantiate the VLM client matching ``config.vlm_backend``."""
        _ensure_vlm_imports()
        from vlm.gemini_client import GeminiClient
        from vlm.claude_client import ClaudeClient
        from vlm.openai_client import OpenAIClient

        backend = config.vlm_backend

        if backend == "gemini":
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY not set in .env")
            return GeminiClient(GOOGLE_API_KEY, max_concurrent=config.max_concurrent)

        if backend in ("claude_haiku", "claude_sonnet"):
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
            model = (
                "claude-haiku-4-5-20251001"
                if backend == "claude_haiku"
                else "claude-sonnet-4-20250514"
            )
            return ClaudeClient(ANTHROPIC_API_KEY, model=model, max_concurrent=config.max_concurrent)

        if backend == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set in .env")
            return OpenAIClient(OPENAI_API_KEY, max_concurrent=config.max_concurrent)

        raise ValueError(f"Unknown VLM backend: {backend}")

    @staticmethod
    def _load_local_model(config: AnnotatorConfig):
        """Load a fine-tuned VideoMAE-v2 model for local inference.

        Expects:
            - ``config.model_path`` pointing to a ``.pt`` / ``.pth`` file.
            - ``class_mapping.json`` in the same directory.
            - ``confused_pairs.json`` (optional) in the same directory.
        """
        model_path = Path(config.model_path)
        if not config.model_path or not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found: {config.model_path}. "
                "Provide a valid --model-path for annotate mode."
            )

        model_dir = model_path.parent

        # Load class mapping
        mapping_path = model_dir / "class_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"class_mapping.json not found in {model_dir}. "
                "This file is required for local model inference."
            )

        import torch

        logger.info("Loading local model from %s …", model_path)
        t0 = time.time()
        model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "model_state_dict" in model:
            # Checkpoint format — caller must reconstruct architecture
            logger.info("Loaded model checkpoint (state_dict)")
        logger.info("Local model loaded in %.2fs", time.time() - t0)

        return model

    # ── single-batch labelling ────────────────────────────────────────

    async def label_batch(
        self,
        frame_paths: List[Tuple[float, str]],
        frame_annotations: List[FrameAnnotation],
    ) -> TaskLabel:
        """Label a single batch of frames.

        **Bootstrap mode:**
            1. Resize and base64-encode frames.
            2. Generate VLM context from ``frame_annotations``.
            3. Send to VLM client.
            4. Parse response into :class:`TaskLabel`.

        **Annotate mode:**
            1. Run local model on frames.
            2. If confidence < ``fallback_threshold`` and fallback is
               enabled, send to VLM as fallback.
            3. Return :class:`TaskLabel`.

        Args:
            frame_paths: ``(timestamp, filepath)`` list for this batch.
            frame_annotations: Corresponding :class:`FrameAnnotation` list.

        Returns:
            :class:`TaskLabel`.
        """
        ts_start = frame_paths[0][0]
        ts_end = frame_paths[-1][0]

        if self.mode == "bootstrap":
            return await self._label_via_vlm(
                frame_paths, frame_annotations, ts_start, ts_end, is_fallback=False
            )
        else:
            return await self._label_via_local(
                frame_paths, frame_annotations, ts_start, ts_end
            )

    # ── full-video labelling ──────────────────────────────────────────

    async def label_video(
        self,
        good_frames: List[Tuple[float, str]],
        frame_annotations: List[FrameAnnotation],
        idle_frames: List[Tuple[float, str]],
    ) -> List[TaskLabel]:
        """Process an entire video's frames.

        1. Split ``good_frames`` into batches of ``config.batch_size``.
        2. Run :meth:`label_batch` on each (async, semaphore-bounded).
        3. Auto-label ``idle_frames`` without any VLM call.
        4. Return all :class:`TaskLabel` in timestamp order.
        """
        batch_size = self.config.batch_size
        labels: List[TaskLabel] = []

        # Build annotation lookup
        ann_by_ts: Dict[float, FrameAnnotation] = {
            a.timestamp: a for a in frame_annotations
        }

        # Create batches
        batches: List[List[Tuple[float, str]]] = []
        for i in range(0, len(good_frames), batch_size):
            batches.append(good_frames[i : i + batch_size])

        logger.info(
            "Labelling %d frames in %d batches (batch_size=%d, mode=%s) …",
            len(good_frames), len(batches), batch_size, self.mode,
        )

        # Process batches with progress bar
        pbar = tqdm(total=len(batches), desc="Task labelling", unit="batch")

        async def _process_batch(idx: int, batch: List[Tuple[float, str]]) -> TaskLabel:
            batch_anns = [ann_by_ts.get(ts) for ts, _ in batch]
            # Filter out None annotations
            batch_anns = [a for a in batch_anns if a is not None]
            result = await self.label_batch(batch, batch_anns)
            pbar.update(1)

            # Print cost every 10 batches
            if self.vlm_client and (idx + 1) % 10 == 0:
                cost = self.vlm_client.total_cost
                logger.info(
                    "Progress: %d/%d batches | running cost: ₹%.2f",
                    idx + 1, len(batches), cost,
                )
            return result

        # Run batches (concurrency controlled by VLM client semaphore)
        tasks = [_process_batch(i, b) for i, b in enumerate(batches)]
        batch_labels = await asyncio.gather(*tasks)
        labels.extend(batch_labels)
        pbar.close()

        # Auto-label idle frames
        for ts, fpath in idle_frames:
            idle_label = TaskLabel(
                action="idle",
                action_description="No productive manipulation activity detected",
                objects_involved=[],
                grasp_type="none",
                hand_used="none",
                manipulation_phase="idle",
                task_hierarchy={
                    "high_level": "idle",
                    "mid_level": "idle",
                    "low_level": "idle",
                },
                confidence=0.95,
                is_idle=True,
                is_transition=False,
                labelling_method=self.mode,
                was_fallback=False,
                vlm_backend=None,
                raw_response={},
                timestamp_start=ts,
                timestamp_end=ts,
            )
            labels.append(idle_label)

        # Sort by timestamp
        labels.sort(key=lambda l: l.timestamp_start)

        logger.info(
            "Labelling complete: %d labels (%d active + %d idle)",
            len(labels), len(batch_labels), len(idle_frames),
        )
        return labels

    # ── cost summary ──────────────────────────────────────────────────

    def get_cost_summary(self) -> Dict[str, Any]:
        """Return cumulative VLM cost statistics."""
        if self.vlm_client is None:
            return {
                "total_calls": 0,
                "total_cost_inr": 0.0,
                "cost_per_call_inr": 0.0,
                "vlm_backend": None,
            }

        total_calls = self.vlm_client.total_calls
        total_cost = self.vlm_client.total_cost

        return {
            "total_calls": total_calls,
            "total_cost_inr": round(total_cost, 2),
            "cost_per_call_inr": (
                round(total_cost / total_calls, 4) if total_calls > 0 else 0.0
            ),
            "vlm_backend": self.vlm_client._backend_name,
        }

    # ── cleanup ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release VLM client resources."""
        if self.vlm_client is not None:
            await self.vlm_client.close()

    # ── internal: VLM labelling ───────────────────────────────────────

    async def _label_via_vlm(
        self,
        frame_paths: List[Tuple[float, str]],
        frame_annotations: List[FrameAnnotation],
        ts_start: float,
        ts_end: float,
        is_fallback: bool,
    ) -> TaskLabel:
        """Encode frames, build context, call VLM, parse response."""
        # 1. Resize & base64 encode
        b64_frames: List[str] = []
        for _, fpath in frame_paths:
            frame = load_frame(fpath)
            frame = resize_frame(frame, self.config.frame_max_size)
            b64_frames.append(frame_to_base64(frame, self.config.jpeg_quality))

        # 2. Generate VLM context from local CV metadata
        context = ""
        if frame_annotations and self._vision_pipeline is not None:
            context = self._vision_pipeline.generate_vlm_context(frame_annotations)
        elif frame_annotations:
            # Inline context generation without full pipeline reference
            context = self._build_inline_context(frame_annotations)

        # 3. Call VLM
        response = await self.vlm_client.predict_batch_with_retry(b64_frames, context)

        # 4. Parse into TaskLabel
        return self._response_to_label(
            response, ts_start, ts_end,
            method="vlm" if not is_fallback else "local_model_vlm_fallback",
            is_fallback=is_fallback,
        )

    async def _label_via_local(
        self,
        frame_paths: List[Tuple[float, str]],
        frame_annotations: List[FrameAnnotation],
        ts_start: float,
        ts_end: float,
    ) -> TaskLabel:
        """Run local model, optionally fall back to VLM."""
        # Placeholder: local model inference
        # In a real implementation this would run the fine-tuned model
        local_confidence = 0.0
        local_action = "unknown"
        local_raw: Dict[str, Any] = {}

        if self.local_model is not None:
            try:
                import torch
                import cv2

                frames_tensor = []
                for _, fpath in frame_paths:
                    f = load_frame(fpath)
                    f = resize_frame(f, 224)
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                    frames_tensor.append(t)

                batch = torch.stack(frames_tensor).unsqueeze(0)  # (1, T, C, H, W)

                if hasattr(self.local_model, "eval"):
                    self.local_model.eval()
                    with torch.no_grad():
                        output = self.local_model(batch)
                    probs = torch.softmax(output, dim=-1)
                    local_confidence = float(probs.max())
                    cls_id = int(probs.argmax())
                    local_action = self._class_mapping.get(cls_id, f"class_{cls_id}")
                    local_raw = {
                        "class_id": cls_id,
                        "confidence": local_confidence,
                        "top5": probs.topk(min(5, probs.shape[-1])).indices.tolist(),
                    }
            except Exception as exc:
                logger.warning("Local model inference failed: %s", exc)
                local_confidence = 0.0

        # Check fallback conditions
        needs_fallback = (
            self.vlm_client is not None
            and local_confidence < self.config.fallback_threshold
        )

        # Check confused pairs
        if not needs_fallback and self.vlm_client is not None:
            for pair in self._confused_pairs:
                if local_action in pair:
                    logger.info(
                        "Action '%s' is in confused-pairs list — triggering fallback",
                        local_action,
                    )
                    needs_fallback = True
                    break

        if needs_fallback:
            logger.info(
                "Local confidence %.3f < threshold %.2f — VLM fallback",
                local_confidence, self.config.fallback_threshold,
            )
            return await self._label_via_vlm(
                frame_paths, frame_annotations, ts_start, ts_end, is_fallback=True
            )

        # Return local-only result
        return TaskLabel(
            action=local_action,
            action_description=f"Locally predicted: {local_action}",
            objects_involved=[],
            grasp_type="unknown",
            hand_used="unknown",
            manipulation_phase="unknown",
            task_hierarchy={
                "high_level": "unknown",
                "mid_level": local_action,
                "low_level": local_action,
            },
            confidence=local_confidence,
            is_idle=local_action == "idle",
            is_transition=local_action == "transition",
            labelling_method="local_model",
            was_fallback=False,
            vlm_backend=None,
            raw_response=local_raw,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
        )

    # ── response parsing ──────────────────────────────────────────────

    def _response_to_label(
        self,
        response: dict,
        ts_start: float,
        ts_end: float,
        method: str,
        is_fallback: bool,
    ) -> TaskLabel:
        """Convert a parsed VLM response dict into a :class:`TaskLabel`."""
        actions = response.get("actions", [])
        if not actions:
            return TaskLabel(
                action="unknown",
                action_description="VLM returned no actions",
                objects_involved=[],
                grasp_type="none",
                hand_used="none",
                manipulation_phase="unknown",
                task_hierarchy={},
                confidence=0.0,
                is_idle=False,
                is_transition=False,
                labelling_method=method,
                was_fallback=is_fallback,
                vlm_backend=self.vlm_client._backend_name if self.vlm_client else None,
                raw_response=response,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
            )

        # Use the first (primary) action
        act = actions[0]

        return TaskLabel(
            action=act.get("action", "unknown"),
            action_description=act.get("action_description", ""),
            objects_involved=act.get("objects_involved", []),
            grasp_type=act.get("grasp_type", "none"),
            hand_used=act.get("hand_used", "none"),
            manipulation_phase=act.get("manipulation_phase", "unknown"),
            task_hierarchy={
                "high_level": act.get("high_level_task", "unknown"),
                "mid_level": act.get("action", "unknown"),
                "low_level": act.get("action_description", ""),
            },
            confidence=float(act.get("confidence", 0.0)),
            is_idle=bool(act.get("is_idle", False)),
            is_transition=bool(act.get("is_transition", False)),
            labelling_method=method,
            was_fallback=is_fallback,
            vlm_backend=self.vlm_client._backend_name if self.vlm_client else None,
            raw_response=response,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
        )

    # ── inline context builder ────────────────────────────────────────

    @staticmethod
    def _build_inline_context(annotations: List[FrameAnnotation]) -> str:
        """Build a lightweight VLM context string without the full pipeline."""
        if not annotations:
            return ""

        lines = [
            f"Local CV metadata for frames {annotations[0].timestamp:.1f}s "
            f"- {annotations[-1].timestamp:.1f}s:"
        ]

        # Hands
        hand_labels = set()
        for ann in annotations:
            for h in ann.hand_pose.hands:
                hand_labels.add(h.handedness)
        if hand_labels:
            lines.append(f"  Hands: {', '.join(sorted(hand_labels))} detected")
        else:
            lines.append("  Hands: none detected")

        # Objects (deduplicated, best confidence)
        best_objs: Dict[str, float] = {}
        for ann in annotations:
            for obj in ann.objects:
                if obj.class_name not in best_objs or obj.confidence > best_objs[obj.class_name]:
                    best_objs[obj.class_name] = obj.confidence
        if best_objs:
            parts = [f"{n} (conf {c:.2f})" for n, c in sorted(best_objs.items(), key=lambda x: -x[1])]
            lines.append(f"  Objects: {', '.join(parts)}")

        # Interactions
        all_ix = []
        for ann in annotations:
            all_ix.extend(ann.interactions)
        if all_ix:
            top = sorted(all_ix, key=lambda i: i.contact_score, reverse=True)[:3]
            parts = [
                f"{ix.hand} hand -> {ix.object.class_name} (score {ix.contact_score:.2f})"
                for ix in top
            ]
            lines.append(f"  Interactions: {'; '.join(parts)}")

        # Mean activity
        scores = [a.activity_score for a in annotations]
        mean_act = sum(scores) / max(len(scores), 1)
        lines.append(f"  Activity score: {mean_act:.2f}")

        return "\n".join(lines)


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)
    print("TaskLabeller module loaded successfully.")
    print("  Modes: bootstrap, annotate")
    print("  VLM backends: gemini, claude_haiku, claude_sonnet, openai")
    print("  Use label_video() for full pipeline execution.")
