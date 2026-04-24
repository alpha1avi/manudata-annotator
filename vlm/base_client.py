"""ManuData Annotator — Base VLM Client.

Abstract base class for all Vision-Language Model API backends.
Provides retry logic, concurrency control, JSON response parsing,
cost tracking, and the shared egocentric system prompt.
"""

import asyncio
import json
import logging
import re
import time
from typing import List

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False

logger = logging.getLogger(__name__)


# ── Egocentric VLM system prompt (shared by all backends) ─────────────

EGOCENTRIC_SYSTEM_PROMPT = """\
You are an expert at recognizing human manipulation actions from EGOCENTRIC (first-person) video frames captured in manufacturing and assembly environments.

You will receive a batch of sequential frames from a helmet-mounted or chest-mounted camera. The camera wearer is a factory worker performing manual tasks. Analyze these frames and provide structured annotations.

## Action Taxonomy

Classify each observed action into this hierarchy:

### High-Level Tasks
- assembly: Putting components together
- disassembly: Taking components apart
- inspection: Examining parts or assemblies
- maintenance: Repairing or servicing equipment
- material_handling: Moving, sorting, or organizing parts
- measurement: Using measuring instruments
- surface_treatment: Cleaning, coating, or finishing
- fastening: Securing parts with fasteners
- wiring: Electrical wiring and connections
- calibration: Adjusting instruments or machines
- idle: No productive activity

### Mid-Level Actions
- pick_up: Grasping and lifting an object
- put_down: Placing an object on a surface
- insert: Pushing a part into another
- remove: Pulling a part out of another
- tighten: Rotating a fastener clockwise
- loosen: Rotating a fastener counter-clockwise
- press: Pushing/pressing a button or component
- hold: Maintaining grasp without movement
- align: Adjusting position for fit
- flip: Turning an object over
- slide: Moving an object along a surface
- pour: Dispensing liquid or granular material
- cut: Severing material
- strip: Removing covering (wire stripping, etc.)
- crimp: Compressing a connector
- solder: Joining with solder
- screw: Driving a screw
- hammer: Striking with a hammer
- clamp: Securing with a clamp
- measure: Taking a measurement reading
- inspect_visual: Looking closely at a part
- wipe: Cleaning a surface
- apply: Applying adhesive, lubricant, etc.
- route: Routing a wire or cable
- connect: Joining electrical connectors
- disconnect: Separating electrical connectors
- hand_over: Passing an object to another person
- reach: Extending hand toward an object
- retract: Pulling hand back
- adjust: Fine-tuning position or setting
- idle: No manipulation activity

### Manipulation Phases
For each action, identify the current phase:
- reach: Hand approaching the object
- grasp: Hand making contact and gripping
- manipulate: Active manipulation in progress
- release: Hand opening and letting go
- retract: Hand withdrawing after release
- hold: Maintaining a static grasp (subset of manipulate)
- transition: Moving between actions

## Output Format

Respond with ONLY a valid JSON object (no markdown, no explanation) with this structure:

{
  "frames_summary": "Brief description of what is happening across all frames",
  "actions": [
    {
      "action": "mid-level action name",
      "action_description": "Specific description of what the worker is doing",
      "high_level_task": "high-level task category",
      "objects_involved": ["object1", "object2"],
      "hand_used": "left|right|both|none",
      "grasp_type": "pinch|power|lateral|hook|spherical|none",
      "manipulation_phase": "reach|grasp|manipulate|release|retract|hold|transition",
      "confidence": 0.85,
      "is_idle": false,
      "is_transition": false
    }
  ],
  "scene_context": {
    "workspace_type": "assembly_station|workbench|floor|machine",
    "lighting": "good|dim|variable",
    "clutter_level": "clean|moderate|cluttered"
  }
}

## Important Guidelines
- Focus on the HANDS and what they are doing — this is egocentric footage.
- If hands are not visible, classify as "idle" or "transition".
- Be specific about objects — use their actual names, not generic terms.
- Confidence should reflect how certain you are (0.0-1.0).
- If multiple actions happen simultaneously (bimanual), list each separately.
- Use the local CV metadata provided to improve your accuracy — it contains hand detection, object detection, and interaction information from local models.
"""

# Required top-level fields in VLM response
REQUIRED_RESPONSE_FIELDS = {"actions"}


# ── Base client ───────────────────────────────────────────────────────


class BaseVLMClient:
    """Abstract base for all VLM API backends."""

    def __init__(self, api_key: str, max_concurrent: int = 3) -> None:
        self.api_key = api_key
        if httpx is None:
            raise ImportError("httpx is required for VLM clients: pip install httpx")
        self.client = httpx.AsyncClient(timeout=60.0)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.total_cost: float = 0.0
        self.total_calls: int = 0
        self.cost_per_call: float = 0.0  # overridden by subclasses
        self._backend_name: str = "base"

    # ── abstract method ───────────────────────────────────────────────

    async def predict_batch(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        """Send a frame batch to the VLM. Must be implemented by subclass."""
        raise NotImplementedError

    # ── retry wrapper ─────────────────────────────────────────────────

    async def predict_batch_with_retry(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        """Call :meth:`predict_batch` with concurrency control and retries.

        - Semaphore limits concurrent requests.
        - Tenacity retry: 3 attempts, exponential back-off (2 s, 4 s, 8 s).
        - Tracks cost and logs call details.
        """
        async with self.semaphore:
            return await self._retryable_predict(frame_base64_list, context)

    async def _retryable_predict(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        t0 = time.time()
        call_num = self.total_calls + 1

        try:
            result = await self.predict_batch(frame_base64_list, context)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                logger.warning(
                    "[%s] Rate limited (429) on call #%d — retrying …",
                    self._backend_name, call_num,
                )
            elif status >= 500:
                logger.warning(
                    "[%s] Server error (%d) on call #%d — retrying …",
                    self._backend_name, status, call_num,
                )
            raise
        except httpx.TimeoutException:
            logger.warning(
                "[%s] Timeout on call #%d — retrying …",
                self._backend_name, call_num,
            )
            raise

        elapsed_ms = (time.time() - t0) * 1000
        self.total_calls += 1
        self.total_cost += self.cost_per_call

        logger.info(
            "[%s] call #%d | %.0fms | cost ₹%.2f | cumulative ₹%.2f | %d frames",
            self._backend_name,
            self.total_calls,
            elapsed_ms,
            self.cost_per_call,
            self.total_cost,
            len(frame_base64_list),
        )
        return result

    # ── response parsing ──────────────────────────────────────────────

    @staticmethod
    def parse_response(raw_text: str) -> dict:
        """Parse VLM JSON response.

        Strips markdown code fences if present, validates that required
        fields exist, and returns the parsed dict.

        Raises:
            ValueError: If the response cannot be parsed or is missing
                required fields.
        """
        text = raw_text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM response is not valid JSON: {exc}\nRaw: {text[:500]}")

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        missing = REQUIRED_RESPONSE_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"VLM response missing required fields: {missing}")

        return data

    # ── cleanup ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
        logger.info(
            "[%s] closed — %d calls, total cost ₹%.2f",
            self._backend_name, self.total_calls, self.total_cost,
        )


# Apply tenacity retry decorator if available
if _HAS_TENACITY and httpx is not None:
    BaseVLMClient._retryable_predict = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TimeoutException, ValueError)
        ),
        reraise=True,
    )(BaseVLMClient._retryable_predict)
