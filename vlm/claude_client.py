"""ManuData Annotator — Anthropic Claude VLM Client.

Sends frame batches to the Anthropic Messages API with base64 image
content blocks and the shared egocentric system prompt.
"""

import logging
from typing import List

import httpx

from vlm.base_client import BaseVLMClient, EGOCENTRIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ClaudeClient(BaseVLMClient):
    """Anthropic Claude API client (Sonnet / Haiku)."""

    # Per-batch cost lookup (INR)
    _MODEL_COSTS = {
        "claude-sonnet-4-20250514": 0.35,
        "claude-haiku-4-5-20251001": 0.08,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 3,
    ) -> None:
        super().__init__(api_key, max_concurrent)
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.cost_per_call = self._MODEL_COSTS.get(model, 0.35)
        self._backend_name = f"claude({model.split('-')[1]})"

    async def predict_batch(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        """Build and send an Anthropic Messages API request.

        Request structure:
            - ``system``: egocentric prompt.
            - ``messages[0].content``: list of image blocks (base64,
              ``media_type image/jpeg``) followed by a text block.
            - ``max_tokens``: 500.

        Returns:
            Parsed JSON dict from the VLM response.
        """
        # Build content blocks
        content: list = []
        for b64 in frame_base64_list:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

        user_text = "Analyze these sequential egocentric frames and provide structured annotations."
        if context:
            user_text += f"\n\n{context}"
        content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "system": EGOCENTRIC_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": content},
            ],
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        response = await self.client.post(
            self.api_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        body = response.json()

        # Extract text from content[0].text
        try:
            raw_text = body["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Claude response structure: %s", exc)
            raise ValueError(f"Cannot extract text from Claude response: {exc}")

        return self.parse_response(raw_text)
