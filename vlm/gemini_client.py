"""ManuData Annotator — Google Gemini Flash VLM Client.

Sends frame batches to the Gemini ``generateContent`` REST API with
inline base64 images and the shared egocentric system prompt.
"""

import logging
from typing import List

import httpx

from vlm.base_client import BaseVLMClient, EGOCENTRIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GeminiClient(BaseVLMClient):
    """Google Gemini Flash API client."""

    def __init__(self, api_key: str, max_concurrent: int = 5) -> None:
        super().__init__(api_key, max_concurrent)
        self.model = "gemini-2.0-flash"
        self.api_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent"
        )
        self.cost_per_call = 0.05  # approximate INR per 5-frame batch
        self._backend_name = "gemini"

    async def predict_batch(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        """Build and send a Gemini ``generateContent`` request.

        Request structure:
            - ``system_instruction`` with the egocentric prompt.
            - ``contents[].parts`` with inline base64 images + context text.
            - API key passed as a query parameter.

        Returns:
            Parsed JSON dict from the VLM response.
        """
        # Build image parts
        parts: list = []
        for b64 in frame_base64_list:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64,
                }
            })

        # Text part with context
        user_text = "Analyze these sequential egocentric frames and provide structured annotations."
        if context:
            user_text += f"\n\n{context}"
        parts.append({"text": user_text})

        payload = {
            "system_instruction": {
                "parts": [{"text": EGOCENTRIC_SYSTEM_PROMPT}]
            },
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }

        response = await self.client.post(
            self.api_url,
            params={"key": self.api_key},
            json=payload,
        )
        response.raise_for_status()

        body = response.json()

        # Extract text from candidates[0].content.parts[0].text
        try:
            raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Gemini response structure: %s", exc)
            raise ValueError(f"Cannot extract text from Gemini response: {exc}")

        return self.parse_response(raw_text)
