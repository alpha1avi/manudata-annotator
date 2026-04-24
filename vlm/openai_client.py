"""ManuData Annotator — OpenAI GPT-4o-mini VLM Client.

Sends frame batches to the OpenAI Chat Completions API with base64
data-URI images and the shared egocentric system prompt.
"""

import logging
from typing import List

import httpx

from vlm.base_client import BaseVLMClient, EGOCENTRIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAIClient(BaseVLMClient):
    """OpenAI GPT-4o-mini client."""

    def __init__(self, api_key: str, max_concurrent: int = 3) -> None:
        super().__init__(api_key, max_concurrent)
        self.model = "gpt-4o-mini"
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.cost_per_call = 0.10  # approximate INR per 5-frame batch
        self._backend_name = "openai"

    async def predict_batch(
        self, frame_base64_list: List[str], context: str
    ) -> dict:
        """Build and send an OpenAI Chat Completions request.

        Request structure:
            - System message with egocentric prompt.
            - User message containing ``image_url`` content blocks
              (base64 data URIs) followed by a text block.
            - ``max_tokens``: 500.

        Returns:
            Parsed JSON dict from the VLM response.
        """
        # Build user content blocks
        user_content: list = []
        for b64 in frame_base64_list:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })

        user_text = "Analyze these sequential egocentric frames and provide structured annotations."
        if context:
            user_text += f"\n\n{context}"
        user_content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": EGOCENTRIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            self.api_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        body = response.json()

        # Extract text from choices[0].message.content
        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected OpenAI response structure: %s", exc)
            raise ValueError(f"Cannot extract text from OpenAI response: {exc}")

        return self.parse_response(raw_text)
