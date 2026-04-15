from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from graphify_rag.logging_utils import get_logger
from graphify_rag.models import ChatTurn


LOGGER = get_logger(__name__)


class OpenAIAPIError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover
            detail = exc.read().decode("utf-8", errors="ignore")
            raise OpenAIAPIError(f"OpenAI API request failed: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover
            raise OpenAIAPIError(f"OpenAI API connection failed: {exc.reason}") from exc

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        payload = self._post("/embeddings", {"model": model, "input": texts})
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise OpenAIAPIError("Unexpected embeddings response payload.")
        return [
            [float(value) for value in item["embedding"]]
            for item in data
            if isinstance(item, dict) and "embedding" in item
        ]

    def chat_completion(self, model: str, system_prompt: str, messages: list[ChatTurn]) -> str:
        payload = self._post(
            "/chat/completions",
            {
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *[{"role": item.role, "content": item.content} for item in messages],
                ],
            },
        )
        choices = payload.get("choices", [])
        if not choices:
            raise OpenAIAPIError("OpenAI response did not contain any choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenAIAPIError("OpenAI response did not contain text content.")
        return content.strip()
