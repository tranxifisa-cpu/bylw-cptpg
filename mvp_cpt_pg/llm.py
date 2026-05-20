from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .config import AgentLLMConfig
from .utils import ensure_dir, extract_json_object, get_env_var, stable_hash


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot return valid JSON."""


class DashScopeClient:
    def __init__(
        self,
        cache_dir: Path,
        config: AgentLLMConfig,
    ) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.timeout_seconds = config.timeout_seconds
        self.endpoint = config.base_url
        self.api_key = get_env_var(*config.api_key_envs)
        self.client = OpenAI(
            api_key=self.api_key or "missing-api-key",
            base_url=self.endpoint,
            timeout=self.timeout_seconds,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, namespace: str, payload: dict[str, Any]) -> Path:
        directory = ensure_dir(self.cache_dir / namespace)
        digest = stable_hash(payload)
        return directory / f"{digest}.json"

    def chat_json(
        self,
        namespace: str,
        system_prompt: str,
        user_prompt: str,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "enable_thinking": self.config.enable_thinking,
            "reasoning_effort": self.config.reasoning_effort,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "namespace": namespace,
        }
        cache_path = self._cache_path(namespace, payload)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["normalized"]
        if not self.enabled:
            raise LLMError("LLM API key is not configured")
        response = self._post_chat(system_prompt, user_prompt)
        try:
            normalized = validator(response)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM response validation failed: {exc}; raw_response={response}") from exc
        cache_path.write_text(
            json.dumps(
                {
                    "payload": payload,
                    "normalized": normalized,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return normalized

    def _post_chat(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.config.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.enable_thinking:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled"},
            }
        try:
            completion = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"LLM request failed via OpenAI SDK: model={self.model}, endpoint={self.endpoint}, body={exc}"
            ) from exc
        try:
            message = completion.choices[0].message
            content = message.content
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Unexpected LLM response: {completion}") from exc
        if isinstance(content, list):
            text_parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            content = "".join(text_parts)
        try:
            return extract_json_object(content)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Unable to parse JSON from LLM content: {content}") from exc
