"""LLM access, behind a thin interface with an offline path.

Every caller asks `is_offline()` first and supplies its own deterministic
fallback when there is no key, rather than having this module fabricate a
plausible-looking answer. A RAG system that quietly invents text when its
model is unreachable is worse than one that says it cannot answer, so the
offline path is always *extractive*: it quotes the corpus, it never writes
prose of its own.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from .config import get_settings

_client = None


class LLMError(RuntimeError):
    pass


def is_offline() -> bool:
    return get_settings().resolved_provider != "gemini"


def provider_name() -> str:
    settings = get_settings()
    return settings.gemini_model if not is_offline() else "offline-extractive"


def _client_or_raise():
    global _client
    if is_offline():
        raise LLMError("No LLM configured (set GEMINI_API_KEY).")
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


def complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    json_schema: dict | None = None,
) -> str:
    from google.genai import types

    client = _client_or_raise()
    settings = get_settings()
    config: dict = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system:
        config["system_instruction"] = system
    if json_schema is not None:
        # Structured output: the model is constrained to the schema, which
        # removes the "parse JSON out of a markdown fence" failure mode that
        # makes metadata extraction flaky.
        config["response_mime_type"] = "application/json"
        config["response_schema"] = json_schema

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )
            return (response.text or "").strip()
        except Exception as exc:
            message = str(exc)
            # A retired model will never succeed; fail fast with the fix.
            if "NOT_FOUND" in message or "no longer available" in message:
                raise LLMError(
                    f"Model '{settings.gemini_model}' is unavailable. List current models with: "
                    f"curl -s 'https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY' "
                    f"then set GEMINI_MODEL in backend/.env. Original error: {message[:200]}"
                ) from exc
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(str(last_error))


def complete_json(prompt: str, system: str | None = None, json_schema: dict | None = None) -> dict:
    raw = complete(prompt, system=system, temperature=0.0, json_schema=json_schema)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return JSON: {raw[:200]}")


def stream(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> Iterator[str]:
    from google.genai import types

    client = _client_or_raise()
    settings = get_settings()
    config: dict = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    if system:
        config["system_instruction"] = system

    response = client.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(**config),
    )
    for piece in response:
        text = getattr(piece, "text", None)
        if text:
            yield text
