"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import json
from typing import Any

import json_repair
from loguru import logger
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def _short_text(value: Any, limit: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _message_summary(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        if isinstance(content, str):
            content_desc = f"str:{len(content)}"
        elif isinstance(content, list):
            content_desc = f"list:{len(content)}"
        elif content is None:
            content_desc = "none"
        else:
            content_desc = type(content).__name__
        has_tool_calls = bool(msg.get("tool_calls"))
        parts.append(f"{role}({content_desc},tools={has_tool_calls})")
    return "[" + ", ".join(parts) + "]"


def _cause_chain(exc: BaseException) -> str:
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        chain.append(f"{type(current).__name__}: {_short_text(current, 200)}")
        next_exc = current.__cause__ or current.__context__
        if next_exc is current:
            break
        current = next_exc
    return " <- ".join(chain)


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
                   reasoning_effort: str | None = None) -> LLMResponse:
        sanitized_messages = self._sanitize_empty_content(messages)
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": sanitized_messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        payload_bytes = len(json.dumps(kwargs, ensure_ascii=False, default=str))
        logger.info(
            "custom_provider request: api_base={} model={} max_tokens={} temperature={} reasoning_effort={} "
            "messages={} tools={} payload_bytes={}",
            self.api_base,
            kwargs["model"],
            kwargs["max_tokens"],
            temperature,
            reasoning_effort,
            _message_summary(sanitized_messages),
            len(tools or []),
            payload_bytes,
        )
        try:
            parsed = self._parse(await self._client.chat.completions.create(**kwargs))
            logger.info(
                "custom_provider response: api_base={} model={} finish_reason={} tool_calls={} usage={} content_chars={}",
                self.api_base,
                kwargs["model"],
                parsed.finish_reason,
                len(parsed.tool_calls),
                _short_text(parsed.usage, 200),
                len(parsed.content or ""),
            )
            return parsed
        except APIStatusError as e:
            logger.opt(exception=e).error(
                "custom_provider status error: api_base={} model={} status={} request_id={} method={} url={} "
                "response_body={} cause_chain={}",
                self.api_base,
                kwargs["model"],
                e.status_code,
                e.request_id,
                e.request.method if e.request else "",
                str(e.request.url) if e.request else "",
                _short_text(e.body),
                _cause_chain(e),
            )
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
        except APIConnectionError as e:
            logger.opt(exception=e).error(
                "custom_provider connection error: api_base={} model={} method={} url={} cause_chain={}",
                self.api_base,
                kwargs["model"],
                e.request.method if e.request else "",
                str(e.request.url) if e.request else "",
                _cause_chain(e),
            )
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
        except Exception as e:
            logger.opt(exception=e).error(
                "custom_provider unexpected error: api_base={} model={} error_type={} cause_chain={}",
                self.api_base,
                kwargs["model"],
                type(e).__name__,
                _cause_chain(e),
            )
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
