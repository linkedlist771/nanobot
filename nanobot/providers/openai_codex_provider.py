"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from loguru import logger

from oauth_cli_kit import get_token as get_codex_token
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "nanobot"
CODEX_KEYCHAIN_SERVICE = "Codex Auth"


@dataclass
class _CodexCredential:
    access: str
    account_id: str
    refresh: str | None = None
    expires: int | None = None


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(self, default_model: str = "openai-codex/gpt-5.1-codex"):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model = model or self.default_model
        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(_get_preferred_codex_token)
        headers = _build_headers(token.account_id, token.access)
        body = _build_codex_request_body(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            input_items=input_items,
            tools=tools,
        )

        url = DEFAULT_CODEX_URL
        logger.debug("Codex request URL: {}", url)
        logger.debug("Codex request headers: {}", json.dumps(_sanitize_headers(headers), ensure_ascii=False))
        logger.debug("Codex request payload: {}", json.dumps(body, ensure_ascii=False))
        should_retry_required = _should_retry_required(messages, tools)

        try:
            try:
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=True)
                logger.debug(f"Content: {content}")
                logger.debug(f"Tool calls: {tool_calls}")
                logger.debug(f"Finish reason: {finish_reason}")
                if should_retry_required and not tool_calls:
                    logger.warning("Codex returned no tool calls in auto mode; retrying once with tool_choice=required")
                    retry_body = dict(body)
                    retry_body["tool_choice"] = "required"
                    logger.debug("Codex retry payload: {}", json.dumps(retry_body, ensure_ascii=False))
                    content, tool_calls, finish_reason = await _request_codex(
                        url, headers, retry_body, verify=True
                    )
                    logger.debug("Codex retry content: {}", content)
                    logger.debug("Codex retry tool calls: {}", tool_calls)
                    logger.debug("Codex retry finish reason: {}", finish_reason)
            except Exception as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("SSL certificate verification failed for Codex API; retrying with verify=False")
                content, tool_calls, finish_reason = await _request_codex(url, headers, body, verify=False)
                if should_retry_required and not tool_calls:
                    logger.warning(
                        "Codex returned no tool calls in auto mode (verify=False); retrying once with tool_choice=required"
                    )
                    retry_body = dict(body)
                    retry_body["tool_choice"] = "required"
                    logger.debug("Codex retry payload (verify=False): {}", json.dumps(retry_body, ensure_ascii=False))
                    content, tool_calls, finish_reason = await _request_codex(
                        url, headers, retry_body, verify=False
                    )
                    logger.debug("Codex retry content (verify=False): {}", content)
                    logger.debug("Codex retry tool calls (verify=False): {}", tool_calls)
                    logger.debug("Codex retry finish reason (verify=False): {}", finish_reason)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error calling Codex: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _get_preferred_codex_token() -> _CodexCredential:
    cli_token = _read_codex_cli_token()
    if cli_token:
        logger.debug("Using Codex CLI credentials from preferred source")
        return cli_token

    token = get_codex_token()
    return _CodexCredential(
        access=token.access,
        refresh=getattr(token, "refresh", None),
        expires=getattr(token, "expires", None),
        account_id=token.account_id,
    )


def _read_codex_cli_token() -> _CodexCredential | None:
    keychain_token = _read_codex_cli_keychain_token()
    if keychain_token:
        logger.debug("Using Codex CLI credentials from macOS keychain")
        return keychain_token

    auth_file_token = _read_codex_cli_auth_file_token()
    if auth_file_token:
        logger.debug("Using Codex CLI credentials from auth.json")
        return auth_file_token

    return None


def _resolve_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    home = Path(configured).expanduser() if configured else Path.home() / ".codex"
    try:
        return home.resolve()
    except Exception:
        return home


def _compute_codex_keychain_account(codex_home: Path) -> str:
    digest = hashlib.sha256(str(codex_home).encode("utf-8")).hexdigest()
    return f"cli|{digest[:16]}"


def _read_codex_cli_keychain_token() -> _CodexCredential | None:
    if os.name != "posix":
        return None
    if os.uname().sysname != "Darwin":
        return None

    codex_home = _resolve_codex_home()
    account = _compute_codex_keychain_account(codex_home)
    try:
        secret = subprocess.check_output(
            [
                "security",
                "find-generic-password",
                "-s",
                CODEX_KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return None

    try:
        payload = json.loads(secret)
    except Exception:
        return None

    tokens = payload.get("tokens") or {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    account_id = tokens.get("account_id")
    if not isinstance(access, str) or not access:
        return None
    if not isinstance(refresh, str) or not refresh:
        return None
    if not isinstance(account_id, str) or not account_id:
        return None

    last_refresh_raw = payload.get("last_refresh")
    try:
        last_refresh = int(float(last_refresh_raw))
    except Exception:
        try:
            last_refresh = int(time.mktime(time.strptime(str(last_refresh_raw), "%Y-%m-%dT%H:%M:%SZ")) * 1000)
        except Exception:
            last_refresh = int(time.time() * 1000)

    return _CodexCredential(
        access=access,
        refresh=refresh,
        expires=last_refresh + 60 * 60 * 1000,
        account_id=account_id,
    )


def _read_codex_cli_auth_file_token() -> _CodexCredential | None:
    auth_path = _resolve_codex_home() / "auth.json"
    if not auth_path.exists():
        return None

    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    tokens = payload.get("tokens") or {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    account_id = tokens.get("account_id")
    if not isinstance(access, str) or not access:
        return None
    if not isinstance(refresh, str) or not refresh:
        return None
    if not isinstance(account_id, str) or not account_id:
        return None

    try:
        expires = int(auth_path.stat().st_mtime * 1000 + 60 * 60 * 1000)
    except Exception:
        expires = int(time.time() * 1000 + 60 * 60 * 1000)

    return _CodexCredential(
        access=access,
        refresh=refresh,
        expires=expires,
        account_id=account_id,
    )


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    safe = dict(headers)
    auth = safe.get("Authorization")
    if isinstance(auth, str) and auth:
        safe["Authorization"] = "Bearer ***"
    return safe


def _build_codex_request_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "store": False,
        "stream": True,
        "instructions": system_prompt,
        "input": input_items,
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": _prompt_cache_key(messages),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if tools:
        body["tools"] = _convert_tools(tools)
    return body


def _should_retry_required(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> bool:
    if not tools:
        return False
    return not any(msg.get("role") == "tool" for msg in messages)


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
) -> tuple[str, list[ToolCallRequest], str]:
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(_friendly_error(response.status_code, text.decode("utf-8", "ignore")))
            return await _consume_sse(response)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex strict-mode format.

    The Codex Responses API always enforces strict mode (``strict: true``),
    which requires every property listed in ``properties`` to also appear in
    ``required``.  Optional parameters (those absent from ``required``) are
    converted to a nullable ``anyOf`` union so the model can legally pass
    ``null`` instead of omitting them, satisfying strict mode without forcing
    the model to invent values for fields like ``working_dir`` or
    action-specific ``cron`` arguments.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = _make_strict_compatible(fn.get("parameters") or {})
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params,
        })
    return converted


def _make_strict_compatible(params: Any) -> dict[str, Any]:
    """Rewrite a JSON-Schema ``object`` so every property is in ``required``.

    Optional properties (present in ``properties`` but absent from
    ``required``) are wrapped in ``{"anyOf": [original, {"type": "null"}]}``
    so the model can legally supply ``null`` when it doesn't want to use them.
    ``additionalProperties`` is set to ``False`` to satisfy the Codex API.
    Nested ``object`` schemas are recursively converted.
    """
    if not isinstance(params, dict):
        return {}
    if params.get("type") != "object":
        return {k: v for k, v in params.items() if k != "additionalProperties"}

    props: dict[str, Any] = {}
    required_set: set[str] = set(params.get("required") or [])
    new_required: list[str] = list(params.get("required") or [])

    for key, schema in (params.get("properties") or {}).items():
        if not isinstance(schema, dict):
            props[key] = schema
            continue
        # Recurse into nested objects.
        converted_schema = _make_strict_compatible(schema) if schema.get("type") == "object" else schema
        if key in required_set:
            props[key] = converted_schema
        else:
            # Wrap in anyOf so the model can pass null for optional params.
            props[key] = {"anyOf": [converted_schema, {"type": "null"}]}
            new_required.append(key)

    result: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": new_required,
        "additionalProperties": False,
    }
    return result


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            # Handle text first.
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                )
            # Then handle tool calls.
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                }
                if item_id:
                    item["id"] = item_id
                input_items.append(item)
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            if not call_id:
                logger.debug("Skipping tool message without valid tool_call_id: {}", msg)
                continue
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"type": "message", "role": "user", "content": converted}
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "", None


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []

    def _flush(buf: list[str]):
        data_lines = [l[5:].strip() for l in buf if l.startswith("data:")]
        if not data_lines:
            return None
        data = "\n".join(data_lines).strip()
        if not data or data == "[DONE]":
            return None
        try:
            event = json.loads(data)
            logger.debug("Codex SSE event: {}", json.dumps(event, ensure_ascii=False))
            return event
        except Exception as e:
            logger.debug("Codex SSE event parse failed: {} | raw={}", e, data)
            return None

    async for line in response.aiter_lines():
        logger.debug("Codex SSE line: {}", line)
        if line == "":
            if buffer:
                event = _flush(buffer)
                buffer = []
                if event is not None:
                    yield event
            continue
        buffer.append(line)

    # Flush any trailing event if the stream ended without a final blank line.
    if buffer:
        event = _flush(buffer)
        if event is not None:
            yield event


async def _consume_sse(response: httpx.Response) -> tuple[str, list[ToolCallRequest], str]:
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    emitted_call_ids: set[str] = set()
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                _update_tool_call_buffer(tool_call_buffers, item)
        elif event_type == "response.output_text.delta":
            content += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                _update_tool_call_buffer(tool_call_buffers, item)
                _emit_tool_call(tool_calls, emitted_call_ids, tool_call_buffers, item)
        elif event_type == "response.completed":
            resp = event.get("response") or {}
            status = resp.get("status")
            finish_reason = _map_finish_reason(status)
            for item in (resp.get("output") or []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function_call":
                    _update_tool_call_buffer(tool_call_buffers, item)
                    _emit_tool_call(tool_calls, emitted_call_ids, tool_call_buffers, item)
                elif item.get("type") == "message" and not content:
                    content += _extract_output_text(item)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    return content, tool_calls, finish_reason


def _update_tool_call_buffer(tool_call_buffers: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return
    current = tool_call_buffers.get(call_id, {})
    tool_call_buffers[call_id] = {
        "id": item.get("id") or current.get("id"),
        "name": item.get("name") or current.get("name"),
        "arguments": item.get("arguments") if item.get("arguments") is not None else current.get("arguments", ""),
    }


def _emit_tool_call(
    tool_calls: list[ToolCallRequest],
    emitted_call_ids: set[str],
    tool_call_buffers: dict[str, dict[str, Any]],
    item: dict[str, Any],
) -> None:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id or call_id in emitted_call_ids:
        return
    buf = tool_call_buffers.get(call_id) or {}
    name = buf.get("name") or item.get("name")
    if not isinstance(name, str) or not name:
        logger.debug("Skipping function_call without valid name: {}", item)
        return
    args_raw = buf.get("arguments") or item.get("arguments") or "{}"
    try:
        args = json.loads(args_raw)
    except Exception:
        logger.debug("Function call arguments are not valid JSON: {}", args_raw)
        args = {"raw": args_raw}
    tool_calls.append(
        ToolCallRequest(
            id=call_id,
            name=name,
            arguments=args,
        )
    )
    emitted_call_ids.add(call_id)


def _extract_output_text(item: dict[str, Any]) -> str:
    parts = item.get("content")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "output_text" and isinstance(part.get("text"), str):
            chunks.append(part.get("text") or "")
    return "".join(chunks)


_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"
