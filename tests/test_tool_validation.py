import json
from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.openai_codex_provider import (
    _build_codex_request_body,
    _compute_codex_keychain_account,
    _consume_sse,
    _convert_messages,
    _read_codex_cli_auth_file_token,
    _should_retry_required,
)


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


class _FakeSSE:
    def __init__(self, events: list[dict]):
        self._lines: list[str] = []
        for event in events:
            self._lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
            self._lines.append("")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_codex_convert_messages_user_shape_and_skip_tool_without_call_id() -> None:
    _, input_items = _convert_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "tool", "name": "exec", "content": "ok"},
        ]
    )

    assert input_items[0]["type"] == "message"
    assert input_items[0]["role"] == "user"
    assert input_items[0]["content"][0]["type"] == "input_text"
    assert all(item.get("type") != "function_call_output" for item in input_items)


def test_codex_convert_messages_keeps_provider_params_controlled_by_caller() -> None:
    _, input_items = _convert_messages([{"role": "user", "content": "ping"}])
    assert input_items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "ping"}],
        }
    ]


def test_codex_request_body_omits_unsupported_generation_params() -> None:
    body = _build_codex_request_body(
        model="openai-codex/gpt-5.3-codex",
        messages=[{"role": "user", "content": "ping"}],
        system_prompt="system",
        input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
        tools=None,
    )

    assert body["model"] == "gpt-5.3-codex"
    assert "max_output_tokens" not in body
    assert "temperature" not in body


def test_codex_reads_cli_auth_json_from_codex_home(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "account_id": "new-account",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    token = _read_codex_cli_auth_file_token()

    assert token is not None
    assert token.access == "new-access"
    assert token.refresh == "new-refresh"
    assert token.account_id == "new-account"


def test_codex_keychain_account_matches_openclaw_shape(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    account = _compute_codex_keychain_account(codex_home)

    assert account.startswith("cli|")
    assert len(account) == 20


def test_codex_retry_required_only_before_tool_output() -> None:
    assert _should_retry_required([{"role": "user", "content": "ping"}], tools=[{"type": "function"}]) is True
    assert (
        _should_retry_required(
            [
                {"role": "user", "content": "ping"},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            tools=[{"type": "function"}],
        )
        is False
    )


@pytest.mark.asyncio
async def test_codex_consume_sse_extracts_function_call_from_stream_events() -> None:
    response = _FakeSSE(
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "call_id": "call_1",
                "delta": '{"command":"ls"}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec",
                    "arguments": '{"command":"ls"}',
                },
            },
            {
                "type": "response.completed",
                "response": {"status": "completed"},
            },
        ]
    )

    content, tool_calls, finish_reason = await _consume_sse(response)  # type: ignore[arg-type]

    assert content == ""
    assert finish_reason == "stop"
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "exec"
    assert tool_calls[0].arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_codex_consume_sse_extracts_function_call_from_completed_fallback() -> None:
    response = _FakeSSE(
        [
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                        {
                            "type": "function_call",
                            "id": "fc_2",
                            "call_id": "call_2",
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    ],
                },
            }
        ]
    )

    content, tool_calls, finish_reason = await _consume_sse(response)  # type: ignore[arg-type]

    assert content == "done"
    assert finish_reason == "stop"
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_2"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].arguments == {"path": "README.md"}


@pytest.mark.asyncio
async def test_codex_consume_sse_invalid_args_fallback_and_deduplicate() -> None:
    response = _FakeSSE(
        [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_bad",
                    "call_id": "call_bad",
                    "name": "exec",
                    "arguments": "{bad",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_bad",
                            "call_id": "call_bad",
                            "name": "exec",
                            "arguments": "{bad",
                        }
                    ],
                },
            },
        ]
    )

    _, tool_calls, _ = await _consume_sse(response)  # type: ignore[arg-type]

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_bad"
    assert tool_calls[0].arguments == {"raw": "{bad"}
