"""Per-request context variables for concurrent tool routing.

Each asyncio.Task gets its own copy of these variables, so multiple
messages can be processed concurrently without shared-state conflicts.
"""

import contextvars

# Set by AgentLoop._process_message for each request; read by tools.
current_channel: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_channel", default=""
)
current_chat_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_chat_id", default=""
)
current_message_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_message_id", default=None
)
# Tracks whether MessageTool.execute() has already sent a response in this turn.
sent_in_turn: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sent_in_turn", default=False
)
