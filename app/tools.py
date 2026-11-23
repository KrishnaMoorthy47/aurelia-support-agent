"""MCP client gateway.

`customer_id` and `approved_by` are removed from the tool schemas the model sees and filled in from
the session, and only the read-only tools are offered to the model.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import get_settings

MODEL_VISIBLE_TOOLS = {"get_account_summary", "get_kyc_status", "get_withdrawal_limits", "list_tickets"}
SERVER_FILLED_ARGS = {"customer_id", "approved_by"}


class ToolError(Exception):
    pass


class McpToolGateway:
    def __init__(self, mock_db_path: str | None = None):
        self._db_path = mock_db_path or get_settings().mock_db_path
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._specs: list[dict] = []

    async def __aenter__(self) -> "McpToolGateway":
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server.server"],
            env={**os.environ, "MOCK_DB_PATH": self._db_path},
        )
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            listed = await self._session.list_tools()
            self._specs = [self._to_openai_spec(t) for t in listed.tools if t.name in MODEL_VISIBLE_TOOLS]
        except BaseException:
            await self._stack.aclose()  # never leave the server subprocess running if startup fails
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    @staticmethod
    def _to_openai_spec(tool: Any) -> dict:
        schema = json.loads(json.dumps(tool.input_schema))  # deep copy (mcp 2.x uses snake_case fields)
        for hidden in SERVER_FILLED_ARGS:
            schema.get("properties", {}).pop(hidden, None)
            if hidden in schema.get("required", []):
                schema["required"].remove(hidden)
        return {"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": schema}}

    @property
    def model_tool_specs(self) -> list[dict]:
        return self._specs

    async def call(
        self, name: str, customer_id: str, args: dict | None = None, *, approved_by: str | None = None, privileged: bool = False
    ) -> dict:
        """Call a tool. `privileged` is only passed by graph code after a human approval."""
        if self._session is None:
            raise ToolError("gateway not started")
        if name not in MODEL_VISIBLE_TOOLS and name != "create_ticket" and not privileged:
            raise ToolError(f"tool {name!r} is not available on this path")
        clean = {k: v for k, v in (args or {}).items() if k not in SERVER_FILLED_ARGS}
        clean["customer_id"] = customer_id
        if approved_by is not None:
            clean["approved_by"] = approved_by
        result = await self._session.call_tool(name, clean)
        if getattr(result, "structured_content", None):
            data = result.structured_content
            return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
        text = "".join(getattr(c, "text", "") for c in result.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": text or "empty_result"}
