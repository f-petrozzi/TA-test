from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import anyio  
except ImportError:  
    anyio = None  

from utils.database import ChatDatabase
from utils.google_tools import GoogleWorkspaceTools
from utils.rag import retrieve_matches

try: 
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server as mcp_stdio_server

    MCP_AVAILABLE = True
except ImportError:  
    mcp_types = None  
    ClientSession = None  
    StdioServerParameters = None  
    stdio_client = None  
    NotificationOptions = None  
    Server = None  
    mcp_stdio_server = None  
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["SimpleMCPClient", "build_mcp_server", "run_mcp_server"]

_PYTHON_BIN = sys.executable or "python3"
SERVER_NAME = "usf_workspace_tools"
SERVER_VERSION = "1.0.0"
DEFAULT_SERVER_CMD = [_PYTHON_BIN, "-m", "utils.mcp", "serve"]
DEFAULT_SERVER_CWD = Path(__file__).resolve().parents[1]

class _ToolRuntime:
    def __init__(
        self,
        chat_db: Optional[ChatDatabase] = None,
        google_tools: Optional[GoogleWorkspaceTools] = None,
    ):
        self._db = chat_db
        self._google = google_tools

    @property
    def db(self) -> ChatDatabase:
        if self._db is None:
            self._db = ChatDatabase()
        return self._db

    @property
    def google(self) -> GoogleWorkspaceTools:
        if self._google is None:
            self._google = GoogleWorkspaceTools()
        return self._google

    # RAG helpers
    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required")
        return retrieve_matches(query, match_count=match_count, extra_filter=extra_filter)

    # Audit helpers
    def log_interaction(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, str]:
        if not session_id or not event_type:
            raise ValueError("session_id and event_type are required to log interactions")
        self.db.log_event(session_id, event_type, payload)
        return {"status": "logged"}

    # Google Workspace helpers
    def list_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        return self.google.list_calendar_events(max_results=max(1, max_results))

    def list_recent_emails(self, query: str = "", max_results: int = 5) -> list[dict[str, str]]:
        return self.google.list_recent_messages(query=query or "", max_results=max(1, max_results))

    def send_email(self, to_address: str, subject: str, body: str) -> str:
        if not to_address or not subject or not body:
            raise ValueError("To, subject, and body are required to send email.")
        return self.google.send_email(to_address, subject, body)

    def create_event(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        description: str = "",
        location: str = "",
    ) -> str:
        return self.google.create_event(
            summary,
            start_iso,
            duration_minutes,
            attendees=attendees,
            description=description,
            location=location,
        )

def _tool_definitions() -> list[mcp_types.Tool]:
    """Return the MCP tool catalog."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not installed; run `pip install mcp` to enable tools.")

    annotations_read_only = mcp_types.ToolAnnotations(readOnlyHint=True, idempotentHint=True)
    annotations_mutating = mcp_types.ToolAnnotations(readOnlyHint=False, destructiveHint=False)

    return [
        mcp_types.Tool(
            name="retrieve_context",
            description="Retrieve semantically relevant USF context from the Supabase vector store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user utterance to embed and search."},
                    "match_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "extra_filter": {"type": "object", "description": "Optional JSON filter applied server-side."},
                },
                "required": ["query"],
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "hits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc": {"type": "string"},
                                "meta": {"type": "object"},
                                "score": {"type": ["number", "null"]},
                            },
                            "required": ["doc", "meta"],
                        },
                    }
                },
                "required": ["hits"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="log_interaction",
            description="Persist an audit trail entry for the current chat session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["session_id", "event_type", "payload"],
            },
            outputSchema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="list_calendar_events",
            description="List upcoming Google Calendar events from the primary calendar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["events"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="list_recent_emails",
            description="List recent Gmail messages matching an optional search query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "default": ""},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["messages"],
            },
            annotations=annotations_read_only,
        ),
        mcp_types.Tool(
            name="send_email",
            description="Send an email via Gmail on behalf of the authenticated USF account.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to_address": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to_address", "subject", "body"],
            },
            outputSchema={
                "type": "object",
                "properties": {"message_id": {"type": "string"}},
                "required": ["message_id"],
            },
            annotations=annotations_mutating,
        ),
        mcp_types.Tool(
            name="create_event",
            description="Create a Google Calendar event with optional attendees and description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "start_iso": {"type": "string", "description": "ISO-8601 start timestamp."},
                    "duration_minutes": {"type": "integer", "minimum": 5, "maximum": 480, "default": 30},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "description": {"type": "string", "default": ""},
                    "location": {"type": "string", "default": ""},
                },
                "required": ["summary", "start_iso", "duration_minutes"],
            },
            outputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
            annotations=annotations_mutating,
        ),
    ]

def build_mcp_server(runtime: Optional[_ToolRuntime] = None) -> Server:
    """
    Build a fully-compliant MCP server that exposes the USF RAG + Google tools.
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("The `mcp` package is required to run the MCP server.")
    if anyio is None:
        raise RuntimeError("anyio is required to run the MCP server transport.")

    runtime = runtime or _ToolRuntime()
    server = Server(SERVER_NAME, version=SERVER_VERSION)
    tools = _tool_definitions()

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        # Return deep copies to avoid accidental mutation between requests
        return [tool.model_copy(deep=True) for tool in tools]

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: Optional[dict[str, Any]]):
        args = arguments or {}
        try:
            return await _execute_tool(runtime, tool_name, args)
        except Exception as exc:  # pragma: no cover - defensive net
            logger.exception("Tool %s failed", tool_name)
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=f"{tool_name} failed: {exc}")],
                isError=True,
            )

    return server

async def _run_blocking(func, *args, **kwargs):
    if anyio is None:
        return func(*args, **kwargs)
    return await anyio.to_thread.run_sync(func, *args, **kwargs)

async def _execute_tool(runtime: _ToolRuntime, tool_name: str, args: dict[str, Any]):
    if tool_name == "retrieve_context":
        hits = await _run_blocking(
            runtime.retrieve_context,
            args.get("query"),
            args.get("match_count"),
            args.get("extra_filter"),
        )
        return {"hits": hits}

    if tool_name == "log_interaction":
        return await _run_blocking(
            runtime.log_interaction,
            args.get("session_id", ""),
            args.get("event_type", ""),
            args.get("payload") or {},
        )

    if tool_name == "list_calendar_events":
        events = await _run_blocking(runtime.list_calendar_events, args.get("max_results", 5))
        return {"events": events}

    if tool_name == "list_recent_emails":
        messages = await _run_blocking(
            runtime.list_recent_emails,
            args.get("query", ""),
            args.get("max_results", 5),
        )
        return {"messages": messages}

    if tool_name == "send_email":
        message_id = await _run_blocking(
            runtime.send_email,
            args.get("to_address", ""),
            args.get("subject", ""),
            args.get("body", ""),
        )
        return {"message_id": message_id}

    if tool_name == "create_event":
        event_id = await _run_blocking(
            runtime.create_event,
            args.get("summary", ""),
            args.get("start_iso", ""),
            int(args.get("duration_minutes", 30)),
            args.get("attendees"),
            args.get("description", ""),
            args.get("location", ""),
        )
        return {"event_id": event_id}

    raise ValueError(f"Unknown tool: {tool_name}")

async def run_mcp_server(
    chat_db: Optional[ChatDatabase] = None,
    google_tools: Optional[GoogleWorkspaceTools] = None,
) -> None:
    """Entry point for `python -m utils.mcp serve`."""
    server = build_mcp_server(_ToolRuntime(chat_db=chat_db, google_tools=google_tools))
    init_options = server.create_initialization_options(
        notification_options=NotificationOptions(tools_changed=True),
        experimental_capabilities={},
    )
    async with mcp_stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            init_options,
        )

class SimpleMCPClient:

    def __init__(
        self,
        chat_db: Optional[ChatDatabase] = None,
        google_tools: Optional[GoogleWorkspaceTools] = None,
        server_command: Optional[Sequence[str]] = None,
        server_cwd: Optional[Path | str] = None,
        server_env: Optional[dict[str, str]] = None,
    ):
        if not MCP_AVAILABLE:
            raise RuntimeError("The `mcp` package is required. Run `pip install -r requirements.txt`.")
        if anyio is None:
            raise RuntimeError("The `anyio` package is required for MCP stdio transport.")
        if os.getenv("USF_DISABLE_MCP", "0") == "1":
            raise RuntimeError("MCP has been disabled via USF_DISABLE_MCP=1.")

        self._runtime = _ToolRuntime(chat_db=chat_db, google_tools=google_tools)
        self._server_command = list(server_command or DEFAULT_SERVER_CMD)
        self._server_cwd = Path(server_cwd or DEFAULT_SERVER_CWD)
        env = dict(os.environ)
        if server_env:
            env.update(server_env)
        self._server_env = env

        self._stdio_params = StdioServerParameters(
            command=self._server_command[0],
            args=self._server_command[1:],
            env=self._server_env,
            cwd=str(self._server_cwd),
        )

    # internal helpers
    def _call_tool(self, tool_name: str, arguments: dict[str, Any]):
        if not self._stdio_params:
            raise RuntimeError("MCP transport has not been initialised.")
        assert anyio is not None

        async def _call():
            async with stdio_client(self._stdio_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    await session.list_tools()
                    return await session.call_tool(tool_name, arguments)

        result = anyio.run(_call)
        if result.isError:
            raise RuntimeError(_extract_error(result))
        return result

    @staticmethod
    def _structured(result, key: str, default: Any):
        data = result.structuredContent or {}
        return data.get(key, default)

    # public API used by the Streamlit app
    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if not query:
            raise ValueError("query is required")
        payload = {"query": query}
        if match_count is not None:
            payload["match_count"] = match_count
        if extra_filter:
            payload["extra_filter"] = extra_filter
        try:
            result = self._call_tool("retrieve_context", payload)
            return self._structured(result, "hits", [])
        except Exception as exc:  # pragma: no cover - fallback safety
            logger.warning("Falling back to direct context retrieval: %s", exc)
        return self._runtime.retrieve_context(query, match_count=match_count, extra_filter=extra_filter)

    def log_interaction(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        try:
            self._call_tool(
                "log_interaction",
                {"session_id": session_id, "event_type": event_type, "payload": payload},
            )
            return
        except Exception as exc:  # pragma: no cover - fallback safety
            logger.warning("MCP log_interaction failed, using direct DB fallback: %s", exc)
        self._runtime.log_interaction(session_id, event_type, payload)

    def list_calendar_events(self, max_results: int = 5) -> list[dict[str, Any]]:
        result = self._call_tool("list_calendar_events", {"max_results": max_results})
        return self._structured(result, "events", [])

    def list_recent_emails(self, query: str = "", max_results: int = 5) -> list[dict[str, str]]:
        result = self._call_tool(
            "list_recent_emails",
            {"query": query or "", "max_results": max_results},
        )
        return self._structured(result, "messages", [])

    def send_email(self, to_address: str, subject: str, body: str) -> str:
        if not to_address or not subject or not body:
            raise ValueError("To, subject, and body are required to send email.")
        result = self._call_tool(
            "send_email",
            {"to_address": to_address, "subject": subject, "body": body},
        )
        return self._structured(result, "message_id", "")

    def create_event(
        self,
        summary: str,
        start_iso: str,
        duration_minutes: int,
        attendees: Optional[list[str]] = None,
        description: str = "",
        location: str = "",
    ) -> str:
        payload = {
            "summary": summary,
            "start_iso": start_iso,
            "duration_minutes": duration_minutes,
            "attendees": attendees or [],
            "description": description,
            "location": location,
        }
        result = self._call_tool("create_event", payload)
        return self._structured(result, "event_id", "")

def _extract_error(result: mcp_types.CallToolResult) -> str:
    for block in result.content:
        if getattr(block, "type", "") == "text":
            return block.text
    return "Tool call failed"

def _main() -> None:
    parser = argparse.ArgumentParser(description="USF MCP server utilities.")
    parser.add_argument("command", choices=["serve"], help="Run the MCP stdio server.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.command == "serve":
        if not MCP_AVAILABLE:
            parser.error("The `mcp` package is not installed.")
        if anyio is None:
            parser.error("The `anyio` package is required to run the MCP server.")
        anyio.run(run_mcp_server)

if __name__ == "__main__":  # pragma: no cover
    _main()