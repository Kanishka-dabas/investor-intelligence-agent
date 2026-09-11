from pathlib import Path

from fastmcp import Client
from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore
from loguru import logger

from src.config import settings

_client: Client | None = None

TOKEN_STORAGE_DIR = Path.home() / ".investor_intelligence_mcp_tokens"


def get_mcp_client() -> Client:
    """
    Returns a singleton FastMCP client connected to the Financial Data
    MCP Server. Uses disk-based token storage so OAuth tokens survive
    process restarts and don't require re-authorization every time.
    """
    global _client
    if _client is None:
        TOKEN_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        token_store = DiskStore(directory=str(TOKEN_STORAGE_DIR))
        oauth = OAuth(mcp_url=settings.mcp.server_url, token_storage=token_store)
        _client = Client(settings.mcp.server_url, auth=oauth)
        logger.info(f"Initialized MCP client for {settings.mcp.server_url}")
    return _client


async def list_available_tools() -> list[str]:
    """Lists all tool names exposed by the MCP server."""
    client = get_mcp_client()
    async with client:
        tools = await client.list_tools()
        return [tool.name for tool in tools]


async def call_mcp_tool(tool_name: str, params: dict) -> dict:
    """Calls a specific tool on the MCP server with the given parameters."""
    client = get_mcp_client()
    async with client:
        result = await client.call_tool(tool_name, params)
        return result