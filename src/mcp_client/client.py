from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from loguru import logger

from src.config import settings

_client: Client | None = None


def get_mcp_client() -> Client:
    """
    Returns a singleton FastMCP client connected to the self-hosted
    Financial Data MCP Server, using static bearer-token auth (no
    interactive OAuth — works headlessly in Docker/production).
    """
    global _client
    if _client is None:
        auth = BearerAuth(token=settings.mcp.auth_token.get_secret_value())
        _client = Client(settings.mcp.server_url, auth=auth)
        logger.info(f"Initialized MCP client for {settings.mcp.server_url}")
    return _client


async def list_available_tools() -> list[str]:
    client = get_mcp_client()
    async with client:
        tools = await client.list_tools()
        return [tool.name for tool in tools]


async def call_mcp_tool(tool_name: str, params: dict) -> dict:
    client = get_mcp_client()
    async with client:
        result = await client.call_tool(tool_name, params)
        return result