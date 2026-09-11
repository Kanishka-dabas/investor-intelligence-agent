from typing import TypedDict, Literal


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """
    query: str
    classification: Literal["rag", "mcp", "both"]
    retrieved_context: str | None
    mcp_tool_result: str | None
    final_answer: str