import asyncio
import json

from loguru import logger

from src.graph.state import AgentState
from src.llm.gateway import LLMGateway
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.chat import build_context, CHAT_SYSTEM_PROMPT
from src.mcp_client.client import call_mcp_tool
from src.caching.redis_cache import RedisCache

ROUTER_SYSTEM_PROMPT = """You classify financial questions into one of three categories:

- "rag": the question is about historical/static data from ingested financial reports
  (e.g. past revenue, net income, risk factors, growth drivers from annual/quarterly reports)
- "mcp": the question is about LIVE/current market data (e.g. current stock price, today's
  quote, real-time company overview, live financial ratios)
- "both": the question needs combining historical report data AND live market data
  (e.g. "compare last quarter's reported revenue to the current stock price reaction")

Respond with ONLY one word: rag, mcp, or both."""


def router_node(state: AgentState, gateway: LLMGateway) -> AgentState:
    """Classifies the query to decide which node(s) should handle it."""
    query = state["query"]
    response = gateway.generate(query, system_prompt=ROUTER_SYSTEM_PROMPT)
    classification = response.strip().lower()

    if classification not in ("rag", "mcp", "both"):
        logger.warning(f"Router returned unexpected classification '{classification}', defaulting to 'rag'.")
        classification = "rag"

    logger.info(f"Router classified query as: {classification}")
    return {**state, "classification": classification}


def rag_node(state: AgentState, retriever: HybridRetriever, cache: RedisCache) -> AgentState:
    """Retrieves relevant chunks from ingested financial reports."""
    query = state["query"]
    retrieved_chunks = retriever.retrieve(query, top_k=10)

    if not retrieved_chunks:
        context = "No relevant context found in ingested reports."
    else:
        context = build_context(retrieved_chunks)

    return {**state, "retrieved_context": context}


def mcp_node(state: AgentState) -> AgentState:
    """
    Calls the appropriate MCP tool based on the query. For now, uses a
    simple heuristic to pick the tool; this can be replaced with an
    LLM-based tool-selection step for more complex queries.
    """
    query = state["query"].lower()

    try:
        if "price" in query or "quote" in query:
            result = asyncio.run(call_mcp_tool("get_stock_quote_tool", {"ticker": "AAPL"}))
        elif "ratio" in query:
            result = asyncio.run(call_mcp_tool("get_financial_ratios_tool", {"ticker": "AAPL"}))
        else:
            result = asyncio.run(call_mcp_tool("get_company_overview_tool", {"ticker": "AAPL"}))

        mcp_result = json.dumps(result.data if hasattr(result, "data") else str(result))
    except Exception as e:
        logger.error(f"MCP tool call failed: {e}")
        mcp_result = f"MCP tool call failed: {e}"

    return {**state, "mcp_tool_result": mcp_result}


async def respond_node(state: AgentState, gateway: LLMGateway) -> AgentState:
    """
    Combines whatever context is available (RAG and/or MCP) into a final
    answer. 
    """
    query = state["query"]
    context_parts = []

    if state.get("retrieved_context"):
        context_parts.append(f"[Report data]\n{state['retrieved_context']}")
    if state.get("mcp_tool_result"):
        context_parts.append(f"[Live market data]\n{state['mcp_tool_result']}")

    full_context = "\n\n".join(context_parts) if context_parts else "No context available."

    prompt = f"{CHAT_SYSTEM_PROMPT}\n\nContext:\n{full_context}\n\nQuestion: {query}"

    chat_model = gateway.get_chat_model("groq")
    full_answer = ""
    async for chunk in chat_model.astream(prompt):
        full_answer += chunk.content

    return {**state, "final_answer": full_answer}