from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes import router_node, rag_node, mcp_node, respond_node
from src.llm.gateway import LLMGateway
from src.retrieval.hybrid_retriever import HybridRetriever
from src.caching.redis_cache import RedisCache


def build_graph(gateway: LLMGateway, retriever: HybridRetriever, cache: RedisCache):
    graph = StateGraph(AgentState)

    graph.add_node("router", lambda state: router_node(state, gateway))
    graph.add_node("rag", lambda state: rag_node(state, retriever, cache))
    graph.add_node("mcp", lambda state: mcp_node(state))

    async def _respond_wrapper(state):
        return await respond_node(state, gateway)

    graph.add_node("respond", _respond_wrapper)

    graph.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        classification = state["classification"]
        if classification == "both":
            return "rag"
        return classification

    graph.add_conditional_edges(
        "router",
        route_decision,
        {"rag": "rag", "mcp": "mcp"},
    )

    def after_rag(state: AgentState) -> str:
        return "mcp" if state["classification"] == "both" else "respond"

    graph.add_conditional_edges("rag", after_rag, {"mcp": "mcp", "respond": "respond"})
    graph.add_edge("mcp", "respond")
    graph.add_edge("respond", END)

    return graph.compile()