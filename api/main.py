import json

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse
from api.dependencies import get_graph_app, get_gateway
from src.security.guardrails import validate_query
from src.security.auth import verify_bearer_token
from src.monitoring.logging_config import configure_logging
from src.monitoring.tracing import configure_langsmith
from src.monitoring.metrics import REQUEST_COUNT, ROUTING_DECISIONS
from prometheus_client import make_asgi_app
from loguru import logger

configure_logging()
configure_langsmith()

app = FastAPI(title="Investor Intelligence Agent API")

# Prometheus /metrics endpoint
app.mount("/metrics", make_asgi_app())


def require_auth(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    if not verify_bearer_token(token):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
def chat(request: ChatRequest, graph_app=Depends(get_graph_app)):
    REQUEST_COUNT.inc()

    is_valid, reason = validate_query(request.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    result = graph_app.invoke({"query": request.query})
    ROUTING_DECISIONS.labels(classification=result["classification"]).inc()

    return ChatResponse(
        query=request.query,
        classification=result["classification"],
        answer=result["final_answer"],
    )


@app.post("/chat/stream", dependencies=[Depends(require_auth)])
async def chat_stream(request: ChatRequest, graph_app=Depends(get_graph_app)):
    REQUEST_COUNT.inc()

    is_valid, reason = validate_query(request.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    async def generate():
        async for event in graph_app.astream_events(
            {"query": request.query}, version="v2"
        ):
            event_type = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node")

            # Node started — send a status update
            if event_type == "on_chain_start" and node_name in ("router", "rag", "mcp", "respond"):
                status_messages = {
                    "router": "Classifying query...",
                    "rag": "Retrieving from knowledge base...",
                    "mcp": "Calling live market data tool...",
                    "respond": "Generating answer...",
                }
                yield json.dumps({"type": "status", "message": status_messages[node_name]}) + "\n"

            # LLM token from the respond node — send as a content chunk
            elif event_type == "on_chat_model_stream" and node_name == "respond":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield json.dumps({"type": "chunk", "content": chunk}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")