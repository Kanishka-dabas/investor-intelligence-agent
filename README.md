# Investor Intelligence Agent

An agentic AI platform for financial report analysis — combining retrieval-augmented generation (RAG) over ingested SEC filings with live market data, orchestrated through a LangGraph-based agentic router. Fully containerized and deployed on AWS EC2, with CI/CD via GitHub Actions.

**Live demo:** http://15.134.23.12:8501

---

## What it does

Ask questions like *"What was Apple's revenue last quarter?"*, *"Compare Amazon and Alphabet's net income"*, or *"What is Apple's current stock price?"* — the agent figures out whether to answer from ingested financial reports, live market data, or both, and responds with a grounded, cited answer streamed token-by-token.

## Architecture

```
                          ┌─────────────────┐
                          │   Streamlit UI   │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │   FastAPI (SSE)   │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  LangGraph Router  │
                          └───┬───────────┬───┘
                              │           │
                    ┌─────────▼───┐   ┌───▼──────────────┐
                    │  RAG Node    │   │   MCP Node        │
                    │ (Qdrant +    │   │ (custom Financial  │
                    │  reranker)   │   │  Data MCP Server)  │
                    └──────────────┘   └───────────────────┘
```

- **Ingestion:** PDF → markdown → structure-aware, table-safe chunking → document-type validation before entering the pipeline.
- **Retrieval:** Hybrid search — dense embeddings + sparse BM25, fused via Reciprocal Rank Fusion in Qdrant, reranked with a cross-encoder.
- **KPI extraction:** Structured LLM output (Pydantic schemas) persisted to PostgreSQL, with idempotent, content-hash-based ingestion tracking.
- **Agentic routing:** A LangGraph StateGraph classifies each query and routes it to the RAG pipeline, to a custom-built Financial Data MCP Server for live market data, or to both.
- **LLM gateway:** Groq (primary) with automatic Gemini fallback, `temperature=0` for deterministic financial answers.
- **Guardrails:** Prompt-injection detection, topic-relevance filtering, bearer-token auth.
- **Observability:** Structured logging, Prometheus metrics, LangSmith tracing across the graph.

## Why it's agentic, not just RAG

The MCP server (built from scratch with FastMCP, exposing tools like `get_stock_quote`, `get_financial_ratios`, `compare_stocks`) is a separately deployed service the agent calls autonomously based on the router's classification — not a hardcoded API integration. Live-data questions and historical-report questions are handled by genuinely different code paths, decided at runtime.

## Evaluation

A custom evaluation suite (`evaluation/`) tests exact-number lookups, comparative questions, adversarial period-confusion cases, and MCP-routing accuracy. It was used throughout development to systematically find and fix real bugs — including a subtle case where multi-period financial tables (e.g. "Three Months" vs "Six Months Ended") got corrupted during PDF-to-markdown conversion, causing the wrong figures to be extracted or retrieved entirely. Full methodology, results, and root-cause writeups are in **[EVALUATION.md](./EVALUATION.md)**.

## Tech stack

| Layer | Technology |
|---|---|
| LLM orchestration | LangChain, LangGraph |
| LLM providers | Groq (primary), Gemini (fallback) |
| Vector search | Qdrant (hybrid dense + sparse) |
| Structured data | PostgreSQL (Supabase) |
| Caching | Redis |
| Live market data | Custom MCP server (FastMCP) |
| API | FastAPI (SSE streaming) |
| Frontend | Streamlit |
| Observability | Prometheus, LangSmith |
| Infra | Docker Compose, AWS EC2 |
| CI/CD | GitHub Actions |

## Project structure

```
src/
  ingestion/      PDF → markdown, chunking, validation
  retrieval/      Qdrant vector store, hybrid retriever
  rag/            KPI extraction, chat/answer generation
  llm/            Provider gateway (Groq/Gemini)
  caching/        Redis-based response & embedding cache
  database/       Postgres models, idempotent ingestion tracking
  security/       Guardrails, auth
  mcp_client/     Client for the Financial Data MCP Server
  graph/          LangGraph router (rag/mcp/respond nodes)
  monitoring/     Logging, metrics, tracing
api/              FastAPI app
frontend/         Streamlit app
evaluation/       Test set, metrics, eval runner
tests/            pytest unit tests
docker/           Dockerfile, docker-compose.yml
.github/workflows/  CI (pytest) and CD (auto-deploy to EC2)
```

## Running locally

```bash
git clone https://github.com/Kanishka-dabas/investor-intelligence-agent.git
cd investor-intelligence-agent
cp .env.example .env   # fill in your API keys
docker compose -f docker/docker-compose.yml up -d --build
```

Then open `http://localhost:8501`.

## Known limitations

- The MCP server's OAuth was originally hosted on FastMCP Cloud, which enforces interactive browser-based auth — incompatible with a headless server. Resolved by self-hosting the MCP server with static bearer-token authentication instead.
- Qdrant's vector data does not persist across fresh deployments (it's a local Docker volume); PostgreSQL (Supabase-hosted) does. A migration/seeding step would be needed for a fully automated first-time deploy.
- See [EVALUATION.md](./EVALUATION.md) for retrieval and generation limitations found during evaluation.

## Related project

**[Financial Data MCP Server](https://github.com/Kanishka-dabas/financial-data-mcp-server)** — a standalone MCP server built with FastMCP, exposing Yahoo Finance-backed tools for stock quotes, company overviews, financial ratios, historical prices, and stock comparisons.