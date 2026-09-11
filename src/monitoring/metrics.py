from prometheus_client import Counter, Histogram

# Request-level metrics
REQUEST_COUNT = Counter(
    "app_requests_total", "Total number of chat requests received"
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency in seconds"
)

# Routing metrics (NEW in v2)
ROUTING_DECISIONS = Counter(
    "app_routing_decisions_total", "Count of router classifications", ["classification"]
)

# MCP metrics (NEW in v2)
MCP_TOOL_CALLS = Counter(
    "app_mcp_tool_calls_total", "Count of MCP tool calls", ["tool_name", "status"]
)

# Cache metrics (NEW in v2)
CACHE_HITS = Counter("app_cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("app_cache_misses_total", "Cache misses", ["cache_type"])