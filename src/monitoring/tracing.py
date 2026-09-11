import os

from src.config import settings


def configure_langsmith() -> None:
    """
    Enables LangSmith tracing for LangChain/LangGraph by setting the
    required environment variables. Call this once at startup, before
    building the graph.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.monitoring.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.monitoring.langsmith_project