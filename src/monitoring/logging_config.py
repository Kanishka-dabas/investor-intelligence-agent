import sys
from pathlib import Path

from loguru import logger

from src.config import settings

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def configure_logging() -> None:
    """
    Configures loguru for structured, rotated JSON logging. Call this
    once at application startup (e.g. in api/main.py).
    """
    logger.remove()  # remove default handler

    # Console output — human-readable, for local dev
    logger.add(
        sys.stderr,
        level=settings.app.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )

    # File output — structured JSON, rotated daily, kept for 14 days
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.jsonl",
        level=settings.app.log_level,
        format="{message}",
        serialize=True,  # JSON output
        rotation="00:00",
        retention="14 days",
        compression="zip",
    )


def log_node_transition(node_name: str, input_summary: str, output_summary: str) -> None:
    """
    Logs a LangGraph node transition — which node ran, with a short
    summary of its input/output. Critical for debugging routing decisions.
    """
    logger.bind(event="node_transition", node=node_name).info(
        f"Node '{node_name}' executed | input: {input_summary[:100]} | output: {output_summary[:100]}"
    )