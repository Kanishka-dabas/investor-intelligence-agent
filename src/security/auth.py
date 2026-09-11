from loguru import logger

from src.config import settings


def verify_bearer_token(provided_token: str) -> bool:
    """
    Verifies a bearer token against the configured API token.
    Returns True if valid, False otherwise.
    """
    expected_token = settings.security.bearer_token.get_secret_value()

    if not expected_token:
        logger.error("SECURITY__BEARER_TOKEN is not configured — rejecting all requests.")
        return False

    is_valid = provided_token == expected_token
    if not is_valid:
        logger.warning("Invalid bearer token provided.")
    return is_valid