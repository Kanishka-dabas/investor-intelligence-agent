from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from src.config import settings


class LLMGatewayError(Exception):
    """Raised when both primary and fallback LLM providers fail."""


class LLMGateway:
    """
    Unified LangChain-based interface over Groq (primary) and Gemini
    (fallback). Exposes both a simple .generate() for plain text calls
    and .get_chat_model() for LangChain-native usage (structured output,
    tool binding, LangGraph nodes).
    """

    def __init__(self):
        self.groq_chat: BaseChatModel = ChatGroq(
            api_key=settings.llm.groq_api_key.get_secret_value(),
            model=settings.llm.groq_model,
            temperature=0,
        )
        self.gemini_chat: BaseChatModel = ChatGoogleGenerativeAI(
            google_api_key=settings.llm.gemini_api_key.get_secret_value(),
            model=settings.llm.gemini_model,
            temperature=0,
            timeout=30,
        )

    def get_chat_model(self, provider: str = "groq") -> BaseChatModel:
        """
        Returns the raw LangChain chat model for a given provider, for
        use with .with_structured_output(), tool binding, or LangGraph
        nodes that need direct LangChain-native access.
        """
        if provider == "groq":
            return self.groq_chat
        if provider == "gemini":
            return self.gemini_chat
        raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a plain-text response, trying Groq first and falling
        back to Gemini on failure. Raises LLMGatewayError if both fail.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.groq_chat.invoke(messages)
            return response.content
        except Exception as e:
            logger.warning(f"Groq failed ({e}), falling back to Gemini...")

        try:
            response = self.gemini_chat.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")
            raise LLMGatewayError("Both Groq and Gemini failed to generate a response.") from e