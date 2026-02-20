"""
LLM factory — returns the appropriate LangChain chat model based on LITELLM_MODE.

  proxy   → ChatOpenAI pointed at the LiteLLM proxy container (dev default)
  library → ChatLiteLLM using the litellm library in-process (production)

Both return the same LangChain BaseChatModel interface, so callers are unaware
of the underlying routing mechanism.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_settings


def get_chat_model(
    *,
    model: str | None = None,
    streaming: bool = True,
    temperature: float = 0.3,
) -> BaseChatModel:
    """
    Return a configured chat model.

    Args:
        model:       Override the model name. Defaults to settings.primary_model.
        streaming:   Enable token-by-token streaming.
        temperature: Sampling temperature.
    """
    settings = get_settings()
    model_name = model or settings.primary_model

    if settings.litellm_mode == "library":
        from langchain_community.chat_models import ChatLiteLLM

        return ChatLiteLLM(
            model=model_name,
            streaming=streaming,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_master_key,
            model=model_name,
            streaming=streaming,
            temperature=temperature,
        )
