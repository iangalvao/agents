# models/provider.py

from models.cloud_openai import get_openai_llm  # type: ignore
from models.cloud_deepseek import get_deepseek_llm  # type: ignore

"""This module provides a function to get an LLM instance based on the specified provider.
It supports OpenAI and DeepSeek as providers, allowing customization of the model and temperature.
"""


def get_llm(provider: str = "openai", model: str = "", temperature: float = 0.3):
    provider = provider.lower()

    if provider == "openai":
        return get_openai_llm(model or "gpt-4", temperature)

    elif provider == "deepseek":
        return get_deepseek_llm(model or "deepseek-chat", temperature)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
