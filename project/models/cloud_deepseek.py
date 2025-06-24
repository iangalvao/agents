# models/cloud_deepseek.py

import os
import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from models.wrapped_response import LLMResponseWrapper
from pydantic import SecretStr
from models.models import load_models

"""
This module provides a function to create a DeepSeek LLM instance with specified parameters.
It retrieves the DeepSeek API key from the environment variables and allows customization of the model and temperature.
"""

load_dotenv()  # Load environment variables from .env file

MODELS = load_models(provider="deepseek")  # Load models.json from the current directory
MODELS_NAMES = {k for k in MODELS.keys()}
COST_PER_1M = {k: v.get("cost1M", [0.0, 0.0]) for k, v in MODELS.items()}
# print("Loaded DeepSeek models:", MODELS.keys())
# print("Cost per 1M tokens:", COST_PER_1M)


def get_deepseek_llm(model: str = "deepseek-chat", temperature: float = 0.3):
    if model not in MODELS_NAMES:
        raise ValueError(
            f"Model '{model}' is not supported. Available models: {', '.join(MODELS_NAMES)}"
        )
    print("Using DeepSeek LLM with model:", model, "and temperature:", temperature)
    api_key = SecretStr(os.getenv("DEEPSEEK_API_KEY") or "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set in the environment")

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model=model,
        temperature=temperature,
    )


def call_deepseek(prompt: list, llm, temperature: float = 0.3):
    """
    Call the DeepSeek LLM with the provided prompt and return the response.
    """
    print("Calling DeepSeek LLM with prompt:", prompt)
    start = time.time()
    response = llm.invoke(prompt)
    end = time.time()
    print("DeepSeek LLM response:", response)
    input_tokens = response.usage_metadata.get("input_tokens", 0)
    cached_input_tokens = response.usage_metadata.get("input_token_details", {}).get(
        "cache_read", 0
    )
    input_tokens -= cached_input_tokens  # Adjust input tokens for cache reads
    if input_tokens < 0:
        input_tokens = 0
    output_tokens = response.usage_metadata.get("output_tokens", 0)
    input_cost = COST_PER_1M.get(llm.model_name, (0.0, 0.0))[0] * (
        input_tokens / 1_000_000
    )
    cached_input_tokens_cost = COST_PER_1M.get(llm.model_name, (0.0, 0.0))[1] * (
        cached_input_tokens / 1_000_000
    )
    output_cost = COST_PER_1M.get(llm.model_name, (0.0, 0.0))[2] * (
        output_tokens / 1_000_000
    )
    return LLMResponseWrapper(
        content=response.content,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_cost=round(input_cost + output_cost + cached_input_tokens_cost, 6),
        latency=round(end - start, 3),
        raw=response,
    )
