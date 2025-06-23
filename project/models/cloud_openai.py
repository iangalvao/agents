# models/cloud_openai.py
"""
This module provides a function to create an OpenAI LLM instance with specified parameters.
It retrieves the OpenAI API key from the environment variables and allows customization of the model and temperature.
"""
# models/cloud_openai.py

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.wrapped_response import LLMResponseWrapper  # type: ignore
from pydantic import SecretStr

load_dotenv()


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    api_key = SecretStr(os.getenv("OPENAI_API_KEY") or "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment")

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


def call_openai(
    prompt: list[HumanMessage], llm: ChatOpenAI, temperature: float = 0.3
) -> LLMResponseWrapper:
    model_name = llm.model_name  # Get the model name from the LLM instance
    print("Using OpenAI LLM with model:", model_name, "and temperature:", temperature)
    if not model_name:
        raise ValueError("Model must be specified in the OpenAI LLM instance")
    start = time.time()
    response = llm.invoke(prompt)
    end = time.time()

    # Token info (LangChain v0.1+ exposes this in `response.response_metadata`)
    metadata = getattr(response, "response_metadata", {})
    usage = metadata.get("token_usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # OpenAI pricing (2024)
    cost_per_1M = {
        "gpt-4.1": (2.00, 8.0),
        "gpt-4.1-mini": (0.40, 1.6),
        "gpt-4.5-preview": (75.0, 150.0),
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "o1-pro": (150.0, 600.0),
        "o1-mini": (1.10, 4.40),
        "o3": (2.00, 8.00),
        "o3-pro": (20.00, 80.0),
        "o3-mini": (1.10, 4.40),
        "o4-mini": (1.10, 4.40),
        "computer-use-preview": (3.0, 12.0),
    }
    pricing = cost_per_1M.get(model_name, (0.0, 0.0))
    cost = (input_tokens / 1000000 * pricing[0]) + (
        output_tokens / 1000000 * pricing[1]
    )

    return LLMResponseWrapper(
        content=response.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost=round(cost, 6),
        latency=round(end - start, 3),
        raw=response,
    )
