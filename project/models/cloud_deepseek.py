# models/cloud_deepseek.py

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

"""
This module provides a function to create a DeepSeek LLM instance with specified parameters.
It retrieves the DeepSeek API key from the environment variables and allows customization of the model and temperature.
"""

load_dotenv()  # Load environment variables from .env file


def get_deepseek_llm(model: str = "deepseek-chat", temperature: float = 0.3):
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
