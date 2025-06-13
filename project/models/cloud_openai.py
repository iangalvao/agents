# models/cloud_openai.py

import os
from langchain_openai import ChatOpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

"""
This module provides a function to create an OpenAI LLM instance with specified parameters.
It retrieves the OpenAI API key from the environment variables and allows customization of the model and temperature.
"""

load_dotenv()  # Load environment variables from .env file


def get_openai_llm(model: str = "gpt-4", temperature: float = 0.3):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment")

    return ChatOpenAI(openai_api_key=api_key, model=model, temperature=temperature)
