# models/wrapped_response.py

from typing import Any
from dataclasses import dataclass


@dataclass
class LLMResponseWrapper:
    content: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    latency: float
    raw: Any  # The full raw response object
    cached_input_tokens: int = 0

    def to_dict(self):
        return {
            "content": self.content,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "latency": self.latency,
        }
