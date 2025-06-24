# agent_core/code_agent.py

from functools import partial
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tools.repl import execute_python
from models.provider import get_llm
from typing import List, Optional
from pydantic import BaseModel

from models.cloud_openai import call_openai
from models.cloud_deepseek import call_deepseek
from models.wrapped_response import LLMResponseWrapper


class CodeAttempt(BaseModel):
    attempt_number: int
    generated_code: str
    result: Optional[str]
    error: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: Optional[float] = 0.0
    latency: Optional[float] = 0.0

    def to_dict(self):
        return {
            "attempt_number": self.attempt_number,
            "generated_code": self.generated_code,
            "result": self.result,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost,
            "latency": self.latency,
        }


class REPLState(BaseModel):
    user_prompt: str
    attempts: List[CodeAttempt] = []
    retry: bool = False
    retries: int = 0

    def get(self, key: str, default=None):
        """Get a value from the state, returning default if not found."""
        return getattr(self, key, default)

    def get_total_cost(self) -> float:
        """Calculate the total cost of all attempts."""
        return sum(
            attempt.total_cost for attempt in self.attempts if attempt.total_cost
        )

    def get_total_tokens(self) -> int:
        """Calculate the total number of tokens used across all attempts."""
        return sum(
            attempt.input_tokens + attempt.output_tokens for attempt in self.attempts
        )

    def get_total_latency(self) -> float:
        """Calculate the total latency of all attempts."""
        return sum(attempt.latency for attempt in self.attempts if attempt.latency)

    def get_total_input_tokens(self) -> int:
        """Calculate the total input tokens used across all attempts."""
        return sum(
            attempt.input_tokens for attempt in self.attempts if attempt.input_tokens
        )

    def get_total_output_tokens(self) -> int:
        """Calculate the total output tokens used across all attempts."""
        return sum(
            attempt.output_tokens for attempt in self.attempts if attempt.output_tokens
        )


def generate_code(state: REPLState, llm, provider="openai") -> REPLState:
    prompt = state.user_prompt
    previous_error = state.get("attempts", [])[-1].error if state.attempts else None

    if not prompt:
        raise ValueError("Missing 'user_prompt' in state")

    system_msg = (
        "You are a Python code generator. Write minimal working code to solve the user's request.\n"
        "Do NOT explain — only output the code block. Dont write any markdown or comments, only code.\n"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    if previous_error:
        messages.append(
            {
                "role": "user",
                "content": f"The previous code failed with this error:\n{previous_error}.\nPlease fix it. This is your previous code:\n{state.attempts[-1].generated_code}",
            }
        )

    print("Generating code with messages:", messages)

    # response = llm.invoke([HumanMessage(content=m["content"]) for m in messages])
    attempt_number = state.retries + 1
    if provider == "deepseek":
        response: LLMResponseWrapper = call_deepseek(
            prompt=[HumanMessage(content=m["content"]) for m in messages],
            llm=llm,
        )
    elif provider == "openai":
        response: LLMResponseWrapper = call_openai(
            prompt=[HumanMessage(content=m["content"]) for m in messages],
            llm=llm,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    state.attempts.append(
        CodeAttempt(
            attempt_number=attempt_number,
            generated_code=response.content,
            result=None,
            error=None,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_cost=response.total_cost,
            latency=response.latency,
        )
    )
    return state


# 2. Run the generated code using REPL tool
def execute_code(state: REPLState) -> REPLState:
    code = state.get("attempts", [])[-1].generated_code if state.attempts else ""
    result = execute_python.invoke(code)
    print("Executing code:", code)
    print("Execution result:", result)
    # Check for errors

    attempt = state.attempts[-1]
    result = execute_python.invoke(attempt.generated_code)

    if isinstance(result, str) and result.startswith("[ERROR]"):
        attempt.error = result
        state.retry = True
        state.retries += 1
        print(
            f"Error in code execution: {result}. Marked to retry. Current retry count: {state.retries}"
        )
    else:
        attempt.result = result
        state.retry = False

    return state


def build_graph(
    max_retries: int = 2,
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
):
    """
    Build the state graph for the code generation and execution workflow.
    Args:
        max_retries (int): Maximum number of retries for code generation.
        llm_provider (str): The LLM provider to use ("openai" or "deepseek").
        model (str): The model to use for the LLM.
        temperature (float): The temperature for the LLM.
    Returns:
        StateGraph: The compiled state graph for the workflow.
    """
    llm = get_llm(provider=llm_provider, model=model, temperature=temperature)
    if not llm:
        raise ValueError(
            f"Failed to initialize LLM with provider: {llm_provider}, model: {model}"
        )

    workflow = StateGraph(REPLState)  # <-- Use plain dict

    generate_code_with_llm = partial(generate_code, llm=llm, provider=llm_provider)
    workflow.add_node("generate_code", generate_code_with_llm)
    workflow.add_node("execute_code", execute_code)

    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "execute_code")

    def decide_next(state: REPLState):
        if state.retry and state.retries < max_retries:
            state.retry = False
            print(
                f"Retrying code generation, attempt {state.retries + 1}/{max_retries}"
            )
            return "generate_code"
        return END

    workflow.add_conditional_edges(
        "execute_code",
        decide_next,
        {"generate_code": "generate_code", END: END},
    )

    return workflow.compile()
