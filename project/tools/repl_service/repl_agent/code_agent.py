# agent_core/code_agent.py

from functools import partial
from langchain_core.messages import HumanMessage  # type: ignore
from langgraph.graph import StateGraph, END  # type: ignore
from tools.repl import execute_python  # type: ignore
from models.provider import get_llm  # type: ignore
from typing import List, Optional
from pydantic import BaseModel


class CodeAttempt(BaseModel):
    attempt_number: int
    generated_code: str
    result: Optional[str]
    error: Optional[str]


class REPLState(BaseModel):
    user_prompt: str
    attempts: List[CodeAttempt] = []
    retry: bool = False
    retries: int = 0

    def get(self, key: str, default=None):
        """Get a value from the state, returning default if not found."""
        return getattr(self, key, default)


def generate_code(state: REPLState, llm) -> REPLState:
    prompt = state.user_prompt
    previous_error = state.get("attempts", [])[-1].error if state.attempts else None

    if not prompt:
        raise ValueError("Missing 'user_prompt' in state")

    system_msg = (
        "You are a Python code generator. Write minimal working code to solve the user's request.\n"
        "Do NOT explain — only output the code block."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    if previous_error:
        messages.append(
            {
                "role": "user",
                "content": f"The previous code failed with this error:\n{previous_error}.\nPlease fix it.",
            }
        )

    print("Generating code with messages:", messages)

    response = llm.invoke([HumanMessage(content=m["content"]) for m in messages])
    attempt_number = state.retries + 1
    generated_code = response.content.strip()
    state.attempts.append(
        CodeAttempt(
            attempt_number=attempt_number,
            generated_code=generated_code,
            result=None,
            error=None,
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
    else:
        attempt.result = result
        state.retry = False

    return state


def build_graph(
    max_retries: int = 2,
    llm_provider: str = "openai",
    model: str = "gpt-4",
    temperature: float = 0.3,
) -> StateGraph:
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

    generate_code_with_llm = partial(generate_code, llm=llm)
    workflow.add_node("generate_code", generate_code_with_llm)
    workflow.add_node("execute_code", execute_code)

    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "execute_code")

    def decide_next(state: REPLState):
        if state.retry:
            state.retries += 1
        if state.retry and state.retries < max_retries:
            return "generate_code"
        return END

    workflow.add_conditional_edges(
        "execute_code",
        decide_next,
        {"generate_code": "generate_code", END: END},
    )

    return workflow.compile()
