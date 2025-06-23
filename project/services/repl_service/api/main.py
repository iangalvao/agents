from agent_core.code_agent import build_graph, REPLState
from services.repl_service.repl_agent.logger import log_attempt, logger
from services.repl_service.repl_agent.storage import save_session
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

"""This module sets up a FastAPI application that provides an endpoint to run a REPL agent.
It defines a POST endpoint `/run` that accepts a user prompt, processes it through a state graph, and logs the attempts made to generate code based on the prompt. The results are saved in a session file."""


app = FastAPI()
openai_model = "gpt-4o-mini"

try:
    graphs = {
        "openai": build_graph(
            llm_provider="openai", model=openai_model, temperature=0.3
        ),
        "deepseek": build_graph(
            llm_provider="deepseek", model="deepseek-chat", temperature=0.3
        ),
    }
except ValueError as ve:
    logger.error(f"Error building graphs: {ve}")
    exit(1)
except Exception as e:
    logger.error(f"Failed to build graphs: {e}")
    exit(1)


class PromptRequest(BaseModel):
    session_id: str
    user_prompt: str
    llm_provider: str = "openai"


@app.post("/run")
def run(prompt: PromptRequest):
    if not prompt.user_prompt.strip():
        logger.error("Received empty prompt")
        raise ValueError("Prompt must not be empty")

    if prompt.llm_provider.lower() not in ["openai", "deepseek"]:
        logger.error(f"Unsupported LLM provider: {prompt.llm_provider}")
        raise ValueError(
            f"Unsupported LLM provider: {prompt.llm_provider}. Supported providers are 'openai' and 'deepseek'."
        )
    logger.info(f"Received prompt from {prompt.llm_provider}")
    provider = prompt.llm_provider.lower()
    if provider not in graphs:
        logger.error(f"Graph for provider {prompt.llm_provider} not found")
        raise ValueError(f"Graph for provider {provider} not found")

    state = REPLState(user_prompt=prompt.user_prompt)
    graph = graphs[provider]

    try:
        raw_output = graph.invoke(state)
        result = REPLState(**raw_output)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise ValueError(f"Graph invocation failed: {e}")

    if not result:
        logger.error("Graph invocation failed")
        raise ValueError("Graph invocation failed")

    logger.info(f"Graph invocation successful for {prompt.llm_provider}")

    for attempt in result.attempts:
        log_attempt(
            prompt=prompt.user_prompt,
            attempt_number=attempt.attempt_number,
            code=attempt.generated_code,
            result=attempt.result or attempt.error or "No output",
        )
    if prompt.session_id:
        session_id = save_session(result, session_id=prompt.session_id)
        print("🧾 Session saved:", session_id)
    logger.info(
        f"Session {session_id} completed with {len(result.attempts)} attempt(s)."
    )

    return result.dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
