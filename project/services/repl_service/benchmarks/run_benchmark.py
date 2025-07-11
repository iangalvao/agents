# benchmark/run_benchmark.py
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

# import mlflow
from agent_core.code_agent import build_graph, CodeAttempt, REPLState

# Ensure the script is running from the project root directory
# and that the necessary directories exist.
# This script is designed to run benchmarks for the REPL agent using LangGraph.

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Adjust as needed
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Project root not found at {PROJECT_ROOT}")
# Ensure the project root is a valid directory
if not PROJECT_ROOT.is_dir():
    raise NotADirectoryError(f"Project root is not a directory: {PROJECT_ROOT}")
# Ensure the script is running from the project root
if not PROJECT_ROOT.name == "project":
    raise RuntimeError(
        f"Script must be run from the project root directory, found: {PROJECT_ROOT.name}"
    )
# Ensure the services directory exists
SERVICES_DIR = PROJECT_ROOT / "services"
if not SERVICES_DIR.exists():
    raise FileNotFoundError(f"Services directory not found at {SERVICES_DIR}")
if not SERVICES_DIR.is_dir():
    raise NotADirectoryError(f"Services path is not a directory: {SERVICES_DIR}")
# Ensure the repl_service directory exists
REPL_SERVICE_DIR = SERVICES_DIR / "repl_service"
if not REPL_SERVICE_DIR.exists():
    raise FileNotFoundError(f"REPL service directory not found at {REPL_SERVICE_DIR}")
if not REPL_SERVICE_DIR.is_dir():
    raise NotADirectoryError(
        f"REPL service path is not a directory: {REPL_SERVICE_DIR}"
    )
# Ensure the project root is in the Python path
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))
# Load prompts and results paths
PROMPTS_PATH = (
    PROJECT_ROOT / "services" / "repl_service" / "benchmarks" / "prompts.jsonl"
)
if not PROMPTS_PATH.exists():
    raise FileNotFoundError(f"Prompts file not found at {PROMPTS_PATH}")
RESULTS_DIR = PROJECT_ROOT / "services" / "repl_service" / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

provider = "deepseek"  # Default provider, can be overridden by environment variable
model = "deepseek-chat"  # Default model, can be overridden by environment variable


def format_output(
    model: str,
    provider: str,
    result: REPLState,
    user_prompt: str,
    expected_result: str = "",
    session_id: str = "",
    test_id: str = "",
    timestamp: str = "",
):
    """
    Save the output of the REPL agent run to a structured format.
    Args:
        result (REPLState): The state object containing the results of the REPL agent run.
        user_prompt (str): The original user prompt.
        expected_result (str, optional): The expected result for comparison.
        session_id (str, optional): Unique identifier for the session.
        test_id (str, optional): Unique identifier for the test.
        timestamp (str, optional): Timestamp of the run.
    """

    # Format result
    last_attempt: CodeAttempt | None = (
        result.get("attempts", [])[-1] if result.get("attempts") else None
    )
    if last_attempt:
        last_attempt = last_attempt.to_dict()  # Convert to dict for JSON serialization
    else:
        last_attempt = {
            "attempt_number": 0,
            "generated_code": None,
            "result": None,
            "error": None,
        }
    output = {
        "id": test_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "prompt": user_prompt,
        "last_attempt": last_attempt,
        "total_tokens": result.get_total_tokens(),
        "total_cost": result.get_total_cost(),
        "total_latency": result.get_total_latency(),
        "total_input_tokens": result.get_total_input_tokens(),
        "total_output_tokens": result.get_total_output_tokens(),
        "retries": result.get("retries", 0),
        "model": model,
        "provider": provider,
    }

    # Check if expected result matches
    if expected_result is not None:
        output["expected_result"] = expected_result
        output["test_passed"] = (
            last_attempt["result"] == expected_result
            if last_attempt["result"] is not None
            else False
        )
    else:
        output["expected_result"] = None
        output["test_passed"] = None
    return output


# Timestamped output file
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
result_path = RESULTS_DIR / f"repl_agent__{timestamp}.jsonl"

# Load prompts
with open(PROMPTS_PATH, "r") as f:
    prompts = [json.loads(line) for line in f]

# Set LLM provider
os.environ["LLM_PROVIDER"] = provider  # or "anthropic", "deepseek", etc.
os.environ["LLM_MODEL"] = (
    model  # Default model, can be overridden by environment variable
)
os.environ["LLM_TEMPERATURE"] = "0.3"  # Adjust temperature
os.environ["LLM_MAX_RETRIES"] = "2"  # Maximum retries for code

# Build LangGraph agent
graph = build_graph(
    llm_provider=os.environ["LLM_PROVIDER"],
    model=model,  # or other model as needed
    temperature=float(os.environ["LLM_TEMPERATURE"]),
    max_retries=int(os.environ["LLM_MAX_RETRIES"]),
)

# Run each prompt

total_cost = 0.0
total_latency = 0.0
total_output_tokens = 0
with open(result_path, "w") as outfile:
    for entry in prompts:
        session_id = str(uuid.uuid4())
        user_prompt = entry["prompt"]
        # system_prompt = "You are a REPL agent that generates and executes Python code based on user prompts. Output only the code, without any mardown, additional text, or explanations."
        expected_result = entry.get("result", None)
        test_id = entry.get("id", session_id)

        # Run REPL agent
        state = {
            "user_prompt": user_prompt,
        }
        result: REPLState = REPLState(**graph.invoke(state))

        # Format output
        output = format_output(
            model=model,
            provider=provider,
            result=result,
            user_prompt=user_prompt,
            expected_result=expected_result,
            session_id=session_id,
            test_id=test_id,
            timestamp=timestamp,
        )

        # Save as JSONL
        outfile.write(json.dumps(output) + "\n")
        total_cost += output["total_cost"]
        total_latency += output["total_latency"]
        total_output_tokens += output["total_output_tokens"]
# Log results to MLflow
# mlflow.set_tracking_uri("http://localhost:5000")  # Adjust as needed
# mlflow.start_run(run_name=f"repl_agent_benchmark_{timestamp}")
# mlflow.log_artifact(str(result_path), artifact_path="benchmark_results")
# mlflow.end_run()

print(f"✅ Benchmark complete. Results saved to: {result_path}")
print(f"Total Cost: ${total_cost:.6f}")
print(f"Total Latency: {total_latency:.3f} seconds")
print(f"Total Output Tokens: {total_output_tokens}")
