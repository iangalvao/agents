import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("REPLAgent")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/repl_agent.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

if not logger.hasHandlers():
    logger.addHandler(file_handler)


def log_attempt(prompt: str, attempt_number: int, code: str, result: str):
    logger.info(f"[Prompt] {prompt}")
    logger.info(f"[Attempt #{attempt_number}] Code:\n{code}")
    logger.info(f"[Attempt #{attempt_number}] Result:\n{result}")
