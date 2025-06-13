from pathlib import Path
import uuid
import json
from agent_core.code_agent import REPLState

SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(state: REPLState, session_id: str = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w") as f:
        f.write(state.model_dump_json())
    return session_id


def load_session(session_id: str) -> REPLState:
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path) as f:
        return REPLState.parse_raw(f.read())
