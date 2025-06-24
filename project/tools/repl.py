# tools/repl.py

import io
import sys
import traceback
from contextlib import redirect_stdout
from langchain.tools import tool


@tool
def execute_python(code: str) -> str:
    """
    Execute a string of Python code and return stdout or error.

    Example:
        >>> execute_python("print(sum([1, 2, 3]))")
        '6'
    """
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            # Create isolated global/local scope for execution
            exec(code, {}, {})
        return f.getvalue().strip() or "(no output)"
    except Exception:
        error_trace = traceback.format_exc()
        return f"[ERROR]\n{error_trace.strip()}"
