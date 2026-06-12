"""claude-in-the-act: capture + replay bad Claude Code responses for evals."""

import os
from pathlib import Path

# Single source of truth for where captures live (overridable for tests).
DATA_DIR = Path(
    os.environ.get(
        "CITA_DATA_DIR", str(Path.home() / "projects" / "claude-in-the-act" / "data")
    )
)
