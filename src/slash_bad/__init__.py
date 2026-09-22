"""slash-bad: your personal benchmark for the things you actually care about."""

import os
from pathlib import Path

__version__ = "0.2.0"

HOME = Path(os.environ.get("SLASH_BAD_HOME", str(Path.home() / ".slash-bad")))
CASES_DIR = HOME / "cases"
REPORTS_DIR = HOME / "reports"

# Set in the environment of every `claude`/`codex` process slash-bad spawns, so
# our own capture hook never fires inside a replay or a judge call.
REPLAY_ENV = "SLASH_BAD_REPLAY"
