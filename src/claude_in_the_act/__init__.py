"""claude-in-the-act: capture + replay bad Claude Code responses for evals."""

import os
from pathlib import Path

# Single source of truth for where captures live (overridable via CITA_DATA_DIR,
# which tests and the dev workflow use). The default is a stable, user-global
# location under ~/.claude — deliberately NOT inside the plugin install dir,
# which `/plugin update` wipes, nor the source checkout, which a friend who
# installs the plugin never has.
DATA_DIR = Path(
    os.environ.get(
        "CITA_DATA_DIR", str(Path.home() / ".claude" / "claude-in-the-act" / "data")
    )
)
