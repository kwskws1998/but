"""Create the local Provo/OB1 environment and verify the installation."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENV_DIR = REPOSITORY_ROOT / ".venv-cognitive-model-comparison"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS = (
    REPOSITORY_ROOT / "cognitive_model_comparsion" / "requirements.txt"
)


def run(command: list[str]) -> None:
    """Print and execute one command from the repository root."""
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    """Create the venv, prepare Provo assets, audit them, and run tests."""
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
        ]
    )
    run([str(VENV_PYTHON), "-m", "pip", "check"])
    run(
        [
            str(VENV_PYTHON),
            "cognitive_model_comparsion/main.py",
            "setup",
            "--corpus",
            "provo",
        ]
    )
    run(
        [
            str(VENV_PYTHON),
            "cognitive_model_comparsion/main.py",
            "audit",
            "--corpus",
            "provo",
        ]
    )
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pytest",
            "-q",
            "cognitive_model_comparsion/tests",
            "cognitive_model_comparsion/behavior_level_validation/tests",
        ]
    )


if __name__ == "__main__":
    main()
