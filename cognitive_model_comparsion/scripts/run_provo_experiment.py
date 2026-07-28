"""Generate T5 token geometry and 100 seeded OB1 Provo trajectories."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = (
    REPOSITORY_ROOT
    / ".venv-cognitive-model-comparison"
    / "bin"
    / "python"
)
EXPERIMENT_DIR = (
    REPOSITORY_ROOT
    / "cognitive_model_comparsion"
    / "outputs"
    / "provo_rebuttal_experiment"
)
ET1_OUTPUT_DIR = EXPERIMENT_DIR / "et1"
OB1_OUTPUT_DIR = EXPERIMENT_DIR / "ob1"
OB1_RUNTIME_DIR = (
    REPOSITORY_ROOT
    / "cognitive_model_comparsion"
    / "data"
    / "ob1_runtime"
    / "provo_rebuttal"
)
OB1_WORKERS = 10
SIGMA_LEFT = 0.3738
SIGMA_RIGHT = 3.21289
SOURCE_ACCURACY = 0.76942
CHECKPOINT_ID = "s11_acc076942_l037380_r321289"


def run(command: list[str]) -> None:
    """Print and execute one command from the repository root."""
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    """Run frozen ET1 geometry extraction and the OB1 simulations."""
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(
            "Run setup_provo_environment.py before this script."
        )
    run(
        [
            str(VENV_PYTHON),
            "cognitive_model_comparsion/main.py",
            "predict-et1",
            "--corpus",
            "provo",
            "--sigma-left",
            str(SIGMA_LEFT),
            "--sigma-right",
            str(SIGMA_RIGHT),
            "--sigma-value-type",
            "effective",
            "--sigma-source-accuracy",
            str(SOURCE_ACCURACY),
            "--checkpoint-id",
            CHECKPOINT_ID,
            "--output-dir",
            str(ET1_OUTPUT_DIR),
        ]
    )
    run(
        [
            str(VENV_PYTHON),
            "cognitive_model_comparsion/main.py",
            "simulate-ob1",
            "--corpus",
            "provo",
            "--runtime-dir",
            str(OB1_RUNTIME_DIR),
            "--output-dir",
            str(OB1_OUTPUT_DIR),
            "--seeds",
            "0:100",
            "--n-trials",
            "55",
            "--python-hash-seed",
            "20260725",
            "--workers",
            str(OB1_WORKERS),
        ]
    )


if __name__ == "__main__":
    main()
