"""Analyze saved Provo/OB1 outputs without rerunning either model."""

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
ANALYSIS_OUTPUT_DIR = EXPERIMENT_DIR / "attention_profile_focused"
LANDSCAPE_POINTS = 41
SIGMA_LEFT = 0.3738
SIGMA_RIGHT = 3.21289
SOURCE_ACCURACY = 0.76942
CHECKPOINT_ID = "s11_acc076942_l037380_r321289"


def run(command: list[str]) -> None:
    """Print and execute one command from the repository root."""
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def require_file(path: Path) -> None:
    """Stop with a clear message when an experiment artifact is absent."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run run_provo_experiment.py first."
        )


def main() -> None:
    """Compute metrics, diagnostics, confidence intervals, and plots."""
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(
            "Run setup_provo_environment.py before this script."
        )
    require_file(ET1_OUTPUT_DIR / "et1_token_values.csv")
    require_file(OB1_OUTPUT_DIR / "ob1_fixations.csv")
    require_file(OB1_OUTPUT_DIR / "ob1_worker_manifest.json")
    run(
        [
            str(VENV_PYTHON),
            "cognitive_model_comparsion/main.py",
            "compare-attention-profile",
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
            "--et1-dir",
            str(ET1_OUTPUT_DIR),
            "--ob1-dir",
            str(OB1_OUTPUT_DIR),
            "--ob1-attention-skew",
            "3",
            "--ob1-attention-skew",
            "4",
            "--fixation-weighting",
            "duration",
            "--profile-component",
            "focused",
            "--candidate-support-policy",
            "fixation_matched",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "20260725",
            "--with-sigma-landscape",
            "--landscape-sigma-min",
            "0.1",
            "--landscape-sigma-max",
            "5.0",
            "--landscape-points",
            str(LANDSCAPE_POINTS),
            "--output-dir",
            str(ANALYSIS_OUTPUT_DIR),
        ]
    )


if __name__ == "__main__":
    main()
