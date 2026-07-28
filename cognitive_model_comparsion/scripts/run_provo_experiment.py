"""Generate T5 token geometry and 100 seeded OB1 Provo trajectories."""

from __future__ import annotations

import shlex
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO


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
LOG_DIR = EXPERIMENT_DIR / "logs"
OB1_RUNTIME_DIR = (
    REPOSITORY_ROOT
    / "cognitive_model_comparsion"
    / "data"
    / "ob1_runtime"
    / "provo_rebuttal"
)
OB1_WORKERS = 24
SIGMA_LEFT = 0.3738
SIGMA_RIGHT = 3.21289
SOURCE_ACCURACY = 0.76942
CHECKPOINT_ID = "s11_acc076942_l037380_r321289"


def emit(message: str, log_handle: TextIO) -> None:
    """Write one message to both the terminal and the experiment log."""
    print(message, flush=True)
    log_handle.write(f"{message}\n")
    log_handle.flush()


def run(command: list[str], log_handle: TextIO) -> None:
    """Execute one command while streaming combined output to the log."""
    emit(f"$ {shlex.join(command)}", log_handle)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess output")
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    """Run ET1 and OB1 while saving a timestamped, real-time log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    log_path = LOG_DIR / (
        f"run_provo_experiment_{started_at.strftime('%Y%m%d_%H%M%S')}.log"
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            emit(f"started_at={started_at.isoformat()}", log_handle)
            emit(f"log_path={log_path}", log_handle)
            emit(
                "configuration="
                f"workers:{OB1_WORKERS},seeds:0:100,n_trials:55,"
                f"sigma_left:{SIGMA_LEFT},sigma_right:{SIGMA_RIGHT}",
                log_handle,
            )
            if not VENV_PYTHON.is_file():
                raise FileNotFoundError(
                    "Run setup_provo_environment.py before this script."
                )
            run(
                [
                    str(VENV_PYTHON),
                    "-u",
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
                ],
                log_handle,
            )
            run(
                [
                    str(VENV_PYTHON),
                    "-u",
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
                ],
                log_handle,
            )
            emit(
                f"finished_at={datetime.now().astimezone().isoformat()}",
                log_handle,
            )
            emit("status=completed", log_handle)
        except BaseException:
            emit(traceback.format_exc().rstrip(), log_handle)
            emit("status=failed", log_handle)
            raise SystemExit(1) from None
    print(f"Experiment log saved to {log_path}", flush=True)


if __name__ == "__main__":
    main()
