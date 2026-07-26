"""Run frozen ET1 and apply fixed learned or symmetric redistribution."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.asym_gaussian_redistributor import AsymGaussianRedistributor


def passage_word_spans(text: str) -> list[dict]:
    """Return whitespace-token word IDs and exact character spans."""
    return [
        {
            "word_id_zero_based": word_id,
            "word_raw": match.group(),
            "character_start": match.start(),
            "character_end": match.end(),
        }
        for word_id, match in enumerate(re.finditer(r"\S+", text))
    ]


def assign_token_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
) -> tuple[list[int | None], list[dict]]:
    """Assign each non-special ET1 token to exactly one passage word."""
    spans = passage_word_spans(text)
    assignments: list[int | None] = []
    for token_index, ((start, end), is_special) in enumerate(
        zip(offsets, special_tokens_mask)
    ):
        if is_special:
            assignments.append(None)
            continue
        if start >= end:
            raise ValueError(
                f"Non-special ET1 token {token_index} has empty offset {(start, end)}"
            )
        overlapping = [
            span
            for span in spans
            if min(end, span["character_end"])
            > max(start, span["character_start"])
        ]
        if len(overlapping) != 1:
            raise ValueError(
                f"ET1 token {token_index} offset {(start, end)} overlaps "
                f"{len(overlapping)} passage words"
            )
        assignments.append(overlapping[0]["word_id_zero_based"])

    assigned_words = {item for item in assignments if item is not None}
    expected_words = {span["word_id_zero_based"] for span in spans}
    if assigned_words != expected_words:
        raise ValueError(
            "ET1 offset coverage does not match passage words: "
            f"missing {sorted(expected_words - assigned_words)}, "
            f"extra {sorted(assigned_words - expected_words)}"
        )
    return assignments, spans


def redistributor_from_log_sigmas(
    log_sigma_left: float,
    log_sigma_right: float,
    min_sigma: float,
) -> AsymGaussianRedistributor:
    """Construct the production redistributor with fixed checkpoint values."""
    module = AsymGaussianRedistributor(
        init_sigma_left=1.0,
        init_sigma_right=1.0,
        min_sigma=min_sigma,
    )
    with torch.no_grad():
        module.log_sigma_left.fill_(log_sigma_left)
        module.log_sigma_right.fill_(log_sigma_right)
    module.log_sigma_left.requires_grad_(False)
    module.log_sigma_right.requires_grad_(False)
    module.eval()
    return module


def redistribute_values(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    sigma_record: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply learned asymmetric and width-matched symmetric redistribution."""
    symmetric, asymmetric = redistributors_from_sigma_record(
        sigma_record,
        values.device,
    )
    return apply_redistributors(values, attention_mask, symmetric, asymmetric)


def redistributors_from_sigma_record(
    sigma_record: dict,
    device: torch.device,
) -> tuple[AsymGaussianRedistributor, AsymGaussianRedistributor]:
    """Build one fixed symmetric/asymmetric module pair for a checkpoint."""
    asymmetric = redistributor_from_log_sigmas(
        sigma_record["log_sigma_left"],
        sigma_record["log_sigma_right"],
        sigma_record["min_sigma"],
    ).to(device)
    symmetric_raw_sigma = (
        sigma_record["sigma_symmetric"] - sigma_record["min_sigma"]
    )
    if symmetric_raw_sigma <= 0:
        raise ValueError("Symmetric sigma must exceed min_sigma")
    symmetric_log_sigma = math.log(symmetric_raw_sigma)
    symmetric = redistributor_from_log_sigmas(
        symmetric_log_sigma,
        symmetric_log_sigma,
        sigma_record["min_sigma"],
    ).to(device)
    return symmetric, asymmetric


def apply_redistributors(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    symmetric: AsymGaussianRedistributor,
    asymmetric: AsymGaussianRedistributor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an already constructed fixed redistributor pair."""
    with torch.inference_mode():
        asymmetric_values = asymmetric(values, attention_mask)
        symmetric_values = symmetric(values, attention_mask)
    return symmetric_values, asymmetric_values


def validate_mass_conservation(
    before: torch.Tensor,
    after: torch.Tensor,
    attention_mask: torch.Tensor,
    tolerance: float = 1e-5,
) -> float:
    """Return mass difference and fail when valid-token mass is not conserved."""
    mask = attention_mask.to(before.dtype)
    before_mass = float((before * mask).sum().item())
    after_mass = float((after * mask).sum().item())
    difference = after_mass - before_mass
    scale = max(1.0, abs(before_mass))
    if abs(difference) > tolerance * scale:
        raise ValueError(
            f"Redistribution mass changed from {before_mass} to {after_mass}"
        )
    return difference


class ET1NativePredictor:
    """Load one frozen ET1 predictor and expose native T5 offsets."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        tokenizer_path: Path | None = None,
        tokenizer_cache_dir: Path | None = None,
    ):
        from models.fixations_predictor_model_1 import FixationsPredictor_1

        self.predictor = FixationsPredictor_1(
            hidden_dim=128,
            drop_out=0.2,
            modelTokenizer=None,
            remap=False,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )

    def predict(self, text: str) -> dict:
        """Predict one passage and return checked native-token metadata."""
        tokenizer = self.predictor.fixTokenizer
        encoded = tokenizer(
            [text],
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        values, attention_mask = self.predictor.forward(sentences=[text])
        values = values.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        input_ids = encoded["input_ids"].detach().cpu()
        encoded_attention = encoded["attention_mask"].detach().cpu()
        if not torch.equal(attention_mask, encoded_attention):
            raise ValueError("ET1 prediction and offset attention masks differ")
        if values.shape != attention_mask.shape:
            raise ValueError(
                f"ET1 output shape {values.shape} does not match "
                f"attention shape {attention_mask.shape}"
            )
        if input_ids.shape != values.shape:
            raise ValueError(
                f"ET1 token shape {input_ids.shape} does not match "
                f"prediction shape {values.shape}"
            )
        offsets = [
            tuple(map(int, item))
            for item in encoded["offset_mapping"][0].tolist()
        ]
        special_tokens_mask = [
            int(item) for item in encoded["special_tokens_mask"][0].tolist()
        ]
        assignments, spans = assign_token_offsets(
            text,
            offsets,
            special_tokens_mask,
        )
        return {
            "values": values,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "offsets": offsets,
            "special_tokens_mask": special_tokens_mask,
            "word_assignments": assignments,
            "word_spans": spans,
            "device": str(self.predictor.device),
            "cache_signature": self.predictor.cache_signature,
        }


def aggregate_tokens_to_words(
    token_values: np.ndarray,
    assignments: list[int | None],
    expected_word_count: int,
) -> dict[int, float]:
    """Sum native ET1 token values into every whitespace-token word."""
    totals = {word_id: 0.0 for word_id in range(expected_word_count)}
    coverage = {word_id: 0 for word_id in range(expected_word_count)}
    for value, word_id in zip(token_values, assignments):
        if word_id is None:
            continue
        totals[word_id] += float(value)
        coverage[word_id] += 1
    missing = [word_id for word_id, count in coverage.items() if count == 0]
    if missing:
        raise ValueError(f"Passage words without ET1 tokens: {missing}")
    return totals


def run_et1_inference(
    passages: pd.DataFrame,
    canonical_words: pd.DataFrame,
    sigma_records: list[dict],
    predictor: ET1NativePredictor,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate raw and checkpoint-specific redistributed ET1 tables."""
    token_rows = []
    word_rows = []
    mass_rows = []
    effective_records = sigma_records or [
        {
            "checkpoint_id": "raw_only",
            "checkpoint": None,
            "sigma_left": None,
            "sigma_right": None,
            "sigma_symmetric": None,
        }
    ]
    redistributors = {
        record["checkpoint_id"]: redistributors_from_sigma_record(
            record,
            torch.device("cpu"),
        )
        for record in effective_records
        if record["checkpoint"] is not None
    }
    canonical_by_passage = {
        int(passage_id): group.sort_values("word_id_zero_based")
        for passage_id, group in canonical_words.groupby(
            "passage_id_zero_based",
            sort=True,
        )
    }

    for passage in passages.sort_values("passage_id_zero_based").itertuples():
        predicted = predictor.predict(passage.passage_text)
        raw = predicted["values"]
        mask = predicted["attention_mask"]
        assignments = predicted["word_assignments"]
        word_count = len(predicted["word_spans"])
        raw_numpy = raw[0].numpy()
        raw_word_totals = aggregate_tokens_to_words(
            raw_numpy,
            assignments,
            word_count,
        )

        for sigma_record in effective_records:
            checkpoint_id = sigma_record["checkpoint_id"]
            if sigma_record["checkpoint"] is None:
                symmetric = asymmetric = None
                symmetric_word_totals = asymmetric_word_totals = {}
            else:
                symmetric, asymmetric = apply_redistributors(
                    raw,
                    mask,
                    *redistributors[checkpoint_id],
                )
                symmetric_difference = validate_mass_conservation(
                    raw,
                    symmetric,
                    mask,
                )
                asymmetric_difference = validate_mass_conservation(
                    raw,
                    asymmetric,
                    mask,
                )
                symmetric_word_totals = aggregate_tokens_to_words(
                    symmetric[0].numpy(),
                    assignments,
                    word_count,
                )
                asymmetric_word_totals = aggregate_tokens_to_words(
                    asymmetric[0].numpy(),
                    assignments,
                    word_count,
                )
                mass_rows.extend(
                    [
                        {
                            "checkpoint_id": checkpoint_id,
                            "passage_id_zero_based": (
                                passage.passage_id_zero_based
                            ),
                            "condition": "et1_symmetric",
                            "valid_token_mass_difference": symmetric_difference,
                        },
                        {
                            "checkpoint_id": checkpoint_id,
                            "passage_id_zero_based": (
                                passage.passage_id_zero_based
                            ),
                            "condition": "et1_asymmetric",
                            "valid_token_mass_difference": asymmetric_difference,
                        },
                    ]
                )

            for token_index, (
                token_id,
                token,
                offset,
                is_special,
                attention,
                word_id,
                raw_value,
            ) in enumerate(
                zip(
                    predicted["input_ids"][0].tolist(),
                    predicted["tokens"],
                    predicted["offsets"],
                    predicted["special_tokens_mask"],
                    mask[0].tolist(),
                    assignments,
                    raw[0].tolist(),
                )
            ):
                token_rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_zero_based": passage.passage_id_zero_based,
                        "token_index": token_index,
                        "token_id": token_id,
                        "token": token,
                        "character_start": offset[0],
                        "character_end": offset[1],
                        "is_special": is_special,
                        "attention_mask": int(attention),
                        "word_id_zero_based": word_id,
                        "et1_raw_token_trt": raw_value,
                        "et1_symmetric_token_trt": (
                            float(symmetric[0, token_index])
                            if symmetric is not None
                            else np.nan
                        ),
                        "et1_asymmetric_token_trt": (
                            float(asymmetric[0, token_index])
                            if asymmetric is not None
                            else np.nan
                        ),
                    }
                )

            passage_words = canonical_by_passage[
                int(passage.passage_id_zero_based)
            ]
            for word in passage_words.itertuples():
                word_id = int(word.word_id_zero_based)
                word_rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_raw": int(word.passage_id_raw),
                        "passage_id_zero_based": int(
                            word.passage_id_zero_based
                        ),
                        "word_id_zero_based": word_id,
                        "word_raw": word.word_raw,
                        "et1_raw_word_trt": raw_word_totals[word_id],
                        "et1_symmetric_word_trt": (
                            symmetric_word_totals[word_id]
                            if symmetric is not None
                            else np.nan
                        ),
                        "et1_asymmetric_word_trt": (
                            asymmetric_word_totals[word_id]
                            if asymmetric is not None
                            else np.nan
                        ),
                    }
                )

    audit = {
        "passages": int(passages["passage_id_zero_based"].nunique()),
        "canonical_words": len(canonical_words),
        "checkpoint_ids": [
            str(record["checkpoint_id"]) for record in effective_records
        ],
        "token_rows": len(token_rows),
        "word_rows": len(word_rows),
        "mass_checks": len(mass_rows),
        "predictor_device": predictor.predictor.device.type,
        "predictor_cache_signature": predictor.predictor.cache_signature,
    }
    return (
        pd.DataFrame(token_rows),
        pd.DataFrame(word_rows),
        pd.DataFrame(mass_rows),
        audit,
    )


def write_et1_outputs(
    output_dir: Path,
    token_frame: pd.DataFrame,
    word_frame: pd.DataFrame,
    mass_frame: pd.DataFrame,
    audit: dict,
) -> None:
    """Write ET1 token, word, mass, and provenance outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token_frame.to_csv(output_dir / "et1_token_values.csv", index=False)
    word_frame.to_csv(output_dir / "et1_word_values.csv", index=False)
    mass_frame.to_csv(output_dir / "et1_mass_audit.csv", index=False)
    with (output_dir / "et1_inference_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
