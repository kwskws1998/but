from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import threading
from typing import Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file
from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
from transformers import RobertaConfig, RobertaModel

from models.et2_checkpoint import (
    ET2_CHECKPOINT_FILENAME,
    ET2_REPO_ID,
    ET2_REVISION,
    default_et2_cache_dir,
    ensure_et2_checkpoint,
)
from models.fixations_aligner import FixationsAligner
from models.et2_tokenizer import ET2_TOKENIZER_SHA256, load_et2_tokenizer


ET2_FEATURE_NAMES = ("nFix", "FFD", "GPT", "TRT", "fixProp")
ET2_WINDOW_SIZE = 512
ET2_OVERLAP = 50
ET2_CACHE_SIGNATURE = (
    f"{ET2_REPO_ID}@{ET2_REVISION}:{ET2_CHECKPOINT_FILENAME}:"
    f"tokenizer_{ET2_TOKENIZER_SHA256}:prefix_space_v1:"
    "whitespace_word_boundary_v1:token_mapping_v2"
)


def create_roberta_base_config() -> RobertaConfig:
    """Return the exact architecture configuration used by roberta-base."""
    return RobertaConfig(
        attention_probs_dropout_prob=0.1,
        bos_token_id=0,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=1,
        type_vocab_size=1,
        vocab_size=50265,
    )


class RobertaRegressionModel(nn.Module):
    """Reproduce the five-output RoBERTa architecture published with ET2."""

    def __init__(self, config: Optional[RobertaConfig] = None):
        super().__init__()
        self.roberta = RobertaModel(config or create_roberta_base_config())
        self.decoder = nn.Linear(
            self.roberta.config.hidden_size,
            len(ET2_FEATURE_NAMES),
        )

    def forward(self, input_ids, attention_mask, predict_mask=None):
        """Return five ET feature predictions for each input token."""
        hidden_states = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        predictions = self.decoder(hidden_states)
        if predict_mask is not None:
            masked = predict_mask.eq(0).unsqueeze(-1).expand_as(predictions)
            predictions = predictions.masked_fill(masked, -1.0)
        return predictions


class FixationsPredictor_2:
    """Load the frozen Hugging Face ET2 model and produce token-level features."""

    def __init__(
        self,
        modelTokenizer=None,
        remap=True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.modelTokenizer = modelTokenizer
        self.remap = remap
        self.tokenizer_lock = threading.Lock()
        self.cache_signature = ET2_CACHE_SIGNATURE

        resolved_cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else default_et2_cache_dir()
        )
        self.fixTokenizer = load_et2_tokenizer(
            tokenizer_path=tokenizer_path,
            cache_dir=resolved_cache_dir,
        )
        if not self.fixTokenizer.is_fast:
            raise RuntimeError(
                "ET2 requires the fast tokenizer bundled in the Hub repo"
            )

        resolved_checkpoint = ensure_et2_checkpoint(
            checkpoint_path=checkpoint_path,
            cache_dir=resolved_cache_dir,
        )
        self.model = RobertaRegressionModel()
        state_dict = load_file(str(resolved_checkpoint), device="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _first_subtoken_mask(text_tokenized_fix, batch_index, valid_length):
        """Match the published RoBERTa word-boundary selection rule."""
        tokens = text_tokenized_fix.tokens(batch_index)[:valid_length]
        mask = torch.zeros(valid_length, dtype=torch.bool)
        current_word_started = False
        for token_index, token in enumerate(tokens):
            if token in ("<s>", "</s>", "<pad>"):
                current_word_started = False
                continue
            if token.startswith("Ġ") or not current_word_started:
                mask[token_index] = True
                current_word_started = True
        return mask

    def _predict_window(self, input_ids, attention_mask):
        """Run one ET2 window without retaining gradients."""
        with torch.inference_mode():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                predict_mask=attention_mask,
            ).squeeze(0)

    def _predict_with_sliding_window(self, input_ids, attention_mask):
        """Predict one unpadded sequence using the published overlap strategy."""
        sequence_length = input_ids.shape[1]
        if sequence_length <= ET2_WINDOW_SIZE:
            return self._predict_window(input_ids, attention_mask)

        predictions = torch.zeros(
            sequence_length,
            len(ET2_FEATURE_NAMES),
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.zeros(
            sequence_length,
            dtype=torch.float32,
            device=self.device,
        )
        stride = ET2_WINDOW_SIZE - ET2_OVERLAP
        start = 0

        while start < sequence_length:
            end = min(start + ET2_WINDOW_SIZE, sequence_length)
            window_predictions = self._predict_window(
                input_ids[:, start:end],
                attention_mask[:, start:end],
            )
            window_length = end - start
            window_weights = torch.ones(
                window_length,
                dtype=torch.float32,
                device=self.device,
            )
            if start > 0:
                ramp_length = min(ET2_OVERLAP, window_length)
                window_weights[:ramp_length] = torch.linspace(
                    0,
                    1,
                    ramp_length,
                    device=self.device,
                )
            if end < sequence_length:
                ramp_length = min(ET2_OVERLAP, window_length)
                window_weights[-ramp_length:] = torch.linspace(
                    1,
                    0,
                    ramp_length,
                    device=self.device,
                )

            predictions[start:end] += window_predictions * window_weights.unsqueeze(-1)
            weights[start:end] += window_weights
            if end == sequence_length:
                break
            start += stride

        nonzero = weights > 0
        predictions[nonzero] /= weights[nonzero].unsqueeze(-1)
        return predictions

    def _predict_token_features(self, text_tokenized_fix):
        """Predict nonnegative ET features on first word pieces for a batch."""
        input_ids = text_tokenized_fix["input_ids"]
        attention_mask = text_tokenized_fix["attention_mask"]
        batch_size, padded_length = input_ids.shape
        features = torch.zeros(
            batch_size,
            padded_length,
            len(ET2_FEATURE_NAMES),
            dtype=torch.float32,
            device=self.device,
        )

        for batch_index in range(batch_size):
            valid_length = int(attention_mask[batch_index].sum().item())
            sequence_ids = input_ids[
                batch_index:batch_index + 1, :valid_length
            ].to(self.device)
            sequence_attention = attention_mask[
                batch_index:batch_index + 1, :valid_length
            ].to(self.device)
            predictions = self._predict_with_sliding_window(
                sequence_ids,
                sequence_attention,
            ).clamp_min(0)
            first_subtoken_mask = self._first_subtoken_mask(
                text_tokenized_fix,
                batch_index,
                valid_length,
            ).to(self.device)
            features[batch_index, :valid_length][first_subtoken_mask] = predictions[
                first_subtoken_mask
            ]

        return features

    def _compute_mapped_fixations(
        self,
        input_ids_original=None,
        attention_mask_original=None,
        sentences=None,
    ):
        """Predict ET2 features and optionally remap them to model tokens."""
        if isinstance(input_ids_original, Mapping):
            if attention_mask_original is None:
                attention_mask_original = input_ids_original.get("attention_mask")
            input_ids_original = input_ids_original.get("input_ids")

        if sentences is None and input_ids_original is None:
            raise ValueError("sentences or input_ids_original must be provided")
        if input_ids_original is not None and self.modelTokenizer is None:
            raise ValueError(
                "modelTokenizer must be provided when input_ids_original is used"
            )
        if self.remap and self.modelTokenizer is None:
            raise ValueError("modelTokenizer must be provided when remap is True")
        if self.remap and input_ids_original is None:
            raise ValueError(
                "input_ids_original must be provided when remap is True"
            )

        if sentences is None:
            sentences = []
            pattern = r"(user|assistant)\r?\n"
            for batch_index, sequence in enumerate(input_ids_original):
                if attention_mask_original is not None:
                    sequence = sequence[attention_mask_original[batch_index].bool()]
                with self.tokenizer_lock:
                    sentence = self.modelTokenizer.decode(
                        sequence,
                        skip_special_tokens=True,
                    )
                sentences.append(re.sub(pattern, "", sentence).strip())
        elif isinstance(sentences, str):
            sentences = [sentences]

        prediction_sentences = [
            " ".join(sentence.strip().split())
            for sentence in sentences
        ]
        with self.tokenizer_lock:
            text_tokenized_model = None
            if self.remap:
                text_tokenized_model = self.modelTokenizer(
                    prediction_sentences,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            text_tokenized_fix = self.fixTokenizer(
                prediction_sentences,
                padding=True,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )

        fixations = self._predict_token_features(text_tokenized_fix)
        fixations_attention_mask = text_tokenized_fix["attention_mask"].to(self.device)
        if not self.remap:
            return (
                fixations,
                fixations_attention_mask,
                None,
                None,
                text_tokenized_fix,
                sentences,
            )

        tokens_id_mapped = TokenizerAligner().align_tokens(
            prediction_sentences,
            text_tokenized_model,
            text_tokenized_fix,
            return_all=False,
        )
        mapped_features = []
        for feature_index in range(len(ET2_FEATURE_NAMES)):
            feature_values = (
                fixations[:, :, feature_index].detach().cpu().tolist()
            )
            mapped = FixationsAligner.map_fixations_between_tokens_correct(
                feature_values,
                tokens_id_mapped,
                input_ids_original,
                text_tokenized_model,
                text_tokenized_fix,
                return_all=False,
            )
            mapped_features.append(
                torch.tensor(
                    mapped,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1)
            )
        mapped_fixations = torch.cat(mapped_features, dim=-1)

        return (
            fixations,
            None,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        )

    def forward(
        self,
        input_ids_original=None,
        attention_mask_original=None,
        sentences=None,
    ):
        """Return mapped or native ET2 features."""
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            _,
            _,
            _,
        ) = self._compute_mapped_fixations(
            input_ids_original=input_ids_original,
            attention_mask_original=attention_mask_original,
            sentences=sentences,
        )
        if self.remap:
            return mapped_fixations
        return fixations, fixations_attention_mask
