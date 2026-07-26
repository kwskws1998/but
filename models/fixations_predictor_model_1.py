from collections.abc import Mapping
from pathlib import Path
import re
import threading
from typing import Optional, Union

import torch
import torch.nn as nn

from models.et_checkpoint import ensure_et1_checkpoint
from models.et1_tokenizer import (
    ET1_TOKENIZER_SIGNATURE,
    load_et1_tokenizer,
)
from models.tokenizer_fingerprint import tokenizer_fingerprint


class BiLSTMRegression(nn.Module):
    """Predict one TRT value per T5 token."""

    def __init__(self, embedding, hidden_dim, drop_out):
        super().__init__()
        self.emb = embedding
        self.emb.requires_grad_(False)
        self.lstm = nn.LSTM(
            input_size=self.emb.weight.size(1),
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=drop_out,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(drop_out)

    def forward(self, input_ids):
        """Return token-level TRT predictions."""
        hidden_states = self.dropout(self.emb(input_ids))
        hidden_states, _ = self.lstm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.head(hidden_states).squeeze(-1)


class FixationsPredictor_1:
    """Load the frozen ET1 model and predict or remap token-level TRT."""

    def __init__(
        self,
        hidden_dim,
        drop_out=0.2,
        modelTokenizer=None,
        remap=True,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tokenizer_cache_dir: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
    ):
        embedding = nn.Embedding(32128, 512)
        self.model = BiLSTMRegression(embedding, hidden_dim, drop_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.modelTokenizer = modelTokenizer
        self.tokenizer_lock = threading.Lock()
        self.remap = remap

        self.fixTokenizer = load_et1_tokenizer(
            tokenizer_path=tokenizer_path,
            cache_dir=tokenizer_cache_dir,
        )
        self.cache_signature = (
            "huangxt39/SelectiveCacheForLM@"
            "eccc93f969745b04ce1e4911d6513d85565cc919:"
            "T5-tokenizer-BiLSTM-TRT-12-concat-3:"
            f"tokenizer_{ET1_TOKENIZER_SIGNATURE}:"
            f"fingerprint_{tokenizer_fingerprint(self.fixTokenizer)}"
        )

        resolved_checkpoint = ensure_et1_checkpoint(checkpoint_path)
        state_dict = torch.load(
            resolved_checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _compute_mapped_fixations(
        self,
        input_ids_original=None,
        sentences=None,
        return_all=False,
    ):
        """Predict TRT and optionally remap it to the reward-model tokenizer."""
        if isinstance(input_ids_original, Mapping):
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
            for sequence in input_ids_original:
                with self.tokenizer_lock:
                    sentence = self.modelTokenizer.decode(
                        sequence,
                        skip_special_tokens=True,
                    )
                sentences.append(re.sub(pattern, "", sentence).strip())
        elif isinstance(sentences, str):
            sentences = [sentences]

        with self.tokenizer_lock:
            text_tokenized_model = None
            if self.remap:
                text_tokenized_model = self.modelTokenizer(
                    sentences,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
            text_tokenized_fix = self.fixTokenizer(
                sentences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

        with torch.inference_mode():
            fixations = self.model(
                text_tokenized_fix["input_ids"].to(self.device)
            )

        if not self.remap:
            return (
                fixations,
                text_tokenized_fix["attention_mask"].to(self.device),
                None,
                None,
                text_tokenized_fix,
                sentences,
            )

        from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner
        from models.fixations_aligner import FixationsAligner

        fixations_list = fixations.detach().cpu().tolist()
        tokens_id_mapped = TokenizerAligner().align_tokens(
            sentences,
            text_tokenized_model,
            text_tokenized_fix,
            return_all=False,
        )
        mapped_fixations = FixationsAligner.map_fixations_between_tokens_correct(
            fixations_list,
            tokens_id_mapped,
            input_ids_original,
            text_tokenized_model,
            text_tokenized_fix,
            return_all=False,
        )
        mapped_fixations_tensor = torch.tensor(
            mapped_fixations,
            dtype=torch.float32,
            device=self.device,
        )

        if not return_all:
            return (
                fixations_list,
                None,
                mapped_fixations_tensor,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            )

        (
            tokens_idx_mapped,
            tokens_id_mapped,
            words_str_mapped,
            tokens_str_mapped,
        ) = TokenizerAligner().align_tokens(
            sentences,
            text_tokenized_model,
            text_tokenized_fix,
            return_all=True,
        )
        mapped_fixations_corrected, mapped_fixations = (
            FixationsAligner.map_fixations_between_tokens_correct(
                fixations_list,
                tokens_id_mapped,
                input_ids_original,
                text_tokenized_model,
                text_tokenized_fix,
                return_all=True,
            )
        )
        return (
            fixations_list,
            None,
            mapped_fixations_corrected,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
            tokens_idx_mapped,
            tokens_id_mapped,
            words_str_mapped,
            tokens_str_mapped,
        )

    def forward(self, input_ids_original=None, sentences=None):
        """Return mapped TRT or native T5-token TRT and its attention mask."""
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            _,
            _,
            _,
        ) = self._compute_mapped_fixations(
            input_ids_original,
            sentences=sentences,
        )
        if self.remap:
            return mapped_fixations
        return fixations, fixations_attention_mask
