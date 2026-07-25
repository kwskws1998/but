import os
import sys
import threading
from datetime import datetime
import pathlib

sys.path.append("../..")

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import (
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel


from typing import (
    TypeVar,
)
from reward_model_base import MyRewardBase

T = TypeVar("T", bound="Module")
import re


def _adopt_pretrained_model_state(target, pretrained_model):
    """Transfer a loaded Hugging Face model into the dynamic wrapper without reinitializing it."""
    PreTrainedModel.__init__(target, pretrained_model.config)
    target.num_labels = pretrained_model.num_labels
    target.model = pretrained_model.model
    target.score = pretrained_model.score
    load_metadata_names = (
        "is_quantized",
        "quantization_method",
        "is_loaded_in_4bit",
        "is_loaded_in_8bit",
        "hf_quantizer",
        "hf_device_map",
        "_is_quantized_training_enabled",
    )
    for metadata_name in load_metadata_names:
        if metadata_name in pretrained_model.__dict__:
            setattr(
                target,
                metadata_name,
                pretrained_model.__dict__[metadata_name],
            )


def _rightmost_unmasked_indices(attention_mask, logits):
    """Return the rightmost valid token index for each row, supporting left or right padding."""
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be rank 2, received shape {tuple(attention_mask.shape)}"
        )
    if tuple(attention_mask.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            "attention_mask and token logits must have the same batch and sequence "
            f"dimensions, received {tuple(attention_mask.shape)} and "
            f"{tuple(logits.shape[:2])}"
        )
    valid_tokens = attention_mask.to(device=logits.device).ne(0)
    if not bool(valid_tokens.any(dim=-1).all()):
        raise ValueError("Each input row must contain at least one unmasked token")
    token_indices = torch.arange(logits.shape[1], device=logits.device)
    return torch.where(valid_tokens, token_indices, -1).max(dim=-1).values


def create_dynamic_class_RewardConcatenate(base_class=LlamaForSequenceClassification):
    class MyRewardConcatenate(base_class, MyRewardBase):
        def __init__(
            self,
            model_name,
            bnb_config=False,
            use_softprompt=True,
            load_local_folder_name=None,
            fixations_model_version=1,
            noise_factor=0.0,
            fp_dropout=[0.0, 0.3],
            load_fix_model=True,
            features_used=[1, 1, 1, 1, 1],
            init_sigma_left=1.0,
            init_sigma_right=1.0,
            use_asym_gaussian_redistributor=True,
            et2_gaze_concat_condition=None,
            *argv,
            **karg,
        ):
            print("loading model", model_name)
            start = datetime.now()
            use_quantization = bnb_config not in (None, False)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                quantization_config=bnb_config if use_quantization else None,
                device_map="auto",
                *argv,
                **karg,
            )
            if not isinstance(model, base_class):
                raise TypeError(
                    f"Loaded model type {type(model).__name__} is incompatible with "
                    f"{base_class.__name__}"
                )
            _adopt_pretrained_model_state(self, model)
            config = self.config
            MyRewardBase.__init__(
                self,
                model_name=model_name,
                features_used=features_used,
                init_sigma_left=init_sigma_left,
                init_sigma_right=init_sigma_right,
                use_asym_gaussian_redistributor=use_asym_gaussian_redistributor,
                et2_gaze_concat_condition=et2_gaze_concat_condition,
            )
            end = datetime.now()
            print("Total time loading model", end.timestamp() - start.timestamp())
            self.model_name = model_name
            self.noise_factor = noise_factor
            self.fp_dropout = fp_dropout
            self.use_softprompt = use_softprompt
            self.fixations_model_version = fixations_model_version
            self.use_quantization = use_quantization
            self.bnb_config = bnb_config
            self._load_tokenizer(load_local_folder_name)
            self.thread_local = threading.local()
            self.load_fix_model = load_fix_model

            if self.use_softprompt:
                if self.fixations_model_version == 1:
                    # TODO:integrate in pipeline the first version
                    # self.load_fx_model(
                    #     config.hidden_size, fp_dropout=self.fp_dropout
                    # )
                    self.load_fx_model_1(
                        config.hidden_size, fp_dropout=self.fp_dropout, remap=False
                    )
                elif self.fixations_model_version == 2:
                    self.load_fx_model_2(
                        config.hidden_size,
                        fp_dropout=self.fp_dropout,
                        remap=False,
                        load_fix_model=self.load_fix_model,
                    )
                else:
                    raise ValueError(
                        f"Fixations model version {self.fixations_model_version} not supported"
                    )
                self.gaze_boundary_embeddings = nn.Embedding(
                    2,
                    config.hidden_size,
                    device=self.model.embed_tokens.weight.device,
                )
                nn.init.normal_(
                    self.gaze_boundary_embeddings.weight,
                    mean=0.0,
                    std=config.initializer_range,
                )
            # we adjust the model embedding layer to the new changes in the tokenizer.
            self.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            if input_ids is None:
                raise ValueError(
                    "input_ids are required because gaze features are derived from token IDs"
                )
            if inputs_embeds is not None:
                raise ValueError(
                    "Pass input_ids only; this model constructs inputs_embeds internally"
                )
            embedding_device = self.model.embed_tokens.weight.device
            inputs_embeds = self.model.embed_tokens(input_ids.to(embedding_device))
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(embedding_device)
            if self.use_softprompt:
                # TODO: change code so the fixations use the chached code
                # mapped_fixations = self.forward_cached(input_ids)
                fixations_normalized, fixations_attention = self.compute_fixations(
                    input_ids,
                    attention_mask,
                    remap=False,
                    fixations_model_version=self.fixations_model_version,
                )
                # concat  fixations
                concat_tokens_embed = self.gaze_boundary_embeddings(
                    torch.arange(2, device=embedding_device)
                ).to(dtype=inputs_embeds.dtype)
                concat_tokens_embed_start = (
                    concat_tokens_embed[0]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(fixations_normalized.shape[0], -1, -1)
                )
                concat_tokens_embed_end = (
                    concat_tokens_embed[1]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(fixations_normalized.shape[0], -1, -1)
                )
                separator_attention_mask = (
                    torch.tensor([1])
                    .expand(fixations_normalized.shape[0], -1)
                    .to(embedding_device)
                )
                fixations_normalized = fixations_normalized.to(
                    device=embedding_device,
                    dtype=inputs_embeds.dtype,
                )
                fixations_attention = fixations_attention.to(embedding_device)

                inputs_embeds = torch.cat(
                    (
                        concat_tokens_embed_start,
                        fixations_normalized,
                        concat_tokens_embed_end,
                        inputs_embeds,
                    ),
                    dim=1,
                )
                # Delete unnecessary tensors to free up memory
                del (
                    concat_tokens_embed_start,
                    concat_tokens_embed_end,
                    fixations_normalized,
                )
                attention_mask = torch.cat(
                    (
                        separator_attention_mask,
                        fixations_attention,
                        separator_attention_mask,
                        attention_mask,
                    ),
                    dim=1,
                )
                # Free memory of unused tensors
                del separator_attention_mask, fixations_attention
            else:
                inputs_embeds = inputs_embeds.float()

            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )
            transformer_outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            token_logits = self.score(hidden_states)
            sequence_indices = _rightmost_unmasked_indices(attention_mask, token_logits)
            batch_indices = torch.arange(
                token_logits.shape[0], device=token_logits.device
            )
            pooled_logits = token_logits[batch_indices, sequence_indices]

            loss = None
            if labels is not None:
                labels = labels.to(token_logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and labels.dtype in (
                        torch.long,
                        torch.int,
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"
                if self.config.problem_type == "regression":
                    if self.num_labels == 1:
                        loss = nn.MSELoss()(pooled_logits.squeeze(), labels.squeeze())
                    else:
                        loss = nn.MSELoss()(pooled_logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss = nn.CrossEntropyLoss()(
                        pooled_logits.view(-1, self.num_labels),
                        labels.view(-1),
                    )
                elif self.config.problem_type == "multi_label_classification":
                    loss = nn.BCEWithLogitsLoss()(pooled_logits, labels)

            if not return_dict:
                output = (pooled_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            torch.cuda.empty_cache()
            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

    return MyRewardConcatenate
