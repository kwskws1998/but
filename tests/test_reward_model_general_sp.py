import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "models" / "reward_model_general_sp.py"


class _Tokenizer:
    pad_token_id = 0

    def __len__(self):
        return 8


class _RewardBase:
    def __init__(self, *args, **kwargs):
        pass

    def _load_tokenizer(self, load_local_folder_name=None):
        self.tokenizer = _Tokenizer()

    def load_fx_model_2(self, *args, **kwargs):
        pass

    def compute_fixations(
        self,
        input_ids,
        attention_mask,
        remap=False,
        fixations_model_version=2,
    ):
        batch_size = input_ids.shape[0]
        fixations = torch.zeros(
            batch_size,
            2,
            self.config.hidden_size,
            device=input_ids.device,
        )
        fixation_mask = torch.tensor(
            [[1, 0], [1, 1]],
            device=input_ids.device,
        )[:batch_size]
        return fixations, fixation_mask


class _Config(PretrainedConfig):
    model_type = "tiny-sequence-classifier"

    def __init__(self):
        super().__init__(
            num_labels=1,
            pad_token_id=0,
            problem_type=None,
            return_dict=True,
        )
        self.hidden_size = 1
        self.initializer_range = 0.02


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 1, padding_idx=0)
        with torch.no_grad():
            self.embed_tokens.weight[:, 0].copy_(torch.arange(8))

    def resize_token_embeddings(self, size):
        if size != self.embed_tokens.num_embeddings:
            raise AssertionError("The test tokenizer must not resize embeddings")
        return self.embed_tokens

    def forward(self, inputs_embeds=None, return_dict=True, **kwargs):
        if return_dict:
            return BaseModelOutputWithPast(last_hidden_state=inputs_embeds)
        return (inputs_embeds,)


class _TinySequenceClassifier(PreTrainedModel):
    config_class = _Config
    init_calls = 0

    def __init__(self, config=None):
        super().__init__(config or _Config())
        type(self).init_calls += 1
        self.num_labels = self.config.num_labels
        self.model = _Backbone()
        self.score = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.score.weight.fill_(1.0)
        self.is_quantized = True
        self.is_loaded_in_4bit = True
        self.is_loaded_in_8bit = False
        self.quantization_method = "bitsandbytes"
        self.hf_quantizer = object()
        self.hf_device_map = {"model": 0}


def _load_module(monkeypatch):
    reward_base_stub = types.ModuleType("reward_model_base")
    reward_base_stub.MyRewardBase = _RewardBase
    module_name = "_reward_model_general_sp_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "reward_model_base", reward_base_stub)
        spec.loader.exec_module(module)
    return module


def _construct_wrapper(monkeypatch, use_softprompt=True):
    module = _load_module(monkeypatch)
    _TinySequenceClassifier.init_calls = 0
    loaded_model = _TinySequenceClassifier()
    loaded_model.forward = lambda *args, **kwargs: "source-forward"
    loaded_model._old_forward = lambda *args, **kwargs: "source-old-forward"
    captured = {}

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured.update(kwargs)
            return loaded_model

    monkeypatch.setattr(module, "AutoModelForSequenceClassification", _AutoModel)
    dynamic_class = module.create_dynamic_class_RewardConcatenate(
        _TinySequenceClassifier
    )
    quantization_config = object()
    wrapped_model = dynamic_class(
        model_name="tiny/model",
        bnb_config=quantization_config,
        use_softprompt=use_softprompt,
        fixations_model_version=2,
    )
    return module, wrapped_model, loaded_model, captured, quantization_config


def test_constructor_keeps_loaded_head_and_quantization_metadata(monkeypatch):
    _, wrapped, loaded, captured, quantization_config = _construct_wrapper(monkeypatch)

    assert _TinySequenceClassifier.init_calls == 1
    assert wrapped._modules is not loaded._modules
    assert wrapped._modules["model"] is loaded._modules["model"]
    assert wrapped._modules["score"] is loaded._modules["score"]
    assert wrapped.model is loaded.model
    assert wrapped.score is loaded.score
    assert "forward" not in wrapped.__dict__
    assert "_old_forward" not in wrapped.__dict__
    assert wrapped.is_quantized is True
    assert wrapped.is_loaded_in_4bit is True
    assert wrapped.is_loaded_in_8bit is False
    assert wrapped.quantization_method == "bitsandbytes"
    assert wrapped.hf_quantizer is loaded.hf_quantizer
    assert wrapped.hf_device_map == {"model": 0}
    assert captured["quantization_config"] is quantization_config
    assert captured["device_map"] == "auto"


def test_forward_pools_rightmost_unmasked_token_after_gaze_concat(monkeypatch):
    _, wrapped, _, _, _ = _construct_wrapper(monkeypatch)
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    labels = torch.tensor([2.0, 5.0])

    output = wrapped(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    torch.testing.assert_close(output.logits[:, 0], torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(output.loss, torch.tensor(0.0))


def test_rightmost_unmasked_indices_rejects_fully_masked_rows(monkeypatch):
    module = _load_module(monkeypatch)
    logits = torch.zeros(2, 3, 1)
    attention_mask = torch.tensor([[1, 0, 0], [0, 0, 0]])

    with pytest.raises(ValueError, match="at least one unmasked token"):
        module._rightmost_unmasked_indices(attention_mask, logits)


def test_adopted_tiny_llama_preserves_state_dict_config_and_save_reload(
    monkeypatch, tmp_path
):
    module = _load_module(monkeypatch)
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        num_labels=1,
        pad_token_id=0,
    )
    loaded_model = LlamaForSequenceClassification(config)
    with torch.no_grad():
        loaded_model.score.weight.copy_(
            torch.arange(8, dtype=loaded_model.score.weight.dtype).unsqueeze(0)
        )
    original_init = LlamaForSequenceClassification.__init__

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return loaded_model

    def _unexpected_second_init(self, config):
        raise AssertionError("The Llama wrapper must not be initialized twice")

    monkeypatch.setattr(module, "AutoModelForSequenceClassification", _AutoModel)
    monkeypatch.setattr(
        LlamaForSequenceClassification,
        "__init__",
        _unexpected_second_init,
    )
    dynamic_class = module.create_dynamic_class_RewardConcatenate(
        LlamaForSequenceClassification
    )
    wrapped_model = dynamic_class(
        model_name="tiny/llama",
        bnb_config=None,
        use_softprompt=False,
        fixations_model_version=2,
    )

    assert wrapped_model.config is loaded_model.config
    assert wrapped_model._modules is not loaded_model._modules
    assert set(wrapped_model.state_dict()) == set(loaded_model.state_dict())
    for name, loaded_tensor in loaded_model.state_dict().items():
        torch.testing.assert_close(
            wrapped_model.state_dict()[name],
            loaded_tensor,
        )

    wrapped_model.save_pretrained(tmp_path, safe_serialization=False)
    monkeypatch.setattr(
        LlamaForSequenceClassification,
        "__init__",
        original_init,
    )
    reloaded_model = LlamaForSequenceClassification.from_pretrained(tmp_path)

    assert reloaded_model.config.hidden_size == config.hidden_size
    assert reloaded_model.config.num_hidden_layers == config.num_hidden_layers
    torch.testing.assert_close(
        reloaded_model.score.weight,
        wrapped_model.score.weight,
    )
