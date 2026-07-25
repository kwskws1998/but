from types import SimpleNamespace

import pytest


pytest.importorskip("trl")

trainer_module = pytest.importorskip("trainers.reward_trainer_general")


class FakeModel:
    tokenizer = object()


class FakeTrainer:
    def __init__(self):
        self.train_called = False

    def train(self):
        self.train_called = True

    def evaluate(self):
        return {"eval_loss": 0.0}


def build_trainer_stub():
    return SimpleNamespace(
        use_quantization=False,
        use_lora=False,
        model_name="model",
        input_layer=None,
        freeze_layer=None,
        freeze=False,
        use_softprompt=True,
        concat=True,
        noise_factor=0.0,
        fp_dropout=[0.1, 0.3],
        fixations_model_version=1,
        load_fix_model=True,
        features_used=[1, 1, 1, 1, 1],
        init_sigma_left=1.0,
        init_sigma_right=1.0,
        sigma_learnable=True,
        use_asym_gaussian_redistributor=True,
        et2_gaze_concat_condition=None,
        dataset_name="OpenAssistant/oasst1",
        dataset_split="train",
        max_length=7000,
        max_tokens=1350,
        grid_search=False,
    )


def test_train_propagates_cli_input_limits(monkeypatch):
    trainer = build_trainer_stub()
    load_calls = []
    trainer.load_dataset = lambda **kwargs: load_calls.append(kwargs)
    trainer.set_trainer = lambda **kwargs: setattr(
        trainer,
        "trainer",
        FakeTrainer(),
    )
    monkeypatch.setattr(
        trainer_module,
        "model_init_func",
        lambda **kwargs: FakeModel(),
    )

    trainer_module.RewardTrainerConstructorGeneral.train_model(trainer)

    assert load_calls == [
        {
            "train_samples": 0,
            "split": "train",
            "eval_mode": False,
            "max_length": 7000,
            "max_tokens": 1350,
        }
    ]
    assert trainer.trainer.train_called


def test_eval_propagates_cli_input_limits(monkeypatch):
    trainer = build_trainer_stub()
    load_calls = []
    trainer.load_dataset = lambda **kwargs: load_calls.append(kwargs)
    trainer.set_trainer_eval = lambda: setattr(
        trainer,
        "trainer",
        FakeTrainer(),
    )
    monkeypatch.setattr(
        trainer_module,
        "model_init_func",
        lambda **kwargs: FakeModel(),
    )

    results = trainer_module.RewardTrainerConstructorGeneral.eval_model(
        trainer,
        folder_name="checkpoint",
    )

    assert load_calls == [
        {
            "split": "train",
            "eval_mode": True,
            "max_length": 7000,
            "max_tokens": 1350,
        }
    ]
    assert results == {"eval_loss": 0.0}
