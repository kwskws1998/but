import pytest

from utils.training_schedule import optimizer_steps_for_trainer


def test_optimizer_steps_include_gradient_accumulation():
    assert (
        optimizer_steps_for_trainer(
            num_samples=160,
            batch_size=1,
            gradient_accumulation_steps=8,
            train_epochs=2,
        )
        == 40
    )


def test_optimizer_steps_match_trainer_floor_per_epoch():
    assert (
        optimizer_steps_for_trainer(
            num_samples=17,
            batch_size=1,
            gradient_accumulation_steps=8,
            train_epochs=2,
        )
        == 4
    )


def test_optimizer_steps_keep_one_update_for_short_epoch():
    assert (
        optimizer_steps_for_trainer(
            num_samples=3,
            batch_size=1,
            gradient_accumulation_steps=8,
            train_epochs=2,
        )
        == 2
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_samples", 0),
        ("batch_size", 0),
        ("gradient_accumulation_steps", 0),
        ("train_epochs", 0),
    ),
)
def test_optimizer_steps_reject_non_positive_values(field, value):
    kwargs = {
        "num_samples": 16,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "train_epochs": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        optimizer_steps_for_trainer(**kwargs)
