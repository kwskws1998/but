import math


DEFAULT_WARMUP_RATIO = 0.02


def optimizer_steps_for_trainer(
    num_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    train_epochs: float,
) -> int:
    """Match the optimizer-step calculation used by Transformers Trainer 4.40."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if train_epochs <= 0:
        raise ValueError("train_epochs must be positive")

    batches_per_epoch = math.ceil(num_samples / batch_size)
    updates_per_epoch = max(
        batches_per_epoch // gradient_accumulation_steps,
        1,
    )
    return math.ceil(train_epochs * updates_per_epoch)
