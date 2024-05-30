from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR


def _get_linear_nonzero_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, 
                                                       num_training_steps: int, lr_decrease_ratio: float):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float((1 - lr_decrease_ratio) * (num_training_steps - current_step) / \
                         (num_training_steps - num_warmup_steps) + lr_decrease_ratio))


def get_linear_nonzero_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_decrease_ratio=0., last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_nonzero_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_decrease_ratio=lr_decrease_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
