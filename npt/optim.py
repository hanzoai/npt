"""Learning rate scheduler.
Forked from https://github.com/OATML/non-parametric-transformers."""
import warnings
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR)
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup)


def clip_gradient(model, clip: float):
    nn.utils.clip_grad_norm_(model.parameters(), clip)


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    From Over9000
    https://github.com/mgrankin/over9000/blob/master/train.py
    """
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps,
                 pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        self.curr_epoch = 0
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.curr_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        self.curr_epoch += 1
        super().step()

    def get_lr(self):
        if self.curr_epoch <= self.step_start:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()

class LRScheduler:
    def __init__(self, config, name, optimizer):
        self.config = config
        self.name = name
        self.optimizer = optimizer
        self.num_steps = 0

        self.construct_auto_scheduler()

        print(f'Initialized "{name}" learning rate scheduler.')

    def construct_auto_scheduler(self):
        total_steps = self.config.training.num_total_steps

        if self.config.training.optimizer_warmup_proportion >= 0:
            num_warmup_steps = (
                    total_steps * self.config.training.optimizer_warmup_proportion
                    )
        else:
            num_warmup_steps = self.config.training.optimizer_warmup_fixed_n_steps

        print(f'Warming up for {num_warmup_steps}/{total_steps} steps.')

        if self.name == 'constant':
            self.scheduler = get_constant_schedule(optimizer=self.optimizer)
        elif self.name == 'linear_warmup':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps)
        elif self.name == 'cosine_cyclic':
            warnings.warn(
                "Depreciated. Falling back to linear_warmup.",
                DeprecationWarning
            )
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps)
        elif self.name == 'polynomial_decay_warmup':
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=1e-7,
                power=1.0)
        elif self.name == 'flat_and_anneal':
            def d(x):
                return 1

            assert self.config.training.optimizer_warmup_proportion >= 0

            # We use exp_optimizer_warmup_proportion to denote the
            # flat LR regime, prior to annealing
            dummy = LambdaLR(self.optimizer, d)
            cosine = CosineAnnealingLR(
                self.optimizer, int(total_steps * (
                    1 - self.config.training.optimizer_warmup_proportion)))
            self.scheduler = ConcatLR(
                self.optimizer, dummy, cosine, total_steps,
                self.config.training.optimizer_warmup_proportion)
        else:
            raise NotImplementedError

    def step(self):
        self.num_steps += 1
        c_lr = self.config.training.lr
        num = self.num_steps
        tot = self.config.training.num_total_steps

        if self.name == 'cosine_cyclic':
            self.scheduler.step_update(num_updates=num)
        else:
            self.scheduler.step()