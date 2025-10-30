"""Utils for model/optimizer initialization and training."""
import pprint

from torch import optim

from npt.utils.optim_utils import Lookahead, Lamb


def init_optimizer(config, model_parameters, device):
    if 'default' in config.training.optimizer:
        optimizer = optim.Adam(
            params=model_parameters,
            lr=config.training.lr)
    elif 'lamb' in config.training.optimizer:
        lamb = Lamb
        optimizer = lamb(
            model_parameters,
            lr=config.training.lr,
            betas=(0.9, 0.999),
            weight_decay=config.training.weight_decay,
            eps=1e-6
        )
    else:
        raise NotImplementedError

    if config.training.optimizer.startswith('lookahead_'):
        optimizer = Lookahead(
            optimizer,
            k=config.training.lookahead_update_cadence
        )

    return optimizer


def get_sorted_params(model):
    param_count_and_name = []
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count_and_name.append((p.numel(), n))

    pprint.pprint(sorted(param_count_and_name, reverse=True))

def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)