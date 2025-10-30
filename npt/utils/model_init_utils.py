from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from npt.model.npt import NPTModel
from npt.utils.encode_utils import get_torch_dtype
from npt.utils.train_utils import count_parameters, init_optimizer


def init_model_opt_scaler(config, metadata, device=None):
    if device is None:
        device = config.system.device

    model = NPTModel(
        config,
        metadata=metadata,
        device=device
    )

    model_torch_dtype = get_torch_dtype(dtype_name=config.model.dtype)
    model = model.to(device=device).type(model_torch_dtype)
    
    print(f'Model has {count_parameters(model)} parameters,'
          f'batch size {config.training.batch_size}.')

    optimizer = init_optimizer(
        config=config, model_parameters=model.parameters(), device=device)
    print(f'Initialized "{config.training.optimizer}" optimizer.')

    # Automatic Mixed Precision (AMP)
    # If config.model.amp is False, the GradScaler call becomes a no-op
    # so we can switch between default/mixed precision without if/else
    # statements.
    scaler = GradScaler(enabled=config.model.amp)
    if config.model.amp:
        print(f'Initialized gradient scaler for Automatic Mixed Precision.')

    return model, optimizer, scaler


def setup_ddp_model(model, config, device):

    print(f'DDP with bucket size of {config.system.bucket_cap_mb} MB.')

    # If we are not using train augmentation, we must "find unused params"
    # to avoid synchronizing gradients on the features
    find_unused_params = (config.model.augmentation_bert_mask_prob['train'] == 0)

    if find_unused_params:
        print('Finding unused params in DDP.')

    # Wrap model
    model = DDP(
        model,
        device_ids=[device],
        find_unused_parameters=find_unused_params
    )

    return model