"""
NPT-AD: Non-Parametric Transformers for Anomaly Detection

A clean, user-friendly implementation of Non-Parametric Transformers 
for anomaly detection on tabular data.
"""

__version__ = "1.0.0"
__author__ = "Hugo Thimonier"
__email__ = "thimonier.hugo@gmail.com"

# Import main classes for easy access
from .config_manager import NPTADConfig
from .configs import *
from .batch_dataset import *
from .cli import *
from .column_encoding_dataset import *
from .loss import *
from .mask import *
from .optim import *
from .simple_trainer import *
from .train import *
from .utils import *
from .model import *

# Conditional imports for torch-dependent modules
try:
    from .simple_trainer import SimpleTrainer, train_anomaly_detector, quick_test
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Dataset registry
from .datasets.dataset_registry import DatasetRegistry, create_simple_dataset

__all__ = [
    'NPTADConfig',
    'DatasetRegistry', 
    'create_simple_dataset',
    'TORCH_AVAILABLE',
]

# Add torch-dependent classes if available
if TORCH_AVAILABLE:
    __all__.extend(['SimpleTrainer', 'train_anomaly_detector', 'quick_test'])

