import os, ast
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Model architecture
    dim_hidden: int = 64
    num_heads: int = 8
    stacking_depth: int = 8
    rff_depth: int = 1

    # Embeddings
    feature_type_embedding: bool = True
    feature_index_embedding: bool = True

    # Attention
    sep_res_embed: bool = True
    mix_heads: bool = True
    att_score_norm: str = 'softmax'

    # Normalization
    embedding_layer_norm: bool = False
    att_block_layer_norm: bool = True
    pre_layer_norm: bool = True
    layer_norm_eps: float = 1e-12

    # Dropout
    hidden_dropout_prob: float = 0.1
    att_score_dropout_prob: float = 0.1
    
    # BERT augmentation
    bert_augmentation: bool = True
    bert_mask_percentage: float = 0.9
    augmentation_bert_mask_prob: Dict[str, float] = field(default_factory=lambda: {
        'train': 0.15, 'val': 0.15, 'test': 0.0
    })

    # Weight init
    init_weights: bool = False
    init_type: str = 'xavier'
    init_params: Any = None

    # Other
    amp: bool = False
    dtype: str = 'float32'
    is_semi_supervised: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Basic training
    batch_size: int = -1  # -1 means full batch
    val_batchsize: int = 1
    num_train_inference: int = 0
    num_total_steps: int = 100000
    lr: float = 1e-3
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0

    # Optimization
    optimizer: str = 'lookahead_lamb'
    scheduler: str = 'flat_and_anneal'
    lookahead_update_cadence: int = 6
    optimizer_warmup_proportion: float = 0.7
    optimizer_warmup_fixed_n_steps: int = 10000
    minibatch_sgd: bool = True

    # Evaluation
    eval_every_n: int = 5
    eval_every_epoch_or_steps: str = 'epochs'
    eval_test_at_end_only: bool = False
    mix_rows_inference: bool = False

    # Checkpointing
    checkpoint_setting: str = 'best_model'
    cache_cadence: int = 1
    checkpoint_save: int = 100000
    load_from_checkpoint: bool = False

    # Early stopping
    patience: int = -1

    # Seeds
    np_seed: int = 42
    torch_seed: int = 42
    original_np_seed: int = 91
    original_torch_seed: int = 91
    n_runs: int = 1

@dataclass
class DataConfig:
    """Data configuration parameters."""
    # Paths
    data_path: str = './data'
    name: str = 'abalone'

    # Data loading
    data_loader_nprocs: int = 0
    dataset_on_cuda: bool = False
    data_force_reload: bool = False
    data_log_mem_usage: bool = False
    clear_tmp_files: bool = False
    data_dtype: str = 'float32'

    # Splits
    val_perc: float = 0.1
    test_perc: float = 0.2

    # Features
    keep_categorical_features: bool = True
    cat_as_num_features: bool = True

    # Anomaly detection specific
    num_reconstruction: int = 15
    deterministic_masks: bool = False
    n_hidden_features: List[int] = field(default_factory=lambda: [1])
    max_n_recon: int = 0
    aggregation: str = 'sum'
    normalize_ad_loss: bool = False
    anomalies_in_inference: bool = False
    full_trainset_inference: bool = None

    # Contamination
    contamination_share_train: float = 0.0
    share_contamination: float = 0.0

    def __post_init__(self):
        if self.data_path == './data':
            self.data_path = os.path.join(self.data_path, self.name)
        elif self.data_path is None:
            print(
                'Data path was not provided, setting default values:\n'
                f'./data/{self.name}')
            self.data_path = os.path.join('./data/', self.name)

@dataclass
class SystemConfig:
    """System configuration parameters."""
    # Device
    use_cuda: bool = True
    device: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    nodes: int = 1
    gpus: int = 1
    nr: int = 0
    no_sync: int = -1
    bucket_cap_mb: int = 25
    
    # Logging
    verbose: bool = False
    print_every_nth_forward: int = False
    
    # Other
    exp_name: Optional[str] = None
    checkpoint_key: str = f'Job_{datetime.now().strftime("%Y%m%d-%H%M%S")}'


@dataclass
class NPTADConfig:
    """Main configuration class for NPT-AD."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Anomaly detection flag
    ad: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_paths()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate model parameters
        if self.model.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.model.dim_hidden % self.model.num_heads != 0:
            raise ValueError("dim_hidden must be divisible by num_heads")

        # Validate training parameters
        if self.training.lr <= 0:
            raise ValueError("learning rate must be positive")
        if self.training.batch_size == 0:
            raise ValueError("batch_size cannot be 0")

        # Validate data parameters
        if not os.path.exists(self.data.data_path):
            print(f"Warning: Data path {self.data.data_path} does not exist")

        # Validate system parameters
        if self.system.gpus <= 0:
            raise ValueError("Number of GPUs must be positive")

    def _setup_paths(self):
        """Setup and create necessary directories."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data.data_path, exist_ok=True)

        # Create results directory
        results_dir = Path("results") / self.data.name
        os.makedirs(results_dir, exist_ok=True)
        self.res_dir = results_dir

        # Create logs directory
        logs_dir = Path("logs") / self.data.name
        os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'NPTADConfig':
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            system=system_config,
            ad=config_dict.get('ad', True)
        )

    @classmethod
    def from_json(cls, json_path: str, dataset: str = None) -> 'NPTADConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        if dataset is not None:
            config_dict['data']['name'] = dataset
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__,
            'ad': self.ad
        }

    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_preset(self, preset_name: str, dataset: str) -> 'NPTADConfig':
        """Get a preset configuration for common use cases."""
        presets = {
            'quick_test': NPTADConfig._get_quick_test_preset(),
            'small_dataset': NPTADConfig._get_small_dataset_preset(
                dataset=dataset),
            'small_dataset_high_d': NPTADConfig._get_small_dataset_high_d_preset(
                dataset=dataset),
            'medium_dataset': NPTADConfig._get_medium_dataset_preset(
                dataset=dataset),
            'medium_dataset_high_d': NPTADConfig._get_medium_dataset_high_d_preset(
                dataset=dataset),
            'large_dataset': NPTADConfig._get_large_dataset_preset(
                dataset=dataset),
            'large_dataset_high_d': NPTADConfig._get_large_dataset_high_d_preset(
                dataset=dataset),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. "
                             f"Available: {list(presets.keys())}")

        return presets[preset_name]

    @classmethod
    def _get_quick_test_preset(cls) -> 'NPTADConfig':
        """Quick test configuration for debugging."""
        config = NPTADConfig()
        config.training.num_total_steps = 1000
        config.training.eval_every_n = 100
        config.model.dim_hidden = 16
        config.model.stacking_depth = 2
        config.model.num_heads = 2
        config.system.verbose = True
        return config

    @classmethod
    def _get_small_dataset_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_small_dataset.json"
        config = cls.from_json(config_dict, dataset)
        return config

    @classmethod
    def _get_small_dataset_high_d_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_small_dataset_high_nb_features.json"
        config = cls.from_json(config_dict, dataset)
        return config

    @classmethod
    def _get_medium_dataset_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_medium_dataset.json"
        config = cls.from_json(config_dict, dataset)
        return config

    @classmethod
    def _get_medium_dataset_high_d_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_medium_dataset_high_nb_features.json"
        config = cls.from_json(config_dict, dataset)
        return config

    @classmethod
    def _get_large_dataset_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_large_dataset.json"
        config = cls.from_json(config_dict, dataset)
        return config

    @classmethod
    def _get_large_dataset_high_d_preset(cls, dataset: str) -> 'NPTADConfig':
        """Configuration optimized for small datasets."""
        config_dict = "./config/base/base_large_dataset_high_nb_features.json"
        config = cls.from_json(config_dict, dataset)
        return config


def create_config_from_args(args) -> NPTADConfig:
    """Create configuration from command line arguments (backward compatibility)."""
    config = NPTADConfig()

    # --- Top-level ---
    if hasattr(args, 'ad'):
        config.ad = args.ad

    # --- DataConfig Mappings ---
    data_mappings = {
        'dataset': 'name',
        'data_path': 'data_path',
        'data_loader_nprocs': 'data_loader_nprocs',
        'dataset_on_cuda': 'dataset_on_cuda',
        'data_force_reload': 'data_force_reload',
        'data_log_mem_usage': 'data_log_mem_usage',
        'data_clear_tmp_files': 'clear_tmp_files',
        'data_dtype': 'data_dtype',
        'exp_val_perc': 'val_perc',
        'exp_test_perc': 'test_perc',
        'exp_keep_categorical_features': 'keep_categorical_features',
        'exp_cat_as_num_features': 'cat_as_num_features',
        'exp_num_reconstruction': 'num_reconstruction',
        'exp_deterministic_masks': 'deterministic_masks',
        'exp_n_hidden_features': 'n_hidden_features',
        'exp_max_n_recon': 'max_n_recon',
        'exp_aggregation': 'aggregation',
        'exp_normalize_ad_loss': 'normalize_ad_loss',
        'anomalies_in_inference': 'anomalies_in_inference',
        'exp_contamination_share_train': 'contamination_share_train',
        'share_contamination': 'share_contamination',
    }
    for arg_name, config_name in data_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            setattr(config.data, config_name, getattr(args, arg_name))

    # --- TrainingConfig Mappings ---
    training_mappings = {
        'exp_batch_size': 'batch_size',
        'exp_val_batchsize': 'val_batchsize',
        'exp_num_train_inference': 'num_train_inference',
        'exp_num_total_steps': 'num_total_steps',
        'exp_lr': 'lr',
        'exp_weight_decay': 'weight_decay',
        'exp_gradient_clipping': 'gradient_clipping',
        'exp_optimizer': 'optimizer',
        'exp_scheduler': 'scheduler',
        'exp_lookahead_update_cadence': 'lookahead_update_cadence',
        'exp_optimizer_warmup_proportion': 'optimizer_warmup_proportion',
        'exp_optimizer_warmup_fixed_n_steps': 'optimizer_warmup_fixed_n_steps',
        'exp_minibatch_sgd': 'minibatch_sgd',
        'exp_eval_every_n': 'eval_every_n',
        'exp_eval_every_epoch_or_steps': 'eval_every_epoch_or_steps',
        'exp_eval_test_at_end_only': 'eval_test_at_end_only',
        'exp_mix_rows_inference': 'mix_rows_inference',
        'exp_checkpoint_setting': 'checkpoint_setting',
        'exp_cache_cadence': 'cache_cadence',
        'exp_checkpoint_save': 'checkpoint_save',
        'exp_load_from_checkpoint': 'load_from_checkpoint',
        'exp_patience': 'patience',
        'np_seed': 'np_seed',
        'torch_seed': 'torch_seed',
        'original_np_seed': 'original_np_seed',
        'original_torch_seed': 'original_torch_seed',
        'n_runs': 'n_runs',
    }
    for arg_name, config_name in training_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            setattr(config.training, config_name, getattr(args, arg_name))

    # --- SystemConfig Mappings ---
    system_mappings = {
        'exp_use_cuda': 'use_cuda',
        'exp_device': 'device',
        'mp_distributed': 'distributed',
        'mp_nodes': 'nodes',
        'mp_gpus': 'gpus',
        'mp_nr': 'nr',
        'mp_no_sync': 'no_sync',
        'mp_bucket_cap_mb': 'bucket_cap_mb',
        'verbose': 'verbose',
        'exp_print_every_nth_forward': 'print_every_nth_forward',
        'exp_name': 'exp_name',
        'model_checkpoint_key': 'checkpoint_key'
    }
    for arg_name, config_name in system_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            setattr(config.system, config_name, getattr(args, arg_name))

    # --- ModelConfig Mappings ---
    model_mappings = {
        'model_dim_hidden': 'dim_hidden',
        'model_num_heads': 'num_heads',
        'model_stacking_depth': 'stacking_depth',
        'model_rff_depth': 'rff_depth',
        'model_feature_type_embedding': 'feature_type_embedding',
        'model_feature_index_embedding': 'feature_index_embedding',
        'model_sep_res_embed': 'sep_res_embed',
        'model_mix_heads': 'mix_heads',
        'model_att_score_norm': 'att_score_norm',
        'model_embedding_layer_norm': 'embedding_layer_norm',
        'model_att_block_layer_norm': 'att_block_layer_norm',
        'model_pre_layer_norm': 'pre_layer_norm',
        'model_layer_norm_eps': 'layer_norm_eps',
        'model_hidden_dropout_prob': 'hidden_dropout_prob',
        'model_att_score_dropout_prob': 'att_score_dropout_prob',
        'model_bert_augmentation': 'bert_augmentation',
        'model_bert_mask_percentage': 'bert_mask_percentage',
        'model_init_weights': 'init_weights',
        'model_init_type': 'init_type',
        'model_init_params': 'init_params',
        'model_amp': 'amp',
        'model_dtype': 'dtype',
        'model_is_semi_supervised': 'is_semi_supervised',
    }
    for arg_name, config_name in model_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            # Special case for model_init_params default
            if arg_name == 'model_init_params' and getattr(args, arg_name) == []:
                continue
            setattr(config.model, config_name, getattr(args, arg_name))

    # --- MODIFICATION: Improved special handling for model_augmentation_bert_mask_prob ---
    if hasattr(args, 'model_augmentation_bert_mask_prob'):
        mask_prob_arg = args.model_augmentation_bert_mask_prob
        if isinstance(mask_prob_arg, str):
            if mask_prob_arg.startswith('dict('):
                try:
                    # Custom parser for "dict(key=val, ...)" format
                    d = {}
                    inner_str = mask_prob_arg.strip('dict(').strip(')')
                    # Split by comma, handling potential spaces
                    pairs = [p.strip() for p in inner_str.split(',') if p.strip()]
                    for pair in pairs:
                        k, v = pair.split('=')
                        d[k.strip()] = float(v.strip())
                    config.model.augmentation_bert_mask_prob = d
                except Exception as e:
                    print(f"Warning: Could not parse custom dict format '{mask_prob_arg}': {e}")
                    config.model.augmentation_bert_mask_prob = mask_prob_arg
            else:
                try:
                    # Use ast.literal_eval for safe evaluation of Python literals like "{...}"
                    config.model.augmentation_bert_mask_prob = ast.literal_eval(mask_prob_arg)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse '{mask_prob_arg}' as dict literal.")
                    config.model.augmentation_bert_mask_prob = mask_prob_arg
        else:
            # It's already a dict (likely from the default value)
            config.model.augmentation_bert_mask_prob = mask_prob_arg

    return config