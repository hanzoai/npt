import os
import time
from typing import Optional, Dict, Any
import random

import torch
import torch.distributed as dist  # Import distributed module
import numpy as np

from .config_manager import NPTADConfig
# Import NPTDataset, which is used for distributed training
from .column_encoding_dataset import (ColumnEncodingDataset,
                                      NPTDataset)
from .train import Trainer
from .utils.model_init_utils import (init_model_opt_scaler,
                                     setup_ddp_model)


class SimpleTrainer:
    """
    Simplified trainer for NPT-AD models.

    This class provides a clean interface for training anomaly detection
    models with sensible defaults and easy configuration.
    """

    def __init__(self, config: NPTADConfig):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
        """
        self.config = config

        # DDP attributes
        self.distributed_args = None
        self.torch_dataset = None
        self.local_rank = 0
        self.world_size = 1
        self.world_rank = 0

        # Setup DDP environment variables first
        self._setup_distributed_vars()
        # Setup device, which now depends on DDP status
        self.device = self._setup_device()
        self.config.system.device = self.device


        # Initialize the DDP process group
        self._init_process_group()

        self.dataset = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.trainer = None

        # Results storage
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'test_metrics': {},
            'training_time': 0
        }

    def _setup_distributed_vars(self) -> None:
        """Read DDP environment variables if enabled."""
        global LOCAL_RANK, WORLD_SIZE, WORLD_RANK
        if self.config.system.distributed and torch.cuda.is_available():
            try:
                # Environment variables set by torch.distributed.launch
                self.local_rank = int(os.environ['LOCAL_RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.world_rank = int(os.environ['RANK'])

                # Set global vars for consistency (optional, but good)
                LOCAL_RANK, WORLD_SIZE, WORLD_RANK = (
                    self.local_rank, self.world_size, self.world_rank)

                self.distributed_args = {
                    'world_size': self.world_size,
                    'rank': self.world_rank,
                    'gpu': self.local_rank
                }
            except KeyError:
                print(
                    "WARNING: system.distributed is True but DDP environment "
                    "variables (LOCAL_RANK, WORLD_SIZE, RANK) are not set. "
                    "Disabling distributed training.")
                self.config.system.distributed = False
        elif self.config.system.distributed:
            print("WARNING: system.distributed is True but CUDA is not available. "
                  "Disabling distributed training.")
            self.config.system.distributed = False

    def _setup_device(self) -> torch.device:
        """Setup computation device, handling DDP."""
        if self.config.system.distributed:
            # DDP is enabled, device is the local rank
            device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
            print(f"Using GPU: {device} (DDP Rank {self.world_rank}/{self.world_size})")
        elif self.config.system.use_cuda and torch.cuda.is_available():
            # Non-DDP CUDA
            if self.config.system.device:
                device = torch.device(self.config.system.device)
            else:
                device = torch.device('cuda:0')
            print(f"Using GPU: {device}")
        else:
            # CPU
            device = torch.device('cpu')
            print("Using CPU")

        return device

    def _init_process_group(self) -> None:
        """Initialize the DDP process group."""
        if self.config.system.distributed and not dist.is_initialized():
            print(f"Initializing process group for rank {self.world_rank}...")
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.world_rank)
            print("Process group initialized.")

    def load_dataset(
            self,
            dataset_name: Optional[str] = None
    ) -> ColumnEncodingDataset:
        """
        Load the dataset.

        Args:
            dataset_name: Name of the dataset to load. If None, uses config value.

        Returns:
            Loaded dataset
        """
        if dataset_name is None:
            dataset_name = self.config.data.name

        print(f"Loading dataset: {dataset_name}")

        # Create dataset configuration
        dataset_config = self.config
        # Note: self.device is set in __init__
        dataset_config.exp_device = str(self.device)

        # Load dataset
        self.dataset = ColumnEncodingDataset(dataset_config)
        self.dataset.load_next_cv_split()

        # Setup for DDP if enabled
        if self.config.system.distributed:
            self.dataset.dataset_gen = None  # As in original script
            self.torch_dataset = NPTDataset(self.dataset)

        print(f"Dataset loaded: {self.dataset.metadata['N']} samples, "
              f"{self.dataset.metadata['D']} features")

        return self.dataset

    def setup_model(self) -> None:
        """Setup the model, optimizer, and scaler."""
        if self.dataset is None:
            raise ValueError("Dataset must be loaded before setting up model")

        print("Setting up model...")

        # Initialize model, optimizer, and scaler
        # In DDP mode, self.device is already set to the correct local_rank
        self.model, self.optimizer, self.scaler = init_model_opt_scaler(
            metadata=self.dataset.metadata,
            config=self.config,
            device=self.device
        )

        # Wrap model for DDP if enabled
        if self.config.system.distributed:
            dist.barrier()  # Wait for all processes to init model
            self.model = setup_ddp_model(
                model=self.model,
                config=self.config,  # 'c' is the config in the original
                device=self.local_rank
            )
            print(f"Model wrapped for DDP on rank {self.world_rank}.")

        print(f"Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters")

    def train(self,) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Training results
        """
        if self.model is None:
            raise ValueError("Model must be set up before training")

        print("Starting training...")
        start_time = time.time()

        # --- Create trainer ---
        # Build keyword arguments for Trainer
        trainer_kwargs = {
            'model': self.model,
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'config': self.config,
            'cv_index': 0,
            'dataset': self.dataset,
        }

        # Add DDP-specific arguments if enabled
        if self.config.system.distributed:
            trainer_kwargs['torch_dataset'] = self.torch_dataset
            trainer_kwargs['distributed_args'] = self.distributed_args

        self.trainer = Trainer(**trainer_kwargs)

        # Train the model
        self.trainer.train_and_eval()

        # Synchronize and clean up DDP
        if self.config.system.distributed:
            dist.barrier()

        # Record training time
        training_time = time.time() - start_time
        self.results['training_time'] = training_time

        print(f"Training completed in {training_time:.2f} seconds")

        return self.results

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")

        print("Evaluating model...")

        # Run evaluation
        self.dataset.set_mode('test', epoch=0)

        # Get evaluation results
        eval_results = self.trainer.eval_model(
            train_loss=None,
            epoch=0,
            end_experiment=True,
            return_dicts=True
        )

        if eval_results[0] is not None:
            self.results['test_metrics'] = eval_results[0]
            print(f"Test metrics: {eval_results[0]}")

        return self.results['test_metrics']

    def save_model(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")

        # In DDP, only save from the main process
        if self.config.system.distributed and self.world_rank != 0:
            print(f"Rank {self.world_rank}: Skipping model save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get the underlying model state if using DDP
        model_state_dict = (
            self.model.module.state_dict()
            if self.config.system.distributed
            else self.model.state_dict()
        )

        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'results': self.results
        }, path)

        print(f"Model saved to {path} (from rank {self.world_rank})")

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        # When loading, DDP should be off, or all processes load
        if self.config.system.distributed:
            print(f"Rank {self.world_rank}: Loading model from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)

        # Load configuration
        self.config = NPTADConfig.from_dict(checkpoint['config'])

        # Re-check DDP status from loaded config
        # Note: This might conflict with runtime DDP flags.
        # For simplicity, we assume DDP status is set at __init__
        # and loading a model respects that.
        print("Model config loaded. Re-loading dataset and setting up model...")

        # Load dataset
        self.load_dataset()

        # Setup model
        self.setup_model()

        # Load model state
        # If DDP is active, load into self.model.module
        if self.config.system.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load results
        self.results = checkpoint.get('results', {})

        print(f"Model loaded from {path}")

    def cleanup(self) -> None:
        """Clean up distributed processes."""
        if self.config.system.distributed:
            dist.destroy_process_group()
            print(f"Destroyed process group for rank {self.world_rank}.")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        return {
            'config': self.config.to_dict(),
            'dataset_info': {
                'name': self.config.data.name,
                'samples': self.dataset.metadata['N'] if self.dataset else None,
                'features': self.dataset.metadata['D'] if self.dataset else None,
            },
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else None,
            },
            'results': self.results
        }

    def synchronize_seeds(self) -> None:
        """
        Generates random seeds on rank 0 and broadcasts them to all
        other ranks to ensure seeds are identical for a given run.
        """
        if not self.config.system.distributed:
            # This should only be called in DDP, but as a safeguard
            # we generate seeds for the single-process case.
            self.config.training.np_seed = random.randint(0, 10000)
            self.config.training.torch_seed = random.randint(0, 10000)
            self.config.training.original_np_seed = random.randint(0, 10000)
            self.config.training.original_torch_seed = random.randint(0, 10000)
            return

        if self.world_rank == 0:
            # Rank 0 generates the seeds
            seeds = [
                random.randint(0, 10000),  # np_seed
                random.randint(0, 10000),  # torch_seed
                random.randint(0, 10000),  # original_np_seed
                random.randint(0, 10000)   # original_torch_seed
            ]
            # Create a tensor of seeds to broadcast
            seed_tensor = torch.tensor(seeds, dtype=torch.long, device=self.device)
        else:
            # Other ranks create an empty tensor to receive seeds
            seed_tensor = torch.empty(4, dtype=torch.long, device=self.device)
        
        # Broadcast the tensor from rank 0 to all ranks
        dist.broadcast(seed_tensor, src=0)
        
        # All ranks now have the synchronized seeds
        synced_seeds = seed_tensor.cpu().tolist()
        
        self.config.training.np_seed = synced_seeds[0]
        self.config.training.torch_seed = synced_seeds[1]
        self.config.training.original_np_seed = synced_seeds[2]
        self.config.training.original_torch_seed = synced_seeds[3]
        
        if self.config.system.verbose:
            print(
                f"Rank {self.world_rank}: Synchronized seeds. "
                f"Torch Seed: {self.config.training.torch_seed}"
            )


def train_anomaly_detector(
        dataset_name: str,
        config: Optional[NPTADConfig] = None,
        save_path: Optional[str] = None) -> SimpleTrainer:
    """
    Convenience function to train an anomaly detector.

    Args:
        dataset_name: Name of the dataset to use
        config: Configuration object. If None, uses default config.
        save_path: Path to save the trained model. If None, doesn't save.

    Returns:
        Trained trainer object
    """
    if config is None:
        config = NPTADConfig()

    # Set dataset name
    config.data.name = dataset_name

    # Create trainer
    trainer = SimpleTrainer(config)
    trainer.synchronize_seeds()

    # Load dataset
    trainer.load_dataset()

    # Setup model
    trainer.setup_model()

    # Train
    trainer.train()

    # Evaluate (only on main process for logging)
    trainer.evaluate()

    # Save if requested (will only run on rank 0)
    if save_path:
        trainer.save_model(save_path)

    return trainer


def quick_test(dataset_name: str = 'separable') -> Dict[str, Any]:
    """
    Run a quick test on a dataset.

    Args:
        dataset_name: Name of the dataset to test

    Returns:
        Test results
    """
    # Use quick test preset
    config = NPTADConfig().get_preset('quick_test')
    config.data.name = dataset_name

    # Train
    trainer = train_anomaly_detector(dataset_name, config)

    # Get summary
    return trainer.get_training_summary()