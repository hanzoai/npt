#!/usr/bin/env python3
"""
Quick start example for NPT-AD.

This script demonstrates how to use the simplified NPT-AD interface
to train an anomaly detector on a dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npt.config_manager import NPTADConfig

# Conditional imports for torch-dependent modules
try:
    from npt.simple_trainer import quick_test, train_anomaly_detector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be limited.")


def main():
    """Run a quick start example."""
    print("NPT-AD Quick Start Example")
    print("=" * 40)
    
    # Example 1: Quick test with default settings
    print("\n1. Running quick test on separable dataset...")
    if not TORCH_AVAILABLE:
        print("Skipping quick test - PyTorch not available")
    else:
        try:
            results = quick_test('separable')
            print(f"Quick test completed!")
            print(f"Training time: {results['results']['training_time']:.2f} seconds")
            if 'test_metrics' in results['results']:
                print(f"Test metrics: {results['results']['test_metrics']}")
        except Exception as e:
            print(f"Quick test failed: {e}")
            print("This is expected if the dataset is not available.")
    
    # Example 2: Custom configuration
    print("\n2. Training with custom configuration...")
    if not TORCH_AVAILABLE:
        print("Skipping custom training - PyTorch not available")
    else:
        try:
            # Create custom config
            config = NPTADConfig()
            config.data.dataset = 'abalone'  # Change dataset
            config.training.num_total_steps = 2000  # Fewer steps for demo
            config.training.eval_every_n = 500
            config.model.dim_hidden = 32
            config.model.stacking_depth = 4
            config.system.verbose = True
            
            # Train
            trainer = train_anomaly_detector('abalone', config)
            
            print("Custom training completed!")
            print(f"Dataset: {trainer.config.data.dataset}")
            print(f"Training time: {trainer.results['training_time']:.2f} seconds")
            
        except Exception as e:
            print(f"Custom training failed: {e}")
            print("This is expected if the dataset is not available.")
    
    # Example 3: Using presets
    print("\n3. Using configuration presets...")
    if not TORCH_AVAILABLE:
        print("Skipping preset training - PyTorch not available")
    else:
        try:
            # Small dataset preset
            config = NPTADConfig().get_preset('small_dataset', dataset='separable')
            
            trainer = train_anomaly_detector('separable', config)
            print("Preset training completed!")
            
        except Exception as e:
            print(f"Preset training failed: {e}")
            print("This is expected if the dataset is not available.")
    
    print("\n" + "=" * 40)
    print("Quick start example completed!")
    print("\nNext steps:")
    print("1. Check the 'results/' directory for output files")
    print("2. Modify the configuration for your specific needs")
    print("3. Add your own datasets using the dataset registry")


if __name__ == "__main__":
    main()