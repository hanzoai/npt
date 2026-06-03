#!/usr/bin/env python3
"""
Command-line interface for NPT-AD.

This module provides a simplified CLI for training and evaluating
NPT-AD models without the complexity of the original argument parser.
"""

import argparse, os, sys, json
import numpy as np
import torch

from .config_manager import NPTADConfig
from .simple_trainer import train_anomaly_detector, quick_test

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='NPT-AD: Non-Parametric Transformers for Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on separable dataset
  python -m npt.cli test --dataset separable

  # Train with custom config
  python -m npt.cli train --dataset abalone --config config.json

  # Train with preset
  python -m npt.cli train --dataset abalone --preset small_dataset

  # List available datasets
  python -m npt.cli list-datasets

  # List available presets
  python -m npt.cli list-presets
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run a quick test')
    test_parser.add_argument('--dataset', default='separable', help='Dataset to test')
    test_parser.add_argument('--preset', default='quick_test', help='Configuration preset')
    test_parser.add_argument('--high_nb_features', default=False, type=bool, help='Configuration preset')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--dataset', required=True, help='Dataset to train on')
    train_parser.add_argument('--config', help='Path to configuration file')
    train_parser.add_argument('--preset', help='Configuration preset to use')
    train_parser.add_argument('--high_nb_features', default=False, type=bool, help='Configuration preset')
    train_parser.add_argument('--output', help='Path to save trained model')
    train_parser.add_argument('--steps', type=int, help='Number of training steps')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--hidden-dim', type=int, help='Hidden dimension')
    train_parser.add_argument('--max_n_reconstruction', type=int, help='Number of inference reconstructions.')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    train_parser.add_argument('--distributed', default=False, help="Distributed environment?")
    train_parser.add_argument('--n_gpus', default=None, help="Number of gpus used.")   
    train_parser.add_argument('--n_runs', default=None, help="Number of runs.")    

    # List commands
    subparsers.add_parser('list-datasets', help='List available datasets')
    subparsers.add_parser('list-presets', help='List available presets')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration utilities')
    config_parser.add_argument('--preset', help='Generate config from preset')
    config_parser.add_argument('--output', help='Output file for generated config')
    
    return parser


def cmd_test(args):
    """Run quick test command."""
    
    print(f"Running quick test on dataset: {args.dataset}")
    print(f"Using preset: {args.preset}")
    
    try:
        results = quick_test(args.dataset)
        print("\nTest completed successfully!")
        print(f"Training time: {results['results']['training_time']:.2f} seconds")
        
        if 'test_metrics' in results['results']:
            print("Test metrics:")
            for metric, value in results['results']['test_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


def cmd_train(args):
    """Run training command."""
    
    print(f"Training model on dataset: {args.dataset}")
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = NPTADConfig.from_json(args.config)
    elif args.preset:
        print(f"Using preset: {args.preset}")
        if args.high_nb_features:
            args.preset += "_high_d"
        config = NPTADConfig().get_preset(args.preset, args.dataset)
    else:
        print("Using default configuration")
        config = NPTADConfig()
    
    # Set dataset
    config.data.name = args.dataset
    if config.data.data_path is None:
        config.data.data_path = f'./data/{args.dataset}'
        print('Setting default path to dataset -> {}'.format(
            config.data.data_path,
        ))
    
    # Override with command-line arguments
    if args.steps:
        config.training.num_total_steps = args.steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.hidden_dim:
        config.model.dim_hidden = args.hidden_dim
    if args.max_n_reconstruction:
        config.data.max_n_recon = args.max_n_reconstruction
    if args.verbose:
        config.system.verbose = True
    if args.n_runs is not None:
        config.training.n_runs = args.n_runs
    if args.distributed:
        config.system.distributed = args.distributed
    if args.n_gpus is not None:
        config.system.gpus = args.n_gpus
    
    if config.training.n_runs == 1:
        trainer = train_anomaly_detector(
            dataset_name=args.dataset,
            config=config,
            save_path=args.output
        )
        
        print("\nTraining completed successfully!")
        print(f"Training time: {trainer.results['training_time']:.2f} seconds")
        
        # Show results
        if 'test_metrics' in trainer.results:
            print("Test metrics:")
            for metric, value in trainer.results['test_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if args.output:
            print(f"Model saved to: {args.output}")

    else:
        metrics = {"F1": [], 'ap': [], "auc": []}
        for _ in range(config.training.n_runs):

            trainer = train_anomaly_detector(
                dataset_name=args.dataset,
                config=config,
                save_path=args.output
            )

            if trainer.world_rank == 0 or torch.cuda.device_count()<2:
                metrics['F1'].append(trainer.results['test_metrics']['F1'].item())
                metrics['auc'].append(trainer.results['test_metrics']['auc'].item())
                metrics['ap'].append(trainer.results['test_metrics']['ap'].item())

        if trainer.world_rank == 0 or torch.cuda.device_count()<2:
            mean_f1 = np.mean(metrics['F1'])
            std_f1 = np.std(metrics['F1'])
            mean_auroc = np.mean(metrics['auc'])
            std_auroc = np.std(metrics['auc'])
            mean_ap = np.mean(metrics['ap'])
            std_ap = np.std(metrics['ap'])
    
            results_string = (
                f"Average metrics for {config.training.n_runs} runs:\n"
                f"- F1 Score: {mean_f1:.4f} ({std_f1:.4f})\n"
                f"- AUROC: {mean_auroc:.4f} ({std_auroc:.4f})\n"
                f"- AP: {mean_ap:.4f} ({std_ap:.4f})"
            )
    
            print(results_string)
            json_file_path = os.path.join(config.res_dir, "all_metrics.json")
            with open(json_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Successfully saved all metrics to {json_file_path}")


def cmd_list_datasets(args):
    """List available datasets."""
    try:
        from .datasets.dataset_registry import DatasetRegistry
        
        datasets = DatasetRegistry.list_datasets()
        print("Available datasets:")
        print("-" * 50)
        
        for dataset_name in sorted(datasets):
            info = DatasetRegistry.get_dataset_info(dataset_name)
            description = info.get('description', 'No description available')
            print(f"{dataset_name:20} - {description}")
        
    except Exception as e:
        print(f"Failed to list datasets: {e}")
        sys.exit(1)


def cmd_list_presets(args):
    """List available presets."""
    presets = [
        'quick_test',
        'small_dataset',
        'small_dataset_high_d',
        'medium_dataset',
        'medium_dataset_high_d',
        'large_dataset',
        'large_dataset_high_d',
        'gpu_optimized'
    ]
    
    print("Available configuration presets:")
    print("-" * 50)
    
    for preset in presets:
        config = NPTADConfig().get_preset(preset, dataset="separable")
        print(f"{preset:20} - {config.training.num_total_steps} steps, "
              f"hidden_dim={config.model.dim_hidden}, "
              f"batch_size={config.training.batch_size}")


def cmd_config(args):
    """Generate configuration file."""
    if args.preset:
        config = NPTADConfig().get_preset(args.preset, dataset="separable")
    else:
        config = NPTADConfig()
    
    output_path = args.output or f"config_{args.preset or 'default'}.json"
    
    config.to_json(output_path)
    print(f"Configuration saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    command_map = {
        'test': cmd_test,
        'train': cmd_train,
        'list-datasets': cmd_list_datasets,
        'list-presets': cmd_list_presets,
        'config': cmd_config,
    }
    
    command_map[args.command](args)


if __name__ == '__main__':
    main()