#!/usr/bin/env python3
"""
Custom dataset example for NPT-AD.

This script demonstrates how to create and use custom datasets
with the simplified NPT-AD interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from npt.config_manager import NPTADConfig
from npt.datasets.dataset_registry import DatasetRegistry, create_simple_dataset
from npt.simple_trainer import SimpleTrainer


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    contamination: float = 0.1,
    random_state: int = 42
):
    """
    Create a synthetic anomaly detection dataset.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        contamination: Fraction of anomalies
        random_state: Random seed
    
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    # Generate normal data
    n_normal = int(n_samples * (1 - contamination))
    n_anomaly = n_samples - n_normal
    
    # Normal data from one cluster
    X_normal, _ = make_blobs(
        n_samples=n_normal, 
        centers=1, 
        n_features=n_features,
        random_state=random_state
    )
    y_normal = np.zeros(n_normal)
    
    # Anomaly data from different cluster
    X_anomaly, _ = make_blobs(
        n_samples=n_anomaly,
        centers=1,
        n_features=n_features,
        center_box=(10, 10),  # Far from normal data
        random_state=random_state + 1
    )
    y_anomaly = np.ones(n_anomaly)
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])
    
    # Shuffle data
    indices = np.random.RandomState(random_state).permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def create_dataframe_dataset():
    """Create a dataset from a pandas DataFrame."""
    # Create synthetic data
    X, y = create_synthetic_dataset(n_samples=500, n_features=5)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['binary'] = np.random.choice([0, 1], size=len(df))
    
    return df


def main():
    """Run custom dataset examples."""
    print("NPT-AD Custom Dataset Example")
    print("=" * 40)
    
    # Example 1: Create dataset from numpy arrays
    print("\n1. Creating dataset from numpy arrays...")
    try:
        # Create synthetic data
        X, y = create_synthetic_dataset(n_samples=1000, n_features=8)
        print(f"Created synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Anomaly ratio: {np.mean(y):.2%}")
        
        # Create dataset class
        SyntheticDataset = create_simple_dataset(
            name='synthetic',
            data_source=(X, y),
            categorical_columns=[],  # All features are numerical
            numerical_columns=list(range(X.shape[1])),
            normal_label=0,
            anomaly_label=1
        )
        
        # Register dataset
        DatasetRegistry.register('synthetic', SyntheticDataset, {
            'description': 'Synthetic anomaly detection dataset',
            'features': 'Numerical only',
            'samples': '1000'
        })
        
        print("Dataset registered successfully!")
        
    except Exception as e:
        print(f"Failed to create numpy dataset: {e}")
    
    # Example 2: Create dataset from DataFrame
    print("\n2. Creating dataset from pandas DataFrame...")
    try:
        # Create DataFrame
        df = create_dataframe_dataset()
        print(f"Created DataFrame dataset: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Create dataset class
        DataFrameDataset = create_simple_dataset(
            name='dataframe_example',
            data_source=df,
            target_column='target',
            categorical_columns=['category', 'binary'],
            numerical_columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'],
            normal_label=0,
            anomaly_label=1
        )
        
        # Register dataset
        DatasetRegistry.register('dataframe_example', DataFrameDataset, {
            'description': 'DataFrame-based anomaly detection dataset',
            'features': 'Mixed (categorical + numerical)',
            'samples': '500'
        })
        
        print("DataFrame dataset registered successfully!")
        
    except Exception as e:
        print(f"Failed to create DataFrame dataset: {e}")
    
    # Example 3: Train on custom dataset
    print("\n3. Training on custom dataset...")
    try:
        # Create configuration
        config = NPTADConfig().get_preset('quick_test', dataset='synthetic')
        
        # Create trainer
        trainer = SimpleTrainer(config)
        
        # Load dataset
        trainer.load_dataset()
        
        # Setup model
        trainer.setup_model()
        
        # Train
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        
        print("Custom dataset training completed!")
        print(f"Training time: {trainer.results['training_time']:.2f} seconds")
        if metrics:
            print(f"Test metrics: {metrics}")
        
    except Exception as e:
        print(f"Custom dataset training failed: {e}")
        print("This might be due to dataset loading issues.")
    
    # Example 4: List available datasets
    print("\n4. Available datasets:")
    try:
        datasets = DatasetRegistry.list_datasets()
        print(f"Registered datasets: {datasets}")
        
        for dataset_name in datasets[:5]:  # Show first 5
            info = DatasetRegistry.get_dataset_info(dataset_name)
            print(f"  - {dataset_name}: {info.get('description', 'No description')}")
        
    except Exception as e:
        print(f"Failed to list datasets: {e}")
    
    print("\n" + "=" * 40)
    print("Custom dataset example completed!")
    print("\nKey takeaways:")
    print("1. You can create datasets from numpy arrays or DataFrames")
    print("2. Datasets are automatically registered and can be used immediately")
    print("3. The system handles both categorical and numerical features")
    print("4. You can specify which features are categorical vs numerical")


if __name__ == "__main__":
    main()
