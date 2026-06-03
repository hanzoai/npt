"""
Dataset registry for NPT-AD.

This module provides a clean interface for registering and managing datasets,
making it easier to add new datasets without modifying core code.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from .base import BaseDataset


class DatasetRegistry:
    """Registry for managing datasets."""
    
    _datasets: Dict[str, type] = {}
    _dataset_info: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: type, info: Optional[Dict] = None):
        """Register a dataset class."""
        if not issubclass(dataset_class, BaseDataset):
            raise ValueError(f"Dataset class must inherit from BaseDataset")
        
        cls._datasets[name] = dataset_class
        cls._dataset_info[name] = info or {}
    
    @classmethod
    def get_dataset_class(cls, name: str) -> type:
        """Get dataset class by name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]
    
    @classmethod
    def get_dataset_info(cls, name: str) -> Dict:
        """Get dataset information."""
        return cls._dataset_info.get(name, {})
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets."""
        return list(cls._datasets.keys())
    
    @classmethod
    def create_dataset(cls, name: str, config) -> BaseDataset:
        """Create a dataset instance."""
        dataset_class = cls.get_dataset_class(name)
        return dataset_class(config)


class SimpleAnomalyDataset(BaseDataset):
    """
    Simplified base class for anomaly detection datasets.
    
    This class provides a cleaner interface for creating anomaly detection
    datasets with common patterns.
    """
    
    def __init__(self, config, normal_label: int = 0, anomaly_label: int = 1):
        super().__init__(fixed_test_set_index=None)
        self.config = config
        self.normal_label = normal_label
        self.anomaly_label = anomaly_label
        self.ad = True
        self.is_data_loaded = False
    
    def load_from_dataframe(self, df: pd.DataFrame, target_column: str, 
                           categorical_columns: Optional[List[str]] = None,
                           numerical_columns: Optional[List[str]] = None):
        """
        Load dataset from a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column (anomaly labels)
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
        """
        # Separate features and target
        self.target = df[target_column].values
        feature_df = df.drop(columns=[target_column])
        
        # Convert to numpy array
        self.data_table = feature_df.values
        
        # Separate normal and anomaly samples
        normal_mask = self.target == self.normal_label
        anomaly_mask = self.target == self.anomaly_label
        
        self.norm_samples = self.data_table[normal_mask]
        self.anom_samples = self.data_table[anomaly_mask]
        
        # Add target column back
        self.norm_samples = np.c_[self.norm_samples, 
                                 np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                 np.ones(self.anom_samples.shape[0])]
        
        # Update data table
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples), axis=0)
        self.N, self.D = self.data_table.shape
        
        # Set up feature types
        if categorical_columns is None:
            categorical_columns = []
        if numerical_columns is None:
            numerical_columns = [col for col in feature_df.columns if col not in categorical_columns]
        
        # Map column names to indices
        col_name_to_idx = {name: idx for idx, name in enumerate(feature_df.columns)}
        
        self.cat_features = [col_name_to_idx[col] for col in categorical_columns if col in col_name_to_idx]
        self.num_features = [col_name_to_idx[col] for col in numerical_columns if col in col_name_to_idx]
        
        # Target column is the last column
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []
        
        # Calculate anomaly ratio
        self.ratio = (100.0 * (0.5 * len(self.norm_samples)) / 
                     ((0.5 * len(self.norm_samples)) + len(self.anom_samples)))
        
        # Initialize missing matrix (no missing values by default)
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True
    
    def load_from_csv(self, csv_path: str, target_column: str,
                     categorical_columns: Optional[List[str]] = None,
                     numerical_columns: Optional[List[str]] = None,
                     **kwargs):
        """Load dataset from CSV file."""
        df = pd.read_csv(csv_path, **kwargs)
        self.load_from_dataframe(df, target_column, categorical_columns, numerical_columns)
    
    def load_from_numpy(self, X: np.ndarray, y: np.ndarray,
                       categorical_indices: Optional[List[int]] = None,
                       numerical_indices: Optional[List[int]] = None):
        """
        Load dataset from numpy arrays.
        
        Args:
            X: Feature matrix (N, D)
            y: Target array (N,)
            categorical_indices: List of categorical feature indices
            numerical_indices: List of numerical feature indices
        """
        # Separate normal and anomaly samples
        normal_mask = y == self.normal_label
        anomaly_mask = y == self.anomaly_label
        
        self.norm_samples = X[normal_mask]
        self.anom_samples = X[anomaly_mask]
        
        # Add target column
        self.norm_samples = np.c_[self.norm_samples, 
                                 np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                 np.ones(self.anom_samples.shape[0])]
        
        # Update data table
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples), axis=0)
        self.N, self.D = self.data_table.shape
        
        # Set up feature types
        if categorical_indices is None:
            categorical_indices = []
        if numerical_indices is None:
            numerical_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
        
        self.cat_features = categorical_indices
        self.num_features = numerical_indices
        
        # Target column is the last column
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []
        
        # Calculate anomaly ratio
        self.ratio = (100.0 * (0.5 * len(self.norm_samples)) / 
                     ((0.5 * len(self.norm_samples)) + len(self.anom_samples)))
        
        # Initialize missing matrix
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.num_normal = len(self.norm_samples)
        self.is_data_loaded = True


def create_simple_dataset(name: str, data_source: Union[str, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
                         target_column: str = 'target',
                         categorical_columns: Optional[List[str]] = None,
                         numerical_columns: Optional[List[str]] = None,
                         normal_label: int = 0, anomaly_label: int = 1,
                         **kwargs) -> type:
    """
    Create a simple dataset class for anomaly detection.
    
    Args:
        name: Name of the dataset
        data_source: Path to CSV file, pandas DataFrame, or tuple of (X, y) arrays
        target_column: Name of the target column (for CSV/DataFrame)
        categorical_columns: List of categorical column names/indices
        numerical_columns: List of numerical column names/indices
        normal_label: Label for normal samples
        anomaly_label: Label for anomaly samples
        **kwargs: Additional arguments for pandas.read_csv
    
    Returns:
        Dataset class that can be registered
    """
    
    class SimpleDataset(SimpleAnomalyDataset):
        def __init__(self, config):
            super().__init__(config, normal_label, anomaly_label)
            self.tmp_file_names = [f"{name}.csv"] if isinstance(data_source, str) else []
        
        def load(self):
            if isinstance(data_source, str):
                # Load from CSV file
                self.load_from_csv(data_source, target_column, 
                                 categorical_columns, numerical_columns, **kwargs)
            elif isinstance(data_source, pd.DataFrame):
                # Load from DataFrame
                self.load_from_dataframe(data_source, target_column,
                                       categorical_columns, numerical_columns)
            elif isinstance(data_source, tuple) and len(data_source) == 2:
                # Load from numpy arrays
                X, y = data_source
                self.load_from_numpy(X, y, categorical_columns, numerical_columns)
            else:
                raise ValueError("data_source must be a file path, DataFrame, or tuple of (X, y) arrays")
    
    return SimpleDataset


# Register built-in datasets
def register_builtin_datasets():
    """Register all built-in datasets."""
    from . import (
        abalone, annthyroid, arrhythmia, backdoor, breastw, campaign, cardio,
        ecoli, forestcoverad, fraud, glass, ionosphere, kdd, kddrev, letter,
        lympho, mammography, mnistad, mulcross, musk, optdigits, pendigits,
        pima, satellite, satimage, seismic, separable, shuttle, speech,
        thyroid, vertebral, vowels, wbc, wine
    )
    
    # Register datasets with their info
    datasets_to_register = [
        ('abalone', abalone.AbaloneDataset, {
            'description': 'Abalone dataset from UCI ML Repository',
            'url': 'https://archive.ics.uci.edu/ml/datasets/abalone',
            'features': 'Mixed (categorical + numerical)',
            'samples': '~4000'
        }),
        ('separable', separable.SeparableDataset, {
            'description': 'Synthetic separable dataset for testing',
            'features': 'Numerical only',
            'samples': '2000'
        }),
        # Add more datasets as needed
    ]
    
    for name, dataset_class, info in datasets_to_register:
        DatasetRegistry.register(name, dataset_class, info)


# Auto-register built-in datasets
register_builtin_datasets()
