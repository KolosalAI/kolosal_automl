"""
Sample Data Loader

Loads sample datasets for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

try:
    from sklearn.datasets import (
        load_iris, load_wine, load_diabetes, 
        load_breast_cancer, make_classification, make_regression
    )
    # Try to import load_boston separately since it was removed in sklearn 1.2+
    try:
        from sklearn.datasets import load_boston
        BOSTON_AVAILABLE = True
    except ImportError:
        BOSTON_AVAILABLE = False
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BOSTON_AVAILABLE = False

try:
    from modules.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class SampleDataLoader:
    """Load sample datasets for demonstration and testing"""
    
    def __init__(self):
        """Initialize the sample data loader"""
        self.datasets = {
            "Iris Classification": {
                "loader": self._load_iris,
                "description": "Classic iris flower classification dataset with 4 features and 3 species",
                "task_type": "classification",
                "target_column": "species"
            },
            "Boston Housing": {
                "loader": self._load_boston,
                "description": "Boston housing price prediction dataset with 13 features",
                "task_type": "regression", 
                "target_column": "price"
            },
            "Wine Quality": {
                "loader": self._load_wine,
                "description": "Wine classification dataset with chemical properties",
                "task_type": "classification",
                "target_column": "target"
            },
            "Diabetes": {
                "loader": self._load_diabetes,
                "description": "Diabetes progression prediction dataset",
                "task_type": "regression",
                "target_column": "target"
            },
            "Breast Cancer": {
                "loader": self._load_breast_cancer,
                "description": "Breast cancer diagnosis classification dataset",
                "task_type": "classification",
                "target_column": "target"
            }
        }
    
    def load_sample_data(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load a sample dataset by name"""
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            dataset_info = self.datasets[dataset_name]
            df = dataset_info["loader"]()
            
            metadata = {
                "name": dataset_name,
                "description": dataset_info["description"],
                "task_type": dataset_info["task_type"],
                "target_column": dataset_info["target_column"],
                "shape": df.shape,
                "columns": df.columns.tolist()
            }
            
            logger.info(f"Successfully loaded sample dataset: {dataset_name}")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading sample dataset {dataset_name}: {e}")
            # Return a simple dummy dataset as fallback
            return self._create_dummy_dataset(), {
                "name": "Dummy Dataset",
                "description": "Fallback dummy dataset",
                "task_type": "classification",
                "target_column": "target",
                "error": str(e)
            }
    
    def get_available_datasets(self) -> list[str]:
        """Get list of available sample datasets"""
        return list(self.datasets.keys())
    
    def _load_iris(self) -> pd.DataFrame:
        """Load Iris dataset"""
        if not SKLEARN_AVAILABLE:
            return self._create_dummy_classification_dataset(n_samples=150, n_features=4, n_classes=3)
        
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df
    
    def _load_boston(self) -> pd.DataFrame:
        """Load Boston Housing dataset"""
        if not SKLEARN_AVAILABLE or not BOSTON_AVAILABLE:
            return self._create_dummy_regression_dataset(n_samples=506, n_features=13, target_name="price")
        
        try:
            from sklearn.datasets import load_boston
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['price'] = data.target
            return df
        except ImportError:
            # Boston dataset was removed in newer sklearn versions
            return self._create_dummy_regression_dataset(n_samples=506, n_features=13, target_name="price")
    
    def _load_wine(self) -> pd.DataFrame:
        """Load Wine dataset"""
        if not SKLEARN_AVAILABLE:
            return self._create_dummy_classification_dataset(n_samples=178, n_features=13, n_classes=3)
        
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    def _load_diabetes(self) -> pd.DataFrame:
        """Load Diabetes dataset"""
        if not SKLEARN_AVAILABLE:
            return self._create_dummy_regression_dataset(n_samples=442, n_features=10)
        
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    def _load_breast_cancer(self) -> pd.DataFrame:
        """Load Breast Cancer dataset"""
        if not SKLEARN_AVAILABLE:
            return self._create_dummy_classification_dataset(n_samples=569, n_features=30, n_classes=2)
        
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    def _create_dummy_classification_dataset(self, n_samples: int = 100, n_features: int = 4, n_classes: int = 3) -> pd.DataFrame:
        """Create a dummy classification dataset"""
        try:
            if SKLEARN_AVAILABLE:
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    n_redundant=0,
                    random_state=42
                )
            else:
                # Manual creation without sklearn
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, n_classes, n_samples)
            
            # Create DataFrame
            feature_names = [f'feature_{i}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating dummy classification dataset: {e}")
            # Ultimate fallback - very simple dataset
            return pd.DataFrame({
                'feature_0': [1, 2, 3, 4, 5],
                'feature_1': [2, 4, 6, 8, 10],
                'target': [0, 1, 0, 1, 0]
            })
    
    def _create_dummy_regression_dataset(self, n_samples: int = 100, n_features: int = 4, target_name: str = "target") -> pd.DataFrame:
        """Create a dummy regression dataset"""
        try:
            if SKLEARN_AVAILABLE:
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=0.1,
                    random_state=42
                )
            else:
                # Manual creation without sklearn
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.1
            
            # Create DataFrame
            feature_names = [f'feature_{i}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df[target_name] = y
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating dummy regression dataset: {e}")
            # Ultimate fallback - very simple dataset
            return pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature_1': [2.0, 4.0, 6.0, 8.0, 10.0],
                'target': [1.5, 3.0, 4.5, 6.0, 7.5]
            })
    
    def _create_dummy_dataset(self) -> pd.DataFrame:
        """Create a simple fallback dummy dataset"""
        return pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5],
            'feature_1': [2, 4, 6, 8, 10],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
