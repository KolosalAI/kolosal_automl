"""
Python backend fallback for Kolosal AutoML.

This module provides the pure Python implementation for environments
where the Rust backend is not available.
"""

import sys
import os
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd

# Add legacy path for imports
_legacy_path = os.path.join(os.path.dirname(__file__), "..", "..", "legacy")
if _legacy_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_legacy_path))

# Import from legacy modules
try:
    from modules.engine.data_preprocessor import DataPreprocessor as _LegacyPreprocessor
    from modules.engine.train_engine import TrainEngine as _LegacyTrainEngine
    from modules.engine.inference_engine import InferenceEngine as _LegacyInferenceEngine
    from modules.optimizer.hyperoptx import HyperOptX as _LegacyHyperOptX
except ImportError as e:
    raise ImportError(
        f"Could not import legacy Python modules: {e}. "
        "Make sure the legacy/ directory exists with the original Python code."
    ) from e


class DataPreprocessor:
    """Data preprocessing pipeline (Python fallback).
    
    This wraps the legacy Python implementation for compatibility.
    """
    
    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        scaling: str = "standard",
        encoding: str = "onehot",
        imputation: str = "mean",
    ):
        """Initialize the preprocessor."""
        self._preprocessor = _LegacyPreprocessor(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            scaling=scaling,
            encoding=encoding,
            imputation=imputation,
        )
        self._fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "DataPreprocessor":
        """Fit the preprocessor to the data."""
        self._preprocessor.fit(X)
        self._fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._preprocessor.transform(X)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._preprocessor.inverse_transform(X)


class TrainEngine:
    """ML training engine (Python fallback)."""
    
    def __init__(
        self,
        task_type: str = "classification",
        cv_folds: int = 5,
        metric: str = "accuracy",
        random_state: Optional[int] = None,
    ):
        """Initialize the training engine."""
        self._engine = _LegacyTrainEngine(
            task_type=task_type,
            cv_folds=cv_folds,
            metric=metric,
            random_state=random_state,
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
        **model_params,
    ) -> Dict[str, Any]:
        """Train a model with cross-validation."""
        return self._engine.train(X, y, model_type, **model_params)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
    ) -> Dict[str, Any]:
        """Perform cross-validation."""
        return self._engine.cross_validate(X, y, model_type)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model."""
        return self._engine.predict(X)


class InferenceEngine:
    """Inference engine (Python fallback)."""
    
    def __init__(self):
        """Initialize the inference engine."""
        self._engine = _LegacyInferenceEngine()
    
    def load_model(self, path: str) -> None:
        """Load a trained model from file."""
        self._engine.load_model(path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self._engine.predict(X)
    
    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Make predictions in batches."""
        return self._engine.predict_batch(X, batch_size)


class HyperOptX:
    """Hyperparameter optimization (Python fallback)."""
    
    def __init__(
        self,
        search_space: Dict[str, tuple],
        sampler: str = "tpe",
        direction: str = "minimize",
        seed: Optional[int] = None,
    ):
        """Initialize the optimizer."""
        self._optimizer = _LegacyHyperOptX(
            search_space=search_space,
            sampler=sampler,
            direction=direction,
            seed=seed,
        )
    
    def optimize(
        self,
        objective: callable,
        n_trials: int = 100,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        return self._optimizer.optimize(objective, n_trials, timeout)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found."""
        return self._optimizer.get_best_params()
    
    def get_trials(self) -> List[Dict[str, Any]]:
        """Get all trial results."""
        return self._optimizer.get_trials()


# Data I/O functions

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path, **kwargs)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file."""
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False, **kwargs)


def get_file_info(path: str) -> Dict[str, Any]:
    """Get information about a data file."""
    if path.endswith('.csv'):
        df = pd.read_csv(path, nrows=0)
        # Count rows separately to avoid loading whole file
        with open(path, 'r') as f:
            n_rows = sum(1 for _ in f) - 1  # Subtract header
    elif path.endswith('.parquet'):
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(path)
        n_rows = parquet_file.metadata.num_rows
        df = parquet_file.schema_arrow.empty_table().to_pandas()
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    return {
        "path": path,
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
