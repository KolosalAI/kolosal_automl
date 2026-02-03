"""
Rust backend wrapper for Kolosal AutoML.

This module wraps the kolosal_rust PyO3 bindings with a clean Python API.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

try:
    import kolosal_rust as _rust
except ImportError as e:
    raise ImportError(
        "Rust backend is not installed. Install with `pip install kolosal[rust]`"
    ) from e


class DataPreprocessor:
    """Data preprocessing pipeline with fit/transform interface.
    
    This is a high-performance Rust implementation of common preprocessing
    operations including scaling, encoding, and imputation.
    
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> preprocessor.fit(X_train)
        >>> X_transformed = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        scaling: str = "standard",
        encoding: str = "onehot",
        imputation: str = "mean",
    ):
        """Initialize the preprocessor.
        
        Args:
            numeric_columns: List of numeric column names (auto-detected if None)
            categorical_columns: List of categorical column names (auto-detected if None)
            scaling: Scaling method - "standard", "minmax", or "robust"
            encoding: Encoding method - "onehot", "label", or "target"
            imputation: Imputation method - "mean", "median", or "mode"
        """
        self._preprocessor = _rust.PyDataPreprocessor(
            numeric_columns=numeric_columns or [],
            categorical_columns=categorical_columns or [],
            scaling=scaling,
            encoding=encoding,
            imputation=imputation,
        )
        self._fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "DataPreprocessor":
        """Fit the preprocessor to the data.
        
        Args:
            X: Training data (numpy array or pandas DataFrame)
            
        Returns:
            self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self._preprocessor.fit_dataframe(X)
        else:
            self._preprocessor.fit(np.asarray(X))
        self._fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed numpy array
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            return self._preprocessor.transform_dataframe(X)
        return self._preprocessor.transform(np.asarray(X))
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed numpy array
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            X: Transformed data
            
        Returns:
            Data in original scale
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._preprocessor.inverse_transform(np.asarray(X))


class TrainEngine:
    """ML training engine with cross-validation support.
    
    Supports training various ML models with automatic cross-validation
    and hyperparameter optimization.
    
    Example:
        >>> engine = TrainEngine(task_type="classification")
        >>> result = engine.train(X, y)
        >>> print(f"Best score: {result['best_score']}")
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        cv_folds: int = 5,
        metric: str = "accuracy",
        random_state: Optional[int] = None,
    ):
        """Initialize the training engine.
        
        Args:
            task_type: "classification" or "regression"
            cv_folds: Number of cross-validation folds
            metric: Evaluation metric ("accuracy", "auc", "mse", "r2", etc.)
            random_state: Random seed for reproducibility
        """
        self._engine = _rust.PyTrainEngine(
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
        """Train a model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with training results including metrics and model
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        result = self._engine.train(X, y, model_type, model_params or {})
        return result
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
    ) -> Dict[str, Any]:
        """Perform cross-validation without final model training.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to evaluate
            
        Returns:
            Cross-validation results including fold scores
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        return self._engine.cross_validate(X, y, model_type)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        return self._engine.predict(np.asarray(X, dtype=np.float64))


class InferenceEngine:
    """High-performance inference engine for batch predictions.
    
    Example:
        >>> engine = InferenceEngine()
        >>> engine.load_model("model.bin")
        >>> predictions = engine.predict(X)
    """
    
    def __init__(self):
        """Initialize the inference engine."""
        self._engine = _rust.PyInferenceEngine()
    
    def load_model(self, path: str) -> None:
        """Load a trained model from file.
        
        Args:
            path: Path to model file
        """
        self._engine.load_model(path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        return self._engine.predict(np.asarray(X, dtype=np.float64))
    
    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Make predictions in batches.
        
        Args:
            X: Features to predict
            batch_size: Size of each batch
            
        Returns:
            Predictions
        """
        return self._engine.predict_batch(np.asarray(X, dtype=np.float64), batch_size)


class HyperOptX:
    """Hyperparameter optimization engine.
    
    Supports TPE, Random, and Grid search with early stopping.
    
    Example:
        >>> optimizer = HyperOptX(search_space={
        ...     "learning_rate": ("float", 0.001, 0.1, True),  # log scale
        ...     "n_estimators": ("int", 10, 200),
        ... })
        >>> best_params = optimizer.optimize(objective_fn, n_trials=100)
    """
    
    def __init__(
        self,
        search_space: Dict[str, tuple],
        sampler: str = "tpe",
        direction: str = "minimize",
        seed: Optional[int] = None,
    ):
        """Initialize the optimizer.
        
        Args:
            search_space: Dictionary of parameter definitions
            sampler: Sampling strategy ("tpe", "random", "grid")
            direction: Optimization direction ("minimize" or "maximize")
            seed: Random seed for reproducibility
        """
        self._optimizer = _rust.PyHyperOptX(
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
        """Run hyperparameter optimization.
        
        Args:
            objective: Objective function to optimize
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            
        Returns:
            Dictionary with best parameters and value
        """
        return self._optimizer.optimize(objective, n_trials, timeout)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found.
        
        Returns:
            Best parameter values
        """
        return self._optimizer.get_best_params()
    
    def get_trials(self) -> List[Dict[str, Any]]:
        """Get all trial results.
        
        Returns:
            List of trial results
        """
        return self._optimizer.get_trials()


# Data I/O functions

def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file using Polars (fast).
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments (delimiter, has_header, etc.)
        
    Returns:
        Pandas DataFrame
    """
    return _rust.load_csv(path, **kwargs)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file using Polars (fast).
    
    Args:
        path: Path to Parquet file
        
    Returns:
        Pandas DataFrame
    """
    return _rust.load_parquet(path)


def save_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save a DataFrame to CSV using Polars (fast).
    
    Args:
        df: DataFrame to save
        path: Output path
        **kwargs: Additional arguments
    """
    _rust.save_csv(df, path, **kwargs)


def get_file_info(path: str) -> Dict[str, Any]:
    """Get information about a data file.
    
    Args:
        path: Path to data file
        
    Returns:
        Dictionary with file information (rows, columns, dtypes, etc.)
    """
    return _rust.get_file_info(path)
