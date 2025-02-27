#!/usr/bin/env python3
import logging
import time
from typing import Callable, Dict, Any, Optional, List, Tuple, Type
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score

# Import CPUAcceleratedModel and its configuration.
try:
    from .engine.inference_engine import CPUAcceleratedModel
except ImportError:
    from modules.engine.inference_engine import CPUAcceleratedModel

try:
    from .configs import CPUAcceleratedModelConfig
except ImportError:
    from configs import CPUAcceleratedModelConfig

logger = logging.getLogger(__name__)

class DefaultEstimator:
    """
    A simple estimator that wraps scikit-learn's LogisticRegression.
    This is just an example and can be replaced with any estimator that
    implements fit, predict, and score methods.
    """
    def __init__(self, **kwargs):
        # You can pass any valid LogisticRegression parameters via kwargs.
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)

class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for a CPUAcceleratedModel using random search.
    This version directly instantiates CPUAcceleratedModel with a configuration
    derived from the sampled hyperparameters.
    
    Features:
      - Time budget control and early stopping.
      - Optional multi-model support.
      - Heuristically computes the search space volume.
    """

    def __init__(
        self,
        model_class: Type[CPUAcceleratedModel],
        hyperparameter_space: Dict[str, Any],
        scoring_function: Optional[Callable[[CPUAcceleratedModel, NDArray[np.float32], NDArray[Any]], float]] = None,
        max_evals: int = 50,
        n_folds: int = 3,
        seed: int = 42,
        loss_threshold: float = 0.01,
        max_runtime: Optional[float] = None,
        early_stop_rounds: int = 10,
        model_candidates: Optional[Dict[str, Type[CPUAcceleratedModel]]] = None,
        task: str = 'classification'
    ) -> None:
        self.model_class = model_class
        self.hyperparameter_space = hyperparameter_space
        self.scoring_function = scoring_function
        self.max_evals = max_evals
        self.n_folds = n_folds
        self.seed = seed
        self.loss_threshold = loss_threshold
        self.max_runtime = max_runtime
        self.early_stop_rounds = early_stop_rounds
        self.model_candidates = model_candidates
        self.task = task

        self.best_model: Optional[CPUAcceleratedModel] = None
        self.best_params: Optional[Dict[str, Any]] = None

        # These will be set when fit() is called.
        self.X: Optional[NDArray[np.float32]] = None
        self.y: Optional[NDArray[Any]] = None

        # Random number generator.
        self.rng = np.random.default_rng(self.seed)

        # Compute search space "area" (volume).
        self.search_space_area = self._compute_search_space_area()

    def _compute_search_space_area(self) -> float:
        area = 1.0
        for key, val in self.hyperparameter_space.items():
            try:
                if isinstance(val, tuple) and len(val) == 2:
                    low, high = val
                    area *= (high - low)
                elif isinstance(val, list):
                    area *= len(val)
                else:
                    area *= 1
            except Exception as e:
                logger.warning(f"Could not compute extent for parameter '{key}': {e}")
                area *= 1
        return area

    def _sample_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, val in self.hyperparameter_space.items():
            if isinstance(val, list):
                params[key] = self.rng.choice(val)
            elif isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                low, high = val
                if isinstance(low, int) and isinstance(high, int):
                    # For integer parameters
                    params[key] = self.rng.integers(low, high + 1)  # +1 to include upper bound
                else:
                    # For float parameters
                    params[key] = self.rng.uniform(low, high)
            else:
                params[key] = val
        return params

    def _objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Build a CPUAcceleratedModelConfig from the sampled parameters.
            config = CPUAcceleratedModelConfig(**params)
            # Instantiate a CPUAcceleratedModel using the config.
            model = self.model_class(config=config)
            start_time = time.time()
            model.fit(self.X, self.y)
            fit_time = time.time() - start_time

            # Evaluate using the provided scoring_function or cross-validation on model.estimator.
            if self.scoring_function:
                score = self.scoring_function(model, self.X, self.y)
            else:
                # Use model's score method if it exists, otherwise fall back to cross-validation
                if hasattr(model, 'score'):
                    score = model.score(self.X, self.y)
                elif hasattr(model, 'estimator'):
                    scores = cross_val_score(model.estimator, self.X, self.y, cv=self.n_folds, 
                                            scoring='accuracy' if self.task == 'classification' else 'r2')
                    score = np.mean(scores)
                else:
                    raise ValueError("Model has neither 'score' method nor 'estimator' attribute")
            
            loss = 1.0 - score

            # Gather additional metrics.
            metrics = {}
            if hasattr(model, 'get_metrics'):
                metrics = model.get_metrics()
            metrics['fit_time'] = fit_time
            metrics['score'] = score

            return {'loss': loss, 'status': "ok", 'metrics': metrics}
        except Exception as e:
            logger.exception(f"Error during hyperparameter evaluation: {e}")
            return {'loss': float('inf'), 'status': "fail", 'exception': str(e)}

    def _optimize_single(self) -> Dict[str, Any]:
        start_time = time.time()
        best_loss = float('inf')
        best_params: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None
        best_model_instance: Optional[CPUAcceleratedModel] = None
        trials = []
        no_improve_counter = 0

        for eval_count in range(1, self.max_evals + 1):
            if self.max_runtime is not None and (time.time() - start_time) > self.max_runtime:
                logger.info("Time budget exceeded. Stopping optimization.")
                break

            params = self._sample_params()
            result = self._objective(params)
            trials.append({'params': params, 'result': result})
            if result['status'] != "ok":
                continue

            current_loss = result['loss']
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params
                best_metrics = result.get('metrics', {})
                # Instantiate and re-fit the best model using the sampled hyperparameters.
                config = CPUAcceleratedModelConfig(**params)
                best_model_instance = self.model_class(config=config)
                best_model_instance.fit(self.X, self.y)
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= self.early_stop_rounds:
                logger.info("Early stopping triggered due to no improvement.")
                break

        if best_model_instance is None:
            logger.warning("No valid model was found during optimization. Falling back to default configuration.")
            default_config = CPUAcceleratedModelConfig()
            best_model_instance = self.model_class(
                config=default_config,
                estimator_class=DefaultEstimator
            )
            best_model_instance.fit(self.X, self.y)
            
            # Evaluate the fallback model
            if hasattr(best_model_instance, 'score'):
                score = best_model_instance.score(self.X, self.y)
            else:
                score = 0.5  # Default score value
            best_loss = 1.0 - score
            
            best_params = {}
            best_metrics = {}
            if hasattr(best_model_instance, 'get_metrics'):
                best_metrics = best_model_instance.get_metrics()
            best_metrics['score'] = score

        logger.info(f"Best hyperparameters found: {best_params}")
        self.best_model = best_model_instance
        self.best_params = best_params

        return {
            'best_hyperparameters': best_params,
            'best_loss': best_loss,
            'best_score': 1.0 - best_loss,
            'trials': trials,
            'best_metrics': best_metrics
        }

    def fit(self, X: NDArray[np.float32], y: NDArray[Any]) -> Dict[str, Any]:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Training data X and y must be NumPy arrays.")
        self.X = X
        self.y = y

        if self.model_candidates is not None:
            candidate_results = {}
            best_overall: Optional[Dict[str, Any]] = None
            best_candidate: Optional[str] = None

            for name, candidate_model in self.model_candidates.items():
                logger.info(f"Optimizing candidate model: {name}")
                self.model_class = candidate_model
                result = self._optimize_single()
                candidate_results[name] = result
                if best_overall is None or result['best_loss'] < best_overall['best_loss']:
                    best_overall = result
                    best_candidate = name

            if best_candidate is not None:
                logger.info(f"Best candidate model: {best_candidate} with hyperparameters: {candidate_results[best_candidate]['best_hyperparameters']}")
                best_overall['best_model_candidate'] = best_candidate
                return best_overall
            else:
                logger.warning("No valid candidate model found")
                return self._optimize_single()  # Fallback to default optimization
        else:
            return self._optimize_single()

    def get_best_model(self) -> CPUAcceleratedModel:
        if self.best_model is None:
            raise ValueError("No model found. Have you called `fit`?")
        return self.best_model

    def get_best_params(self) -> Dict[str, Any]:
        if self.best_params is None:
            raise ValueError("No hyperparameters found. Have you called `fit`?")
        return self.best_params


# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(100, 9).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=100)  # Binary classification target

    optimizer = HyperparameterOptimizer(
        model_class=CPUAcceleratedModel,
        hyperparameter_space={'alpha': (0.0, 10.0)},
        max_evals=20,
        seed=42
    )
    result = optimizer.fit(X_dummy, y_dummy)
    best_model = optimizer.get_best_model()
    predictions = best_model.predict(np.random.randn(10, 9).astype(np.float32))
    logger.info("Predictions: %s", predictions)
