#!/usr/bin/env python3

import logging
import time
from typing import Callable, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score

# Assuming CPUAcceleratedModel and ModelConfig are in the same directory
try:
    from .engine.engine import CPUAcceleratedModel
except ImportError:
    from engine.engine import CPUAcceleratedModel

from .configs import ModelConfig

# For ensemble majority voting (if classification)
from scipy.stats import mode

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    A simple ensemble that aggregates predictions from several models.
    For classification, it uses majority vote; for regression, it averages predictions.
    """
    def __init__(self, models: list, task: str = 'classification'):
        self.models = models
        self.task = task

    def predict(self, X: NDArray[np.float32]) -> NDArray:
        predictions = [model.predict(X) for model in self.models]
        predictions = np.array(predictions)
        if self.task == 'classification':
            # Majority vote along the first axis
            maj_vote, _ = mode(predictions, axis=0, keepdims=False)
            return maj_vote
        else:
            # For regression, take the mean prediction
            return np.mean(predictions, axis=0)


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for a CPUAcceleratedModel using Hyperopt,
    with additional features inspired by FLAML:
      - Time budget control and early stopping.
      - Option to search over multiple candidate model classes.
      - Optional ensemble of top-performing models.
    """

    def __init__(
        self,
        model_class: type[CPUAcceleratedModel],
        hyperparameter_space: Dict[str, Any],
        scoring_function: Optional[Callable[[CPUAcceleratedModel, NDArray[np.float32], NDArray[Any]], float]] = None,
        max_evals: int = 50,
        n_folds: int = 3,
        seed: int = 42,
        loss_threshold: float = 0.01,
        max_runtime: Optional[float] = None,          # maximum runtime (seconds) for optimization
        early_stop_rounds: int = 10,                  # stop if no improvement after these many evaluations
        model_candidates: Optional[Dict[str, type[CPUAcceleratedModel]]] = None,  # Optional multi-model support
        ensemble_size: int = 0,                       # If > 0, store evaluated models for ensembling
        task: str = 'classification'                  # 'classification' or 'regression'
    ):
        """
        Initializes the HyperparameterOptimizer. Training data is *not* passed here; 
        call `fit(X, y)` later to run hyperparameter optimization.

        Args:
            model_class: The class of the CPUAcceleratedModel to optimize (if single model).
            hyperparameter_space: A dictionary defining the hyperparameter search space for Hyperopt.
            scoring_function: Optional callable that takes a model, X, and y and returns a score.
                              If None, cross-validation (accuracy) is used by default.
            max_evals: Maximum number of evaluations for Hyperopt.
            n_folds: Number of cross-validation folds if scoring_function is None.
            seed: Random seed for reproducibility.
            loss_threshold: A threshold for the loss metric (if you have a custom stopping criterion).
            max_runtime: Maximum runtime (in seconds) for the entire optimization.
            early_stop_rounds: Number of evaluations with no improvement after which to stop.
            model_candidates: Optional dict mapping candidate names to model classes for model selection.
            ensemble_size: If > 0, the top ensemble_size models will be stored for ensembling.
            task: The problem type, 'classification' or 'regression'.
        """
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
        self.ensemble_size = ensemble_size
        self.task = task

        self.best_model = None   # Store the best model
        self.best_params = None  # Store the best hyperparams

        # To store intermediate models (for ensemble if desired)
        # Each entry: (model, loss, params, metrics)
        self.model_history: list[tuple[CPUAcceleratedModel, float, Dict[str, Any], Dict[str, Any]]] = []

        # Will be set when `fit` is called
        self.X = None
        self.y = None

    def _objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for Hyperopt. This function trains and evaluates the model
        with the given hyperparameters.

        Args:
            params: A dictionary of hyperparameters to use for training the model.

        Returns:
            A dictionary containing the loss, status, and additional metrics.
        """
        try:
            # Convert hyperopt parameters to ModelConfig
            config = ModelConfig(**params)

            # Instantiate and train the model
            model = self.model_class(estimator_class=None, config=config)
            
            start_time = time.time()
            model.fit(self.X, self.y)
            fit_time = time.time() - start_time

            # Evaluate the model
            if self.scoring_function:
                score = self.scoring_function(model, self.X, self.y)
            else:
                scores = cross_val_score(
                    model.estimator, self.X, self.y, 
                    cv=self.n_folds, scoring='accuracy'
                )
                score = np.mean(scores)
            
            loss = 1.0 - score  # Hyperopt minimizes, so use 1 - score as the 'loss'

            # Collect metrics
            metrics = model.get_metrics()
            metrics['fit_time'] = fit_time
            metrics['score'] = score

            # Optionally store model history for ensembling
            if self.ensemble_size > 0:
                self.model_history.append((model, loss, params, metrics))

            return {'loss': loss, 'status': STATUS_OK, 'metrics': metrics}

        except Exception as e:
            logger.exception(f"Error during hyperparameter evaluation: {e}")
            return {'loss': float('inf'), 'status': STATUS_FAIL, 'exception': str(e)}

    def _optimize_single(self) -> Dict[str, Any]:
        """
        Runs the hyperparameter optimization loop for the current model_class
        with early stopping based on a time budget and lack of improvement.
        """
        start_time = time.time()
        trials = Trials()
        best_loss = float('inf')
        no_improve_counter = 0
        eval_count = 0

        best = None  # Will hold the best result from fmin
        while eval_count < self.max_evals:
            new_eval = eval_count + 1
            best = fmin(
                fn=self._objective,
                space=self.hyperparameter_space,
                algo=tpe.suggest,
                max_evals=new_eval,
                trials=trials,
                rstate=np.random.default_rng(self.seed)
            )
            eval_count = len(trials.trials)
            current_best_loss = min(trial['result']['loss'] for trial in trials.trials)

            if current_best_loss < best_loss:
                best_loss = current_best_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            # Early stopping if no improvement
            if no_improve_counter >= self.early_stop_rounds:
                logger.info("Early stopping triggered due to no improvement.")
                break

            # Stop if max runtime is exceeded
            if self.max_runtime is not None and (time.time() - start_time) > self.max_runtime:
                logger.info("Time budget exceeded. Stopping optimization.")
                break

        best_hyperparams = space_eval(self.hyperparameter_space, best)
        logger.info(f"Best hyperparameters found: {best_hyperparams}")

        # Retrieve best loss and corresponding trial
        best_loss = min(trial['result']['loss'] for trial in trials.trials)
        best_trial = next(trial for trial in trials.trials if trial['result']['loss'] == best_loss)

        # Instantiate and fit the best model on the full data
        best_config = ModelConfig(**best_hyperparams)
        self.best_model = self.model_class(estimator_class=None, config=best_config)
        self.best_model.fit(self.X, self.y)
        self.best_params = best_hyperparams

        return {
            'best_hyperparameters': best_hyperparams,
            'best_loss': best_loss,
            'best_score': 1.0 - best_loss,
            'trials': trials,
            'best_metrics': best_trial['result']['metrics']
        }

    def fit(self, X: NDArray[np.float32], y: NDArray[Any]) -> Dict[str, Any]:
        """
        Performs hyperparameter optimization. If multiple model candidates are
        provided, each candidate is optimized and the one with the best performance
        is selected. Otherwise, a single model is optimized.

        Args:
            X: Training features
            y: Training labels (classification or regression)

        Returns:
            A dictionary containing the best hyperparameters found, the best score,
            and the optimization trials. If model candidates were used, the best 
            candidate name is also included.
        """
        # Store the data for internal usage
        self.X = X
        self.y = y

        if self.model_candidates is not None:
            candidate_results = {}
            best_overall = None
            best_candidate = None

            for name, candidate_model in self.model_candidates.items():
                logger.info(f"Optimizing candidate model: {name}")
                self.model_class = candidate_model  # update current candidate
                # Reset model history for this candidate
                self.model_history = []
                result = self._optimize_single()
                candidate_results[name] = result

                if best_overall is None or result['best_loss'] < best_overall['best_loss']:
                    best_overall = result
                    best_candidate = name

            logger.info(
                f"Best candidate model: {best_candidate} "
                f"with hyperparameters: {candidate_results[best_candidate]['best_hyperparameters']}"
            )
            best_overall['best_model_candidate'] = best_candidate
            return best_overall
        else:
            return self._optimize_single()

    def get_best_model(self) -> CPUAcceleratedModel:
        """
        Returns the best trained model found during optimization.
        """
        if self.best_model is None:
            raise ValueError("No model found. Have you called `fit`?")
        return self.best_model

    def get_best_params(self) -> Dict[str, Any]:
        """
        Returns the best hyperparameters found during optimization.
        """
        if self.best_params is None:
            raise ValueError("No hyperparameters found. Have you called `fit`?")
        return self.best_params

    def get_ensemble(self) -> EnsembleModel:
        """
        Returns an ensemble model built from the top-performing evaluated models.
        
        Raises:
            ValueError: If ensemble_size is not set or no models were stored.
        """
        if self.ensemble_size > 0 and self.model_history:
            # Sort stored models by loss (ascending order)
            sorted_models = sorted(self.model_history, key=lambda x: x[1])
            best_models = [entry[0] for entry in sorted_models[:self.ensemble_size]]
            return EnsembleModel(best_models, task=self.task)
        else:
            raise ValueError(
                "No ensemble available. Set ensemble_size > 0 and ensure "
                "that models were evaluated during optimization."
            )
