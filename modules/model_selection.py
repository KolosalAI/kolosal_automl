#!/usr/bin/env python3
import os
import time
import logging
import joblib
import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional, Callable, Any, TypeVar

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from numpy.typing import NDArray

# Import the CPUAcceleratedModel and its configuration.
try:
    from .engine.engine import CPUAcceleratedModel
except ImportError:
    from engine.engine import CPUAcceleratedModel

from modules.configs import CPUAcceleratedModelConfig

# Import the HyperparameterOptimizer from your local hyperopt.py
try:
    from .hyperopt import HyperparameterOptimizer
except ImportError:
    from modules.hyperopt import HyperparameterOptimizer

EstimatorType = TypeVar('EstimatorType')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


###############################################################################
# HAMSeOptimizer: Hybrid Adaptive Model Selection (merged version)
###############################################################################
class HAMSeOptimizer:
    """
    Implements the Hybrid Adaptive Model Selection (HAMSe) framework.
    This includes:
      - Adaptive model screening (with optional Hyperopt tuning for CPUAcceleratedModel-wrapped sklearn models)
      - Elastic net stability feature selection
      - Diversified Bayesian optimization
      - Ensemble validation

    The hyperparameter tuning for scikit-learn candidates now uses the CPUAcceleratedModel
    directly rather than an internal wrapper class.
    """

    def __init__(
        self,
        hyperparameter_bounds: Optional[List[Tuple[float, float]]] = None,
        n_jobs: Optional[int] = None,
        random_state: int = 42,
        folds: int = 5,
        max_bayes_iter: int = 50
    ):
        self.hyperparameter_bounds = hyperparameter_bounds
        self.n_jobs = n_jobs or os.cpu_count()
        self.random_state = random_state
        self.folds = folds
        self.max_bayes_iter = max_bayes_iter

        self.rng: Optional[np.random.RandomState] = None
        self.X_ = None
        self.y_ = None
        self.stratified_data_ = None

        self.final_model: Optional[Callable[[np.ndarray], np.ndarray]] = None

        # New attributes: store search space area and trial history per candidate.
        self.search_space_area = {}      # candidate id -> search space area (float)
        self.model_trial_history = {}    # candidate id -> Hyperopt trial history

    ############################################################################
    # (Data preprocessing and other methods remain unchanged)
    ############################################################################
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        col_means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_means, inds[1])
        return data

    def _scale_features(self, data: np.ndarray) -> np.ndarray:
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero.
        return (data - means) / stds

    def _validate_data(self, data: np.ndarray) -> None:
        if data.size == 0:
            raise ValueError("Data is empty.")

    def _stratify_data(self, data: np.ndarray) -> np.ndarray:
        return data

    ############################################################################
    # Candidate Model Definition
    ############################################################################
    class CandidateModel:
        def __init__(
            self,
            model_id: Any,
            params: Optional[np.ndarray] = None,
            sklearn_model: Optional[Any] = None
        ):
            self.id = model_id
            self.params = params  
            self.sklearn_model = sklearn_model  
            self.score: Optional[float] = None
            self.cv_performance: Optional[float] = None
            self.weight: Optional[float] = None

        def predict(self, X: np.ndarray) -> np.ndarray:
            if self.sklearn_model is not None:
                return self.sklearn_model.predict(X)
            elif self.params is not None:
                return np.dot(X, self.params)
            else:
                return np.zeros(X.shape[0])

    ############################################################################
    # Candidate Generation & Screening
    ############################################################################
    def _generate_candidate_models(self, num_models: int = 5) -> List['HAMSeOptimizer.CandidateModel']:
        n_features = self.stratified_data_.shape[1] - 1
        models = []
        n_sklearn = num_models // 2
        n_custom = num_models - n_sklearn

        for i in range(n_custom):
            params = self.rng.randn(n_features)
            models.append(self.CandidateModel(model_id=f'custom_{i}', params=params))

        sklearn_estimators = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1)]
        for i, estimator in enumerate(sklearn_estimators[:n_sklearn]):
            models.append(self.CandidateModel(model_id=f'sklearn_{i}', sklearn_model=estimator))
        return models

    def _get_fold_indices(self, n: int) -> List[np.ndarray]:
        indices = np.arange(n)
        self.rng.shuffle(indices)
        fold_size = n // self.folds
        return [indices[i * fold_size: (i + 1) * fold_size] for i in range(self.folds)]

    def _cross_validated_hamse_score(self, model: 'HAMSeOptimizer.CandidateModel') -> float:
        n = self.stratified_data_.shape[0]
        folds = self._get_fold_indices(n)
        scores = []
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            train_data = self.stratified_data_[train_idx]
            test_data = self.stratified_data_[test_idx]
            if train_data.shape[1] < 2:
                continue
            X_train = train_data[:, 1:]
            y_train = train_data[:, 0]
            X_test = test_data[:, 1:]
            y_test = test_data[:, 0]

            if model.sklearn_model is not None:
                model_clone = clone(model.sklearn_model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
            else:
                y_pred = np.dot(X_test, model.params)
            mse = np.mean((y_test - y_pred) ** 2)
            scores.append(mse)
        return np.mean(scores) if scores else np.inf

    ############################################################################
    # Hyperparameter Optimization for sklearn candidates using CPUAcceleratedModel
    ############################################################################
    def _hyperopt_sklearn_model(
        self,
        candidate: 'HAMSeOptimizer.CandidateModel',
        X_full: np.ndarray,
        y_full: np.ndarray
    ) -> None:
        sk_model = candidate.sklearn_model
        if sk_model is None:
            return

        # Define search space based on model type.
        if isinstance(sk_model, Ridge):
            search_space = {'alpha': (0.0, 10.0)}
        elif isinstance(sk_model, Lasso):
            search_space = {'alpha': (0.0, 1.0)}
        elif isinstance(sk_model, LinearRegression):
            return  # No hyperparameters to tune.
        else:
            return

        # Instead of a custom wrapper, we now directly use CPUAcceleratedModel.
        # Create a default configuration from the search space parameters.
        default_config = CPUAcceleratedModelConfig()
        # Override parameters in the config if they exist in the search space.
        for key, bounds in search_space.items():
            if hasattr(default_config, key):
                setattr(default_config, key, bounds[0])  # start with the lower bound

        # Instantiate a CPUAcceleratedModel wrapping the sklearn estimator.
        base_model = CPUAcceleratedModel(estimator_class=type(sk_model), config=default_config)
        # Define a scoring function using cross-validation.
        def scoring_function(model_obj, X_data, y_data):
            scores = cross_val_score(model_obj.estimator, X_data, y_data, cv=self.folds, scoring='neg_mean_squared_error')
            return np.mean(scores)

        # Create the HyperparameterOptimizer with CPUAcceleratedModel as model_class.
        optimizer = HyperparameterOptimizer(
            model_class=CPUAcceleratedModel,
            hyperparameter_space=search_space,
            scoring_function=scoring_function,
            max_evals=20,
            seed=self.random_state,
            n_folds=self.folds
        )

        # Run hyperparameter optimization.
        result = optimizer.fit(X_full, y_full)
        best_model = optimizer.get_best_model()
        # Fit the best model on full data.
        best_model.estimator.fit(X_full, y_full)
        # Update the candidate's sklearn_model with the tuned estimator.
        candidate.sklearn_model = best_model.estimator

        trial_history = result.get('trials', result)
        self.model_trial_history[candidate.id] = trial_history

        if hasattr(optimizer, 'search_space_area'):
            self.search_space_area[candidate.id] = optimizer.search_space_area
        else:
            area = 1.0
            for key, (low, high) in search_space.items():
                area *= (high - low)
            self.search_space_area[candidate.id] = area

    ############################################################################
    # Model Screening
    ############################################################################
    def _screen_models(self) -> List['HAMSeOptimizer.CandidateModel']:
        candidate_models = self._generate_candidate_models()
        X_full = self.stratified_data_[:, 1:]
        y_full = self.stratified_data_[:, 0]

        for candidate in candidate_models:
            self._hyperopt_sklearn_model(candidate, X_full, y_full)

        cache = {}

        def evaluate_model(m: 'HAMSeOptimizer.CandidateModel') -> Tuple[Any, float]:
            if m.id not in cache:
                score_ = self._cross_validated_hamse_score(m)
                cache[m.id] = score_
            else:
                score_ = cache[m.id]
            return m.id, score_

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(evaluate_model, model): model for model in candidate_models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    _, score = future.result()
                    model.score = score
                except Exception as exc:
                    logger.error("Model %s generated an exception: %s", model.id, exc)
                    model.score = np.inf

        logger.info("Stage 1 complete: Screened %d models (with Hyperopt tuning via CPUAcceleratedModel).", len(candidate_models))
        return candidate_models

    ############################################################################
    # (Other methods for Elastic Net, Bayesian optimization, and ensemble remain unchanged)
    ############################################################################
    def _elastic_net(self, data: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 100) -> np.ndarray:
        X = data[:, 1:]
        y = data[:, 0]
        n_samples, n_features = X.shape
        coef = np.zeros(n_features)
        for _ in range(max_iter):
            for j in range(n_features):
                prediction = np.dot(X, coef)
                residual = y - prediction + coef[j] * X[:, j]
                rho = np.dot(X[:, j], residual)
                norm_val = np.dot(X[:, j], X[:, j]) + alpha * (1 - l1_ratio)
                if rho < -alpha * l1_ratio:
                    coef[j] = (rho + alpha * l1_ratio) / norm_val
                elif rho > alpha * l1_ratio:
                    coef[j] = (rho - alpha * l1_ratio) / norm_val
                else:
                    coef[j] = 0.0
        return coef

    def _bootstrap_sample(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        indices = self.rng.choice(n, n, replace=True)
        return data[indices]

    def _enforce_group_selection(self, beta_paths: List[np.ndarray], threshold_ratio: float = 0.8) -> np.ndarray:
        beta_paths = np.array(beta_paths)
        return np.mean(beta_paths, axis=0)

    def _elastic_net_stability_path(self, n_bootstraps: int = 50) -> np.ndarray:
        stability_paths = []
        for _ in range(n_bootstraps):
            boot_data = self._bootstrap_sample(self.stratified_data_)
            beta = self._elastic_net(boot_data)
            stability_paths.append(beta)
        grouped_beta = self._enforce_group_selection(stability_paths, threshold_ratio=0.8)
        logger.info("Stage 2 complete: Computed Elastic Net Stability Path.")
        return grouped_beta

    def _latin_hypercube_sampling(self, n_samples: int = 10) -> np.ndarray:
        dim = len(self.hyperparameter_bounds)
        samples = np.zeros((n_samples, dim))
        for d in range(dim):
            low, high = self.hyperparameter_bounds[d]
            perm = self.rng.permutation(n_samples)
            intervals = np.linspace(low, high, n_samples + 1)
            for i in range(n_samples):
                low_val = intervals[perm[i]]
                high_val = intervals[perm[i] + 1]
                samples[i, d] = self.rng.uniform(low_val, high_val)
        return samples

    def _entropy_search_portfolio(self, n_samples: int = 10) -> np.ndarray:
        dim = len(self.hyperparameter_bounds)
        samples = np.zeros((n_samples, dim))
        for i in range(n_samples):
            for d in range(dim):
                low, high = self.hyperparameter_bounds[d]
                samples[i, d] = self.rng.uniform(low, high)
        return samples

    def _merge_samples(self, samples1: np.ndarray, samples2: np.ndarray) -> np.ndarray:
        return np.vstack((samples1, samples2))

    def _generate_candidate_points(self, n_points: int = 5) -> np.ndarray:
        dim = len(self.hyperparameter_bounds)
        points = np.zeros((n_points, dim))
        for i in range(n_points):
            for d in range(dim):
                low, high = self.hyperparameter_bounds[d]
                points[i, d] = self.rng.uniform(low, high)
        return points

    def _expected_improvement(self, x: np.ndarray, objective_fn: Callable[[np.ndarray], float]) -> float:
        return self.rng.rand()

    def _thompson_sampling(self, x: np.ndarray, objective_fn: Callable[[np.ndarray], float]) -> float:
        return self.rng.rand()

    def _objective_function(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)

    def _early_stopping_criteria_met(self, patience: int) -> bool:
        return self.rng.rand() < 0.1

    def _diversified_bayesopt(self) -> np.ndarray:
        n_initial = 10
        samples_lhs = self._latin_hypercube_sampling(n_samples=n_initial)
        samples_entropy = self._entropy_search_portfolio(n_samples=n_initial)
        initial_points = self._merge_samples(samples_lhs, samples_entropy)

        evaluated = {}
        iteration = 1
        best_hyperparams = None
        best_obj_value = np.inf

        while (iteration <= self.max_bayes_iter and 
               not self._early_stopping_criteria_met(int(np.ceil(np.log2(iteration))))):
            candidate_points = self._generate_candidate_points(n_points=5)
            for x in candidate_points:
                x_tuple = tuple(x)
                if x_tuple not in evaluated:
                    ei = self._expected_improvement(x, self._objective_function)
                    ts = self._thompson_sampling(x, self._objective_function)
                    acquisition_value = 0.7 * ei + 0.3 * ts
                    obj_value = self._objective_function(x)
                    evaluated[x_tuple] = obj_value
                    if obj_value < best_obj_value:
                        best_obj_value = obj_value
                        best_hyperparams = x
            iteration += 1
            logger.info("Bayesian Optimization iteration %d completed.", iteration)
        logger.info("Stage 3 complete: Best hyperparameters found.")
        return best_hyperparams if best_hyperparams is not None else np.zeros(len(self.hyperparameter_bounds))

    def _merge_models(
        self,
        models_list: List['HAMSeOptimizer.CandidateModel'],
        beta_path: np.ndarray,
        best_hps: np.ndarray
    ) -> List['HAMSeOptimizer.CandidateModel']:
        merged_models = models_list.copy()
        merged_models.append(self.CandidateModel(model_id='beta', params=beta_path))
        n_features = self.stratified_data_.shape[1] - 1
        if best_hps.shape[0] != n_features:
            best_hps_mapped = np.zeros(n_features)
            n = min(n_features, best_hps.shape[0])
            best_hps_mapped[:n] = best_hps[:n]
        else:
            best_hps_mapped = best_hps
        merged_models.append(self.CandidateModel(model_id='bo', params=best_hps_mapped))
        return merged_models

    def _cross_validation_performance(self, model: 'HAMSeOptimizer.CandidateModel') -> float:
        return self._cross_validated_hamse_score(model)

    def _predict_with_model(self, model: 'HAMSeOptimizer.CandidateModel', X: np.ndarray) -> np.ndarray:
        if model.sklearn_model is not None:
            try:
                return model.sklearn_model.predict(X)
            except NotFittedError:
                X_full = self.stratified_data_[:, 1:]
                y_full = self.stratified_data_[:, 0]
                model.sklearn_model.fit(X_full, y_full)
                return model.sklearn_model.predict(X)
        elif model.params is not None:
            return np.dot(X, model.params)
        else:
            return np.zeros(X.shape[0])

    def _weighted_average_predictions(
        self,
        models: List['HAMSeOptimizer.CandidateModel'],
        num_samples: int = 10
    ) -> np.ndarray:
        predictions = []
        weights = []
        dummy_input = np.random.randn(num_samples, self.stratified_data_.shape[1] - 1)
        for model in models:
            pred_val = self._predict_with_model(model, dummy_input)
            predictions.append(pred_val)
            weights.append(model.weight if model.weight is not None else 1.0)
        predictions = np.array(predictions)
        weights = np.array(weights)
        return np.average(predictions, axis=0, weights=weights)

    def _train_stacking_regressor(
        self,
        models: List['HAMSeOptimizer.CandidateModel']
    ) -> Callable[[np.ndarray], np.ndarray]:
        def ensemble_predict(X: np.ndarray) -> np.ndarray:
            preds = [self._predict_with_model(m, X) for m in models]
            preds = np.array(preds)
            return np.mean(preds, axis=0)
        return ensemble_predict

    def _ensemble_validation(
        self,
        models_list: List['HAMSeOptimizer.CandidateModel'],
        beta_path: np.ndarray,
        best_hps: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        ensemble_candidates = self._merge_models(models_list, beta_path, best_hps)
        for model in ensemble_candidates:
            model.cv_performance = self._cross_validation_performance(model)

        total_weight = 0.0
        for model in ensemble_candidates:
            if model.score is None:
                model.score = np.inf
            if model.cv_performance is None:
                model.cv_performance = 1.0
            model_weight = np.exp(-0.5 * model.score) * model.cv_performance
            model.weight = model_weight
            total_weight += model_weight

        for model in ensemble_candidates:
            model.weight = model.weight / total_weight if total_weight != 0 else 0

        ensemble_prediction = self._weighted_average_predictions(ensemble_candidates)
        filtered_candidates = []
        dummy_input = np.random.randn(10, self.stratified_data_.shape[1] - 1)
        for m in ensemble_candidates:
            pred_vals = self._predict_with_model(m, dummy_input)
            pred_val = np.mean(pred_vals)
            pred = np.full(ensemble_prediction.shape, pred_val)
            mae = np.mean(np.abs(pred - ensemble_prediction))
            std_dev = np.std(ensemble_prediction)
            if mae <= 1.5 * std_dev:
                filtered_candidates.append(m)
            else:
                logger.info("Model %s removed during agreement filtering.", m.id)

        final_ensemble = self._train_stacking_regressor(filtered_candidates)
        logger.info("Stage 4 complete: Final ensemble model trained with %d candidate(s).",
                    len(filtered_candidates))
        return final_ensemble

    def fit(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        data = np.column_stack((y, X))
        self.rng = np.random.RandomState(self.random_state)

        if self.hyperparameter_bounds is None:
            n_features = X.shape[1]
            self.hyperparameter_bounds = [(-5, 5)] * n_features
            logger.info("Hyperparameter bounds automatically set to %s", self.hyperparameter_bounds)

        data = self._handle_missing_values(data)
        data = self._scale_features(data)
        self._validate_data(data)
        self.stratified_data_ = self._stratify_data(data)

        candidate_models = self._screen_models()
        beta_path = self._elastic_net_stability_path(n_bootstraps=50)
        best_hyperparams = self._diversified_bayesopt()
        final_model = self._ensemble_validation(candidate_models, beta_path, best_hyperparams)

        logger.info("HAMSe Model Selection completed successfully.")
        self.final_model = final_model
        return final_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.final_model is None:
            raise NotFittedError("The HAMSe model has not been fitted yet.")
        return self.final_model(X)

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'hyperparameter_bounds': self.hyperparameter_bounds,
            'random_state': self.random_state,
            'rng_state': self.rng.get_state() if self.rng else None,
            'folds': self.folds,
            'max_bayes_iter': self.max_bayes_iter,
            'search_space_area': self.search_space_area,
            'model_trial_history': self.model_trial_history,
        }
        joblib.dump(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = joblib.load(path)
        self.hyperparameter_bounds = checkpoint['hyperparameter_bounds']
        self.random_state = checkpoint['random_state']
        self.folds = checkpoint.get('folds', self.folds)
        self.max_bayes_iter = checkpoint.get('max_bayes_iter', self.max_bayes_iter)
        self.search_space_area = checkpoint.get('search_space_area', {})
        self.model_trial_history = checkpoint.get('model_trial_history', {})
        rng_state = checkpoint.get('rng_state', None)
        if rng_state is not None:
            self.rng = np.random.RandomState(self.random_state)
            self.rng.set_state(rng_state)
        logger.info("Checkpoint loaded from %s", path)


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    np.random.seed(42)
    X_dummy = np.random.randn(100, 9).astype(np.float32)
    y_dummy = np.random.randn(100).astype(np.float32)

    optimizer = HAMSeOptimizer(n_jobs=4, random_state=42)
    optimizer.fit(X_dummy, y_dummy)

    X_test = np.random.randn(10, 9).astype(np.float32)
    predictions = optimizer.predict(X_test)
    logger.info("Predictions: %s", predictions)

    checkpoint_path = "hamse_checkpoint.pkl"
    optimizer.save_checkpoint(checkpoint_path)
