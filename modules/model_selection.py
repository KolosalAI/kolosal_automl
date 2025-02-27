#!/usr/bin/env python3
import os
import time
import logging
import joblib
import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional, Callable, Any, TypeVar, Dict, Union
from dataclasses import dataclass, field

# Third-party imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from numpy.typing import NDArray

# Import the CPUAcceleratedModel and its configuration.
try:
    from .engine.inference_engine import CPUAcceleratedModel
except ImportError:
    from modules.engine.inference_engine import CPUAcceleratedModel

try:
    from .configs import CPUAcceleratedModelConfig
except ImportError:
    from modules.configs import CPUAcceleratedModelConfig

# Import the HyperparameterOptimizer from your local hyperopt.py
try:
    from .hyperopt import HyperparameterOptimizer
except ImportError:
    from modules.hyperopt import HyperparameterOptimizer

EstimatorType = TypeVar('EstimatorType', bound=BaseEstimator)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CandidateModel:
    """
    Represents a candidate model in the HAMSe framework.
    
    Attributes:
        id: Unique identifier for the model
        params: NumPy array of model parameters (for custom models)
        sklearn_model: Scikit-learn estimator (for scikit-learn models)
        score: Performance score (lower is better)
        cv_performance: Cross-validation performance
        weight: Weight in the ensemble
    """
    id: Any
    params: Optional[np.ndarray] = None
    sklearn_model: Optional[BaseEstimator] = None
    score: Optional[float] = None
    cv_performance: Optional[float] = None
    weight: Optional[float] = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using this candidate model."""
        if self.sklearn_model is not None:
            return self.sklearn_model.predict(X)
        elif self.params is not None:
            return np.dot(X, self.params)
        else:
            return np.zeros(X.shape[0])


class HAMSeOptimizer:
    """
    Implements the Hybrid Adaptive Model Selection (HAMSe) framework.
    
    This includes:
      - Adaptive model screening (with Hyperopt tuning for CPUAcceleratedModel-wrapped sklearn models)
      - Elastic net stability feature selection
      - Diversified Bayesian optimization
      - Ensemble validation
    
    Parameters:
        hyperparameter_bounds: List of (min, max) tuples for each hyperparameter
        n_jobs: Number of parallel jobs for model evaluation
        random_state: Random seed for reproducibility
        folds: Number of cross-validation folds
        max_bayes_iter: Maximum iterations for Bayesian optimization
        early_stopping_patience: Patience for early stopping
        elastic_net_bootstraps: Number of bootstrap samples for elastic net
        candidate_models_count: Number of candidate models to generate
    """
    def __init__(
        self,
        hyperparameter_bounds: Optional[List[Tuple[float, float]]] = None,
        n_jobs: Optional[int] = None,
        random_state: int = 42,
        folds: int = 5,
        max_bayes_iter: int = 50,
        early_stopping_patience: int = 5,
        elastic_net_bootstraps: int = 50,
        candidate_models_count: int = 5
    ):
        self.hyperparameter_bounds = hyperparameter_bounds
        self.n_jobs = n_jobs or max(1, os.cpu_count() or 1)
        self.random_state = random_state
        self.folds = folds
        self.max_bayes_iter = max_bayes_iter
        self.early_stopping_patience = early_stopping_patience
        self.elastic_net_bootstraps = elastic_net_bootstraps
        self.candidate_models_count = candidate_models_count

        # State variables
        self.rng: Optional[np.random.Generator] = None
        self._feature_scaler: Optional[StandardScaler] = None
        self._target_scaler: Optional[StandardScaler] = None
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.stratified_data_: Optional[np.ndarray] = None
        
        self.final_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.selected_models: List[CandidateModel] = []

        # Search space tracking
        self.search_space_area: Dict[Any, float] = {}
        self.model_trial_history: Dict[Any, Any] = {}
        
        # Optimization trajectory
        self.bayes_opt_history: List[Dict[str, Any]] = []

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data by handling missing values and scaling.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Preprocessed data with target in first column
        """
        # Initialize scalers if they don't exist
        if self._feature_scaler is None:
            self._feature_scaler = StandardScaler()
            self._target_scaler = StandardScaler()
            
        # Handle missing values in X
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            
        # Handle missing values in y
        if np.isnan(y).any():
            y = np.nan_to_num(y, nan=np.nanmean(y))
            
        # Scale features and target
        X_scaled = self._feature_scaler.fit_transform(X)
        y_scaled = self._target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Combine into a single array
        return np.column_stack((y_scaled, X_scaled))

    def _generate_candidate_models(self) -> List[CandidateModel]:
        """
        Generate a list of candidate models for evaluation.
        
        Returns:
            List of candidate models
        """
        n_features = self.stratified_data_.shape[1] - 1
        models = []
        n_sklearn = self.candidate_models_count // 2
        n_custom = self.candidate_models_count - n_sklearn
        
        # Generate custom models with random parameters
        for i in range(n_custom):
            params = self.rng.standard_normal(n_features)
            models.append(CandidateModel(model_id=f'custom_{i}', params=params))
        
        # Generate scikit-learn models
        sklearn_estimators = [
            LinearRegression(),
            Ridge(alpha=1.0, random_state=self.random_state),
            Lasso(alpha=0.1, random_state=self.random_state),
            ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        ]
        
        for i, estimator in enumerate(sklearn_estimators[:n_sklearn]):
            models.append(CandidateModel(model_id=f'sklearn_{i}', sklearn_model=estimator))
            
        return models

    def _get_fold_indices(self, n: int) -> List[np.ndarray]:
        """
        Generate cross-validation fold indices.
        
        Args:
            n: Number of samples
            
        Returns:
            List of arrays containing test indices for each fold
        """
        indices = np.arange(n)
        self.rng.shuffle(indices)
        fold_size = n // self.folds
        return [indices[i * fold_size: (i + 1) * fold_size] for i in range(self.folds)]

    def _cross_validated_score(self, model: CandidateModel) -> float:
        """
        Compute cross-validated score for a candidate model.
        
        Args:
            model: The candidate model to evaluate
            
        Returns:
            Mean squared error score (lower is better)
        """
        n = self.stratified_data_.shape[0]
        folds = self._get_fold_indices(n)
        scores = []
        
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            train_data = self.stratified_data_[train_idx]
            test_data = self.stratified_data_[test_idx]
            
            X_train = train_data[:, 1:]
            y_train = train_data[:, 0]
            X_test = test_data[:, 1:]
            y_test = test_data[:, 0]
            
            if model.sklearn_model is not None:
                model_clone_obj = clone(model.sklearn_model)
                model_clone_obj.fit(X_train, y_train)
                y_pred = model_clone_obj.predict(X_test)
            else:
                y_pred = np.dot(X_test, model.params)
                
            mse = np.mean((y_test - y_pred) ** 2)
            scores.append(mse)
            
        return np.mean(scores) if scores else np.inf

    def _hyperopt_sklearn_model(
        self,
        candidate: CandidateModel,
        X_full: np.ndarray,
        y_full: np.ndarray
    ) -> None:
        """
        Optimize hyperparameters for a scikit-learn model.
        
        Args:
            candidate: The candidate model to optimize
            X_full: Feature matrix
            y_full: Target vector
        """
        sk_model = candidate.sklearn_model
        if sk_model is None:
            return

        # Define search space based on model type
        if isinstance(sk_model, Ridge):
            search_space = {'alpha': (0.0001, 10.0)}
        elif isinstance(sk_model, Lasso):
            search_space = {'alpha': (0.0001, 1.0)}
        elif isinstance(sk_model, ElasticNet):
            search_space = {
                'alpha': (0.0001, 1.0),
                'l1_ratio': (0.0, 1.0)
            }
        elif isinstance(sk_model, LinearRegression):
            # No hyperparameters to tune
            return
        else:
            logger.warning(f"Unsupported model type: {type(sk_model).__name__}")
            return

        # Create a default configuration
        default_config = CPUAcceleratedModelConfig()
        
        # Override parameters if they exist in the search space
        for key, bounds in search_space.items():
            if hasattr(default_config, key):
                setattr(default_config, key, bounds[0])
        
        # Define the model class
        base_model = CPUAcceleratedModel(
            estimator_class=type(sk_model),
            config=default_config
        )
        
        # Define scoring function
        def scoring_function(model_obj, X_data, y_data):
            scores = cross_val_score(
                model_obj.estimator,
                X_data, y_data,
                cv=self.folds,
                scoring='neg_mean_squared_error'
            )
            return np.mean(scores)
        
        # Create and run the optimizer
        optimizer = HyperparameterOptimizer(
            model_class=CPUAcceleratedModel,
            hyperparameter_space=search_space,
            scoring_function=scoring_function,
            max_evals=20,
            seed=self.random_state,
            n_folds=self.folds,
            task='regression'
        )
        
        try:
            # Run hyperparameter optimization
            result = optimizer.fit(X_full, y_full)
            best_model = optimizer.get_best_model()
            
            # Update the candidate with the optimized model
            if hasattr(best_model, 'estimator'):
                candidate.sklearn_model = best_model.estimator
                # Ensure the model is fitted
                candidate.sklearn_model.fit(X_full, y_full)
            
            # Store results
            self.model_trial_history[candidate.id] = result.get('trials', [])
            
            # Store search space area
            if hasattr(optimizer, 'search_space_area'):
                self.search_space_area[candidate.id] = optimizer.search_space_area
            else:
                # Calculate area manually
                area = 1.0
                for key, (low, high) in search_space.items():
                    area *= (high - low)
                self.search_space_area[candidate.id] = area
                
            logger.info(f"Hyperopt complete for {candidate.id} with score: {result.get('best_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization for {candidate.id}: {str(e)}")

    def _screen_models(self) -> List[CandidateModel]:
        """
        Screen candidate models to find the best performers.
        
        Returns:
            List of candidate models with performance scores
        """
        candidate_models = self._generate_candidate_models()
        X_full = self.stratified_data_[:, 1:]
        y_full = self.stratified_data_[:, 0]
        
        # Optimize scikit-learn models
        for candidate in candidate_models:
            if candidate.sklearn_model is not None:
                self._hyperopt_sklearn_model(candidate, X_full, y_full)
        
        # Function to evaluate a model
        def evaluate_model(model: CandidateModel) -> Tuple[Any, float]:
            try:
                score = self._cross_validated_score(model)
                return model.id, score
            except Exception as e:
                logger.error(f"Error evaluating model {model.id}: {str(e)}")
                return model.id, np.inf
        
        # Evaluate models in parallel
        scores_dict = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_model = {executor.submit(evaluate_model, model): model for model in candidate_models}
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        model_id, score = future.result()
                        scores_dict[model_id] = score
                    except Exception as e:
                        logger.error(f"Model {model.id} evaluation failed: {str(e)}")
                        scores_dict[model.id] = np.inf
        except Exception as e:
            logger.error(f"Error in parallel model evaluation: {str(e)}")
        
        # Update scores in models
        for model in candidate_models:
            model.score = scores_dict.get(model.id, np.inf)
        
        # Sort models by score (lower is better)
        candidate_models.sort(key=lambda m: m.score if m.score is not None else np.inf)
        
        logger.info(f"Screened {len(candidate_models)} models. Best score: {candidate_models[0].score if candidate_models else 'N/A'}")
        return candidate_models

    def _elastic_net(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4
    ) -> np.ndarray:
        """
        Simplified Elastic Net implementation.
        
        Args:
            X: Feature matrix
            y: Target vector
            alpha: Regularization strength
            l1_ratio: Mixing parameter (0 = ridge, 1 = lasso)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Coefficient vector
        """
        # Use scikit-learn's implementation for robustness
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            random_state=self.random_state
        )
        model.fit(X, y)
        return model.coef_

    def _elastic_net_stability_path(self) -> np.ndarray:
        """
        Compute the elastic net stability path using bootstrap samples.
        
        Returns:
            Average coefficient vector
        """
        X = self.stratified_data_[:, 1:]
        y = self.stratified_data_[:, 0]
        stability_paths = []
        
        for i in range(self.elastic_net_bootstraps):
            # Generate bootstrap sample
            boot_indices = resample(
                np.arange(len(y)),
                replace=True,
                n_samples=len(y),
                random_state=self.random_state + i
            )
            X_boot = X[boot_indices]
            y_boot = y[boot_indices]
            
            # Compute elastic net on bootstrap sample
            try:
                beta = self._elastic_net(X_boot, y_boot)
                stability_paths.append(beta)
            except Exception as e:
                logger.warning(f"Error in bootstrap {i}: {str(e)}")
        
        if not stability_paths:
            logger.warning("No valid stability paths. Using zeros.")
            return np.zeros(X.shape[1])
        
        # Average coefficients
        beta_mean = np.mean(stability_paths, axis=0)
        logger.info(f"Elastic Net Stability Path completed with {len(stability_paths)} valid bootstraps.")
        return beta_mean

    def _latin_hypercube_sampling(self, n_samples: int = 10) -> np.ndarray:
        """
        Generate Latin Hypercube samples for initial points in Bayesian optimization.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of sample points
        """
        if not self.hyperparameter_bounds:
            return np.array([])
            
        dim = len(self.hyperparameter_bounds)
        result = np.zeros((n_samples, dim))
        
        # Generate samples for each dimension
        for d in range(dim):
            low, high = self.hyperparameter_bounds[d]
            # Create evenly spaced bins
            bins = np.linspace(low, high, n_samples + 1)
            # Generate random positions within each bin
            positions = bins[:-1] + self.rng.random(n_samples) * (bins[1:] - bins[:-1])
            # Shuffle the positions
            self.rng.shuffle(positions)
            result[:, d] = positions
            
        return result

    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            params: Hyperparameter vector
            
        Returns:
            Negative mean squared error (higher is better)
        """
        X = self.stratified_data_[:, 1:]
        y = self.stratified_data_[:, 0]
        
        # Create a candidate model with these parameters
        candidate = CandidateModel(model_id='bo_candidate', params=params)
        
        # Evaluate using cross-validation
        try:
            score = self._cross_validated_score(candidate)
            return -score  # Negate for maximization
        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return -np.inf

    def _expected_improvement(
        self,
        x: np.ndarray,
        best_value: float,
        history: List[Tuple[np.ndarray, float]],
        xi: float = 0.01
    ) -> float:
        """
        Compute expected improvement acquisition function.
        
        Args:
            x: Point to evaluate
            best_value: Current best objective value
            history: History of evaluated points and their values
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Expected improvement value
        """
        # Simple implementation using RBF kernel for surrogate model
        if not history:
            return 1.0
            
        # Extract points and values
        X_history = np.array([p for p, _ in history])
        y_history = np.array([v for _, v in history])
        
        # Compute RBF kernel distances
        dists = np.sum((X_history[:, np.newaxis, :] - X_history[np.newaxis, :, :]) ** 2, axis=2)
        median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2 * median_dist)
        
        # Compute kernel between x and history points
        k = np.exp(-gamma * np.sum((X_history - x) ** 2, axis=1))
        
        # Predict mean and standard deviation
        mu = np.sum(k * y_history) / (np.sum(k) + 1e-8)
        sigma2 = 1.0 - np.sum(k) / (np.sum(k) + 1e-8)
        sigma = np.sqrt(max(0, sigma2))
        
        if sigma == 0:
            return 0
            
        # Compute improvement
        z = (mu - best_value - xi) / sigma
        
        # Expected improvement
        ei = (mu - best_value - xi) * norm_cdf(z) + sigma * norm_pdf(z)
        return max(0, ei)

    def _diversified_bayesopt(self) -> np.ndarray:
        """
        Run diversified Bayesian optimization to find optimal hyperparameters.
        
        Returns:
            Best hyperparameter vector found
        """
        if not self.hyperparameter_bounds:
            return np.array([])
            
        # Generate initial points
        n_initial = min(10, 5 * len(self.hyperparameter_bounds))
        initial_points = self._latin_hypercube_sampling(n_samples=n_initial)
        
        # Evaluate initial points
        history = []
        best_value = -np.inf
        best_params = None
        
        for i, x in enumerate(initial_points):
            value = self._objective_function(x)
            history.append((x, value))
            if value > best_value:
                best_value = value
                best_params = x.copy()
                
            # Log progress
            if i % 5 == 0 or i == len(initial_points) - 1:
                logger.info(f"Initial evaluation {i+1}/{len(initial_points)}, best value: {best_value}")
                
        # Main optimization loop
        no_improvement_count = 0
        for iteration in range(self.max_bayes_iter):
            # Generate candidate points
            candidates = self.rng.normal(
                size=(10, len(self.hyperparameter_bounds))
            )
            
            # Scale to bounds
            for d, (low, high) in enumerate(self.hyperparameter_bounds):
                candidates[:, d] = low + (high - low) * (0.5 + 0.5 * np.tanh(candidates[:, d]))
            
            # Compute acquisition function values
            acq_values = [
                self._expected_improvement(x, best_value, history)
                for x in candidates
            ]
            
            # Select best candidate
            best_idx = np.argmax(acq_values)
            next_point = candidates[best_idx]
            
            # Evaluate
            value = self._objective_function(next_point)
            history.append((next_point, value))
            
            # Update best
            improvement = False
            if value > best_value:
                improvement = True
                best_value = value
                best_params = next_point.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Store in history
            self.bayes_opt_history.append({
                'iteration': iteration,
                'params': next_point.tolist(),
                'value': value,
                'best_value': best_value,
                'improvement': improvement
            })
            
            # Early stopping
            if no_improvement_count >= self.early_stopping_patience:
                logger.info(f"Early stopping at iteration {iteration} due to no improvement")
                break
                
            # Log progress
            if iteration % 5 == 0 or iteration == self.max_bayes_iter - 1:
                logger.info(f"Bayes iteration {iteration+1}/{self.max_bayes_iter}, best value: {best_value}")
                
        logger.info(f"Bayesian optimization completed with best value: {best_value}")
        return best_params if best_params is not None else np.zeros(len(self.hyperparameter_bounds))

    def _ensemble_validation(
        self,
        candidate_models: List[CandidateModel],
        beta_path: np.ndarray,
        bayes_best: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Build an ensemble of the best models.
        
        Args:
            candidate_models: List of candidate models
            beta_path: Coefficient vector from elastic net stability
            bayes_best: Best hyperparameters from Bayesian optimization
            
        Returns:
            Function that makes ensemble predictions
        """
        # Sort candidates by score
        top_models = sorted(
            [m for m in candidate_models if m.score is not None and not np.isinf(m.score)],
            key=lambda m: m.score
        )[:5]  # Take top 5 models
        
        # Add elastic net stability path model
        if beta_path is not None and len(beta_path) > 0:
            en_model = CandidateModel(model_id='elastic_net', params=beta_path)
            en_model.score = self._cross_validated_score(en_model)
            top_models.append(en_model)
        
        # Add Bayesian optimization model
        if bayes_best is not None and len(bayes_best) > 0:
            n_features = self.stratified_data_.shape[1] - 1
            if len(bayes_best) != n_features:
                # Resize if dimensions don't match
                bayes_params = np.zeros(n_features)
                min_len = min(len(bayes_best), n_features)
                bayes_params[:min_len] = bayes_best[:min_len]
            else:
                bayes_params = bayes_best
                
            bayes_model = CandidateModel(model_id='bayes_opt', params=bayes_params)
            bayes_model.score = self._cross_validated_score(bayes_model)
            top_models.append(bayes_model)
        
        # Compute CV performance for each model
        X = self.stratified_data_[:, 1:]
        y = self.stratified_data_[:, 0]
        
        for model in top_models:
            # Check if sklearn model needs fitting
            if model.sklearn_model is not None and not hasattr(model.sklearn_model, 'coef_'):
                try:
                    model.sklearn_model.fit(X, y)
                except Exception as e:
                    logger.error(f"Error fitting model {model.id}: {str(e)}")
            
            # Get cross-validation performance
            try:
                model.cv_performance = -self._cross_validated_score(model)  # Negate to make positive
            except Exception as e:
                logger.error(f"Error computing CV performance for {model.id}: {str(e)}")
                model.cv_performance = 0.0
        
        # Compute weights
        total_perf = sum(max(0, m.cv_performance) for m in top_models)
        if total_perf > 0:
            for model in top_models:
                model.weight = max(0, model.cv_performance) / total_perf
        else:
            # Equal weights if all performances are negative
            for model in top_models:
                model.weight = 1.0 / len(top_models)
        
        # Filter models with too low weight
        min_weight = 0.05
        filtered_models = [m for m in top_models if m.weight >= min_weight]
        
        if not filtered_models:
            # If all filtered out, keep the best one
            filtered_models = [top_models[0]] if top_models else []
        
        # Create ensemble prediction function
        def ensemble_predict(X_input: np.ndarray) -> np.ndarray:
            if not filtered_models:
                return np.zeros(X_input.shape[0])
                
            # Apply feature scaling if needed
            if self._feature_scaler is not None:
                X_input = self._feature_scaler.transform(X_input)
                
            # Get predictions from each model
            predictions = []
            weights = []
            
            for model in filtered_models:
                try:
                    pred = self._predict_with_model(model, X_input)
                    predictions.append(pred)
                    weights.append(model.weight)
                except Exception as e:
                    logger.warning(f"Error in model {model.id} prediction: {str(e)}")
            
            # Combine predictions with weights
            if not predictions:
                return np.zeros(X_input.shape[0])
                
            # Stack predictions and compute weighted average
            stacked_preds = np.stack(predictions, axis=0)
            weights_array = np.array(weights)
            weighted_preds = np.average(stacked_preds, axis=0, weights=weights_array)
            
            # Inverse transform if using target scaling
            if self._target_scaler is not None:
                weighted_preds = self._target_scaler.inverse_transform(
                    weighted_preds.reshape(-1, 1)
                ).ravel()
                
            return weighted_preds
            
        # Store selected models for inspection
        self.selected_models = filtered_models
        logger.info(f"Created ensemble with {len(filtered_models)} models")
        
        return ensemble_predict
        
    def _predict_with_model(self, model: CandidateModel, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a candidate model.
        
        Args:
            model: The model to use for prediction
            X: Feature matrix
            
        Returns:
            Predictions from the model
        """
        if model.sklearn_model is not None:
            try:
                return model.sklearn_model.predict(X)
            except NotFittedError:
                # Fit the model if needed
                if self.X_ is not None and self.y_ is not None:
                    X_full = self.stratified_data_[:, 1:]
                    y_full = self.stratified_data_[:, 0]
                    model.sklearn_model.fit(X_full, y_full)
                    return model.sklearn_model.predict(X)
                else:
                    raise RuntimeError("Cannot fit model: training data not available")
        elif model.params is not None:
            return np.dot(X, model.params)
        else:
            return np.zeros(X.shape[0])
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HAMSeOptimizer':
        """
        Fit the HAMSe optimizer to the training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            Self, for method chaining
        """
        start_time = time.time()
        
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            X = np.asarray(X)
            y = np.asarray(y)
            
        if X.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape {X.shape}")
            
        if y.ndim != 1:
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError(f"y should be 1D array, got shape {y.shape}")
                
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y dimensions don't match: {X.shape[0]} vs {y.shape[0]}")
            
        # Initialize random generator
        self.rng = np.random.default_rng(self.random_state)
        
        # Set up hyperparameter bounds if not provided
        if self.hyperparameter_bounds is None:
            n_features = X.shape[1]
            self.hyperparameter_bounds = [(-5, 5)] * n_features
            logger.info(f"Hyperparameter bounds automatically set to {self.hyperparameter_bounds[0]} for all {n_features} features")
            
        # Store and preprocess data
        self.X_ = X
        self.y_ = y
        self.stratified_data_ = self._preprocess_data(X, y)
        
        # Stage 1: Model Screening
        logger.info("Stage 1: Model Screening")
        candidate_models = self._screen_models()
        
        # Stage 2: Elastic Net Stability Path
        logger.info("Stage 2: Computing Elastic Net Stability Path")
        beta_path = self._elastic_net_stability_path()
        
        # Stage 3: Bayesian Optimization
        logger.info("Stage 3: Diversified Bayesian Optimization")
        best_hyperparams = self._diversified_bayesopt()
        
        # Stage 4: Ensemble Validation
        logger.info("Stage 4: Ensemble Validation")
        final_model = self._ensemble_validation(candidate_models, beta_path, best_hyperparams)
        
        # Store the final model
        self.final_model = final_model
        
        # Report completion
        elapsed_time = time.time() - start_time
        logger.info(f"HAMSe Model Selection completed in {elapsed_time:.2f} seconds")
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the final ensemble model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if self.final_model is None:
            raise NotFittedError("The HAMSe model has not been fitted yet.")
            
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            
        if X.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape {X.shape}")
            
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.X_.shape[1]}")
            
        # Apply the ensemble model
        return self.final_model(X)
        
    def save_checkpoint(self, path: str) -> None:
        """
        Save optimizer state to a file.
        
        Args:
            path: File path to save to
        """
        if not self.X_ is not None:
            raise ValueError("Cannot save checkpoint: model has not been fitted")
            
        checkpoint = {
            # Configuration
            'hyperparameter_bounds': self.hyperparameter_bounds,
            'random_state': self.random_state,
            'folds': self.folds,
            'max_bayes_iter': self.max_bayes_iter,
            'early_stopping_patience': self.early_stopping_patience,
            'elastic_net_bootstraps': self.elastic_net_bootstraps,
            'candidate_models_count': self.candidate_models_count,
            
            # State tracking
            'search_space_area': self.search_space_area,
            'model_trial_history': self.model_trial_history,
            'bayes_opt_history': self.bayes_opt_history,
            
            # Model information
            'selected_models_info': [
                {
                    'id': model.id,
                    'score': model.score,
                    'cv_performance': model.cv_performance,
                    'weight': model.weight
                }
                for model in self.selected_models
            ],
            
            # Preprocessing
            'feature_scaler': self._feature_scaler,
            'target_scaler': self._target_scaler,
        }
        
        # Save sklearn models separately if needed
        sklearn_models = {}
        for i, model in enumerate(self.selected_models):
            if model.sklearn_model is not None:
                sklearn_models[f"model_{i}_{model.id}"] = model.sklearn_model
                
        if sklearn_models:
            checkpoint['sklearn_models'] = sklearn_models
        
        try:
            joblib.dump(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def load_checkpoint(self, path: str) -> 'HAMSeOptimizer':
        """
        Load optimizer state from a file.
        
        Args:
            path: File path to load from
            
        Returns:
            Self, for method chaining
        """
        try:
            checkpoint = joblib.load(path)
            
            # Load configuration
            self.hyperparameter_bounds = checkpoint.get('hyperparameter_bounds', self.hyperparameter_bounds)
            self.random_state = checkpoint.get('random_state', self.random_state)
            self.folds = checkpoint.get('folds', self.folds)
            self.max_bayes_iter = checkpoint.get('max_bayes_iter', self.max_bayes_iter)
            self.early_stopping_patience = checkpoint.get('early_stopping_patience', self.early_stopping_patience)
            self.elastic_net_bootstraps = checkpoint.get('elastic_net_bootstraps', self.elastic_net_bootstraps)
            self.candidate_models_count = checkpoint.get('candidate_models_count', self.candidate_models_count)
            
            # Load state tracking
            self.search_space_area = checkpoint.get('search_space_area', {})
            self.model_trial_history = checkpoint.get('model_trial_history', {})
            self.bayes_opt_history = checkpoint.get('bayes_opt_history', [])
            
            # Load preprocessing
            self._feature_scaler = checkpoint.get('feature_scaler')
            self._target_scaler = checkpoint.get('target_scaler')
            
            # Initialize RNG
            self.rng = np.random.default_rng(self.random_state)
            
            logger.info(f"Checkpoint loaded from {path}")
            return self
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
            
    def get_selected_models(self) -> List[Dict[str, Any]]:
        """
        Get information about the selected models in the ensemble.
        
        Returns:
            List of dictionaries with model information
        """
        if not self.selected_models:
            raise NotFittedError("No models have been selected yet. Run fit() first.")
            
        return [
            {
                'id': model.id,
                'type': 'sklearn' if model.sklearn_model is not None else 'custom',
                'score': model.score,
                'cv_performance': model.cv_performance,
                'weight': model.weight
            }
            for model in self.selected_models
        ]
        
    def get_feature_importances(self) -> Dict[str, np.ndarray]:
        """
        Get feature importances from the models.
        
        Returns:
            Dictionary mapping model IDs to their feature importance vectors
        """
        if not self.selected_models:
            raise NotFittedError("No models have been selected yet. Run fit() first.")
            
        result = {}
        for model in self.selected_models:
            if model.sklearn_model is not None and hasattr(model.sklearn_model, 'coef_'):
                result[model.id] = np.abs(model.sklearn_model.coef_)
            elif model.params is not None:
                result[model.id] = np.abs(model.params)
                
        # Add ensemble importance
        if result:
            # Weighted average of all importances
            all_imps = np.vstack([imp for imp in result.values()])
            weights = np.array([m.weight for m in self.selected_models if m.id in result])
            if weights.sum() > 0:
                result['ensemble'] = np.average(all_imps, axis=0, weights=weights)
                
        return result


# Helper functions for Bayesian optimization
def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * x * (1 + 0.044715 * x * x)))

def norm_pdf(x):
    """Standard normal probability density function."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # Set up logging with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 200, 10
    
    # Create data with specific pattern
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -0.5, 0.8, 0.0, 0.0, 0.0, 0.3, 0.0, -0.7, 0.2])
    y = np.dot(X, true_weights) + 0.1 * np.random.randn(n_samples)
    
    # Split into train/test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and fit the HAMSe optimizer
    optimizer = HAMSeOptimizer(
        n_jobs=4,
        random_state=42,
        max_bayes_iter=30,
        elastic_net_bootstraps=30,
        candidate_models_count=6
    )
    
    # Fit the model
    optimizer.fit(X_train, y_train)
    
    # Make predictions
    y_pred = optimizer.predict(X_test)
    
    # Evaluate performance
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R²: {r2:.4f}")
    
    # Show selected models
    selected_models = optimizer.get_selected_models()
    logger.info(f"Selected models: {selected_models}")
    
    # Get feature importances
    importances = optimizer.get_feature_importances()
    if 'ensemble' in importances:
        logger.info(f"Ensemble feature importances: {importances['ensemble']}")
        logger.info(f"True weights for comparison: {true_weights}")
    
    # Save checkpoint
    checkpoint_path = "hamse_checkpoint.pkl"
    optimizer.save_checkpoint(checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
