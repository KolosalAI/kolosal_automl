import os
import time
import logging
import joblib
import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional, Callable, Any

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

# --------------------------------------------------------------------------------
# Import the HyperparameterOptimizer from your local hyperopt.py
# --------------------------------------------------------------------------------
from .hyperopt import HyperparameterOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class HAMSeOptimizer:
    """
    HAMSeOptimizer implements the Hybrid Adaptive Model Selection (HAMSe) framework.
    The framework integrates:
      - Adaptive model screening
      - Elastic net stability feature selection
      - Diversified Bayesian optimization
      - Ensemble validation
    """

    def __init__(self,
                 hyperparameter_bounds: Optional[List[Tuple[float, float]]] = None,
                 n_jobs: Optional[int] = None,
                 random_state: int = 42,
                 folds: int = 5,
                 max_bayes_iter: int = 50):
        self.hyperparameter_bounds = hyperparameter_bounds
        self.n_jobs = n_jobs or os.cpu_count()
        self.random_state = random_state
        self.folds = folds
        self.max_bayes_iter = max_bayes_iter

        self.rng = None
        self.X_ = None
        self.y_ = None
        self.stratified_data_ = None

        self.final_model = None

    ############################################################################
    # Data Preprocessing Methods
    ############################################################################

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Replace NaNs with column means."""
        col_means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_means, inds[1])
        return data

    def _scale_features(self, data: np.ndarray) -> np.ndarray:
        """Standardize each column to have mean 0 and standard deviation 1."""
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero.
        return (data - means) / stds

    def _validate_data(self, data: np.ndarray) -> None:
        """Validate that data is nonempty."""
        if data.size == 0:
            raise ValueError("Data is empty.")

    def _stratify_data(self, data: np.ndarray) -> np.ndarray:
        """
        Stratify data if necessary. For demonstration, simply returns the data.
        """
        return data

    ############################################################################
    # NEW: Compare Various scikit-learn Models
    ############################################################################
    def compare_sklearn_models(self,
                               model_list: List[Any],
                               X: np.ndarray,
                               y: np.ndarray,
                               scoring: str = "neg_mean_squared_error",
                               cv: Optional[int] = None) -> List[dict]:
        """
        Compare the performance of multiple scikit-learn models using cross-validation.

        Parameters
        ----------
        model_list : List[Any]
            A list of sklearn estimator instances (e.g., [LinearRegression(), Ridge(), ...]).
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        scoring : str, default="neg_mean_squared_error"
            Scoring method to use (any valid sklearn scoring string).
        cv : int, optional
            Number of folds for cross-validation. Defaults to self.folds if not provided.

        Returns
        -------
        results : List[dict]
            A list of dictionaries with keys: "model", "mean_score", "std_score".
        """
        if cv is None:
            cv = self.folds

        results = []
        for model in model_list:
            # Clone the model to avoid side-effects
            model_clone = clone(model)
            scores = cross_val_score(model_clone, X, y, scoring=scoring, cv=cv, n_jobs=self.n_jobs)
            results.append({
                "model": model.__class__.__name__,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores)
            })
            logging.info("Model: %s | Mean %s: %.4f | Std: %.4f",
                         model.__class__.__name__,
                         scoring,
                         np.mean(scores),
                         np.std(scores))
        return results

    ############################################################################
    # Candidate Model Definition
    ############################################################################

    class CandidateModel:
        """
        Represents a candidate model. The candidate can either be a simple linear model
        represented by a coefficient vector (params) or a scikit-learn model (sklearn_model).
        """
        def __init__(self, model_id: Any,
                     params: Optional[np.ndarray] = None,
                     sklearn_model: Optional[Any] = None):
            self.id = model_id
            self.params = params  # NumPy array of coefficients.
            self.sklearn_model = sklearn_model  # Any sklearn estimator.
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
    # Model Generation and Screening
    ############################################################################

    def _generate_candidate_models(self, num_models: int = 5) -> List['HAMSeOptimizer.CandidateModel']:
        """
        Generate candidate models with random coefficients and sklearn models.
        For demonstration, half the candidates are custom-parameter-based,
        and the rest are sklearn models (LinearRegression, Ridge, Lasso).
        """
        n_features = self.stratified_data_.shape[1] - 1
        models = []
        n_sklearn = num_models // 2
        n_custom = num_models - n_sklearn

        # Custom candidate models (params only).
        for i in range(n_custom):
            params = self.rng.randn(n_features)
            models.append(self.CandidateModel(model_id=f'custom_{i}', params=params))

        # Sklearn candidate models.
        sklearn_estimators = [LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.1)]
        for i, estimator in enumerate(sklearn_estimators[:n_sklearn]):
            models.append(self.CandidateModel(model_id=f'sklearn_{i}', sklearn_model=estimator))

        return models

    def _get_fold_indices(self, n: int) -> List[np.ndarray]:
        """
        Splits indices into self.folds folds.
        """
        indices = np.arange(n)
        self.rng.shuffle(indices)
        fold_size = n // self.folds
        return [indices[i * fold_size: (i + 1) * fold_size] for i in range(self.folds)]

    def _cross_validated_hamse_score(self, model: 'HAMSeOptimizer.CandidateModel') -> float:
        """
        Compute a cross-validated HAMSe score using mean squared error as a proxy.
        If the candidate wraps a sklearn model, it is cloned and fit on each fold.
        """
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
                # Clone and fit the sklearn model on the current fold
                model_clone = clone(model.sklearn_model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
            else:
                # Custom param-based model
                y_pred = np.dot(X_test, model.params)

            mse = np.mean((y_test - y_pred) ** 2)
            scores.append(mse)

        return np.mean(scores) if scores else np.inf

    # ----------------------------------------------------------------------
    # Helper to run Hyperopt for each sklearn model (Ridge, Lasso, etc.)
    # ----------------------------------------------------------------------
    def _hyperopt_sklearn_model(self,
                                candidate: 'HAMSeOptimizer.CandidateModel',
                                X_full: np.ndarray,
                                y_full: np.ndarray) -> None:
        """
        If candidate is a scikit-learn model with known hyperparameters, use
        HyperparameterOptimizer to tune them. Then update the candidate's model.
        """
        # Identify the underlying sklearn model type:
        sk_model = candidate.sklearn_model
        if sk_model is None:
            return  # Not an sklearn model, do nothing.

        # Define a hyperparameter space based on the model type:
        if isinstance(sk_model, Ridge):
            # Example: single parameter alpha in [0.0, 10.0]
            search_space = {
                'alpha': ('uniform', 0.0, 10.0)
            }
        elif isinstance(sk_model, Lasso):
            # Example: single parameter alpha in [0.0, 1.0]
            search_space = {
                'alpha': ('uniform', 0.0, 1.0)
            }
        elif isinstance(sk_model, LinearRegression):
            # LinearRegression rarely has relevant hyperparameters to tune,
            # but we could add 'fit_intercept' or others if needed.
            return
        else:
            return

        # Minimal wrapper so HyperparameterOptimizer can train it:
        class SklearnWrapperModel:
            """
            Minimal wrapper to let HyperparameterOptimizer fit/predict
            a scikit-learn estimator.
            """
            def __init__(self, base_estimator):
                self.estimator = clone(base_estimator)

            def fit(self, X, y):
                self.estimator.fit(X, y)

            def predict(self, X):
                return self.estimator.predict(X)

            def get_metrics(self):
                # Return any custom metrics you want HyperparameterOptimizer to record
                return {}

        # Scoring function for HyperparameterOptimizer:
        def scoring_function(model_obj, X_data, y_data):
            cv_scores = cross_val_score(
                model_obj.estimator, X_data, y_data,
                cv=self.folds,
                scoring='neg_mean_squared_error'
            )
            # scikit-learn's 'neg_mean_squared_error' returns negative MSE,
            # so we can return mean of these scores directly:
            return np.mean(cv_scores)

        # Run HyperparameterOptimizer:
        optimizer = HyperparameterOptimizer(
            model_class=SklearnWrapperModel,
            hyperparameter_space=search_space,
            scoring_function=scoring_function,
            max_evals=20,  # example
            seed=self.random_state,
            n_folds=self.folds
        )

        optimizer.fit(X_full, y_full)
        best_model = optimizer.get_best_model()  # Returns best SklearnWrapperModel
        candidate.sklearn_model = best_model.estimator  # Update the candidate's final sklearn model

    def _screen_models(self) -> List['HAMSeOptimizer.CandidateModel']:
        """
        Screen candidate models using a HAMSe-inspired cross-validation score.
        We also run Hyperopt for each scikit-learn candidate to find best hyperparams.
        """
        candidate_models = self._generate_candidate_models()

        # Separate out the data: first col is y, rest are X
        X_full = self.stratified_data_[:, 1:]
        y_full = self.stratified_data_[:, 0]

        # 1) Before scoring, hyperopt-tune each sklearn model.
        for candidate in candidate_models:
            self._hyperopt_sklearn_model(candidate, X_full, y_full)

        # 2) Evaluate cross-validated HAMSe score in parallel
        cache = {}

        def evaluate_model(m: 'HAMSeOptimizer.CandidateModel') -> Tuple[Any, float]:
            if m.id not in cache:
                score_ = self._cross_validated_hamse_score(m)
                cache[m.id] = score_
            else:
                score_ = cache[m.id]
            return (m.id, score_)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(evaluate_model, model): model for model in candidate_models}
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    _, score = future.result()
                    model.score = score
                except Exception as exc:
                    logging.error("Model %s generated an exception: %s", model.id, exc)
                    model.score = np.inf

        logging.info("Stage 1 complete: Screened %d models (with Hyperopt for sklearn).", len(candidate_models))
        return candidate_models

    ############################################################################
    # Elastic Net Stability Path Methods
    ############################################################################

    def _elastic_net(self, data: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 100) -> np.ndarray:
        """
        Perform coordinate descent for Elastic Net regression (simple example).
        """
        X = data[:, 1:]
        y = data[:, 0]
        n_samples, n_features = X.shape
        coef = np.zeros(n_features)
        for _ in range(max_iter):
            for j in range(n_features):
                prediction = np.dot(X, coef)
                # remove the contribution of coef[j] from the residual
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
        """Generate a bootstrap sample from the data."""
        n = data.shape[0]
        indices = self.rng.choice(n, n, replace=True)
        return data[indices]

    def _enforce_group_selection(self, beta_paths: List[np.ndarray], threshold_ratio: float = 0.8) -> np.ndarray:
        """Dummy group selection: returns the mean of the coefficient paths."""
        beta_paths = np.array(beta_paths)
        return np.mean(beta_paths, axis=0)

    def _elastic_net_stability_path(self, n_bootstraps: int = 50) -> np.ndarray:
        """
        Compute the Elastic Net stability path over multiple bootstrap samples.
        """
        stability_paths = []
        for _ in range(n_bootstraps):
            boot_data = self._bootstrap_sample(self.stratified_data_)
            beta = self._elastic_net(boot_data)
            stability_paths.append(beta)
        grouped_beta = self._enforce_group_selection(stability_paths, threshold_ratio=0.8)
        logging.info("Stage 2 complete: Computed Elastic Net Stability Path.")
        return grouped_beta

    ############################################################################
    # Diversified Bayesian Optimization Methods
    ############################################################################

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
        # Dummy placeholder
        return self.rng.rand()

    def _thompson_sampling(self, x: np.ndarray, objective_fn: Callable[[np.ndarray], float]) -> float:
        # Dummy placeholder
        return self.rng.rand()

    def _objective_function(self, x: np.ndarray) -> float:
        # Dummy placeholder objective
        return np.sum(x ** 2)

    def _early_stopping_criteria_met(self, patience: int) -> bool:
        # Dummy placeholder
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
            logging.info("Bayesian Optimization iteration %d completed.", iteration)
        logging.info("Stage 3 complete: Best hyperparameters found.")
        return best_hyperparams if best_hyperparams is not None else np.zeros(len(self.hyperparameter_bounds))

    ############################################################################
    # Ensemble Validation
    ############################################################################

    def _merge_models(self,
                      models_list: List['HAMSeOptimizer.CandidateModel'],
                      beta_path: np.ndarray,
                      best_hps: np.ndarray) -> List['HAMSeOptimizer.CandidateModel']:
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
        """Helper method to handle not-fitted scikit-learn models (fit on full data if needed)."""
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

    def _weighted_average_predictions(self,
                                      models: List['HAMSeOptimizer.CandidateModel'],
                                      num_samples: int = 10) -> np.ndarray:
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

    def _train_stacking_regressor(self,
                                  models: List['HAMSeOptimizer.CandidateModel']) -> Callable[[np.ndarray], np.ndarray]:
        def ensemble_predict(X: np.ndarray) -> np.ndarray:
            preds = []
            for m in models:
                pred = self._predict_with_model(m, X)
                preds.append(pred)
            preds = np.array(preds)
            return np.mean(preds, axis=0)
        return ensemble_predict

    def _ensemble_validation(self,
                            models_list: List['HAMSeOptimizer.CandidateModel'],
                            beta_path: np.ndarray,
                            best_hps: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        ensemble_candidates = self._merge_models(models_list, beta_path, best_hps)

        # Evaluate cross-validation performance
        for model in ensemble_candidates:
            model.cv_performance = self._cross_validation_performance(model)

        # Compute weights
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
            if total_weight != 0:
                model.weight /= total_weight
            else:
                model.weight = 0

        # Weighted average predictions on dummy input
        ensemble_prediction = self._weighted_average_predictions(ensemble_candidates)
        filtered_candidates = []
        dummy_input = np.random.randn(10, self.stratified_data_.shape[1] - 1)

        # Agreement filtering
        for m in ensemble_candidates:
            pred_vals = self._predict_with_model(m, dummy_input)
            pred_val = np.mean(pred_vals)
            pred = np.full(ensemble_prediction.shape, pred_val)
            mae = np.mean(np.abs(pred - ensemble_prediction))
            std_dev = np.std(ensemble_prediction)
            if mae <= 1.5 * std_dev:
                filtered_candidates.append(m)
            else:
                logging.info("Model %s removed during agreement filtering.", m.id)

        final_ensemble = self._train_stacking_regressor(filtered_candidates)
        logging.info("Stage 4 complete: Final ensemble model trained with %d candidate(s).",
                     len(filtered_candidates))
        return final_ensemble

    ############################################################################
    # Main Optimization Pipeline
    ############################################################################

    def fit(self, X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Execute the HAMSe model selection process and return the final ensemble predictor.
        """
        # Combine X and y into a single array: first column is target, rest are features.
        data = np.column_stack((y, X))

        # Initialize RNG here
        self.rng = np.random.RandomState(self.random_state)

        # If hyperparameter_bounds not provided, set defaults
        if self.hyperparameter_bounds is None:
            n_features = X.shape[1]
            self.hyperparameter_bounds = [(-5, 5)] * n_features
            logging.info("Hyperparameter bounds automatically set to %s", self.hyperparameter_bounds)

        # Preprocess data
        data = self._handle_missing_values(data)
        data = self._scale_features(data)
        self._validate_data(data)
        self.stratified_data_ = self._stratify_data(data)

        # Stage 1: Model Screening (includes Hyperopt for sklearn models)
        candidate_models = self._screen_models()

        # Stage 2: Elastic Net Stability Path
        beta_path = self._elastic_net_stability_path(n_bootstraps=50)

        # Stage 3: Diversified Bayesian Optimization
        best_hyperparams = self._diversified_bayesopt()

        # Stage 4: Ensemble Validation
        final_model = self._ensemble_validation(candidate_models, beta_path, best_hyperparams)

        logging.info("HAMSe Model Selection completed successfully.")
        self.final_model = final_model
        return final_model

    ############################################################################
    # Checkpointing Methods
    ############################################################################

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'hyperparameter_bounds': self.hyperparameter_bounds,
            'random_state': self.random_state,
            'rng_state': self.rng.get_state() if self.rng else None,
            'folds': self.folds,
            'max_bayes_iter': self.max_bayes_iter,
        }
        joblib.dump(checkpoint, path)
        logging.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = joblib.load(path)
        self.hyperparameter_bounds = checkpoint['hyperparameter_bounds']
        self.random_state = checkpoint['random_state']
        self.folds = checkpoint.get('folds', self.folds)
        self.max_bayes_iter = checkpoint.get('max_bayes_iter', self.max_bayes_iter)
        rng_state = checkpoint.get('rng_state', None)
        if rng_state is not None:
            self.rng = np.random.RandomState(self.random_state)
            self.rng.set_state(rng_state)
        logging.info("Checkpoint loaded from %s", path)


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # Generate dummy data: 100 samples, 9 features, plus 1 target column
    np.random.seed(42)
    dummy_data = np.random.randn(100, 10)  # 10 columns: [target, 9 features]
    X, y = dummy_data[:, 1:], dummy_data[:, 0]

    # Initialize and run the HAMSe optimizer
    optimizer = HAMSeOptimizer(n_jobs=4, random_state=42)
    
    # --- (1) Compare different sklearn models individually (optional) ---
    model_list = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.1)
        # Add more sklearn estimators if desired
    ]
    compare_results = optimizer.compare_sklearn_models(model_list, X, y, scoring="neg_mean_squared_error", cv=5)
    logging.info("Comparison of base models: %s", compare_results)

    # --- (2) Fit the entire HAMSe pipeline ---
    ensemble_predictor = optimizer.fit(X, y)

    # Generate dummy test data: 10 samples, 9 features
    test_X = np.random.randn(10, 9)
    predictions = ensemble_predictor(test_X)
    logging.info("Ensemble predictions: %s", predictions)
