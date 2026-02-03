import numpy as np
import math
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

class ASHTOptimizer:
    """Adaptive Surrogate-Assisted Hyperparameter Tuning implementation"""
    
    def __init__(self, estimator, param_space, max_iter=50, cv=5, scoring=None, 
                 random_state=None, n_jobs=1, verbose=0):
        self.estimator = estimator
        self.param_space = param_space
        self.max_iter = max_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = -float('inf')
        self.best_estimator_ = None
        self.cv_results_ = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'split0_test_score': [],
            'budget': []
        }
        
        # Initialize surrogate model (RandomForest for better performance)
        self.surrogate_model = RandomForestRegressor(
            n_estimators=10, 
            max_depth=5, 
            random_state=random_state
        )
        
        # Parameter space analysis
        self.param_types = self._analyze_param_space()
        self.param_bounds = self._get_param_bounds()
        self.scaler = StandardScaler()
        
        # Cache for evaluated configurations
        self.evaluated_configs = {}
        
    def _analyze_param_space(self):
        """Analyze parameter space to determine parameter types"""
        param_types = {}
        for param_name, param_value in self.param_space.items():
            if isinstance(param_value, list):
                param_types[param_name] = 'categorical'
            elif isinstance(param_value, tuple) and len(param_value) == 2:
                param_types[param_name] = 'numerical'
            elif hasattr(param_value, 'rvs'):  # scipy distribution
                param_types[param_name] = 'distribution'
            else:
                param_types[param_name] = 'unknown'
        return param_types
    
    def _get_param_bounds(self):
        """Get bounds for numerical parameters for optimization"""
        bounds = {}
        for param_name, param_type in self.param_types.items():
            if param_type == 'numerical':
                bounds[param_name] = self.param_space[param_name]
        return bounds
    
    def _validate_params(self, params):
        """Validate and fix parameter compatibility issues"""
        fixed_params = params.copy()
        
        # Check for common incompatibilities
        if 'solver' in fixed_params and 'penalty' in fixed_params:
            solver = fixed_params['solver']
            penalty = fixed_params['penalty']
            
            # Fix incompatible solver/penalty combinations
            if solver == 'lbfgs' and penalty not in ['l2', None]:
                fixed_params['penalty'] = 'l2'
            elif solver == 'newton-cg' and penalty not in ['l2', None]:
                fixed_params['penalty'] = 'l2'
            elif solver == 'sag' and penalty not in ['l2', None]:
                fixed_params['penalty'] = 'l2'
            elif solver == 'saga' and penalty not in ['l1', 'l2', 'elasticnet', None]:
                fixed_params['penalty'] = 'l2'
            elif solver == 'liblinear' and penalty not in ['l1', 'l2']:
                fixed_params['penalty'] = 'l2'
        
        # Check for C/alpha compatibility
        if 'alpha' in fixed_params and fixed_params['alpha'] <= 0:
            fixed_params['alpha'] = 1e-4
        if 'C' in fixed_params and fixed_params['C'] <= 0:
            fixed_params['C'] = 1.0
            
        # Check for max_iter
        if 'max_iter' in fixed_params and fixed_params['max_iter'] <= 0:
            fixed_params['max_iter'] = 100
            
        return fixed_params
    
    def _objective_func(self, params, budget):
        """Evaluate a configuration with the given budget"""
        # Validate and fix parameters
        params = self._validate_params(params)
        
        # Check cache first to avoid redundant evaluations
        param_key = str(sorted(params.items()))
        budget_key = round(budget, 3)  # Round to avoid floating point issues
        
        if param_key in self.evaluated_configs and budget_key in self.evaluated_configs[param_key]:
            return self.evaluated_configs[param_key][budget_key]
        
        # Set parameters on a clone of the estimator
        estimator = clone(self.estimator)
        try:
            estimator.set_params(**params)
        except Exception as e:
            if self.verbose:
                print(f"Parameter setting error: {e}")
                print(f"Problematic params: {params}")
            # Return a very low score for invalid configurations
            return -float('inf')
        
        # Create a subset of data based on budget if needed
        # For simplicity, we'll use budget to determine CV folds
        actual_cv = max(2, min(self.cv, int(self.cv * budget)))
        
        # For compatibility with the engine's data format
        X, y = self.X, self.y
        
        # Perform cross-validation with error handling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    estimator, X, y, 
                    cv=actual_cv, 
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    error_score=np.nan
                )
            
            # Handle NaN scores
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                mean_score = -float('inf')
                std_score = 0
            else:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
        except Exception as e:
            if self.verbose:
                print(f"Cross-validation error: {e}")
            mean_score = -float('inf')
            std_score = 0
            scores = [-float('inf')]
        
        # Store results
        self.cv_results_['params'].append(params)
        self.cv_results_['mean_test_score'].append(mean_score)
        self.cv_results_['std_test_score'].append(std_score)
        self.cv_results_['split0_test_score'].append(scores[0] if len(scores) > 0 else -float('inf'))
        self.cv_results_['budget'].append(budget)
        
        # Update best if needed
        if mean_score > self.best_score_:
            self.best_score_ = mean_score
            self.best_params_ = params
        
        # Cache the result
        if param_key not in self.evaluated_configs:
            self.evaluated_configs[param_key] = {}
        self.evaluated_configs[param_key][budget_key] = mean_score
        
        return mean_score
    
    def _sample_random_configs(self, n):
        """Sample n random configurations from parameter space"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        samples = []
        for _ in range(n):
            config = {}
            for pname, pval in self.param_space.items():
                # Numerical range
                if isinstance(pval, tuple) and len(pval) == 2:
                    low, high = pval
                    # If both are ints, sample an int
                    if isinstance(low, int) and isinstance(high, int):
                        config[pname] = np.random.randint(low, high + 1)
                    else:
                        config[pname] = np.random.uniform(low, high)
                # Categorical
                elif isinstance(pval, list):
                    config[pname] = np.random.choice(pval)
                # Distribution with .rvs()
                elif hasattr(pval, 'rvs'):
                    config[pname] = pval.rvs(random_state=self.random_state)
                else:
                    # Unknown - just store as is
                    config[pname] = pval
            
            # Validate the configuration
            config = self._validate_params(config)
            samples.append(config)
            
        return samples
    
    def _encode_config(self, config):
        """Encode a configuration into a numerical vector for surrogate model"""
        features = []
        for param_name, param_type in self.param_types.items():
            if param_name in config:
                if param_type == 'numerical' or param_type == 'distribution':
                    features.append(float(config[param_name]))
                elif param_type == 'categorical':
                    # One-hot encoding for categorical parameters
                    categories = self.param_space[param_name]
                    one_hot = [1.0 if val == config[param_name] else 0.0 for val in categories]
                    features.extend(one_hot)
                else:
                    # For unknown types, use a hash
                    features.append(float(hash(str(config[param_name])) % 1000) / 1000)
        return np.array(features).reshape(1, -1)
    
    def _configs_to_features(self, configs):
        """Convert a list of configurations to a feature matrix for the surrogate model"""
        if not configs:
            return np.array([]).reshape(0, 0)
            
        # Get all features for the first config to determine dimensionality
        first_features = self._encode_config(configs[0])
        n_features = first_features.shape[1]
        
        # Create feature matrix
        X = np.zeros((len(configs), n_features))
        X[0] = first_features
        
        # Fill in the rest
        for i in range(1, len(configs)):
            X[i] = self._encode_config(configs[i])
        
        return X
    
    def _decode_vector_to_config(self, vector, param_names):
        """Decode a numerical vector back to a configuration"""
        config = {}
        idx = 0
        
        for param_name in param_names:
            param_type = self.param_types[param_name]
            
            if param_type == 'numerical':
                # Clip to bounds
                low, high = self.param_bounds[param_name]
                value = np.clip(vector[idx], low, high)
                
                # Handle integer parameters
                if isinstance(low, int) and isinstance(high, int):
                    value = int(round(value))
                
                config[param_name] = value
                idx += 1
            elif param_type == 'categorical':
                categories = self.param_space[param_name]
                n_categories = len(categories)
                
                # Get one-hot part of the vector
                one_hot = vector[idx:idx+n_categories]
                
                # Find the index of the maximum value
                category_idx = np.argmax(one_hot)
                config[param_name] = categories[category_idx]
                
                idx += n_categories
            elif param_type == 'distribution':
                # For distributions, we just store the raw value
                config[param_name] = vector[idx]
                idx += 1
            else:
                # For unknown parameters, use the raw value
                config[param_name] = vector[idx]
                idx += 1
                
        # Validate the configuration
        config = self._validate_params(config)
        return config
    
    def _train_surrogate_model(self, configs, scores):
        """Train a surrogate model to predict scores from configurations"""
        if not configs:
            return self.surrogate_model
            
        # Convert configs to feature matrix
        X_surrogate = self._configs_to_features(configs)
        
        if X_surrogate.shape[0] == 0 or X_surrogate.shape[1] == 0:
            return self.surrogate_model
            
        # Filter out -inf scores
        valid_indices = np.isfinite(scores)
        if np.sum(valid_indices) < 2:  # Need at least 2 valid samples
            # Create a dummy model that returns the mean of valid scores
            mean_score = np.mean(np.array(scores)[valid_indices]) if np.any(valid_indices) else 0
            
            class DummySurrogate:
                def predict(self, X):
                    return np.full(X.shape[0], mean_score)
                    
                def predict_with_std(self, X):
                    return np.full(X.shape[0], mean_score), np.full(X.shape[0], 0.1)
                    
            return DummySurrogate()
            
        X_valid = X_surrogate[valid_indices]
        y_valid = np.array(scores)[valid_indices]
        
        # Scale features for better surrogate performance
        X_scaled = self.scaler.fit_transform(X_valid)
        
        # Train the surrogate
        try:
            self.surrogate_model.fit(X_scaled, y_valid)
            
            # Add predict_with_std method if not present
            if not hasattr(self.surrogate_model, 'predict_with_std'):
                original_predict = self.surrogate_model.predict
                
                def predict_with_std(X):
                    # For random forest, we can use the std of individual tree predictions
                    if hasattr(self.surrogate_model, 'estimators_'):
                        preds = np.array([tree.predict(X) for tree in self.surrogate_model.estimators_])
                        return np.mean(preds, axis=0), np.std(preds, axis=0)
                    else:
                        # Fallback for other models
                        return original_predict(X), np.ones(X.shape[0]) * 0.1
                        
                self.surrogate_model.predict_with_std = predict_with_std
                
        except Exception as e:
            if self.verbose:
                print(f"Surrogate training error: {e}")
            # Return a simple model that predicts the mean
            mean_score = np.mean(y_valid)
            
            class SimpleSurrogate:
                def predict(self, X):
                    return np.full(X.shape[0], mean_score)
                    
                def predict_with_std(self, X):
                    return np.full(X.shape[0], mean_score), np.full(X.shape[0], 0.1)
                    
            return SimpleSurrogate()
        
        return self.surrogate_model
    
    def _phi(self, z):
        """Standard Normal PDF."""
        return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * z * z)
    
    def _Phi(self, z):
        """Standard Normal CDF."""
        return 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
    
    def _expected_improvement(self, x, surrogate, best_f, param_names, xi=0.01):
        """
        Expected Improvement acquisition function.
        
        Args:
            x: The point at which to evaluate EI
            surrogate: The surrogate model
            best_f: The best observed value
            param_names: List of parameter names
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            -1 * Expected Improvement (negated for minimization)
        """
        # Convert x to a configuration
        config = self._decode_vector_to_config(x, param_names)
        
        # Convert config to features
        features = self._encode_config(config)
        
        # Handle empty features
        if features.size == 0:
            return 0
            
        features_scaled = self.scaler.transform(features)
        
        # Predict with the surrogate model
        if hasattr(surrogate, 'predict_with_std'):
            mu, sigma = surrogate.predict_with_std(features_scaled)
            if isinstance(mu, np.ndarray):
                mu = mu[0]
            if isinstance(sigma, np.ndarray):
                sigma = sigma[0]
        else:
            mu = surrogate.predict(features_scaled)[0]
            sigma = 0.1  # Default uncertainty
        
        # Handle case where sigma is 0
        if sigma <= 0:
            return 0
        
        # Calculate improvement
        imp = mu - best_f - xi
        
        # Calculate Z score
        z = imp / sigma
        
        # Calculate expected improvement using the phi and Phi functions
        ei = imp * self._Phi(z) + sigma * self._phi(z)
        
        # Return negative EI (for minimization)
        return -1.0 * ei
    
    def _optimize_acquisition(self, surrogate, best_f, param_names, n_restarts=5):
        """
        Optimize the acquisition function to find the next point to evaluate.
        
        Args:
            surrogate: The surrogate model
            best_f: The best observed value
            param_names: List of parameter names
            n_restarts: Number of random restarts for optimization
            
        Returns:
            The configuration that maximizes the acquisition function
        """
        # Determine dimensionality of the optimization space
        dim = 0
        for param_name in param_names:
            if self.param_types[param_name] == 'numerical' or self.param_types[param_name] == 'distribution':
                dim += 1
            elif self.param_types[param_name] == 'categorical':
                dim += len(self.param_space[param_name])
            else:
                dim += 1
        
        # Define bounds for optimization
        bounds = []
        for param_name in param_names:
            if self.param_types[param_name] == 'numerical':
                low, high = self.param_bounds[param_name]
                bounds.append((low, high))
            elif self.param_types[param_name] == 'categorical':
                # For categorical parameters, use [0, 1] bounds for each category
                categories = self.param_space[param_name]
                bounds.extend([(0, 1) for _ in range(len(categories))])
            elif self.param_types[param_name] == 'distribution':
                # For distributions, use [0, 1] as default bounds
                bounds.append((0, 1))
            else:
                # Default bounds for unknown parameters
                bounds.append((0, 1))
        
        # If no bounds, return a random configuration
        if not bounds:
            return self._sample_random_configs(1)[0]
            
        # Run optimization with multiple random starts
        best_x = None
        best_acq = -np.inf
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.zeros(dim)
            for i, (low, high) in enumerate(bounds):
                x0[i] = np.random.uniform(low, high)
            
            # Run optimization
            try:
                result = minimize(
                    lambda x: self._expected_improvement(x, surrogate, best_f, param_names),
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x
            except Exception as e:
                if self.verbose:
                    print(f"Optimization error: {e}")
                # If optimization fails, continue with next restart
                continue
        
        # If optimization failed, sample randomly
        if best_x is None:
            # Simple approach: just sample a random configuration
            return self._sample_random_configs(1)[0]
        
        # Convert the optimal point to a configuration
        return self._decode_vector_to_config(best_x, param_names)
    
    def _refine_param_space(self, param_space, surrogate):
        """Refine parameter space based on surrogate model insights"""
        # If surrogate is a tree-based model, we can use feature importances
        if hasattr(surrogate, 'feature_importances_'):
            importances = surrogate.feature_importances_
            
            # Build a mapping param -> number of columns in the feature matrix
            param_names = list(param_space.keys())
            col_mapping = {}
            col_index = 0
            
            for pname in param_names:
                pval = param_space[pname]
                if isinstance(pval, tuple) and len(pval) == 2:
                    # numeric -> 1 col
                    col_mapping[pname] = [col_index]
                    col_index += 1
                elif isinstance(pval, list):
                    # categorical -> len(pval) columns
                    col_mapping[pname] = list(range(col_index, col_index + len(pval)))
                    col_index += len(pval)
                elif hasattr(pval, 'rvs'):
                    # distribution -> 1 col
                    col_mapping[pname] = [col_index]
                    col_index += 1
                else:
                    # fallback -> 1 col
                    col_mapping[pname] = [col_index]
                    col_index += 1
            
            # Check if we have enough importances
            if len(importances) < col_index:
                return param_space
                
            # Define a threshold for low importance
            threshold = np.percentile(importances, 20)
            
            # Create a refined space
            refined_space = param_space.copy()
            
            # Evaluate importance per parameter by summing columns
            for pname, cols in col_mapping.items():
                param_importance = sum(importances[c] for c in cols if c < len(importances))
                
                if param_importance < threshold:
                    # Narrow the parameter space for less important parameters
                    if isinstance(refined_space[pname], tuple) and len(refined_space[pname]) == 2:
                        low, high = refined_space[pname]
                        mid = (low + high) / 2.0
                        # Narrow to ± half the original range
                        new_low = mid - (mid - low) * 0.5
                        new_high = mid + (high - mid) * 0.5
                        refined_space[pname] = (new_low, new_high)
                    elif isinstance(refined_space[pname], list) and len(refined_space[pname]) > 3:
                        # Reduce categories if more than 3
                        # Keep the most promising categories based on surrogate predictions
                        categories = refined_space[pname]
                        category_scores = []
                        
                        for cat in categories:
                            # Create a test config with this category
                            test_config = {pname: cat}
                            # Add median values for other parameters
                            for other_param, other_value in refined_space.items():
                                if other_param != pname:
                                    if isinstance(other_value, tuple) and len(other_value) == 2:
                                        test_config[other_param] = (other_value[0] + other_value[1]) / 2
                                    elif isinstance(other_value, list):
                                        test_config[other_param] = other_value[0]
                                    else:
                                        test_config[other_param] = other_value
                            
                            # Predict score for this config
                            features = self._encode_config(test_config)
                            if features.size > 0:
                                features_scaled = self.scaler.transform(features)
                                score = surrogate.predict(features_scaled)[0]
                                category_scores.append((cat, score))
                            else:
                                category_scores.append((cat, 0))
                        
                        # Sort by predicted score
                        category_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # Keep top 3 categories
                        refined_space[pname] = [cat for cat, _ in category_scores[:3]]
            
            return refined_space
        
        # Default: return original space
        return param_space
    
    def _propose_using_surrogate(self, surrogate, param_space):
        """Propose a promising configuration using the surrogate model"""
        # Get parameter names in a consistent order
        param_names = sorted(list(param_space.keys()))
        
        # Use gradient-based optimization of acquisition function
        proposed_config = self._optimize_acquisition(
            surrogate, 
            self.best_score_, 
            param_names
        )
        
        return proposed_config
    
    def _propose_batch_using_surrogate(self, surrogate, param_space, batch_size=5):
        """Propose multiple diverse configurations using the surrogate model"""
        # First proposal: optimize acquisition function
        proposals = [self._propose_using_surrogate(surrogate, param_space)]
        
        # Additional proposals with diversity promotion
        for _ in range(batch_size - 1):
            # Sample candidates
            candidates = self._sample_random_configs(100)
            
            # Convert to features
            X_candidates = self._configs_to_features(candidates)
            
            # Skip if no features
            if X_candidates.shape[0] == 0 or X_candidates.shape[1] == 0:
                proposals.append(self._sample_random_configs(1)[0])
                continue
                
            X_scaled = self.scaler.transform(X_candidates)
            
            # Predict scores
            predicted_scores = surrogate.predict(X_scaled)
            
            # Calculate diversity penalty
            diversity_scores = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                # Calculate minimum distance to existing proposals
                min_distance = float('inf')
                candidate_features = self._encode_config(candidate)
                
                if candidate_features.size == 0:
                    diversity_scores[i] = -float('inf')
                    continue
                    
                candidate_scaled = self.scaler.transform(candidate_features)
                
                for proposal in proposals:
                    proposal_features = self._encode_config(proposal)
                    
                    if proposal_features.size == 0:
                        continue
                        
                    proposal_scaled = self.scaler.transform(proposal_features)
                    
                    # Euclidean distance in scaled feature space
                    distance = np.sqrt(np.sum((candidate_scaled - proposal_scaled) ** 2))
                    min_distance = min(min_distance, distance)
                
                # Combine predicted score with diversity
                diversity_scores[i] = predicted_scores[i] + 0.1 * min_distance
            
            # Select the candidate with the best combined score
            best_idx = np.argmax(diversity_scores)
            proposals.append(candidates[best_idx])
        
        return proposals
    
    def fit(self, X, y):
        """Run the ASHT optimization process"""
        self.X = X
        self.y = y
        
        # Phase 1: Initial Exploration with low-fidelity evaluations
        R = 1.0  # full resource
        B = R * 0.1  # start with 10% of full resource
        
        # Adaptive initial exploration based on parameter space size
        param_space_size = np.prod([
            len(v) if isinstance(v, list) else 10 
            for v in self.param_space.values()
        ])
        N = min(self.max_iter // 4, int(np.log10(param_space_size) * 10))
        N = max(5, N)  # At least 5 initial points
        
        if self.verbose:
            print(f"ASHT Phase 1: Exploring {N} initial configurations with budget {B:.2f}")
        
        initial_configs = self._sample_random_configs(N)
        results = []
        for config in tqdm(initial_configs, disable=not self.verbose):
            score = self._objective_func(config, budget=B)
            results.append((config, score))
        
        # Keep top configurations based on score
        results.sort(key=lambda x: x[1], reverse=True)
        top_k = max(3, N // 2)  # Keep at least 3 configurations
        promising_configs = [cfg for (cfg, _) in results[:top_k]]
        promising_scores = [s for (_, s) in results[:top_k]]
        
        # Train surrogate model
        surrogate = self._train_surrogate_model(promising_configs, promising_scores)
        
        # Refine parameter space
        refined_param_space = self._refine_param_space(self.param_space, surrogate)
        
        # Phase 2: Iterative focused search
        remaining_iter = self.max_iter - N
        
        if self.verbose:
            print(f"ASHT Phase 2: Focused search with {remaining_iter} iterations")
        
        all_results = results[:]  # Store all results
        
        while remaining_iter > 0:
            # Determine batch size based on remaining iterations
            batch_size = min(5, remaining_iter)
            
            # Propose batch of configurations
            candidates = self._propose_batch_using_surrogate(surrogate, refined_param_space, batch_size)
            
            # Increase budget
            B = min(R, B * 1.5)
            
            if self.verbose:
                print(f"  Iteration with budget {B:.2f}, {len(candidates)} candidates")
            
            # Evaluate candidates
            new_results = []
            for config in tqdm(candidates, disable=not self.verbose):
                score = self._objective_func(config, budget=B)
                new_results.append((config, score))
                remaining_iter -= 1
                
                # Early stopping if we've used all iterations
                if remaining_iter <= 0:
                    break
            
            # Update surrogate model with all results
            all_configs = [cfg for (cfg, _) in all_results + new_results]
            all_scores = [s for (_, s) in all_results + new_results]
            surrogate = self._train_surrogate_model(all_configs, all_scores)
            
            # Refine parameter space again
            refined_param_space = self._refine_param_space(refined_param_space, surrogate)
            
            all_results += new_results
        
        # Set the best estimator
        if self.best_params_ is not None:
            try:
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_.set_params(**self.best_params_)
                self.best_estimator_.fit(X, y)
            except Exception as e:
                if self.verbose:
                    print(f"Error fitting best estimator: {e}")
                # Fallback: fit with default parameters
                self.best_estimator_ = clone(self.estimator)
                self.best_estimator_.fit(X, y)
        else:
            # No good parameters found, use default
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.fit(X, y)
        
        return self

