# Fix 1: Completely rewrite the _update_surrogate_predictions method to handle each model type correctly
def _update_surrogate_predictions(self, surrogate_models):
    """Update surrogate predictions in cv_results_"""
    if not surrogate_models:
        return
        
    # Get configurations and convert to features
    configs = self.cv_results_['params']
    X_configs = self._configs_to_features(configs)
    
    if X_configs.shape[0] == 0 or X_configs.shape[1] == 0:
        return
        
    X_scaled = self.scaler.transform(X_configs)
    
    # Get predictions from surrogate models
    if self.ensemble_surrogate and len(self.active_surrogates) > 1:
        # Use all models with weights
        means = []
        stds = []
        
        for i, model_name in enumerate(self.active_surrogates):
            if model_name in surrogate_models:
                model = surrogate_models[model_name]
                
                # Different prediction method based on model type
                if model_name == 'gp':
                    # Gaussian Process has return_std parameter
                    try:
                        mean, std = model.predict(X_scaled, return_std=True)
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        if self.verbose > 1:
                            print(f"Error in GP prediction: {e}")
                
                elif model_name == 'rf':
                    # Random Forest - manually calculate std from trees
                    try:
                        # First get the mean prediction
                        mean = model.predict(X_scaled)
                        
                        # Then get predictions from individual trees
                        tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                        std = np.std(tree_preds, axis=0)
                        
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        if self.verbose > 1:
                            print(f"Error in RF prediction: {e}")
                
                elif model_name == 'nn':
                    # Neural Network - use distance-based uncertainty
                    try:
                        mean = model.predict(X_scaled)
                        
                        # Simple uncertainty based on distance to training data
                        if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                            dists = np.min(np.sum((X_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                            std = np.sqrt(dists) * 0.1 + 0.05
                        else:
                            std = np.ones_like(mean) * 0.1
                            
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        if self.verbose > 1:
                            print(f"Error in NN prediction: {e}")
                
                elif model_name == 'dummy':
                    # Dummy model
                    try:
                        mean = model.predict(X_scaled)
                        std = np.ones_like(mean) * 0.1
                        means.append(mean)
                        stds.append(std)
                    except Exception as e:
                        if self.verbose > 1:
                            print(f"Error in dummy prediction: {e}")
        
        # Skip ensemble calculation if no predictions available
        if not means:
            return
                
        # Calculate ensemble prediction
        if self.surrogate_weights is not None:
            # Make sure we have the right number of weights
            if len(self.surrogate_weights) != len(means):
                # If mismatch, just use equal weights
                self.surrogate_weights = np.ones(len(means)) / len(means)
                
            # Weighted mean and variance
            weights = self.surrogate_weights
            mu = np.zeros_like(means[0])
            
            # Compute weighted mean
            for i, mean in enumerate(means):
                mu += weights[i] * mean
                
            # Compute weighted variance (including model disagreement)
            total_var = np.zeros_like(stds[0])
            
            # Within-model variance
            for i, std in enumerate(stds):
                total_var += weights[i] * (std ** 2)
                
            # Between-model variance (disagreement)
            for i, mean in enumerate(means):
                total_var += weights[i] * ((mean - mu) ** 2)
                
            sigma = np.sqrt(total_var)
        else:
            # Equal weights if no weights provided
            mu = np.mean(means, axis=0)
            
            # Combine within-model and between-model variance
            within_var = np.mean([std**2 for std in stds], axis=0)
            between_var = np.var(means, axis=0)
            total_var = within_var + between_var
            
            sigma = np.sqrt(total_var)
            
        # Update cv_results_
        for i in range(len(self.cv_results_['surrogate_prediction'])):
            if i < len(mu):
                self.cv_results_['surrogate_prediction'][i] = mu[i]
                self.cv_results_['surrogate_uncertainty'][i] = sigma[i]
    else:
        # Use single best model
        if not self.active_surrogates:
            return
            
        model_name = self.active_surrogates[0]
        if model_name not in surrogate_models:
            return
            
        model = surrogate_models[model_name]
        
        # Different prediction method based on model type
        if model_name == 'gp':
            try:
                means, stds = model.predict(X_scaled, return_std=True)
            except Exception:
                return
        elif model_name == 'rf':
            try:
                means = model.predict(X_scaled)
                
                # Get std from individual trees
                tree_preds = np.array([tree.predict(X_scaled) for tree in model.estimators_])
                stds = np.std(tree_preds, axis=0)
            except Exception:
                return
        elif model_name == 'nn':
            try:
                means = model.predict(X_scaled)
                
                # Simple uncertainty
                if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                    dists = np.min(np.sum((X_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                    stds = np.sqrt(dists) * 0.1 + 0.05
                else:
                    stds = np.ones_like(means) * 0.1
            except Exception:
                return
        else:
            # Dummy or unknown model
            try:
                means = model.predict(X_scaled)
                stds = np.ones_like(means) * 0.1
            except Exception:
                return
                
        # Store predictions
        for i in range(len(self.cv_results_['surrogate_prediction'])):
            if i < len(means):
                self.cv_results_['surrogate_prediction'][i] = means[i]
                self.cv_results_['surrogate_uncertainty'][i] = stds[i]

# Fix 2: Remove predict_with_std from _initialize_surrogate_models since we handle each model type directly in _update_surrogate_predictions
def _initialize_surrogate_models(self):
    """Initialize surrogate models based on meta-learning or ensemble strategy"""
    self.surrogate_models = {}
    
    # Base GP model with Matern kernel (good for hyperparameter optimization)
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
    base_gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=self.random_state
    )
    
    # Random Forest model (robust for mixed parameter types)
    base_rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=3,
        random_state=self.random_state,
        n_jobs=min(self.n_jobs, 4)  # Limit RF internal parallelism
    )
    
    # Neural network model for complex landscapes
    base_nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        early_stopping=True,
        random_state=self.random_state
    )
    
    # Register surrogate models
    self.surrogate_models['gp'] = base_gp
    self.surrogate_models['rf'] = base_rf
    self.surrogate_models['nn'] = base_nn
    
    # For ensembling we use a meta-model
    if self.ensemble_surrogate:
        self.meta_model = Ridge(alpha=1.0)
        self.surrogate_weights = None
        self.active_surrogates = ['gp', 'rf']  # Start with these two
    else:
        # Default to Gaussian Process
        self.active_surrogates = ['gp']
    
    # Current surrogate model state
    self.trained_surrogates = {}
    
    # Initialize meta-model selection tracking
    self.surrogate_performances = {model: [] for model in self.surrogate_models}
    
    # Store training data for uncertainty estimation
    self.X_valid = None
    self.y_valid = None

# Fix 3: Update _train_surrogate_models to store training data without adding predict_with_std
def _train_surrogate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Train surrogate models on evaluated configurations"""
    trained_models = {}
    
    if X.shape[0] < 2:  # Need at least 2 samples
        return trained_models
        
    # Scale features
    X_scaled = self.scaler.fit_transform(X)
    
    # Direction adjustment for minimization
    y_adj = y.copy()
    if not self.maximize:
        y_adj = -y_adj
        
    # Filter out non-finite values
    finite_mask = np.isfinite(y_adj)
    if np.sum(finite_mask) < 2:  # Need at least 2 valid samples
        return trained_models
        
    X_valid = X_scaled[finite_mask]
    y_valid = y_adj[finite_mask]
    
    # Store for uncertainty estimation
    self.X_valid = X_valid
    self.y_valid = y_valid
    
    # Train each surrogate model in the active set
    model_errors = {}
    
    for model_name in self.active_surrogates:
        model = self.surrogate_models[model_name]
        
        try:
            # Train the model
            model.fit(X_valid, y_valid)
            
            # Calculate cross-validation error for model selection
            if len(X_valid) >= 5:  # Need enough data for CV
                from sklearn.model_selection import cross_val_predict
                y_pred = cross_val_predict(model, X_valid, y_valid, cv=min(5, len(X_valid)))
                mse = np.mean((y_valid - y_pred) ** 2)
                model_errors[model_name] = mse
                
                # Store for meta-learning
                self.surrogate_performances[model_name].append(mse)
            
            # Add to trained models
            trained_models[model_name] = model
            
        except Exception as e:
            if self.verbose > 1:
                print(f"Error training {model_name} surrogate: {e}")
    
    # Meta-model selection based on performance
    if model_errors and self.meta_learning:
        # Check if we should switch models
        current_best = min(model_errors, key=model_errors.get)
        
        # Record for meta-learning
        if X.shape[1] > 0:  # Skip if no features
            self.meta_learning_data['problem_features'].append({
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'y_mean': np.mean(y_valid),
                'y_std': np.std(y_valid),
                'iteration': self.iteration_count
            })
            self.meta_learning_data['best_surrogate'].append(current_best)
            self.meta_learning_data['surrogate_performance'].append(model_errors)
        
        # Update active surrogates
        if self.ensemble_surrogate:
            # Keep all models, but update weights in acquisition
            self.active_surrogates = list(trained_models.keys())
            
            # Compute weights inversely proportional to error
            errors = np.array([model_errors.get(m, float('inf')) for m in self.active_surrogates])
            if np.all(np.isfinite(errors)) and np.sum(errors) > 0:
                weights = 1.0 / (errors + 1e-10)
                self.surrogate_weights = weights / np.sum(weights)
            else:
                self.surrogate_weights = np.ones(len(self.active_surrogates)) / len(self.active_surrogates)
        else:
            # Use single best model
            self.active_surrogates = [current_best]
    
    # If no models could be trained, create a dummy surrogate
    if not trained_models:
        trained_models['dummy'] = self._create_dummy_surrogate(y)
        self.active_surrogates = ['dummy']
    
    return trained_models

# Fix 4: Update the _acquisition_function to handle each model type correctly
def _acquisition_function(self, x: np.ndarray, models: Dict[str, Any], best_f: float, xi: float = 0.01) -> float:
    """
    Advanced acquisition function with ensemble support and adaptive exploration
    
    Parameters:
    -----------
    x : array-like
        Point to evaluate
    
    models : dict
        Dictionary of trained surrogate models
    
    best_f : float
        Best function value observed so far
    
    xi : float
        Exploration-exploitation parameter
        
    Returns:
    --------
    acquisition_value : float
        Value of acquisition function
    """
    # Reshape x if needed
    if x.ndim == 1:
        x = x.reshape(1, -1)
        
    # Scale features
    x_scaled = self.scaler.transform(x)
    
    # Check which acquisition function to use based on iteration
    acq_type = self._select_acquisition_function()
    
    # Compute predictions from all models
    all_means = []
    all_stds = []
    
    for i, model_name in enumerate(self.active_surrogates):
        if model_name not in models:
            continue
            
        model = models[model_name]
        
        # Different prediction method based on model type
        if model_name == 'gp':
            try:
                mean, std = model.predict(x_scaled, return_std=True)
                all_means.append(mean)
                all_stds.append(std)
            except Exception as e:
                if self.verbose > 1:
                    print(f"Error in GP acquisition prediction: {e}")
        
        elif model_name == 'rf':
            try:
                # Mean prediction
                mean = model.predict(x_scaled)
                
                # Get std from individual trees
                tree_preds = np.array([tree.predict(x_scaled) for tree in model.estimators_])
                std = np.std(tree_preds, axis=0)
                
                all_means.append(mean)
                all_stds.append(std)
            except Exception as e:
                if self.verbose > 1:
                    print(f"Error in RF acquisition prediction: {e}")
        
        elif model_name == 'nn':
            try:
                mean = model.predict(x_scaled)
                
                # Simple uncertainty
                if hasattr(self, 'X_valid') and self.X_valid is not None and self.X_valid.shape[0] > 0:
                    dists = np.min(np.sum((x_scaled[:, np.newaxis, :] - self.X_valid[np.newaxis, :, :]) ** 2, axis=2), axis=1)
                    std = np.sqrt(dists) * 0.1 + 0.05
                else:
                    std = np.ones_like(mean) * 0.1
                    
                all_means.append(mean)
                all_stds.append(std)
            except Exception as e:
                if self.verbose > 1:
                    print(f"Error in NN acquisition prediction: {e}")
        
        else:  # dummy or other
            try:
                mean = model.predict(x_scaled)
                std = np.ones_like(mean) * 0.1
                
                all_means.append(mean)
                all_stds.append(std)
            except Exception as e:
                if self.verbose > 1:
                    print(f"Error in dummy acquisition prediction: {e}")
    
    # If no models provided valid predictions, return a default value
    if not all_means:
        return 0.0
        
    # Compute ensemble prediction
    if self.ensemble_surrogate and self.surrogate_weights is not None and len(all_means) > 1:
        # Ensure we have correct number of weights
        if len(self.surrogate_weights) != len(all_means):
            # Equal weights as fallback
            weights = np.ones(len(all_means)) / len(all_means)
        else:
            weights = self.surrogate_weights
            
        # Weighted mean
        mu = np.zeros_like(all_means[0])
        for i, mean in enumerate(all_means):
            mu += weights[i] * mean
            
        # Weighted variance (including model disagreement)
        total_var = np.zeros_like(all_stds[0])
        
        # Within-model variance
        for i, std in enumerate(all_stds):
            total_var += weights[i] * (std ** 2)
            
        # Between-model variance (disagreement)
        for i, mean in enumerate(all_means):
            total_var += weights[i] * ((mean - mu) ** 2)
            
        sigma = np.sqrt(total_var)
    else:
        # Single model or equal weights
        mu = all_means[0]
        sigma = all_stds[0]
    
    # Handle zero uncertainty
    if np.all(sigma < 1e-6):
        sigma = np.ones_like(sigma) * 1e-6
    
    # Compute acquisition value based on selected function
    if acq_type == 'ei':  # Expected Improvement
        # Adjust for maximization/minimization
        if self.maximize:
            imp = mu - best_f - xi
        else:
            # For minimization problems, we negate predictions
            imp = best_f - mu - xi
            
        # Z-score
        z = imp / sigma
        
        # EI formula: (imp * CDF(z) + sigma * pdf(z))
        cdf = 0.5 * (1 + np.array([np.math.erf(val / np.sqrt(2)) for val in z]))
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        
        ei = imp * cdf + sigma * pdf
        acquisition = ei[0]  # Take first element for single point
        
    elif acq_type == 'ucb':  # Upper Confidence Bound
        # Adaptive beta based on iteration count
        beta = 0.5 + np.log(1 + self.iteration_count)
        
        if self.maximize:
            acquisition = mu[0] + beta * sigma[0]
        else:
            acquisition = -(mu[0] - beta * sigma[0])
            
    elif acq_type == 'poi':  # Probability of Improvement
        if self.maximize:
            z = (mu - best_f - xi) / sigma
        else:
            z = (best_f - mu - xi) / sigma
            
        # POI is just the CDF
        acquisition = 0.5 * (1 + np.math.erf(z[0] / np.sqrt(2)))
        
    elif acq_type == 'thompson':  # Thompson Sampling
        # Sample from posterior
        if self.maximize:
            acquisition = mu[0] + sigma[0] * np.random.randn()
        else:
            acquisition = -(mu[0] + sigma[0] * np.random.randn())
            
    else:  # Default to EI
        if self.maximize:
            imp = mu - best_f - xi
        else:
            imp = best_f - mu - xi
            
        z = imp / sigma
        cdf = 0.5 * (1 + np.array([np.math.erf(val / np.sqrt(2)) for val in z]))
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        
        ei = imp * cdf + sigma * pdf
        acquisition = ei[0]
    
    # Return negative for minimization with scipy optimize
    return -acquisition

# Fix 5: Make sure fit strictly adheres to max_iter
def fit(self, X, y):
    """
    Run the HyperOptX optimization process
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
        
    y : array-like, shape (n_samples,)
        Target values
    
    Returns:
    --------
    self : object
        Returns self
    """
    self.X = X
    self.y = y
    
    # Start timing
    self.start_time = time.time()
    
    # Track current iteration number to enforce max_iter
    current_iter = 0
    
    # Extract problem features for meta-learning
    if self.meta_learning:
        problem_features = self._extract_problem_features(X, y)
        if self.verbose:
            print(f"Problem features: {problem_features}")
    
    # Create multi-fidelity budget schedule
    budget_schedule = self._multi_fidelity_schedule(self.max_iter)
    
    # Track scores and configs for early stopping
    all_scores = []
    all_configs = []
    eval_times = []
    
    # Strategy determination
    selected_strategy = self.optimization_strategy
    if selected_strategy == 'auto':
        # Choose strategy based on problem size
        if X.shape[1] > 50 or len(self.param_space) > 10:
            # High-dimensional problems: prefer evolutionary
            selected_strategy = 'evolutionary'
        elif len([p for p, t in self.param_types.items() if t == 'categorical']) > len(self.param_types) / 2:
            # Many categorical parameters: prefer bayesian
            selected_strategy = 'bayesian'
        else:
            # Default to hybrid
            selected_strategy = 'hybrid'
            
    if self.verbose:
        print(f"Selected optimization strategy: {selected_strategy}")
        print(f"Initializing HyperOptX with {self.max_iter} maximum iterations")
    
    # Strict enforcement of max_iter
    # Phase 1: Initial exploration with fewer samples to stay within max_iter
    n_initial = max(1, min(self.max_iter // 4, 5))
    
    if self.verbose:
        print(f"Phase 1: Initial exploration with {n_initial} configurations")
    
    # Generate initial configurations
    initial_configs = self._sample_configurations(n_initial, 'mixed')
    
    # Evaluate initial configs
    initial_results = []
    for config in initial_configs:
        if current_iter >= self.max_iter:
            break
            
        score = self._objective_func(config, budget=0.5)  # 50% budget
        initial_results.append((config, score))
        all_scores.append(score)
        all_configs.append(config)
        
        # Update counters
        current_iter += 1
        self.iteration_count += 1
        
        # Update evolutionary population
        if selected_strategy in ['evolutionary', 'hybrid']:
            self._update_population([config], [score])
    
    # Sort by score
    if self.maximize:
        initial_results.sort(key=lambda x: x[1], reverse=True)
    else:
        initial_results.sort(key=lambda x: x[1])
    
    # Initialize surrogate models with these results
    X_init = self._configs_to_features([cfg for cfg, _ in initial_results])
    y_init = np.array([score for _, score in initial_results])
    
    # Train initial surrogate models
    surrogate_models = self._train_surrogate_models(X_init, y_init)
    
    # Iterative optimization phase
    remaining_iter = self.max_iter - current_iter
    
    if remaining_iter > 0 and self.verbose:
        print(f"Phase 2: Iterative optimization with {remaining_iter} iterations")
    
    while current_iter < self.max_iter:
        # Current budget from schedule
        current_budget = budget_schedule[min(current_iter, len(budget_schedule)-1)]
        
        # Different proposal strategies based on selected strategy
        if selected_strategy == 'bayesian':
            # Bayesian optimization: use acquisition function
            current_best = max(all_scores) if self.maximize else min(all_scores)
            next_config = self._optimize_acquisition(surrogate_models, current_best)
            candidates = [next_config]
            
        elif selected_strategy == 'evolutionary':
            # Evolutionary optimization: generate offspring
            if len(self.population) >= 5:
                candidates = self._evolutionary_search(n_offspring=1)  # Reduced to just 1
            else:
                # Not enough population yet, use mixed strategy
                candidates = self._sample_configurations(1, 'mixed')  # Reduced to just 1
                
        elif selected_strategy == 'hybrid':
            # Hybrid: mix of bayesian and evolutionary
            candidates = []
            
            # Bayesian candidate
            current_best = max(all_scores) if self.maximize else min(all_scores)
            bayes_config = self._optimize_acquisition(surrogate_models, current_best)
            candidates.append(bayes_config)
            
            # Don't add more candidates if we're near max_iter
            if current_iter < self.max_iter - 1:
                # Add just one more candidate
                if len(self.population) >= 5:
                    evo_config = self._evolutionary_search(n_offspring=1)[0]
                    candidates.append(evo_config)
                else:
                    random_config = self._sample_configurations(1, 'random')[0]
                    candidates.append(random_config)
        
        else:  # Default to mixed sampling
            candidates = self._sample_configurations(1, 'mixed')  # Reduced to just 1
        
        # Limit candidates to respect remaining iterations
        candidates = candidates[:max(1, self.max_iter - current_iter)]
        
        # Evaluate candidates
        batch_scores = []
        batch_times = []
        
        if self.verbose > 0 and (current_iter % 5 == 0 or self.verbose >= 2):
            print(f"  Iteration {current_iter+1}/{self.max_iter}: " + 
                  f"budget={current_budget:.2f}, evaluating {len(candidates)} candidates")
        
        for config in candidates:
            # Check for early stopping
            if self._needs_early_stopping(current_iter, all_scores, eval_times):
                if self.verbose:
                    print(f"Early stopping at iteration {current_iter+1}")
                break
            
            # Check if we've reached max_iter
            if current_iter >= self.max_iter:
                break
                
            # Evaluate with current budget
            eval_start = time.time()
            score = self._objective_func(config, budget=current_budget)
            eval_time = time.time() - eval_start
            
            batch_scores.append(score)
            batch_times.append(eval_time)
            
            all_scores.append(score)
            all_configs.append(config)
            eval_times.append(eval_time)
            
            # Update evolutionary population
            if selected_strategy in ['evolutionary', 'hybrid']:
                self._update_population([config], [score])
            
            # Update iteration counter
            current_iter += 1
            self.iteration_count += 1
        
        # Retrain surrogate models with all data
        X_all = self._configs_to_features(all_configs)
        y_all = np.array(all_scores)
        
        # Only retrain if we haven't reached max_iter yet
        if current_iter < self.max_iter and len(all_configs) >= 2:
            surrogate_models = self._train_surrogate_models(X_all, y_all)
            
            # Update surrogate predictions in cv_results
            self._update_surrogate_predictions(surrogate_models)
        
        # Check for early stopping
        if self._needs_early_stopping(current_iter, all_scores, eval_times):
            if self.verbose:
                print(f"Early stopping at iteration {current_iter}")
            break
    
    # Final evaluation of best configurations with full budget
    if current_iter >= self.max_iter and self.verbose:
        print("Phase 3: Final evaluation of top configurations with full budget")
    
        # Get top configurations
        indices = np.argsort(all_scores)
        if self.maximize:
            indices = indices[::-1]  # Reverse for maximization
            
        # Take top 3 configurations
        top_k = min(3, len(all_configs))
        top_configs = [all_configs[i] for i in indices[:top_k]]
        
        # Re-evaluate with full budget if needed
        for config in top_configs:
            # Skip if already evaluated with full budget
            param_key = frozenset(config.items())
            if param_key in self.evaluated_configs and 1.0 in self.evaluated_configs[param_key]:
                continue
                
            # We don't count this towards max_iter since it's a final evaluation
            self._objective_func(config, budget=1.0)
    
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
    
    # Display final results
    if self.verbose:
        print("\nOptimization completed:")
        print(f"Total iterations: {current_iter}")
        print(f"Best score: {self.best_score_:.6f}")
        print(f"Best parameters: {self.best_params_}")
        print(f"Time elapsed: {time.time() - self.start_time:.2f} seconds")
    
    return self

# Fix 6: Fix the early stopping method
def _needs_early_stopping(self, iteration: int, scores: List[float], times: List[float]) -> bool:
    """Determine if optimization should be stopped early"""
    try:
        if not self.early_stopping:
            return False
            
        # Need at least 10 iterations to detect trends
        if iteration < 10:
            return False
            
        # Stop if time budget exceeded
        if self.time_budget is not None and (time.time() - self.start_time) > self.time_budget:
            return True
            
        # Check for score convergence
        if len(scores) >= 10:
            recent_scores = scores[-10:]
            
            # Extract valid scores
            valid_scores = [s for s in recent_scores if np.isfinite(s)]
            if len(valid_scores) < 5:  # Need enough valid scores
                return False
                
            # Convert to numpy array for operations
            valid_scores = np.array(valid_scores)
            
            # Use simple linear fit to detect trend
            x = np.arange(len(valid_scores)).reshape(-1, 1)
            
            try:
                # Reset and fit the model
                self.learning_curve_model = Ridge(alpha=1.0)
                self.learning_curve_model.fit(x, valid_scores)
                slope = self.learning_curve_model.coef_[0]
                
                # Calculate improvement percentage
                if len(valid_scores) > 1:
                    score_range = np.max(valid_scores) - np.min(valid_scores)
                    if score_range == 0:  # No variation in scores
                        return True
                        
                    # Calculate relative slope
                    norm_slope = slope / score_range
                    
                    # If maximizing, check if slope is very small positive or negative
                    if self.maximize:
                        return bool(norm_slope < 0.001)
                    # If minimizing, check if slope is very small negative or positive
                    else:
                        return bool(norm_slope > -0.001)
            except Exception:
                # If error in slope calculation, don't trigger early stopping
                return False
                
        return False
    except Exception:
        # Safety catch-all to avoid halting optimization due to early stopping error
        return False