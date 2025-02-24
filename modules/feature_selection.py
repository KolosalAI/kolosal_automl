import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score


class MRMRSelector:
    """
    Enhanced mRMR feature selector with:
    
    - Precomputation of feature-feature mutual information (MI).
    - Parallel processing of candidate feature selection.
    - Support for an unsupervised (clustering) mode.
    
    For supervised tasks (classification/regression), the relevance score is computed 
    as the mutual information (MI) between each feature and the target. 
    
    For unsupervised tasks (clustering), where no target vector is provided, 
    the feature entropy is used as a proxy for relevance.
    
    Parameters
    ----------
    n_features_to_select : int, None, or "auto", default=10
        Number of features to select.
        - If an integer > 0, the process stops once that many features are selected.
        - If None or "auto", features are added until the candidate's mRMR
          score (relevance - redundancy) falls below `score_threshold`.
    task : str, default="classification"
        Type of task. Options are "classification", "regression", or "clustering". 
        For clustering, y should be None (or it will be ignored).
    discrete_features : bool or array-like of shape (n_features,), default=False
        Indicates whether features are discrete (True) or continuous (False).
        May be a single boolean or an array of booleans of length n_features for mixed data.
    score_threshold : float, default=0.0
        Minimum improvement in mRMR score (relevance - redundancy) for a new feature 
        to be added in auto mode.
    precompute_mi : bool or 'auto', default='auto'
        Whether to precompute the MI matrix for all feature-feature pairs. 
        If 'auto', precomputation is used when n_features <= mi_threshold.
    mi_threshold : int, default=1000
        Maximum number of features for which MI precomputation is performed when
        `precompute_mi` is 'auto'.
    n_jobs : int, default=None
        Number of parallel jobs to use for candidate evaluation in each selection iteration.
        If None, uses a single core.
        
    Attributes
    ----------
    selected_features_ : ndarray of shape (n_selected_features,)
        Indices of the selected features in ascending order.
    _mi_matrix : ndarray of shape (n_features, n_features) or None
        Precomputed mutual information (MI) matrix for all feature pairs, if applicable.
    _mi_cache : dict
        A dictionary cache for on-the-fly MI computations when precomputation is disabled.
    """

    def __init__(
        self,
        n_features_to_select=None,
        task="classification",
        discrete_features=False,
        score_threshold=0.0,
        precompute_mi='auto',
        mi_threshold=1000,
        n_jobs=None
    ):
        self.n_features_to_select = n_features_to_select
        self.task = task
        self.discrete_features = discrete_features
        self.score_threshold = score_threshold
        self.precompute_mi = precompute_mi
        self.mi_threshold = mi_threshold
        self.n_jobs = n_jobs
        
        self.selected_features_ = None
        self._mi_matrix = None
        self._mi_cache = {}

    def _check_task_and_labels(self, X, y):
        """
        Validate inputs based on the task type.
        Returns the properly checked X, y, and a boolean indicating unsupervised mode.
        """
        valid_tasks = ("classification", "regression", "clustering")
        if self.task not in valid_tasks:
            raise ValueError(
                f"Unknown task type '{self.task}'. "
                f"Choose from {valid_tasks}."
            )
        
        if self.task == "clustering":
            # Ignore y if provided; treat as None for unsupervised
            return check_array(X), None, True
        else:
            # Supervised learning task
            X, y = check_X_y(X, y)
            return X, y, False

    def _check_discrete_features(self, n_features):
        """
        Validate or expand self.discrete_features into an array of bool of shape (n_features,).
        """
        discrete_features = self.discrete_features
        if isinstance(discrete_features, bool):
            # Single bool -> expand
            discrete_features_ = np.array([discrete_features] * n_features, dtype=bool)
        else:
            # Must be array-like of length n_features
            discrete_features_ = np.array(discrete_features, dtype=bool)
            if len(discrete_features_) != n_features:
                raise ValueError(
                    "Length of 'discrete_features' array must match the number of features."
                )
        return discrete_features_

    def _compute_entropy(self, feature, bins=10, discrete=False):
        """
        Compute the entropy of a single feature.
        
        If 'discrete=True', uses the unique counts approach.
        Otherwise, uses histogram binning for continuous data.
        """
        feature = np.ravel(feature)
        if discrete:
            values, counts = np.unique(feature, return_counts=True)
            probabilities = counts / counts.sum()
        else:
            counts, _ = np.histogram(feature, bins=bins, density=False)
            probabilities = counts / counts.sum()
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log(probabilities))

    def _compute_pairwise_mi(self, X, i, j, disc_i, disc_j):
        """
        Compute the mutual information between features i and j.
        
        We use:
        - mutual_info_score if both features are discrete;
        - mutual_info_regression if both are continuous;
        - a 'mixed' approach if one is discrete and the other is continuous.

        For the mixed approach (discrete + continuous), we treat the continuous one
        as 'X' and the discrete one as 'y' in mutual_info_classif. This is a heuristic
        (scikit-learn doesn't provide a built-in discrete-continuous MI estimator).
        """
        # Check cache
        key = (i, j) if i < j else (j, i)
        if key in self._mi_cache:
            return self._mi_cache[key]

        Xi = X[:, i]
        Xj = X[:, j]

        # Both discrete
        if disc_i and disc_j:
            # mutual_info_score expects 1D discrete arrays
            mi_val = mutual_info_score(Xi, Xj)
        # Both continuous
        elif not disc_i and not disc_j:
            # mutual_info_regression: X must be 2D, y is 1D
            mi_val = mutual_info_regression(Xi.reshape(-1, 1), Xj, discrete_features=False)[0]
        else:
            # Mixed case: one discrete, one continuous
            # Heuristic: use mutual_info_classif with discrete y
            # Identify which is discrete
            if disc_i:
                # Xi is discrete, Xj is continuous
                mi_val = mutual_info_classif(
                    Xj.reshape(-1, 1), Xi, discrete_features=False
                )[0]
            else:
                # Xj is discrete, Xi is continuous
                mi_val = mutual_info_classif(
                    Xi.reshape(-1, 1), Xj, discrete_features=False
                )[0]

        self._mi_cache[key] = mi_val
        return mi_val

    def _build_mi_matrix(self, X, discrete_features):
        """
        Precompute the full pairwise MI matrix for features in X.
        """
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features), dtype=float)

        # Optionally parallelize the pairwise computations
        def compute_and_fill(i, j):
            mi_ij = self._compute_pairwise_mi(X, i, j, discrete_features[i], discrete_features[j])
            mi_matrix[i, j] = mi_ij
            mi_matrix[j, i] = mi_ij

        # Use a simple double-loop or joblib parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(compute_and_fill)(i, j)
            for i in range(n_features) for j in range(i + 1, n_features)
        )
        return mi_matrix

    def _compute_mi_vector(self, X, y, discrete_features):
        """
        Compute the relevance vector between each feature in X and target y.
        
        For classification: uses mutual_info_classif
        For regression: uses mutual_info_regression
        """
        task = self.task
        if task == "classification":
            # mutual_info_classif can handle discrete_features array
            return mutual_info_classif(X, y, discrete_features=discrete_features)
        elif task == "regression":
            # mutual_info_regression (discrete_features parameter is also available in newer sklearn)
            return mutual_info_regression(X, y, discrete_features=discrete_features)
        else:
            raise ValueError("Unknown supervised task type: must be classification or regression.")

    def fit(self, X, y=None):
        """
        Fit the mRMR selector to the data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.
        y : ndarray of shape (n_samples,), optional
            The target vector. For clustering (unsupervised), y should be None.

        Returns
        -------
        self : object
            The fitted MRMRSelector.
        """
        # Check inputs and determine supervised vs unsupervised
        X, y, unsupervised = self._check_task_and_labels(X, y)
        n_samples, n_features = X.shape

        # Validate discrete_features into a boolean array
        discrete_features = self._check_discrete_features(n_features)

        # Decide whether to precompute the MI matrix
        if self.precompute_mi == 'auto':
            precompute = (n_features <= self.mi_threshold)
        else:
            precompute = bool(self.precompute_mi)

        # Precompute pairwise MI if desired
        if precompute:
            self._mi_matrix = self._build_mi_matrix(X, discrete_features)
        else:
            self._mi_matrix = None  # not used

        # Compute "relevance" for each feature
        if unsupervised:
            # Use entropy as a proxy for relevance in clustering mode
            relevance = np.array([
                self._compute_entropy(X[:, i], bins=10, discrete=discrete_features[i])
                for i in range(n_features)
            ])
        else:
            # Compute MI between each feature and the target
            relevance = self._compute_mi_vector(X, y, discrete_features)

        # Forward selection:
        # 1) Select the feature with the highest relevance as the first one
        # 2) Iteratively pick the feature f among the remaining that maximizes
        #    ( relevance[f] - mean_{s in selected}( MI(f, s) ) )
        auto_mode = (self.n_features_to_select is None or self.n_features_to_select == "auto")

        # Initialize selected feature list
        selected = []
        candidates = list(range(n_features))

        # First feature: highest relevance
        best_first = int(np.argmax(relevance))
        selected.append(best_first)
        candidates.remove(best_first)

        # Iterative selection
        while candidates:
            # Stop if we already have the desired number of features in fixed mode
            if (not auto_mode) and (len(selected) >= self.n_features_to_select):
                break

            if self._mi_matrix is not None:
                # We have precomputed pairwise MIs
                # Compute average redundancy with selected features
                c_array = np.array(candidates)
                s_array = np.array(selected)
                # shape (n_candidates, n_selected) -> mean over axis=1
                redundancies = self._mi_matrix[c_array][:, s_array].mean(axis=1)
                # compute score
                scores = relevance[c_array] - redundancies
                best_idx = int(np.argmax(scores))
                best_feature = c_array[best_idx]
                best_score = scores[best_idx]
            else:
                # No precomputation; compute on the fly
                def compute_score(feature_idx):
                    # average redundancy to already selected
                    if len(selected) == 0:
                        redundancy = 0.0
                    else:
                        mis = [
                            self._compute_pairwise_mi(
                                X, feature_idx, sel, 
                                discrete_features[feature_idx],
                                discrete_features[sel]
                            )
                            for sel in selected
                        ]
                        redundancy = np.mean(mis) if len(mis) > 0 else 0.0
                    return relevance[feature_idx] - redundancy, feature_idx

                scores_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(compute_score)(f) for f in candidates
                )
                best_score, best_feature = max(scores_list, key=lambda x: x[0])

            # In auto mode, stop if the best score is below threshold
            if auto_mode and best_score < self.score_threshold:
                break

            selected.append(best_feature)
            candidates.remove(best_feature)

        # Sort selected feature indices for consistency
        self.selected_features_ = np.sort(selected)
        return self

    def transform(self, X):
        """
        Transform the data by selecting only the features chosen by mRMR.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input feature matrix.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            The feature matrix containing only the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("MRMRSelector has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        """
        Fit to the data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.
        y : ndarray of shape (n_samples,), optional
            Target vector (ignored if clustering).

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            The transformed feature matrix with only the selected features.
        """
        return self.fit(X, y).transform(X)


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=100, 
        n_features=20, 
        n_informative=5, 
        random_state=42
    )
    
    # Example 1: Fixed number of features (e.g., 5 features)
    selector_fixed = MRMRSelector(
        n_features_to_select=5, 
        task="classification",
        discrete_features=False
    )
    selector_fixed.fit(X, y)
    print("Selected feature indices (fixed):", selector_fixed.selected_features_)
    X_selected_fixed = selector_fixed.transform(X)
    print("Shape of the transformed data (fixed):", X_selected_fixed.shape)
    
    # Example 2: Automatic feature selection based on score threshold
    # (stops when best_score < 0.0).
    selector_auto = MRMRSelector(
        n_features_to_select="auto", 
        task="classification",
        discrete_features=False, 
        score_threshold=0.0
    )
    selector_auto.fit(X, y)
    print("Selected feature indices (auto):", selector_auto.selected_features_)
    X_selected_auto = selector_auto.transform(X)
    print("Shape of the transformed data (auto):", X_selected_auto.shape)
