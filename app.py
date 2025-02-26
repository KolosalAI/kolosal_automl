import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import datetime
import os
import joblib
import logging

from typing import List, Dict, Any, Optional, Tuple

# Sklearn imports for demonstration
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score,
    make_scorer  # <-- FIX: Import make_scorer for custom clustering scoring.
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC, SVR

# ------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom scorer for clustering
def clustering_scorer(estimator, X, y=None):
    """
    Custom scorer for clustering using silhouette_score.
    y is ignored since silhouette is an unsupervised metric.
    """
    try:
        labels = estimator.fit_predict(X)
        return silhouette_score(X, labels)
    except Exception:
        return -1

# ------------------------------------------------------------------------
# Import configuration objects from the updated configs.py module
from modules.configs import (
    AutoMLModelConfig as ModelConfig,
    TaskType,
    HyperParameterConfig,
    TargetMetrics,
    DEFAULT_HYPERPARAMETERS,
    CPUAcceleratedModelConfig
)

# Utility: Convert HyperParameterConfig to dictionary domain
def get_tune_domain(hp_config: HyperParameterConfig) -> Dict[str, Any]:
    """
    Example utility for building a hyperparameter domain (if you need it).
    """
    domain_array = np.arange(hp_config.min_val, hp_config.max_val + hp_config.step, hp_config.step)
    return {
        "domain": domain_array.tolist(),
        "init_value": hp_config.default,
        "low_cost_init_value": hp_config.default
    }

# ------------------------------------------------------------------------
# Data Preprocessing & Visualization
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and one-hot encode categorical features."""
    if df.empty:  # <-- FIX: Handle empty DataFrame case
        return df
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            # Fallback to first available mode if there's missing data
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return pd.get_dummies(df, drop_first=True)

def display_target_visualization(df: pl.DataFrame, config: ModelConfig):
    """Display a simple plot of the target distribution based on task type."""
    st.subheader("Target Distribution")
    df_pd = df.to_pandas()
    target = config.target_column
    if target not in df_pd.columns:
        st.error("Target column not found in the dataset.")
        return

    if config.task_type == TaskType.CLASSIFICATION:
        fig, ax = plt.subplots()
        sns.countplot(x=target, data=df_pd, ax=ax)
        ax.set_title(f"Countplot of {target}")
        st.pyplot(fig)

    elif config.task_type == TaskType.REGRESSION:
        fig, ax = plt.subplots()
        sns.histplot(df_pd[target], kde=True, ax=ax)
        ax.set_title(f"Histogram of {target}")
        st.pyplot(fig)

    else:  # For clustering or unknown
        st.info("No direct visualization available for the selected task.")

# ------------------------------------------------------------------------
# Model Instantiation & Evaluation Functions
def get_model_instance(model_name: str, config: ModelConfig):
    rs = config.random_seed
    # Clustering
    if config.task_type == TaskType.CLUSTERING:
        if model_name == "kmeans":
            return KMeans(n_clusters=config.n_clusters, random_state=rs)
        elif model_name == "dbscan":
            return DBSCAN()
        else:
            raise ValueError(f"Clustering model '{model_name}' not implemented.")

    # Classification & Regression
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=rs)
    elif model_name == "linear_regression":
        return LinearRegression()
    elif model_name == "ridge":
        return Ridge(random_state=rs)
    elif model_name == "lasso":
        return Lasso(random_state=rs)
    elif model_name == "rf":
        if config.task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier(random_state=rs)
        else:
            return RandomForestRegressor(random_state=rs)
    elif model_name == "svm":
        if config.task_type == TaskType.CLASSIFICATION:
            return SVC(probability=True, random_state=rs)
        else:
            return SVR()
    elif model_name in ["xgboost", "catboost", "lgbm"]:
        # Placeholder: Replace with actual third-party implementations if needed
        if config.task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier(random_state=rs)
        else:
            return RandomForestRegressor(random_state=rs)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

def evaluate_model(model, X_test, y_test, task_type: TaskType, metric: str) -> Dict[str, float]:
    """
    Evaluate classification or regression models on the given test set.
    Returns a dictionary with metric scores.
    """
    if X_test is None or y_test is None or len(X_test) == 0:
        return {}

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {}

    report = {}
    if task_type == TaskType.CLASSIFICATION:
        # Attempt to retrieve predicted probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                # <-- IMPROVEMENT: handle multi-class; for binary, use proba[:,1]
                if proba.shape[1] == 2:
                    y_pred_proba = proba[:, 1]
                else:
                    # For multi-class, we keep the entire array for optional metrics (e.g. ROC AUC).
                    y_pred_proba = proba
            except Exception as e:
                logger.warning(f"Could not compute predict_proba: {e}")
                y_pred_proba = None

        # Basic classification metrics
        report["accuracy"] = accuracy_score(y_test, y_pred)
        report["f1_macro"] = f1_score(y_test, y_pred, average="macro")
        report["precision_macro"] = precision_score(y_test, y_pred, average="macro")
        report["recall_macro"] = recall_score(y_test, y_pred, average="macro")

        # If probabilities are available, compute ROC AUC (handle multi-class if needed)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2 and y_pred_proba.ndim == 1:
                    # Binary classification
                    report["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
                else:
                    # Multi-class scenario
                    report["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
            except Exception as e:
                logger.warning(f"ROC AUC computation failed: {e}")

    elif task_type == TaskType.REGRESSION:
        # Regression metrics
        report["r2"] = r2_score(y_test, y_pred)
        report["mse"] = mean_squared_error(y_test, y_pred)
        report["mae"] = mean_absolute_error(y_test, y_pred)
        report["rmse"] = np.sqrt(report["mse"])

    return report

def evaluate_clustering(model, X, config: ModelConfig) -> Dict[str, float]:
    """
    Evaluate a clustering model using the specified metric in config (default silhouette).
    """
    try:
        # Some clustering models store labels_ after fit; others require predict.
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
    except Exception as e:
        st.error(f"Error retrieving clustering labels: {e}")
        logger.error(f"Error retrieving clustering labels: {e}")
        return {}

    report = {}
    if config.metric_name == "silhouette":
        try:
            report["silhouette"] = silhouette_score(X, labels)
        except Exception as e:
            st.error(f"Silhouette computation error: {e}")
            logger.error(f"Silhouette computation error: {e}")
    return report

def select_best_model(results: Dict[str, Dict[str, float]], primary_metric: str) -> Optional[str]:
    """
    Given a dictionary of {model_name: {metric_name: score}},
    selects the best model based on the primary_metric.
    """
    best_model, best_score = None, float("-inf")
    # If the metric is an error metric (lower is better), we invert it
    lower_is_better = primary_metric in ["mse", "mae", "rmse"]
    for model_name, metrics in results.items():
        score = metrics.get(primary_metric)
        if score is None:
            continue
        # Invert the score if lower is better
        if lower_is_better:
            score = -score
        if score > best_score:
            best_score, best_model = score, model_name
    return best_model

def get_shap_explanation(model, X_test) -> Tuple[Optional[Any], Optional[plt.Figure]]:
    """
    Generate SHAP explanations for the first 50 samples of X_test.
    Returns SHAP values and a matplotlib Figure.
    """
    if X_test is None or len(X_test) == 0:
        return None, None

    sample_X = X_test.iloc[:50, :].reset_index(drop=True)
    try:
        # Use TreeExplainer for tree-based models; otherwise KernelExplainer
        if hasattr(model, "predict_proba") and "rf" in str(model).lower():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_X)
        else:
            explainer = shap.KernelExplainer(model.predict, sample_X)
            shap_values = explainer.shap_values(sample_X)

        plt.figure()
        # shap_values can be a list for multi-class problems
        if isinstance(shap_values, list):
            # Show summary for the first class if multi-class
            shap.summary_plot(shap_values[0], sample_X, show=False)
            shap_array = np.abs(shap_values[0])
        else:
            shap.summary_plot(shap_values, sample_X, show=False)
            shap_array = np.abs(shap_values)

        fig = plt.gcf()
        mean_abs = np.mean(shap_array, axis=0)
        top_features = [sample_X.columns[i] for i in np.argsort(mean_abs)[::-1][:5]]
        st.info(f"Top contributing features: {', '.join(top_features)}")

        return shap_values, fig
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        st.error("Error generating SHAP explanation.")
        return None, None

# ------------------------------------------------------------------------
# Main Application Class
class GentaAutoMLApp:
    def __init__(self) -> None:
        st.set_page_config(page_title="Genta AutoML", layout="wide")
        st.title("Genta AutoML: Automated Machine Learning Platform")
        self._initialize_session_state()

    def _initialize_session_state(self):
        for key in [
            'data', 'model', 'X_pd', 'y_pd', 'target_metrics',
            'X_selected', 'trained_model', 'evaluation_report',
            'shap_values', 'shap_fig', 'model_config',
            'search_space_area', 'model_trial_history'
        ]:
            if key not in st.session_state:
                st.session_state[key] = None

    # --- Data Loading ---
    def load_data(self) -> Optional[pl.DataFrame]:
        st.header("1. Upload Your Dataset")
        uploaded = st.file_uploader(
            "Upload CSV, Parquet, or Excel file",
            type=['csv', 'parquet', 'xlsx', 'xls'],
            help="Supported formats: CSV, Parquet, Excel"
        )
        if uploaded:
            try:
                ext = uploaded.name.split('.')[-1].lower()
                if ext == 'csv':
                    df = pl.read_csv(uploaded)
                elif ext == 'parquet':
                    df = pl.read_parquet(uploaded)
                elif ext in ['xlsx', 'xls']:
                    df_pd = pd.read_excel(uploaded)
                    df = pl.from_pandas(df_pd)
                else:
                    st.error("Unsupported file format.")
                    return None

                st.session_state['data'] = df
                st.success("Data loaded successfully!")
                self._display_data_preview(df)
                return df

            except Exception as e:
                st.error(f"Error loading data: {e}")
                logger.error(f"Error loading data: {e}")
        else:
            st.info("Please upload a dataset.")
        return None

    def _display_data_preview(self, df: pl.DataFrame):
        st.subheader("Data Preview")
        try:
            st.dataframe(df.head(10).to_pandas())
        except Exception as e:
            st.error(f"Error displaying preview: {e}")

        st.subheader("Data Statistics")
        try:
            # Polars df.describe() can sometimes fail on non-numeric columns
            desc = df.describe()
            st.write(desc.to_pandas())
        except Exception as e:
            st.error(f"Error computing statistics: {e}")
            logger.error(f"Error in df.describe(): {e}")

    # --- Model Configuration ---
    def get_available_models(self, task_type: TaskType) -> Tuple[List[str], List[str]]:
        """
        Returns a tuple of (all_available_models, default_models_for_task).
        """
        options = {
            TaskType.CLASSIFICATION: ["logistic_regression", "rf", "svm", "xgboost", "catboost", "lgbm"],
            TaskType.REGRESSION: ["linear_regression", "ridge", "lasso", "rf", "svm", "xgboost", "catboost", "lgbm"],
            TaskType.CLUSTERING: list(DEFAULT_HYPERPARAMETERS.keys())  # kmeans, dbscan in the default config
        }
        defaults = {
            TaskType.CLASSIFICATION: ["logistic_regression", "rf"],
            TaskType.REGRESSION: ["linear_regression", "rf"],
            TaskType.CLUSTERING: ["kmeans"]
        }
        return options[task_type], defaults[task_type]

    def get_target_metrics(self, task_type: TaskType) -> Tuple[Optional[str], Optional[float]]:
        """Prompt the user to select a target metric and a target score."""
        metric_options = {
            TaskType.CLASSIFICATION: ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"],
            TaskType.REGRESSION: ["r2", "mse", "mae", "rmse"],
            TaskType.CLUSTERING: ["silhouette"]
        }
        metric_explanations = {
            "accuracy": "Proportion of correct predictions (Classification).",
            "f1_macro": "Macro-averaged F1 score (Classification).",
            "precision_macro": "Macro-averaged precision (Classification).",
            "recall_macro": "Macro-averaged recall (Classification).",
            "roc_auc": "Area Under the ROC Curve (Classification).",
            "r2": "Coefficient of determination (Regression).",
            "mse": "Mean squared error (Regression).",
            "mae": "Mean absolute error (Regression).",
            "rmse": "Root mean squared error (Regression).",
            "silhouette": "Silhouette score (Clustering)."
        }

        st.subheader("Select Target Metric and Score")
        metrics = metric_options[task_type]
        metric_name = st.selectbox("Performance Metric", options=metrics)
        if metric_name:
            st.info(metric_explanations.get(metric_name, "No explanation available."))
            # Default target for many metrics is 0.8, or 0.0 for error-based
            default_target = 0.8 if metric_name in [
                "accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"
            ] else 0.0
            target_score = st.number_input(
                f"Enter target {metric_name} score",
                min_value=0.0,
                max_value=1.0 if metric_name in [
                    "accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"
                ] else 1e5,
                value=default_target,
                step=0.01
            )
            return metric_name, float(target_score)
        return None, None

    def get_hyperparameter_ranges(self, selected_models: List[str]) -> Dict[str, List[HyperParameterConfig]]:
        """
        Build a dictionary of {model_name: [HyperParameterConfig, ...]} for the selected models,
        optionally allowing the user to override defaults in an 'advanced' mode.
        """
        st.subheader("Configure Hyperparameters (Optional)")
        use_advanced = st.checkbox("Enable advanced hyperparameter configuration", value=False)
        hyper_configs = {}
        for model in selected_models:
            if model in DEFAULT_HYPERPARAMETERS:
                defaults = DEFAULT_HYPERPARAMETERS[model]
                if use_advanced:
                    st.write(f"### {model.upper()} Hyperparameters")
                    user_configs = []
                    for hp in defaults:
                        with st.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                min_val = st.number_input(
                                    f"{hp.name} min [{model}]",
                                    value=hp.min_val,
                                    step=hp.step
                                )
                            with col2:
                                max_val = st.number_input(
                                    f"{hp.name} max [{model}]",
                                    value=hp.max_val,
                                    step=hp.step
                                )
                            with col3:
                                default_val = st.number_input(
                                    f"{hp.name} default [{model}]",
                                    value=hp.default,
                                    step=hp.step
                                )
                            user_configs.append(
                                HyperParameterConfig(hp.name, min_val, max_val, hp.step, default_val)
                            )
                    hyper_configs[model] = user_configs
                else:
                    hyper_configs[model] = defaults
        return hyper_configs

    def get_model_config(self, df: pl.DataFrame) -> Optional[ModelConfig]:
        """
        Collect model configuration from the user:
        - Task type
        - Target column
        - SHAP usage
        - Time budget
        - Number of clusters (if clustering)
        - Random seed, verbosity
        - Auto model selection
        - Cross validation
        - Chosen models
        - Metric name, target score
        - Hyperparameters
        """
        st.header("2. Model Configuration")
        if df is None or len(df.columns) == 0:
            st.warning("Please upload data with columns first.")
            return None

        with st.expander("Model Settings", expanded=True):
            # Target column is irrelevant for clustering, but we still prompt
            target_column = st.selectbox("Select target column (ignored for clustering)", options=df.columns)

            task_str = st.selectbox("Select task type", options=[t.value for t in TaskType])
            task_type = TaskType(task_str)

            show_shap = st.checkbox("Show SHAP Explanations (for Classification/Regression)", value=True)
            time_budget = st.number_input("Time budget (seconds)", min_value=60, max_value=3600, value=300, step=60)
            random_seed = st.number_input("Random seed", min_value=0, max_value=1000, value=42, step=1)
            verbosity = st.slider("Verbosity level", min_value=0, max_value=3, value=1)
            auto_select = st.checkbox("Enable Auto Model Selection", value=False)
            use_cv = st.checkbox("Enable Cross-Validation", value=False)

            # If clustering, ask for n_clusters
            n_clusters = 0
            if task_type == TaskType.CLUSTERING:
                n_clusters = st.number_input("Number of clusters", min_value=2, max_value=20, value=3)

            available_models, defaults = self.get_available_models(task_type)

            if auto_select:
                st.info("Auto model selection enabled; all available models will be used in an ensemble approach.")
                selected_models = available_models
            else:
                selected_models = st.multiselect("Select models", available_models, default=defaults)
                if not selected_models:
                    st.warning("Select at least one model.")
                    return None

            metric_name, target_score = self.get_target_metrics(task_type)
            if not metric_name:
                st.warning("Select a performance metric.")
                return None

            hyper_configs = self.get_hyperparameter_ranges(selected_models)

            config = ModelConfig(
                target_column=target_column,
                task_type=task_type,
                models=selected_models,
                time_budget=time_budget,
                n_clusters=n_clusters,
                random_seed=random_seed,
                verbosity=verbosity,
                show_shap=show_shap,
                metric_name=metric_name,
                target_score=target_score,
                hyperparameter_configs=hyper_configs,
                auto_model_selection=auto_select,
                use_cv=use_cv
            )
            st.session_state['model_config'] = config

            # Display target visualization
            display_target_visualization(df, config)
            return config

    # --- Training & Evaluation ---
    def train_and_evaluate_model(self, config: ModelConfig):
        """Train and evaluate the selected models based on the provided ModelConfig."""
        if config is None:
            st.error("Missing model configuration.")
            return

        df = st.session_state.get('data')
        if df is None:
            st.error("No data loaded.")
            return

        st.header("3. Training and Evaluation")
        df_pd = df.to_pandas()

        # Classification/Regression => split data into X, y
        if config.task_type != TaskType.CLUSTERING:
            df_pd = preprocess_data(df_pd)  # <-- IMPROVEMENT: ensures data is clean before splitting

            if config.target_column not in df_pd.columns:
                st.error(f"Target column '{config.target_column}' not found in dataset.")
                return

            X = df_pd.drop(columns=[config.target_column])
            y = df_pd[config.target_column]
            if config.task_type != TaskType.CLASSIFICATION:
                y = y.astype(np.int32)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.random_seed
            )
            st.session_state['X_pd'] = X
            st.session_state['y_pd'] = y

        else:  # Clustering => no split needed
            X_train, X_test, y_train, y_test = df_pd, None, None, None
            st.session_state['X_pd'] = df_pd
            st.session_state['y_pd'] = None

        # --- Ensemble training using the new HAMSeOptimizer (if auto_model_selection & not clustering) ---
        if config.auto_model_selection and config.task_type != TaskType.CLUSTERING:
            st.write("### Training with HAMSe Ensemble")
            try:
                from modules.model_selection import HAMSeOptimizer  # <-- FIX: Potential missing import guarded
            except ImportError as e:
                st.error(f"HAMSeOptimizer not found. Please install or provide the correct module. {e}")
                return

            hamse_model = HAMSeOptimizer(n_jobs=4, random_state=config.random_seed)
            hamse_model.fit(
                X_train.to_numpy().astype(np.float32),
                y_train.to_numpy().astype(np.float32)
            )
            predictions = hamse_model.predict(X_test.to_numpy().astype(np.float32))
            report = evaluate_model(hamse_model, X_test, y_test, config.task_type, config.metric_name)

            st.session_state['trained_model'] = hamse_model
            st.session_state['evaluation_report'] = report
            st.write("**Ensemble Evaluation Report:**", report)

        else:
            # Train each model separately
            results = {}
            progress_bar = st.progress(0)
            total = len(config.models)

            for idx, model_name in enumerate(config.models):
                st.write(f"### Training Model: {model_name.upper()}")
                try:
                    base_model = get_model_instance(model_name, config)
                    tuned_model = base_model

                    # Hyperparameter Tuning if available
                    if model_name in config.hyperparameter_configs:
                        hp_configs = config.hyperparameter_configs[model_name]
                        param_grid = {
                            hp.name: np.arange(
                                hp.min_val,
                                hp.max_val + hp.step,
                                hp.step
                            ).tolist()
                            for hp in hp_configs
                        }
                        st.write(f"Tuning {model_name} with grid: {param_grid}")

                        # Use custom scorer for clustering or metric_name for classification/regression
                        if config.task_type == TaskType.CLUSTERING:
                            scoring_function = make_scorer(clustering_scorer, greater_is_better=True)
                        else:
                            scoring_function = config.metric_name

                        grid = GridSearchCV(base_model, param_grid, scoring=scoring_function, cv=3)

                        if config.task_type == TaskType.CLUSTERING:
                            grid.fit(X_train)
                        else:
                            grid.fit(X_train, y_train)

                        tuned_model = grid.best_estimator_

                        st.session_state.setdefault('model_trial_history', {})[model_name] = grid.cv_results_
                        st.session_state.setdefault('search_space_area', {})[model_name] = len(grid.cv_results_['params'])

                    # Fit final model, evaluate
                    if config.task_type == TaskType.CLUSTERING:
                        tuned_model.fit(X_train)
                        report = evaluate_clustering(tuned_model, X_train, config)
                    else:
                        if config.use_cv:
                            scores = cross_val_score(tuned_model, X_train, y_train, cv=5, scoring=config.metric_name)
                            report = {config.metric_name: float(np.mean(scores))}
                            # Train on full train set after CV
                            tuned_model.fit(X_train, y_train)
                        else:
                            tuned_model.fit(X_train, y_train)
                            report = evaluate_model(tuned_model, X_test, y_test, config.task_type, config.metric_name)

                    results[model_name] = report
                    st.session_state['trained_model'] = tuned_model  # The last trained model is stored here
                    st.write("**Evaluation Report:**", report)

                    # SHAP explanations (if classification/regression and user enabled)
                    if config.show_shap and config.task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
                        shap_vals, shap_fig = get_shap_explanation(tuned_model, X_test)
                        if shap_vals is not None and shap_fig is not None:
                            st.subheader("SHAP Summary Plot")
                            st.pyplot(shap_fig)
                            st.session_state['shap_values'] = shap_vals
                            st.session_state['shap_fig'] = shap_fig

                except Exception as e:
                    st.error(f"Error with model {model_name}: {e}")
                    logger.error(f"Error with model {model_name}: {e}")

                progress_bar.progress((idx + 1) / total)
            progress_bar.empty()

            # If auto_model_selection is checked but we are in clustering mode, skip
            if config.auto_model_selection and config.task_type != TaskType.CLUSTERING:
                best = select_best_model(results, config.metric_name)
                if best:
                    st.success(f"Best model selected: '{best}'.")

            st.session_state['evaluation_report'] = results

    # --- Report Generation & Model Saving ---
    def generate_markdown_report(self) -> str:
        """Generate a Markdown report summarizing the dataset, configs, and results."""
        lines = ["# Genta AutoML Report"]
        # Dataset Overview
        data = st.session_state.get("data")
        if data:
            df = data.to_pandas()
            lines.extend([
                "## 1. Dataset Overview",
                f"- **Rows:** {df.shape[0]}",
                f"- **Columns:** {df.shape[1]}",
                f"- **Columns List:** {list(df.columns)}"
            ])
        else:
            lines.append("## 1. Dataset Overview\n_No dataset loaded._")

        # Model Configuration
        config = st.session_state.get("model_config")
        lines.append("## 2. Model Configuration")
        if config:
            lines.extend([
                f"- **Task Type:** {config.task_type.value}",
                f"- **Target Column:** {config.target_column}",
                f"- **Selected Models:** {config.models}",
                f"- **Metric:** {config.metric_name}",
                f"- **Auto Model Selection:** {config.auto_model_selection}",
                f"- **Random Seed:** {config.random_seed}",
                f"- **Time Budget:** {config.time_budget} seconds"
            ])
            if config.task_type == TaskType.CLUSTERING:
                lines.append(f"- **Number of Clusters:** {config.n_clusters}")
            lines.append(f"- **Cross-Validation:** {config.use_cv}")

            lines.append("### Hyperparameters Used")
            if config.hyperparameter_configs:
                for model, hp_list in config.hyperparameter_configs.items():
                    lines.append(f"**Model:** {model}")
                    lines.append("| Param Name | Min | Max | Step | Default |")
                    lines.append("|------------|-----|-----|------|---------|")
                    for hp in hp_list:
                        lines.append(f"| {hp.name} | {hp.min_val} | {hp.max_val} | {hp.step} | {hp.default} |")
            else:
                lines.append("_No custom hyperparameters configured._")
        else:
            lines.append("_No model configuration found._")

        # Evaluation Results
        lines.append("## 3. Evaluation Results")
        eval_report = st.session_state.get("evaluation_report")
        if eval_report:
            if isinstance(eval_report, dict):
                # It could be {model_name: {metric: score}} or single model
                for model, metrics in eval_report.items():
                    if isinstance(metrics, dict):
                        lines.append(f"**Model: {model}**")
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                lines.append(f"- {metric}: {value:.4f}")
                            else:
                                lines.append(f"- {metric}: {value}")
                    else:
                        # Single model scenario
                        lines.append(f"**Model**: {model}")
                        lines.append(f"- Score: {metrics}")
            else:
                # Possibly a single model dictionary
                for k, v in eval_report.items():
                    lines.append(f"- {k}: {v}")
        else:
            lines.append("_No evaluation results available._")

        # SHAP Analysis
        lines.append("## 4. SHAP Analysis")
        if st.session_state.get("shap_values") is not None:
            lines.append("SHAP summary plot and top contributing features have been generated.")
        else:
            lines.append("_No SHAP analysis performed._")

        lines.append("\n---\nReport generated by Genta AutoML.")
        return "\n".join(lines)

    def download_markdown_report(self):
        """Allow the user to download the generated Markdown report."""
        md = self.generate_markdown_report()
        if not os.path.exists("result"):
            os.makedirs("result")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"genta_automl_report_{timestamp}.md"
        filepath = os.path.join("result", filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)
            st.download_button("Download Markdown Report", data=md, file_name=filename, mime="text/markdown")
            st.success(f"Report saved locally at: {filepath}")
        except Exception as e:
            st.error(f"Error while saving or downloading report: {e}")
            logger.error(f"Report saving error: {e}")

    def save_trained_model(self):
        """Save the trained model to disk as a pickle file."""
        model = st.session_state.get("trained_model")
        if model is None:
            st.warning("No trained model found.")
            return
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trained_model_{timestamp}.pkl"
        filepath = os.path.join("saved_models", filename)
        try:
            joblib.dump(model, filepath)
            st.success(f"Model saved to: {filepath}")
        except Exception as e:
            st.error(f"Error saving model: {e}")
            logger.error(f"Error saving model: {e}")

    # --- Main Execution ---
    def run(self):
        df = self.load_data()
        if df is None:
            return
        config = self.get_model_config(df)
        if config:
            st.success("Model configuration completed.")
            st.write("### Current Model Configuration")
            st.write(config)

            if st.button("Train and Evaluate Model"):
                self.train_and_evaluate_model(config)
                st.info("Training and evaluation finished.")
                if st.button("Save Trained Model"):
                    self.save_trained_model()

            if st.button("Generate Final Report"):
                md_report = self.generate_markdown_report()
                st.markdown(md_report)
                if st.button("Download Final Report"):
                    self.download_markdown_report()

# ------------------------------------------------------------------------
# Entry point
if __name__ == "__main__":
    app = GentaAutoMLApp()
    app.run()
