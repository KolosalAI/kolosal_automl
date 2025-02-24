import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import shap
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import json
import os  # For creating the result directory if needed

# NEW: Additional imports for model comparison
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import numpy as np
# ------------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Enums and Dataclasses for configuration

class TaskType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"

    def __str__(self):
        return self.value

@dataclass
class HyperParameterConfig:
    name: str
    min_val: float
    max_val: float
    step: float
    default: float

@dataclass
class ModelConfig:
    target_column: str
    task_type: TaskType
    models: List[str]
    time_budget: int
    n_clusters: int
    random_seed: int
    verbosity: int
    show_shap: bool
    metric_name: str
    target_score: float
    hyperparameter_configs: Dict[str, List[HyperParameterConfig]]
    auto_model_selection: bool

@dataclass
class TargetMetrics:
    metric_name: str
    target_score: float
    achieved_value: float
    is_achieved: bool

# ------------------------------------------------------------------------------
# Default hyperparameters for various models (for clustering only)
DEFAULT_HYPERPARAMETERS: Dict[str, List[HyperParameterConfig]] = {
    "kmeans": [
        HyperParameterConfig("n_clusters", 2, 10, 1, 3),
        HyperParameterConfig("max_iter", 100, 500, 100, 300)
    ],
    "gmm": [
        HyperParameterConfig("n_components", 2, 10, 1, 3),
        HyperParameterConfig("max_iter", 50, 200, 50, 100)
    ],
    "hierarchical": [
        HyperParameterConfig("n_clusters", 2, 10, 1, 3)
    ],
    "dbscan": [
        HyperParameterConfig("eps", 0.1, 1.0, 0.1, 0.5),
        HyperParameterConfig("min_samples", 2, 10, 1, 5)
    ],
    "hdbscan": [
        HyperParameterConfig("min_cluster_size", 2, 10, 1, 5)
    ]
}

# ------------------------------------------------------------------------------
def get_tune_domain(hp_config: HyperParameterConfig) -> Dict[str, Any]:
    logger.debug(f"Creating hyperparameter domain for {hp_config.name}")
    domain_array = np.arange(hp_config.min_val, hp_config.max_val + hp_config.step, hp_config.step)
    logger.debug(f"Domain array: {domain_array}")
    return {
        "domain": domain_array.tolist(),
        "init_value": hp_config.default,
        "low_cost_init_value": hp_config.default
    }

# ------------------------------------------------------------------------------
# Streamlit AutoML App Class
class GentaAutoMLApp:
    def __init__(self) -> None:
        logger.debug("Initializing GentaAutoMLApp...")
        st.set_page_config(page_title="Genta AutoML", layout="wide")
        st.title("Genta AutoML: Automated Machine Learning Platform")
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes session state variables if they don't exist."""
        keys = [
            'data',
            'model',
            'X_pd',
            'target_metrics',
            'y_pd',
            'X_selected',
            'trained_model',
            'evaluation_report',
            'shap_values',
            'shap_fig',
            'model_config'
        ]
        for key in keys:
            if key not in st.session_state:
                logger.debug(f"Initializing session_state['{key}'] to None")
                st.session_state[key] = None

    def load_data(self) -> Optional[pl.DataFrame]:
        logger.debug("Entering load_data function...")
        st.header("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV, Parquet, or Excel file",
            type=['csv', 'parquet', 'xlsx', 'xls'],
            help="Supported file formats: CSV, Parquet, Excel"
        )

        if uploaded_file is not None:
            logger.debug(f"File uploaded: {uploaded_file.name}")
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                logger.debug(f"File extension: {file_extension}")

                if file_extension == 'csv':
                    df = pl.read_csv(uploaded_file)
                elif file_extension == 'parquet':
                    df = pl.read_parquet(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    try:
                        df = pl.from_pandas(pd.read_excel(uploaded_file))
                    except Exception as e:
                        st.error(f"Error reading Excel file: {e}")
                        logger.error(f"Error reading Excel file: {e}")
                        return None
                else:
                    st.error("Unsupported file format. Please upload a CSV, Parquet, or Excel file.")
                    logger.debug("Unsupported file format encountered.")
                    return None

                st.session_state['data'] = df
                st.success("Data loaded successfully!")
                self._display_data_preview(df)
                return df

            except pl.exceptions.ComputeError as e:
                st.error(f"Data type error: {e}. Ensure correct data formatting.")
                logger.error(f"Polars ComputeError: {e}")
                return None
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error loading data: {str(e)}")
                return None
        else:
            logger.debug("No file uploaded yet.")
            st.info("Please upload a dataset to begin.")
            return None

    def _display_data_preview(self, df: pl.DataFrame):
        """Displays a preview of the data and statistics."""
        st.subheader("Data Preview")
        try:
            st.dataframe(df.head(10).to_pandas())
        except Exception as e:
            st.error(f"Error displaying dataframe: {e}")
            logger.error(f"Error displaying dataframe: {e}")

        st.subheader("Data Statistics")
        try:
            st.write(df.describe().to_pandas())
        except Exception as e:
            st.error(f"Error generating data statistics: {e}")
            logger.error(f"Error generating data statistics: {e}")

    def get_available_models(self, task_type: TaskType) -> Tuple[List[str], List[str]]:
        logger.debug(f"Getting available models for task type: {task_type}")
        model_options = {
            TaskType.CLASSIFICATION: ["lgbm", "rf", "xgboost", "catboost"],
            TaskType.REGRESSION: ["lgbm", "rf", "xgboost", "catboost"],
            TaskType.CLUSTERING: list(DEFAULT_HYPERPARAMETERS.keys())
        }

        default_models = {
            TaskType.CLASSIFICATION: ["lgbm", "rf"],
            TaskType.REGRESSION: ["lgbm", "rf"],
            TaskType.CLUSTERING: ["kmeans", "hierarchical"]
        }

        available = model_options.get(task_type, [])
        defaults = default_models.get(task_type, [])
        logger.debug(f"Available models: {available}, Default models: {defaults}")
        return available, defaults

    def get_target_metrics(self, task_type: TaskType) -> Tuple[Optional[str], Optional[float]]:
        logger.debug(f"Entering get_target_metrics with task_type: {task_type}")
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
            "roc_auc": "Area under the ROC curve (Classification).",
            "r2": "Coefficient of determination (Regression).",
            "mse": "Mean squared error (Regression).",
            "mae": "Mean absolute error (Regression).",
            "rmse": "Root mean squared error (Regression).",
            "silhouette": "Silhouette score (Clustering)."
        }

        st.subheader("Select Target Metric and Score")
        available_metrics = metric_options.get(task_type, [])

        if not available_metrics:
            logger.debug("No metrics available for this task type.")
            st.warning("No metrics available for the selected task type.")
            return None, None

        metric_name = st.selectbox(
            "Select performance metric",
            options=available_metrics,
            help="Choose the metric to evaluate model performance."
        )
        logger.debug(f"Selected metric: {metric_name}")

        if metric_name:
            st.info(metric_explanations.get(metric_name, "No explanation available."))
            default_target = self._get_default_target(metric_name)
            target_score = st.number_input(
                f"Enter target {metric_name} score",
                min_value=0.0,
                max_value=self._get_max_target(metric_name),
                value=default_target,
                step=0.01,
                help=f"Set the target score for {metric_name}."
            )
            logger.debug(f"Target score for {metric_name} is {target_score}")
            return metric_name, float(target_score)
        else:
            logger.debug("No metric was selected.")
            return None, None

    def _get_default_target(self, metric_name: str) -> float:
        """Returns a sensible default target score for a given metric."""
        if metric_name in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"]:
            return 0.8
        else:
            return 0.0

    def _get_max_target(self, metric_name: str) -> float:
        """Returns a maximum allowed value for the target score."""
        if metric_name in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"]:
            return 1.0
        else:
            return 10000.0

    def get_hyperparameter_ranges(self, selected_models: List[str]) -> Dict[str, List[HyperParameterConfig]]:
        logger.debug("Entering get_hyperparameter_ranges...")
        st.subheader("Configure Hyperparameters (Optional)")
        use_advanced = st.checkbox(
            "Enable advanced hyperparameter configuration",
            value=False,
            help="Allows you to customize the hyperparameter search space for each model."
        )
        logger.debug(f"Advanced configuration: {use_advanced}")
        hyperparameter_configs: Dict[str, List[HyperParameterConfig]] = {}

        for model in selected_models:
            logger.debug(f"Processing hyperparameters for model: {model}")
            if model not in DEFAULT_HYPERPARAMETERS:
                logger.debug(f"No default hyperparameters found for model: {model}")
                continue

            default_configs = DEFAULT_HYPERPARAMETERS[model]
            hyperparameter_configs[model] = []

            if use_advanced:
                st.write(f"### {model.upper()} Hyperparameters")
                for hp in default_configs:
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            min_val = st.number_input(
                                f"{hp.name} min", value=hp.min_val, step=hp.step,
                                key=f"{model}_{hp.name}_min"
                            )
                        with col2:
                            max_val = st.number_input(
                                f"{hp.name} max", value=hp.max_val, step=hp.step,
                                key=f"{model}_{hp.name}_max"
                            )
                        with col3:
                            default_val = st.number_input(
                                f"{hp.name} default", value=hp.default, step=hp.step,
                                key=f"{model}_{hp.name}_default"
                            )

                        logger.debug(f"Hyperparameter {hp.name} for {model}: min={min_val}, max={max_val}, default={default_val}")
                        hyperparameter_configs[model].append(
                            HyperParameterConfig(hp.name, min_val, max_val, hp.step, default_val)
                        )
            else:
                logger.debug(f"Using default hyperparameters for {model}")
                hyperparameter_configs[model] = default_configs

        logger.debug("Finished building hyperparameter configs.")
        return hyperparameter_configs

    def get_model_config(self, df: pl.DataFrame) -> Optional[ModelConfig]:
        logger.debug("Entering get_model_config...")
        st.header("Configure Your Model")

        if df is None:
            logger.debug("No data available; prompting user to upload.")
            st.warning("Please upload data first.")
            return None

        with st.expander("Model Settings", expanded=True):
            target_column = st.selectbox(
                "Select target column",
                options=df.columns,
                help="Choose the column to predict (ignored for clustering)."
            )
            logger.debug(f"Selected target column: {target_column}")

            task_type_str = st.selectbox(
                "Select task type",
                options=[t.value for t in TaskType],
                help="Choose the type of machine learning task."
            )
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                st.error(f"Invalid task type selected: {task_type_str}")
                return None
            logger.debug(f"Selected task type: {task_type}")

            auto_model_selection = st.checkbox(
                "Enable Auto Model Selection (Advanced)",
                value=False,
                help="If enabled, the system will train all available models and automatically choose the best one based on the performance metric."
            )
            logger.debug(f"Auto model selection enabled: {auto_model_selection}")

            available_models, default_models = self.get_available_models(task_type)

            if auto_model_selection:
                models = available_models
                st.info("Auto model selection enabled. Using all available models for evaluation.")
            else:
                models = st.multiselect(
                    "Select models to train",
                    options=available_models,
                    default=default_models,
                    help="Choose one or more models to train."
                )
                if not models:
                    st.warning("Please select at least one model.")
                    logger.debug("No models were selected.")
                    return None
            logger.debug(f"Models to be used: {models}")

            time_budget = st.number_input(
                "Time budget (seconds)",
                min_value=10,
                max_value=3600,
                value=60,
                step=10,
                help="Set the maximum time for training."
            )
            logger.debug(f"Time budget: {time_budget}")

            n_clusters = st.number_input(
                "Number of clusters (for clustering)",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Set the number of clusters (used by some clustering algorithms)."
            ) if task_type == TaskType.CLUSTERING else 3
            logger.debug(f"Number of clusters: {n_clusters}")

            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                value=42,
                step=1,
                help="Set the random seed for reproducibility."
            )
            logger.debug(f"Random seed: {random_seed}")

            verbosity = st.slider(
                "Verbosity level",
                min_value=0,
                max_value=3,
                value=1,
                help="Set the logging verbosity level."
            )
            logger.debug(f"Verbosity level: {verbosity}")

            show_shap = st.checkbox(
                "Show SHAP explanations (classification/regression only)",
                value=False,
                help="Enable SHAP value explanations for model predictions."
            ) if task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION] else False
            logger.debug(f"Show SHAP: {show_shap}")

            metric_name, target_score = self.get_target_metrics(task_type)
            if metric_name is None or target_score is None:
                st.warning("Please select a metric and target score.")
                logger.debug("No metric or target score selected.")
                return None

            hyperparameter_configs = self.get_hyperparameter_ranges(models)
            logger.debug("Model configuration collected successfully.")

            return ModelConfig(
                target_column=target_column,
                task_type=task_type,
                models=models,
                time_budget=time_budget,
                n_clusters=n_clusters,
                random_seed=random_seed,
                verbosity=verbosity,
                show_shap=show_shap,
                metric_name=metric_name,
                target_score=target_score,
                hyperparameter_configs=hyperparameter_configs,
                auto_model_selection=auto_model_selection
            )

    # --------------------------------------------------------------------------
    # NEW: Compare multiple scikit-learn models side-by-side
    # --------------------------------------------------------------------------
    def compare_sklearn_models_app(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   model_names: List[str],
                                   scoring: str,
                                   cv: int = 5):
        """
        Compare the performance of multiple scikit-learn models using cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray
            Target array.
        model_names : List[str]
            List of model names (e.g., ["LinearRegression", "Ridge", "Lasso"]).
        scoring : str
            Sklearn-compatible scoring method (e.g. "accuracy", "neg_mean_squared_error").
        cv : int
            Number of cross-validation folds.
        """
        st.subheader("Compare Base scikit-learn Models")

        # Map user-friendly strings to actual model objects
        # You can add or remove as desired:
        model_map = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC(),
            "SVR": SVR(),
            "RandomForestClassifier": RandomForestClassifier(),
            "RandomForestRegressor": RandomForestRegressor()
        }

        valid_models = []
        for name in model_names:
            if name in model_map:
                valid_models.append(model_map[name])
            else:
                st.warning(f"'{name}' is not recognized or not supported in this demo.")

        if not valid_models:
            st.info("No valid scikit-learn models selected for comparison.")
            return

        results = []
        for model in valid_models:
            model_clone = clone(model)
            try:
                scores = cross_val_score(model_clone, X, y, scoring=scoring, cv=cv, n_jobs=-1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                results.append({
                    "Model": model.__class__.__name__,
                    "Mean Score": mean_score,
                    "Std Dev": std_score
                })
            except ValueError as ve:
                st.error(f"Error evaluating {model.__class__.__name__}: {ve}")
                logger.error(f"Error evaluating {model.__class__.__name__}: {ve}")

        if results:
            results_df = pd.DataFrame(results)
            st.write("### Comparison Results")
            st.dataframe(results_df)

    # --------------------------------------------------------------------------
    # Training (placeholder)
    # --------------------------------------------------------------------------
    def train_model(self, model_config: ModelConfig, df: pl.DataFrame):
        """
        Placeholder for model training logic.
        Replace this function with your actual training process.
        """
        st.subheader("Training in Progress")
        logger.info("Starting model training...")
        import time
        progress_bar = st.progress(0)
        for percent_complete in range(0, 101, 10):
            time.sleep(0.2)
            progress_bar.progress(percent_complete)
        st.success("Training completed successfully!")
        logger.info("Model training completed.")

        # ----------------------------------------------------------------------
        # For demonstration, we create a *placeholder* model
        # ----------------------------------------------------------------------
        placeholder_model = {"model_name": "fake_model_object"}  # Replace with real model
        st.session_state['trained_model'] = placeholder_model

        # Convert Polars df to Pandas for convenience in SHAP, etc.
        df_pandas = df.to_pandas()

        # If classification/regression, separate X, y for SHAP
        if model_config.task_type != TaskType.CLUSTERING:
            target = df_pandas[model_config.target_column].values
            features = df_pandas.drop(columns=[model_config.target_column])
            st.session_state['X_pd'] = features
            st.session_state['y_pd'] = target
        else:
            st.session_state['X_pd'] = df_pandas
            st.session_state['y_pd'] = None

    # --------------------------------------------------------------------------
    # SHAP Analysis
    # --------------------------------------------------------------------------
    def generate_shap_analysis(self, model_config: ModelConfig):
        """
        Generate SHAP analysis if show_shap is True (for Classification/Regression only).
        This is a placeholder approach. Replace with your actual model and predictor.
        """
        if not model_config.show_shap:
            return  # Skip if SHAP is not enabled

        if 'trained_model' not in st.session_state or st.session_state['trained_model'] is None:
            st.warning("No trained model found for SHAP analysis.")
            return

        model = st.session_state['trained_model']
        X = st.session_state['X_pd']
        y = st.session_state['y_pd']

        if X is None or y is None:
            st.warning("No training data found for SHAP analysis.")
            return

        st.subheader("SHAP Analysis")

        try:
            # Example placeholder for demonstration: random predictor
            def model_predict(data):
                return np.random.rand(data.shape[0])

            explainer = shap.KernelExplainer(model_predict, X.iloc[:50, :])
            shap_values = explainer.shap_values(X.iloc[:50, :], nsamples=50)

            st.session_state['shap_values'] = shap_values

            plt.figure()
            shap.summary_plot(shap_values, X.iloc[:50, :], plot_type="bar", show=False)
            fig = plt.gcf()
            st.pyplot(fig, bbox_inches='tight')
            st.session_state['shap_fig'] = fig

            st.success("SHAP analysis completed!")
        except Exception as e:
            st.error(f"Error generating SHAP explanations: {e}")
            logger.error(f"Error in SHAP generation: {e}")

    def evaluate_best_model(self) -> Dict[str, Any]:
        """
        Evaluates multiple models by splitting the data into training and test sets,
        computes training and test scores based on the selected metric, and generates
        an evaluation report including an evaluation table.
        """
        st.subheader("Model Evaluation Report")
        logger.info("Evaluating models using a train/test split...")

        model_config = st.session_state.get("model_config")
        if model_config is None:
            st.error("Model configuration not found!")
            return {}

        # For non-clustering tasks, get X and y from session state
        if model_config.task_type != TaskType.CLUSTERING:
            X = st.session_state.get("X_pd")
            y = st.session_state.get("y_pd")
            if X is None or y is None:
                st.error("Training data not available!")
                return {}
            
            # Automatically split the data (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=model_config.random_seed
            )
        else:
            st.warning("Evaluation with train/test split is not applicable for clustering tasks.")
            return {}

        # Define metric functions mapping
        metric_name = model_config.metric_name
        metric_funcs = {
            "accuracy": accuracy_score,
            "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
            "precision_macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
            "recall_macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
            "roc_auc": roc_auc_score,  # Assumes probability outputs; may need adjustment
            "r2": r2_score,
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        }

        if metric_name not in metric_funcs:
            st.error(f"Selected metric '{metric_name}' is not supported for evaluation.")
            return {}

        metric_func = metric_funcs[metric_name]

        # Create a mapping for model instantiation for demonstration.
        # For classification and regression, you may use popular estimators.
        # Here we assume:
        #   - For Classification: lgbm -> LightGBMClassifier, rf -> RandomForestClassifier, etc.
        #   - For Regression: lgbm -> LightGBMRegressor, rf -> RandomForestRegressor, etc.
        # For the demo, we'll use scikit-learn models where possible.
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression

        model_map = {}
        if model_config.task_type == TaskType.CLASSIFICATION:
            # For demonstration, we map common names to scikit-learn models.
            model_map = {
                "lgbm": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(random_state=model_config.random_seed),
                "xgboost": LogisticRegression(max_iter=1000),  # Placeholder; replace with XGBClassifier if available
                "catboost": LogisticRegression(max_iter=1000)  # Placeholder; replace with CatBoostClassifier if available
            }
        elif model_config.task_type == TaskType.REGRESSION:
            model_map = {
                "lgbm": LinearRegression(),
                "rf": RandomForestRegressor(random_state=model_config.random_seed),
                "xgboost": LinearRegression(),  # Placeholder; replace with XGBRegressor if available
                "catboost": LinearRegression()  # Placeholder; replace with CatBoostRegressor if available
            }

        evaluation_results = []
        best_model_name = None
        best_test_score = -np.inf  # Assume higher is better for most metrics; adjust if needed

        for model_name in model_config.models:
            if model_name not in model_map:
                st.warning(f"Model '{model_name}' is not supported in the evaluation demo. Skipping.")
                continue

            model = model_map[model_name]
            # Train the model on training data
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                logger.error(f"Error training {model_name}: {e}")
                continue

            # For classification tasks with roc_auc, we need probability estimates.
            if model_config.task_type == TaskType.CLASSIFICATION and metric_name == "roc_auc":
                try:
                    y_pred_train = model.predict_proba(X_train)[:, 1]
                    y_pred_test = model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    st.warning(f"Model '{model_name}' does not support probability estimates. Using predictions instead.")
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
            else:
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            # Compute metric scores for train and test splits
            try:
                train_score = metric_func(y_train, y_pred_train)
                test_score = metric_func(y_test, y_pred_test)
            except Exception as e:
                st.error(f"Error computing metric for {model_name}: {e}")
                logger.error(f"Error computing metric for {model_name}: {e}")
                continue

            evaluation_results.append({
                "Model": model_name,
                "Train Score": train_score,
                "Test Score": test_score
            })

            # For error metrics (mse, mae, rmse) lower is better, so invert the test score for selection
            if metric_name in ["mse", "mae", "rmse"]:
                score_for_selection = -test_score
            else:
                score_for_selection = test_score

            if score_for_selection > best_test_score:
                best_test_score = score_for_selection
                best_model_name = model_name

        if evaluation_results:
            results_df = pd.DataFrame(evaluation_results)
            st.write("### Evaluation Table")
            st.dataframe(results_df)
        else:
            st.error("No evaluation results available.")
            return {}

        # Create a report dictionary
        report = {
            "Best Model": best_model_name if best_model_name is not None else "N/A",
            "Best Test Score": best_test_score if best_model_name is not None else "N/A",
            "Metric": metric_name,
            "Evaluation Table": results_df.to_dict(orient="records")
        }

        st.write("### Final Evaluation Report")
        st.write(report)
        st.session_state['evaluation_report'] = report
        return report

    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """
        Create a comprehensive Markdown report that includes the main points of the evaluation,
        model configuration, SHAP analysis, and all other available session state information.
        """
        import json

        lines = []
        lines.append("# Genta AutoML Comprehensive Report\n")
        lines.append("## Overview\n")
        lines.append("This report provides an overview of the model evaluation, configuration, SHAP analysis, and additional session state details gathered during the AutoML process.\n")

        # Evaluation Summary
        lines.append("## Model Evaluation Summary")
        lines.append(f"- **Best Model**: {report.get('Best Model', 'N/A')}")
        if 'Performance Score' in report:
            lines.append(f"- **Performance Score**: {report.get('Performance Score', 'N/A')}")
        elif 'Best Test Score' in report:
            lines.append(f"- **Best Test Score**: {report.get('Best Test Score', 'N/A')}")
        lines.append(f"- **Metric**: {report.get('Metric', 'N/A')}\n")

        # SHAP Analysis Section
        if st.session_state.get('shap_values') is not None:
            lines.append("## SHAP Analysis")
            lines.append("SHAP values were computed. A summary plot is shown in the UI, but is not embedded here.\n")
        else:
            lines.append("*(No SHAP analysis was performed or no SHAP values were found.)*\n")

        # Additional Session State Details
        lines.append("## Additional Session State Details")
        # Loop through session state keys and include relevant ones
        for key, value in st.session_state.items():
            # Optionally skip large or non-textual objects (like dataframes or figures)
            if key in ['shap_fig', 'data']:
                continue
            lines.append(f"### {key}")
            try:
                # Try to serialize to JSON; if fails, fallback to string representation
                value_str = json.dumps(value, indent=2, default=str)
            except Exception:
                value_str = str(value)
            lines.append("```json")
            lines.append(value_str)
            lines.append("```\n")

        return "\n".join(lines)


    def download_markdown_report_button(self, md_content: str):
        """
        Provides a button to download the evaluation report as Markdown
        and saves it locally in the `result` directory.
        """
        if not os.path.exists("result"):
            os.makedirs("result")

        local_filename = os.path.join("result", "report.md")
        with open(local_filename, "w", encoding="utf-8") as f:
            f.write(md_content)

        st.download_button(
            label="Download Markdown Report",
            data=md_content,
            file_name="report.md",
            mime="text/markdown"
        )
        st.success(f"Markdown report also saved locally to `{local_filename}`.")

    def run(self):
        """Main execution function of the app."""
        df = self.load_data()
        if df is not None:
            model_config = self.get_model_config(df)
            if model_config:
                st.session_state['model_config'] = model_config
                st.success("Model configuration complete. Ready to train!")
                st.write("## Configuration")
                st.write(model_config)

                # --- NEW: Optional step to compare base scikit-learn models (Classification/Regression only) ---
                if model_config.task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
                    st.info("You can optionally compare some base sklearn models before full training.")
                    base_model_names = [
                        "LinearRegression", "Ridge", "Lasso",
                        "LogisticRegression", "SVC", "SVR",
                        "RandomForestClassifier", "RandomForestRegressor"
                    ]

                    # Map your chosen metric_name to scikit-learn's scoring
                    # For example: "mse" => "neg_mean_squared_error", etc.
                    scikit_metric_map = {
                        "accuracy": "accuracy",
                        "f1_macro": "f1_macro",
                        "precision_macro": "precision_macro",
                        "recall_macro": "recall_macro",
                        "roc_auc": "roc_auc",
                        "r2": "r2",
                        "mse": "neg_mean_squared_error",
                        "mae": "neg_mean_absolute_error",
                        "rmse": "neg_root_mean_squared_error"
                    }
                    # Make sure the chosen metric is valid for scikit
                    chosen_metric = scikit_metric_map.get(model_config.metric_name, None)

                    selected_base_models = st.multiselect(
                        "Select base sklearn models to compare",
                        base_model_names,
                        default=["LinearRegression"] if model_config.task_type == TaskType.REGRESSION else ["LogisticRegression"]
                    )

                    if st.button("Compare Base Models"):
                        # Convert Polars to Pandas
                        df_pd = df.to_pandas()
                        # For clustering, there's no y, so skip
                        if model_config.task_type != TaskType.CLUSTERING:
                            y = df_pd[model_config.target_column]
                            X = df_pd.drop(columns=[model_config.target_column])
                            if chosen_metric:
                                self.compare_sklearn_models_app(X, y, selected_base_models, scoring=chosen_metric)
                            else:
                                st.warning("Selected metric is not currently supported by scikit-learn scoring.")
                        else:
                            st.warning("Base model comparison is not available for clustering tasks.")

                # -------------------------------------------------------------------
                # Proceed with the normal "Train" button
                # -------------------------------------------------------------------
                if st.button("Train"):
                    logger.info("Train button pressed.")
                    self.train_model(model_config, df)

                    # If user requested SHAP, generate it
                    if model_config.show_shap:
                        self.generate_shap_analysis(model_config)

                    # Automatically evaluate the best model after training
                    report = self.evaluate_best_model()

                    # Generate a Markdown string
                    md_report = self.generate_markdown_report(report)
                    # Provide button to download the markdown report
                    self.download_markdown_report_button(md_report)

                else:
                    st.info("Press the 'Train' button to start model training.")

# ------------------------------------------------------------------------------
# Main execution
if __name__ == "__main__":
    app = GentaAutoMLApp()
    app.run()
