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
import os

# Sklearn imports for demonstration
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN

# ------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
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

# ------------------------------------------------------------------------
# Default hyperparameters (example for clustering)
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

# ------------------------------------------------------------------------
def get_tune_domain(hp_config: HyperParameterConfig) -> Dict[str, Any]:
    """Utility to convert a HyperParameterConfig into a dictionary domain."""
    domain_array = np.arange(
        hp_config.min_val,
        hp_config.max_val + hp_config.step,
        hp_config.step
    )
    return {
        "domain": domain_array.tolist(),
        "init_value": hp_config.default,
        "low_cost_init_value": hp_config.default
    }

# ------------------------------------------------------------------------
class GentaAutoMLApp:
    def __init__(self) -> None:
        # Streamlit page config and title
        st.set_page_config(page_title="Genta AutoML", layout="wide")
        st.title("Genta AutoML: Automated Machine Learning Platform")
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes session state variables if they don't exist."""
        keys = [
            'data',
            'model',
            'X_pd',
            'y_pd',
            'target_metrics',
            'X_selected',
            'trained_model',
            'evaluation_report',
            'shap_values',
            'shap_fig',
            'model_config'
        ]
        for key in keys:
            if key not in st.session_state:
                st.session_state[key] = None

    # --------------------------------------------------------------------
    # 1) Load Data
    # --------------------------------------------------------------------
    def load_data(self) -> Optional[pl.DataFrame]:
        st.header("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV, Parquet, or Excel file",
            type=['csv', 'parquet', 'xlsx', 'xls'],
            help="Supported file formats: CSV, Parquet, Excel"
        )
        if uploaded_file:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = pl.read_csv(uploaded_file)
                elif file_extension == 'parquet':
                    df = pl.read_parquet(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    df = pl.from_pandas(pd.read_excel(uploaded_file))
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
                return None
        else:
            st.info("Please upload a dataset to begin.")
            return None

    def _display_data_preview(self, df: pl.DataFrame):
        st.subheader("Data Preview")
        try:
            st.dataframe(df.head(10).to_pandas())
        except Exception as e:
            st.error(f"Error displaying dataframe: {e}")

        st.subheader("Data Statistics")
        try:
            st.write(df.describe().to_pandas())
        except Exception as e:
            st.error(f"Error generating data statistics: {e}")

    # --------------------------------------------------------------------
    # 2) Model Configuration
    # --------------------------------------------------------------------
    def get_available_models(self, task_type: TaskType) -> Tuple[List[str], List[str]]:
        """
        Return a tuple (available_models, default_models) based on the task type.
        """
        model_options = {
            TaskType.CLASSIFICATION: [
                "logistic_regression", "rf", "xgboost", "catboost", "lgbm"
            ],
            TaskType.REGRESSION: [
                "linear_regression", "ridge", "lasso", "rf", "xgboost", "catboost", "lgbm"
            ],
            TaskType.CLUSTERING: list(DEFAULT_HYPERPARAMETERS.keys())  # e.g. kmeans, dbscan, etc.
        }
        default_models = {
            TaskType.CLASSIFICATION: ["logistic_regression", "rf"],
            TaskType.REGRESSION: ["linear_regression", "rf"],
            TaskType.CLUSTERING: ["kmeans"]
        }
        return model_options[task_type], default_models[task_type]

    def get_target_metrics(self, task_type: TaskType) -> Tuple[Optional[str], Optional[float]]:
        """
        Let the user pick a metric and target score.
        """
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
        available_metrics = metric_options[task_type]

        metric_name = st.selectbox("Select performance metric", options=available_metrics)
        if metric_name:
            st.info(metric_explanations.get(metric_name, "No explanation available."))
            default_target = self._get_default_target(metric_name)
            target_score = st.number_input(
                f"Enter target {metric_name} score",
                min_value=0.0,
                max_value=self._get_max_target(metric_name),
                value=default_target,
                step=0.01
            )
            return metric_name, float(target_score)
        else:
            return None, None

    def _get_default_target(self, metric_name: str) -> float:
        if metric_name in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"]:
            return 0.8
        else:
            return 0.0

    def _get_max_target(self, metric_name: str) -> float:
        if metric_name in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "r2", "silhouette"]:
            return 1.0
        else:
            return 1e5

    def get_hyperparameter_ranges(self, selected_models: List[str]) -> Dict[str, List[HyperParameterConfig]]:
        """
        Lets the user adjust hyperparameters for each selected model (optional).
        Only relevant for clustering defaults in this example.
        """
        st.subheader("Configure Hyperparameters (Optional)")
        use_advanced = st.checkbox(
            "Enable advanced hyperparameter configuration",
            value=False
        )
        hyperparameter_configs = {}

        for model in selected_models:
            if model in DEFAULT_HYPERPARAMETERS:
                default_configs = DEFAULT_HYPERPARAMETERS[model]
                if use_advanced:
                    st.write(f"### {model.upper()} Hyperparameters")
                    user_configs = []
                    for hp in default_configs:
                        with st.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                min_val = st.number_input(
                                    f"{hp.name} min [{model}]",
                                    value=hp.min_val, step=hp.step
                                )
                            with col2:
                                max_val = st.number_input(
                                    f"{hp.name} max [{model}]",
                                    value=hp.max_val, step=hp.step
                                )
                            with col3:
                                default_val = st.number_input(
                                    f"{hp.name} default [{model}]",
                                    value=hp.default, step=hp.step
                                )
                            user_configs.append(
                                HyperParameterConfig(
                                    name=hp.name,
                                    min_val=min_val,
                                    max_val=max_val,
                                    step=hp.step,
                                    default=default_val
                                )
                            )
                    hyperparameter_configs[model] = user_configs
                else:
                    # Use defaults if advanced config is not used
                    hyperparameter_configs[model] = default_configs
        return hyperparameter_configs

    def get_model_config(self, df: pl.DataFrame) -> Optional[ModelConfig]:
        st.header("Configure Your Model")
        if df is None:
            st.warning("Please upload data first.")
            return None

        with st.expander("Model Settings", expanded=True):
            # Target column selection
            if len(df.columns) == 0:
                st.error("No columns found in the dataset.")
                return None

            target_column = st.selectbox(
                "Select target column (ignored for clustering)",
                options=df.columns
            )

            # Task Type
            task_type_str = st.selectbox(
                "Select task type",
                options=[t.value for t in TaskType]
            )
            task_type = TaskType(task_type_str)

            # Basic settings
            show_shap = st.checkbox("Show SHAP Explanations (Classification/Regression only)", value=True)
            time_budget = st.number_input("Time budget (seconds)", min_value=60, max_value=3600, value=300, step=60)
            n_clusters = 0
            if task_type == TaskType.CLUSTERING:
                n_clusters = st.number_input("Number of clusters", min_value=2, max_value=20, value=3)

            random_seed = st.number_input("Random seed", min_value=0, max_value=1000, value=42, step=1)
            verbosity = st.slider("Verbosity level", min_value=0, max_value=3, value=1)
            auto_model_selection = st.checkbox("Enable Auto Model Selection", value=False)

            # Model selection
            available_models, default_models = self.get_available_models(task_type)
            if auto_model_selection:
                st.info("Auto model selection enabled: all available models will be used.")
                selected_models = available_models
            else:
                selected_models = st.multiselect("Select models", available_models, default=default_models)
                if not selected_models:
                    st.warning("Please select at least one model.")
                    return None

            # Metric & Target Score
            metric_name, target_score = self.get_target_metrics(task_type)
            if not metric_name:
                st.warning("Please select a metric.")
                return None

            # Hyperparameters (optional)
            hyperparameter_configs = self.get_hyperparameter_ranges(selected_models)

            # Construct and save ModelConfig
            model_config = ModelConfig(
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
                hyperparameter_configs=hyperparameter_configs,
                auto_model_selection=auto_model_selection
            )
            st.session_state['model_config'] = model_config
            return model_config

    # --------------------------------------------------------------------
    # 3) Train and Evaluate Model
    # --------------------------------------------------------------------
    def train_and_evaluate_model(self, model_config: ModelConfig) -> None:
        """
        Main logic for training, evaluating, and optionally selecting the best model.
        """
        if not model_config:
            st.error("Model configuration is missing.")
            return

        df = st.session_state.get('data')
        if df is None:
            st.error("No data loaded. Please upload data first.")
            return

        st.subheader("Model Training and Evaluation")

        # Convert Polars -> Pandas for scikit-learn
        df_pandas = df.to_pandas()

        # If clustering, skip splitting. For classification/regression, do train/test split.
        if model_config.task_type != TaskType.CLUSTERING:
            X = df_pandas.drop(columns=[model_config.target_column])
            y = df_pandas[model_config.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=model_config.random_seed
            )
            st.session_state['X_pd'] = X
            st.session_state['y_pd'] = y
        else:
            # For clustering, we consider X as the entire dataset
            X_train = df_pandas
            X_test = None
            y_train = None
            y_test = None
            st.session_state['X_pd'] = df_pandas
            st.session_state['y_pd'] = None

        results = {}
        for model_name in model_config.models:
            st.write(f"### Training: {model_name.upper()}")
            try:
                model = self.get_model_instance(model_name, model_config)

                if model_config.task_type == TaskType.CLUSTERING:
                    # Clustering training
                    model.fit(X_train)
                    evaluation_report = self.evaluate_clustering(model, X_train, model_config)
                else:
                    # Classification or Regression
                    model.fit(X_train, y_train)
                    evaluation_report = self.evaluate_model(
                        model, X_test, y_test, model_config.task_type, model_config.metric_name
                    )

                results[model_name] = evaluation_report
                # Store the latest trained model
                st.session_state['trained_model'] = model
                st.session_state['evaluation_report'] = evaluation_report

                # Display the evaluation
                st.write("**Evaluation Report**:")
                st.write(evaluation_report)

                # If SHAP is enabled (Classification/Regression)
                if model_config.show_shap and model_config.task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
                    try:
                        shap_values, shap_fig = self.get_shap_explanation(model, X_test)
                        if shap_values is not None and shap_fig is not None:
                            st.subheader("SHAP Summary Plot")
                            st.pyplot(shap_fig)
                            st.session_state['shap_values'] = shap_values
                            st.session_state['shap_fig'] = shap_fig
                    except Exception as e:
                        st.error(f"Error generating SHAP explanation: {e}")
                        logger.error(f"Error generating SHAP explanation: {e}")

            except Exception as e:
                st.error(f"Error training/evaluating model {model_name}: {e}")
                logger.error(f"Error training/evaluating model {model_name}: {e}")

        # Auto model selection
        if model_config.auto_model_selection and model_config.task_type != TaskType.CLUSTERING:
            best_model = self.select_best_model(results, model_config.metric_name)
            if best_model is not None:
                st.success(f"Auto Model Selection: Best model is '{best_model}'.")

    def get_model_instance(self, model_name: str, model_config: ModelConfig):
        """
        Returns an instance of the specified model, based on the task type.
        Extend this for XGBoost, CatBoost, LightGBM as needed.
        """
        rs = model_config.random_seed
        if model_config.task_type == TaskType.CLUSTERING:
            # Minimal examples of clustering models
            if model_name == "kmeans":
                # Possibly use user hyperparameters from model_config.hyperparameter_configs if you like
                return KMeans(n_clusters=model_config.n_clusters, random_state=rs)
            elif model_name == "dbscan":
                return DBSCAN()
            else:
                raise ValueError(f"Clustering model '{model_name}' not implemented.")

        # Classification or Regression
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=rs)
        elif model_name == "linear_regression":
            return LinearRegression()
        elif model_name == "ridge":
            return Ridge(random_state=rs)
        elif model_name == "lasso":
            return Lasso(random_state=rs)
        elif model_name == "rf":
            if model_config.task_type == TaskType.CLASSIFICATION:
                return RandomForestClassifier(random_state=rs)
            else:
                return RandomForestRegressor(random_state=rs)
        elif model_name == "xgboost":
            # Example placeholders, replace with xgboost.XGBClassifier/Regressor if installed
            if model_config.task_type == TaskType.CLASSIFICATION:
                return RandomForestClassifier(random_state=rs)  # placeholder
            else:
                return RandomForestRegressor(random_state=rs)  # placeholder
        elif model_name == "catboost":
            # Example placeholders, replace with catboost.CatBoostClassifier/Regressor if installed
            if model_config.task_type == TaskType.CLASSIFICATION:
                return RandomForestClassifier(random_state=rs)  # placeholder
            else:
                return RandomForestRegressor(random_state=rs)  # placeholder
        elif model_name == "lgbm":
            # Example placeholders, replace with lightgbm.LGBMClassifier/Regressor if installed
            if model_config.task_type == TaskType.CLASSIFICATION:
                return RandomForestClassifier(random_state=rs)  # placeholder
            else:
                return RandomForestRegressor(random_state=rs)  # placeholder
        else:
            raise ValueError(f"Model '{model_name}' not recognized or implemented.")

    # --------------------------------------------------------------------
    # 4) Evaluation
    # --------------------------------------------------------------------
    def evaluate_model(self, model, X_test, y_test, task_type: TaskType, metric_name: str) -> Dict[str, float]:
        """
        Compute multiple metrics, but highlight the one the user picked as 'primary'.
        """
        if X_test is None or y_test is None:
            return {}

        y_pred = model.predict(X_test)
        eval_report = {}

        if task_type == TaskType.CLASSIFICATION:
            # Probability estimates for certain metrics (e.g. roc_auc)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_pred_proba = None

            eval_report["accuracy"] = accuracy_score(y_test, y_pred)
            eval_report["f1_macro"] = f1_score(y_test, y_pred, average="macro")
            eval_report["precision_macro"] = precision_score(y_test, y_pred, average="macro")
            eval_report["recall_macro"] = recall_score(y_test, y_pred, average="macro")

            if y_pred_proba is not None:
                eval_report["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

        elif task_type == TaskType.REGRESSION:
            eval_report["r2"] = r2_score(y_test, y_pred)
            eval_report["mse"] = mean_squared_error(y_test, y_pred)
            eval_report["mae"] = mean_absolute_error(y_test, y_pred)
            eval_report["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))

        return eval_report

    def evaluate_clustering(self, model, X, model_config: ModelConfig) -> Dict[str, float]:
        """
        Evaluate clustering using silhouette score (or other metrics).
        """
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        eval_report = {}
        if model_config.metric_name == "silhouette":
            try:
                score = silhouette_score(X, labels)
                eval_report["silhouette"] = score
            except Exception as e:
                st.error(f"Error computing silhouette score: {e}")
                logger.error(f"Error computing silhouette score: {e}")
        return eval_report

    def select_best_model(self, results: Dict[str, Dict[str, float]], primary_metric: str) -> Optional[str]:
        """
        Choose the best model based on the specified 'primary_metric'.
        If the metric is an error metric (like 'mse'), lower is better; otherwise, higher is better.
        """
        best_model = None
        best_score = float('-inf')
        # If it's an error metric, we invert the logic
        lower_is_better = primary_metric in ["mse", "mae", "rmse"]

        for model_name, metrics in results.items():
            metric_val = metrics.get(primary_metric)
            if metric_val is None:
                continue

            # Decide how to compare
            if lower_is_better:
                # Convert to negative so that a lower metric_val => higher "score"
                metric_val = -metric_val

            if metric_val > best_score:
                best_score = metric_val
                best_model = model_name

        return best_model

    # --------------------------------------------------------------------
    # 5) SHAP Explanation
    # --------------------------------------------------------------------
    def get_shap_explanation(self, model, X_test) -> Tuple[Optional[Any], Optional[plt.Figure]]:
        """
        Generates a SHAP explanation for the provided model using a subset of X_test.
        Resets the index to avoid index issues that might trigger conversion errors.
        """
        if X_test is None or len(X_test) == 0:
            return None, None

        # Limit to a smaller subset and reset index to avoid potential indexing issues
        sample_X = X_test.iloc[:50, :].reset_index(drop=True)
        try:
            explainer = shap.Explainer(model, sample_X)
            shap_values = explainer(sample_X)
            plt.figure()
            shap.summary_plot(shap_values, sample_X, show=False)
            fig = plt.gcf()
            return shap_values, fig
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return None, None

    # --------------------------------------------------------------------
    # 6) Generate Markdown Report
    # --------------------------------------------------------------------
    def generate_markdown_report(self) -> str:
        """
        Generate a comprehensive Markdown report that includes:
        - Dataset overview
        - Model configuration details (including hyperparameters)
        - Evaluation table for all trained models
        - Best model highlights
        - SHAP analysis info (if available)
        """
        import json

        lines = []
        lines.append("# Genta AutoML Report\n")

        # ----------------------------------------------------------------------
        # 1) Dataset Overview
        # ----------------------------------------------------------------------
        data = st.session_state.get("data")
        if data is not None:
            df_pandas = data.to_pandas()
            rows, cols = df_pandas.shape
            lines.append("## 1. Dataset Overview\n")
            lines.append(f"- **Number of Rows**: {rows}")
            lines.append(f"- **Number of Columns**: {cols}")
            lines.append(f"- **Columns**: {list(df_pandas.columns)}\n")
        else:
            lines.append("## 1. Dataset Overview\n")
            lines.append("_No dataset found in session state._\n")

        # ----------------------------------------------------------------------
        # 2) Model Configuration
        # ----------------------------------------------------------------------
        model_config = st.session_state.get("model_config")
        lines.append("## 2. Model Configuration\n")
        if model_config is not None:
            lines.append(f"- **Task Type**: {model_config.task_type.value}")
            lines.append(f"- **Target Column**: {model_config.target_column}")
            lines.append(f"- **Selected Models**: {model_config.models}")
            lines.append(f"- **Metric Name**: {model_config.metric_name}")
            lines.append(f"- **Auto Model Selection**: {model_config.auto_model_selection}")
            lines.append(f"- **Random Seed**: {model_config.random_seed}")
            lines.append(f"- **Time Budget (seconds)**: {model_config.time_budget}")
            if model_config.task_type == model_config.task_type.CLUSTERING:
                lines.append(f"- **Number of Clusters**: {model_config.n_clusters}")
            lines.append("")

            # ------------------------------------------------------------------
            # 2.1) Hyperparameter Configuration
            # ------------------------------------------------------------------
            lines.append("### Hyperparameters Used")
            if model_config.hyperparameter_configs:
                # We'll show a sub-table for each model's hyperparameters
                for model_name, hp_list in model_config.hyperparameter_configs.items():
                    if not hp_list:
                        continue
                    lines.append(f"**Model**: {model_name}")

                    # Build a Markdown table
                    table_header = "| Param Name | Min Value | Max Value | Step | Default |"
                    table_sep    = "|------------|----------|----------|------|---------|"
                    table_rows   = []
                    for hp in hp_list:
                        row = f"| {hp.name} | {hp.min_val} | {hp.max_val} | {hp.step} | {hp.default} |"
                        table_rows.append(row)
                    
                    lines.append(table_header)
                    lines.append(table_sep)
                    lines.extend(table_rows)
                    lines.append("")  # blank line after table
            else:
                lines.append("_No custom hyperparameter configuration found._\n")
        else:
            lines.append("_No model configuration found in session state._\n")

        # ----------------------------------------------------------------------
        # 3) Evaluation Results
        # ----------------------------------------------------------------------
        lines.append("## 3. Evaluation Results\n")
        evaluation_report = st.session_state.get("evaluation_report", {})
        if evaluation_report:
            # If your evaluation_report is structured as a dictionary of 
            # { "modelA": {...metrics...}, "modelB": {...metrics...}, "Best Model": "...", ... } 
            # we can build a table of all model metrics:
            best_model = evaluation_report.get("Best Model")

            # Filter out special keys like "Best Model" or "Metric"
            # Assume each key is a model name mapping to a dictionary of metrics
            # Example: evaluation_report = {
            #   "modelA": {"accuracy": 0.91, "f1_macro": 0.90}, 
            #   "modelB": {"accuracy": 0.87, "f1_macro": 0.85},
            #   "Best Model": "modelA",
            #   ...
            # }
            # Adjust logic as needed if your actual structure differs.

            # Collect only model-based metric dicts
            model_metric_dicts = {
                k: v for k, v in evaluation_report.items()
                if isinstance(v, dict)  # metric dictionaries
            }

            if model_metric_dicts:
                # We can gather all distinct metric names across models
                all_metrics = set()
                for metrics in model_metric_dicts.values():
                    all_metrics.update(metrics.keys())
                all_metrics = list(all_metrics)

                # Build a Markdown table
                table_header = "| Model | " + " | ".join(all_metrics) + " |"
                table_sep = "|-------|" + "|".join(["----------"] * len(all_metrics)) + "|"
                table_rows = []
                for model_name, metrics in model_metric_dicts.items():
                    row_values = []
                    for m in all_metrics:
                        val = metrics.get(m, "N/A")
                        # Round float values if you want
                        if isinstance(val, float):
                            val = f"{val:.4f}"
                        row_values.append(str(val))
                    row_str = " | ".join(row_values)
                    table_rows.append(f"| {model_name} | {row_str} |")

                lines.append("### Overall Evaluation Table")
                lines.append(table_header)
                lines.append(table_sep)
                lines.extend(table_rows)
                lines.append("")  # blank line

            # Show best model
            if best_model:
                lines.append(f"- **Best Model**: {best_model}\n")

            # Optionally show the full report as JSON (if desired)
            # lines.append("```json")
            # lines.append(json.dumps(evaluation_report, indent=2, default=str))
            # lines.append("```\n")

        else:
            lines.append("_No evaluation results found in session state._\n")

        # ----------------------------------------------------------------------
        # 4) SHAP Analysis
        # ----------------------------------------------------------------------
        lines.append("## 4. SHAP Analysis\n")
        shap_values = st.session_state.get("shap_values")
        if shap_values is not None:
            lines.append(
                "SHAP values were computed to interpret feature contributions. "
                "A summary plot was displayed in the UI but not embedded in this Markdown."
            )
        else:
            lines.append("_No SHAP analysis was performed or no SHAP values found._")

        lines.append("")


        lines.append("---")
        lines.append("**Report generated by Genta AutoML**")

        return "\n".join(lines)


    def download_markdown_report_button(self) -> None:
        """
        Generates a new markdown report, shows it, and provides a button to download.
        Saves locally with a timestamped filename.
        """
        import datetime
        import os

        # 1) Generate the Markdown content
        md_content = self.generate_markdown_report()
        
        # 2) Display it (optional) as raw Markdown in the UI
        st.markdown(md_content)

        # 3) Create the result folder if not exist
        if not os.path.exists("result"):
            os.makedirs("result")

        # 4) Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"genta_automl_report_{timestamp}.md"
        local_filepath = os.path.join("result", filename)

        # 5) Save the file locally
        with open(local_filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 6) Provide download button to the user
        st.download_button(
            label="Download Markdown Report",
            data=md_content,
            file_name=filename,
            mime="text/markdown"
        )
        st.success(f"Report saved locally to: {local_filepath}")


    # --------------------------------------------------------------------
    # 7) Main Execution
    # --------------------------------------------------------------------
    def run(self):
        """Main entry point for the Streamlit app."""
        df = self.load_data()
        if df is not None:
            model_config = self.get_model_config(df)
            if model_config:
                st.success("Model configuration complete. Review details below and click 'Train and Evaluate Model'.")
                st.write("### Current Model Configuration")
                st.write(model_config)

                if st.button("Train and Evaluate Model"):
                    self.train_and_evaluate_model(model_config)
                    st.info("Training and evaluation completed.")

                # Optional: Generate a final "best model" report
                if st.button("Generate Final Report"):
                    # If you have a separate best-model selection logic or a final summary:
                    final_report = {}
                    # For example, if you stored something in st.session_state['evaluation_report']:
                    if 'evaluation_report' in st.session_state and st.session_state['evaluation_report']:
                        final_report = st.session_state['evaluation_report']

                    # Or you can store additional info in `final_report` as needed
                    final_report["Best Model"] = "N/A"
                    if 'trained_model' in st.session_state and st.session_state['trained_model']:
                        final_report["Best Model"] = str(st.session_state['trained_model'])

                    # Convert that to a Markdown report
                    md_report = self.generate_markdown_report()
                    st.markdown(md_report)
                    if st.button("Generate & Download Final Report"):
                        self.download_markdown_report_button()

                else:
                    st.info("Click 'Generate Final Report' to create a downloadable Markdown summary.")

# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    app = GentaAutoMLApp()
    app.run()
