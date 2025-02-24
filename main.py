import streamlit as st
import polars as pl
import pandas as pd
import io
import numpy as np
import logging
import matplotlib.pyplot as plt
import shap

from flaml import AutoML
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple, NamedTuple
from flaml import tune

import hdbscan

# Configure logging
logging.basicConfig(level=logging.INFO)

# Hyperparameter configuration class
@dataclass
class HyperParameterConfig:
    param_name: str
    min_value: float
    max_value: float
    log_scale: bool = False

class TaskType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    CLUSTERING = "Clustering"

@dataclass
class ModelConfig:
    task_type: TaskType
    target_column: str
    time_budget: Optional[int] = 60  # 0 means train until target reached; None for clustering tasks
    n_clusters: Optional[int] = 3
    selected_models: List[str] = None
    target_score: float = None
    metric_name: str = None
    random_seed: int = 42
    verbose: int = 0
    show_shap: bool = True
    hyperparameter_ranges: Dict[str, Dict[str, Dict]] = None

class TargetMetrics(NamedTuple):
    metric_name: str
    target_value: float
    achieved_value: float
    is_achieved: bool

# Default hyperparameter ranges for each model
DEFAULT_HYPERPARAMETERS = {
    'lgbm': {
        'n_estimators': HyperParameterConfig('n_estimators', 10, 1000, True),
        'learning_rate': HyperParameterConfig('learning_rate', 0.01, 1.0, True),
        'max_depth': HyperParameterConfig('max_depth', 3, 12, False),
        'num_leaves': HyperParameterConfig('num_leaves', 2, 256, True),
    },
    'xgboost': {
        'n_estimators': HyperParameterConfig('n_estimators', 10, 1000, True),
        'learning_rate': HyperParameterConfig('learning_rate', 0.01, 1.0, True),
        'max_depth': HyperParameterConfig('max_depth', 3, 12, False),
    },
    'rf': {
        'n_estimators': HyperParameterConfig('n_estimators', 10, 500, True),
        'max_depth': HyperParameterConfig('max_depth', 3, 20, False),
        'min_samples_split': HyperParameterConfig('min_samples_split', 2, 20, False),
    },
    'kmeans': {
        'n_init': HyperParameterConfig('n_init', 5, 20, False),
        'max_iter': HyperParameterConfig('max_iter', 100, 1000, True),
    },
    'dbscan': {
        'eps': HyperParameterConfig('eps', 0.1, 2.0, True),
        'min_samples': HyperParameterConfig('min_samples', 2, 20, False),
    }
}

@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, file_name: str) -> Optional[pl.DataFrame]:
    """
    Load data from bytes depending on file type.
    """
    try:
        if file_name.lower().endswith('.parquet'):
            df = pl.read_parquet(io.BytesIO(file_bytes))
        elif file_name.lower().endswith('.xlsx'):
            df = pl.from_pandas(pd.read_excel(io.BytesIO(file_bytes)))
        else:
            df = pl.read_csv(io.BytesIO(file_bytes))
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logging.exception("Error loading data")
        return None
def get_tune_domain(param_name: str, param_config: HyperParameterConfig):
    # Decide whether it's an integer param or float param
    # (You can do this by name, or store a flag in your param_config.)
    integer_params = {
        "n_estimators", "max_depth", "num_leaves", 
        "min_samples_split", "n_init", "min_samples"
    }
    is_integer = param_name in integer_params

    # Get the lower and upper limits
    lower = param_config.min_value
    upper = param_config.max_value

    # For integer parameters
    if is_integer:
        lower_int = int(lower)
        upper_int = int(upper)
        if param_config.log_scale:
            return tune.lograndint(lower=lower_int, upper=upper_int+1)
        else:
            return tune.randint(lower=lower_int, upper=upper_int+1)
    else:
        # For float parameters
        if param_config.log_scale:
            return tune.loguniform(lower=lower, upper=upper)
        else:
            return tune.uniform(lower=lower, upper=upper)
class GentaAutoMLApp:
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()

    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="Genta AutoML",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Genta AutoML: Next-Generation Automated Machine Learning")
        st.sidebar.header("About")
        st.sidebar.info(
            "Genta AutoML is an enhanced AutoML framework featuring advanced configurations, "
            "improved model insights, and support for multiple data formats. "
            "Upload your dataset, configure your target, models and hyperparameters, "
            "and let Genta AutoML do the heavy lifting!"
        )

    @staticmethod
    def initialize_session_state():
        for key in ['data', 'model', 'results', 'target_metrics']:
            if key not in st.session_state:
                st.session_state[key] = None

    def load_data(self) -> Optional[pl.DataFrame]:
        """
        Step 1: Upload Dataset and preview the data.
        """
        with st.expander("📁 Step 1: Upload Dataset", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload CSV/Parquet/Excel file",
                type=["csv", "parquet", "xlsx"],
                help="Dataset should be numeric except for target column"
            )
            if uploaded_file is not None:
                df = load_data_from_bytes(uploaded_file.read(), uploaded_file.name)
                if df is not None:
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    st.write(f"Shape: {df.shape} | Columns: {', '.join(df.columns)}")
                return df
            return None

    def get_available_models(self, task_type: TaskType) -> List[str]:
        """
        Provide a list of models based on the task type.
        """
        model_options = {
            TaskType.CLASSIFICATION: ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],
            TaskType.REGRESSION: ['lgbm', 'xgboost', 'rf', 'extra_tree'],
            TaskType.CLUSTERING: ['kmeans', 'dbscan', 'hdbscan', 'gmm']
        }
        return st.multiselect(
            "Select Models",
            options=model_options[task_type],
            default=[],
            help="Leave empty to use all available models"
        )

    def get_target_metrics(self, task_type: TaskType) -> Tuple[str, float]:
        with st.expander("🎯 Step 4: Set Target Performance", expanded=True):
            col1, col2 = st.columns(2)
            if task_type == TaskType.CLASSIFICATION:
                metric_options = ['accuracy', 'f1', 'precision', 'recall', 'auc']
            elif task_type == TaskType.REGRESSION:
                metric_options = ['rmse', 'mae', 'r2', 'mse']
            else:
                metric_options = ['silhouette_score']

            with col1:
                metric_name = st.selectbox("Select Performance Metric", options=metric_options, index=0)

            with col2:
                if task_type == TaskType.CLASSIFICATION:
                    target_score = st.slider(
                        "Target Score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        help="Target performance score to achieve"
                    )
                elif task_type == TaskType.REGRESSION:
                    # Make sure step is a float:
                    target_score = st.number_input(
                        "Target Score",
                        min_value=0.0,
                        value=0.5,
                        step=0.1,  # changed from step=1
                        help="Target performance score to achieve"
                    )
                else:
                    target_score = st.slider(
                        "Target Silhouette Score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05
                    )
        return metric_name, target_score

    


    def get_hyperparameter_ranges(self, selected_models: List[str]) -> Dict[str, Dict[str, Dict]]:
        st.markdown("## Hyperparameter Ranges")
        st.info("Configure hyperparameter ranges for each model (use the settings below).")

        hyperparameter_ranges = {}
        selected_models = selected_models or list(DEFAULT_HYPERPARAMETERS.keys())

        for model_name in selected_models:
            if model_name not in DEFAULT_HYPERPARAMETERS:
                st.warning(f"Unknown model {model_name} in hyperparameter configuration.")
                continue

            st.subheader(f"{model_name.upper()} Parameters")
            model_params = {}
            params = DEFAULT_HYPERPARAMETERS[model_name]

            for param_name, param_config in params.items():
                default_min = float(param_config.min_value)
                default_max = float(param_config.max_value)
                # **Use float for step_val to avoid mixed numeric types**
                step_val = 0.1 if param_config.log_scale else 1.0

                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"**{param_name}**")
                    st.caption(f"Log scale: {param_config.log_scale}")
                with cols[1]:
                    min_val = st.number_input(
                        f"Min ({model_name}-{param_name})",
                        key=f"{model_name}_{param_name}_min",
                        value=default_min,
                        step=step_val
                    )
                with cols[2]:
                    max_val = st.number_input(
                        f"Max ({model_name}-{param_name})",
                        key=f"{model_name}_{param_name}_max",
                        value=default_max,
                        step=step_val
                    )

                if min_val >= max_val:
                    st.error("Max value must be greater than min value.")
                    continue

                model_params[param_name] = {
                    "min": float(min_val),
                    "max": float(max_val),
                    "log_scale": param_config.log_scale
                }

            hyperparameter_ranges[model_name] = model_params

        return hyperparameter_ranges


    def get_model_config(self, df: pl.DataFrame) -> Optional[ModelConfig]:
        """
        Step 2: Select target column, task type, models and advanced options.
        """
        with st.expander("🎯 Step 2: Select Target & Configure Model", expanded=True):
            target_col = st.selectbox(
                "Select target variable (y)",
                options=df.columns,
                index=len(df.columns) - 1
            )
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                task_type = TaskType(st.selectbox(
                    "Machine Learning Task",
                    options=[t.value for t in TaskType]
                ))
            with col2:
                if task_type == TaskType.CLUSTERING:
                    n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3)
                    time_budget = None
                else:
                    time_budget = st.slider(
                        "Time Budget (seconds) (0 means train until target reached)",
                        min_value=0,
                        max_value=3600,
                        value=60
                    )
                    n_clusters = None
            with col3:
                selected_models = self.get_available_models(task_type)
        metric_name, target_score = self.get_target_metrics(task_type)
        with st.expander("🔧 Step 3: Advanced Options", expanded=False):
            random_seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
            verbose = st.selectbox("Verbose Level", options=[0, 1, 2], index=0)
            show_shap = st.checkbox("Show SHAP Explanation (if applicable)", value=True)
            hyperparameter_ranges = self.get_hyperparameter_ranges(selected_models)
        return ModelConfig(
            task_type=task_type,
            target_column=target_col,
            time_budget=time_budget,
            n_clusters=n_clusters,
            selected_models=selected_models,
            target_score=target_score,
            metric_name=metric_name,
            random_seed=random_seed,
            verbose=verbose,
            show_shap=show_shap,
            hyperparameter_ranges=hyperparameter_ranges
        )

    def train_model(self, df: pl.DataFrame, config: ModelConfig):
        """
        Train the model based on the task type.
        """
        X = df.drop(config.target_column)
        if config.task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            y = df[config.target_column]
            return self.train_automl(X, y, config)
        else:
            return self.train_clustering(X, config)



    def train_automl(self, X: pl.DataFrame, y: pl.Series, config: ModelConfig):
        """
        Train classification/regression models using FLAML.
        """
        automl = AutoML()
        settings = {
            "time_budget": config.time_budget,
            "metric": config.metric_name,
            "task": config.task_type.value.lower(),
            "estimator_list": config.selected_models if config.selected_models else 'auto',
            "verbose": config.verbose,
            "seed": config.random_seed
        }
        if config.metric_name == "auc":
            st.warning("Metric 'auc' is not supported; remapping to 'roc_auc'.")
            settings["metric"] = "roc_auc"
            config.metric_name = "roc_auc"
        if config.hyperparameter_ranges:
            custom_hp = {}
            for model_name, params in config.hyperparameter_ranges.items():
                model_hp = {}
                for param_name, param_range_dict in params.items():
                    # param_range_dict looks like {"min": <>, "max": <>, "log_scale": <>}
                    # You originally stored the param metadata in a HyperParameterConfig
                    # so adapt accordingly:
                    param_config = HyperParameterConfig(
                        param_name=param_name,
                        min_value=param_range_dict["min"],
                        max_value=param_range_dict["max"],
                        log_scale=param_range_dict["log_scale"]
                    )
                    domain = get_tune_domain(param_name, param_config)

                    # Build the dictionary as FLAML expects
                    model_hp[param_name] = {"domain": domain}
                
                custom_hp[model_name] = model_hp

            settings["custom_hp"] = custom_hp


        X_pd, y_pd = X.to_pandas(), y.to_pandas()
        try:
            automl.fit(X_pd, y_pd, **settings)
        except ValueError as e:
            if "auc is neither" in str(e):
                st.warning("AUC metric is not supported by FLAML. Setting validation loss to np.inf.")
                automl.best_loss = np.inf
            else:
                logging.exception("Error during AutoML training")
                raise e

        # Ensure best_loss is a float
        best_loss = automl.best_loss[0] if isinstance(automl.best_loss, tuple) else automl.best_loss
        automl.best_loss = best_loss

        # Compute achieved score based on metric
        if config.metric_name in ['accuracy', 'f1', 'roc_auc']:
            achieved_score = 1 - best_loss
        else:
            achieved_score = best_loss

        st.session_state.target_metrics = TargetMetrics(
            metric_name=config.metric_name,
            target_value=config.target_score,
            achieved_value=achieved_score,
            is_achieved=(achieved_score >= config.target_score
                         if config.metric_name in ['accuracy', 'f1', 'roc_auc']
                         else achieved_score <= config.target_score)
        )
        return automl

    def train_clustering(self, X: pl.DataFrame, config: ModelConfig) -> List[Dict]:
        """
        Train clustering algorithms and return a list of results.
        """
        X_pd = X.to_pandas()
        results = []
        clustering_algorithms = {
            'kmeans': lambda: KMeans(n_clusters=config.n_clusters, random_state=config.random_seed),
            'dbscan': lambda: DBSCAN(eps=0.5, min_samples=5),
            'hdbscan': lambda: hdbscan.HDBSCAN(min_cluster_size=5),
            'gmm': lambda: GaussianMixture(n_components=config.n_clusters, random_state=config.random_seed)
        }
        selected_algos = config.selected_models if config.selected_models else ['kmeans']
        for algo_name in selected_algos:
            if algo_name not in clustering_algorithms:
                st.warning(f"Clustering algorithm {algo_name} is not supported.")
                continue
            try:
                model = clustering_algorithms[algo_name]()
                labels = model.fit_predict(X_pd)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_pd, labels)
                    results.append({'algorithm': algo_name, 'labels': labels, 'silhouette_score': score})
                else:
                    st.warning(f"Clustering algorithm {algo_name} resulted in only one cluster. Skipping...")
            except Exception as e:
                st.error(f"Error training {algo_name}: {e}")
                logging.exception(f"Error in clustering algorithm {algo_name}")
        return results

    def show_model_results(self, automl):
        """
        Display performance metrics and best model details.
        """
        if st.session_state.target_metrics:
            target_metrics = st.session_state.target_metrics
            st.write("### Model Performance")
            st.write(f"Metric: {target_metrics.metric_name}")
            st.write(f"Target: {target_metrics.target_value}")
            st.write(f"Achieved: {target_metrics.achieved_value}")
            st.write(f"Achieved Target: {target_metrics.is_achieved}")

        if hasattr(automl, 'best_model') and automl.best_model:
            st.write("### Best Model")
            st.write(automl.best_model)

        if hasattr(automl, 'best_loss'):
            st.write("### Best Loss")
            st.write(automl.best_loss)

    def show_clustering_results(self, results: List[Dict], X: pl.DataFrame):
        """
        Display clustering results and visualize the clusters using t-SNE.
        """
        st.write("### Clustering Results")
        X_pd = X.to_pandas()
        # Compute t-SNE embedding once for all clustering visualizations
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X_pd)

        for result in results:
            st.write(f"**Algorithm:** {result['algorithm']}")
            st.write(f"**Silhouette Score:** {result['silhouette_score']:.4f}")
            # Plot clusters using the precomputed t-SNE embedding
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                                 c=result['labels'], cmap='viridis', alpha=0.7)
            ax.set_title(f"t-SNE Visualization for {result['algorithm']}")
            st.pyplot(fig)
            plt.close(fig)

    def visualize_shap_summary(self, automl, X: pl.DataFrame, config: ModelConfig):
        """
        Generate and display a SHAP summary plot.
        """
        if config.show_shap and config.task_type != TaskType.CLUSTERING:
            X_pd = X.to_pandas()
            try:
                explainer = shap.TreeExplainer(automl.best_estimator)
                shap_values = explainer.shap_values(X_pd)
                st.subheader("SHAP Summary Plot")
                # Create a new figure for the SHAP plot
                fig = plt.figure()
                shap.summary_plot(shap_values, X_pd, show=False)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
                logging.exception("SHAP plot generation failed.")

if __name__ == "__main__":
    app = GentaAutoMLApp()
    df = app.load_data()

    if df is not None:
        config = app.get_model_config(df)
        if st.button("🚀 Train Model"):
            if config is None:
                st.error("Please configure the model before training.")
            else:
                with st.spinner("Training in progress..."):
                    if config.task_type != TaskType.CLUSTERING:
                        automl = app.train_model(df, config)
                        if automl:
                            app.show_model_results(automl)
                            # Pass the features (X) for SHAP visualization
                            app.visualize_shap_summary(automl, df.drop(config.target_column), config)
                            st.success("Training completed successfully!")
                    else:
                        # For clustering, pass the full feature set (X) to visualize results
                        results = app.train_model(df, config)
                        if results:
                            app.show_clustering_results(results, df.drop(config.target_column))
                            st.success("Clustering completed successfully!")
