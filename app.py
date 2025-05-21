import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import json
import requests
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Any

# -----------------------------------------------------------------------------
# Optional Kolosal imports
# -----------------------------------------------------------------------------
try:
    from modules.configs import (
        MLTrainingEngineConfig,
        TaskType,
        OptimizationStrategy,
        BatchProcessorConfig,
        PreprocessorConfig,
    )
    from modules.engine.train_engine import MLTrainingEngine
    from modules.model_manager import SecureModelManager

    KOLOSAL_IMPORTS_SUCCESS = True
except ImportError as e:
    KOLOSAL_IMPORTS_SUCCESS = False
    import_error = str(e)

# -----------------------------------------------------------------------------
# Streamlit page configuration & helpers
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kolosal AutoML UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#4B56D2"
SECONDARY_COLOR = "#82C3EC"
BACKGROUND_COLOR = "#F5F5F5"
TEXT_COLOR = "#333333"

st.markdown(
    f"""
    <style>
    .main .block-container{{padding-top:2rem;padding-bottom:2rem;}}
    .stButton>button{{background-color:{PRIMARY_COLOR};color:#fff;border-radius:6px;padding:0.5rem 1rem;font-weight:500;}}
    .stButton>button:hover{{background-color:{SECONDARY_COLOR};color:{TEXT_COLOR};}}
    h1,h2,h3{{color:{PRIMARY_COLOR};}}
    .card{{background-color:#fff;border-radius:8px;padding:1.5rem;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session‑state initialisation
# -----------------------------------------------------------------------------
def _init_session_state() -> None:
    """Ensure all required keys exist in session_state."""
    defaults = {
        "trained_models": {},
        "current_model": None,
        "data_uploaded": False,
        "X_train": None,
        "y_train": None,
        "feature_names": None,
        "target_name": None,
        "engine": None,
        "task_type": "classification",
        "optimization_strategy": "random_search",
        "df": None,
        "sample_data_description": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session_state()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def create_training_engine(config: Dict[str, Any]) -> MLTrainingEngine:
    """Instantiate a MLTrainingEngine given a user config dict."""
    model_path = config.get("model_path", "./models")
    os.makedirs(model_path, exist_ok=True)

    task_type = TaskType.CLASSIFICATION if config.get("task_type") == "classification" else TaskType.REGRESSION

    opt_map = {
        "grid_search": OptimizationStrategy.GRID_SEARCH,
        "random_search": OptimizationStrategy.RANDOM_SEARCH,
        "bayesian_optimization": OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        "hyperx": OptimizationStrategy.HYPERX,
    }
    opt_strategy = opt_map.get(config.get("optimization_strategy", "random_search"), OptimizationStrategy.RANDOM_SEARCH)

    pre_cfg = PreprocessorConfig(
        normalization=config.get("normalization", "standard"),
        handle_nan=config.get("handle_nan", True),
        handle_inf=True,
        detect_outliers=config.get("detect_outliers", True),
    )

    batch_cfg = BatchProcessorConfig(initial_batch_size=config.get("batch_size", 64), enable_memory_optimization=True)

    engine_cfg = MLTrainingEngineConfig(
        task_type=task_type,
        random_state=config.get("random_seed", 42),
        n_jobs=config.get("n_jobs", -1),
        verbose=config.get("verbose", 1),
        cv_folds=config.get("cv_folds", 5),
        test_size=config.get("test_size", 0.2),
        stratify=config.get("stratify", True),
        optimization_strategy=opt_strategy,
        optimization_iterations=config.get("optimization_iterations", 20),
        early_stopping=config.get("early_stopping", True),
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        feature_selection=config.get("feature_selection", True),
        model_path=model_path,
        experiment_tracking=True,
        auto_save=True,
        preprocessing_config=pre_cfg,
        batch_processing_config=batch_cfg,
    )
    return MLTrainingEngine(engine_cfg)


def load_data(file) -> Optional[pd.DataFrame]:
    """Load uploaded file into a DataFrame (supports CSV, Excel, Parquet, JSON)."""
    ext = Path(file.name).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file)
    if ext == ".parquet":
        return pd.read_parquet(file)
    if ext == ".json":
        return pd.read_json(file)
    st.error(f"Unsupported file type: {ext}")
    return None


def _feature_importance_plot(importances: np.ndarray | Dict[str, float], feature_names: List[str]) -> plt.Figure:
    if isinstance(importances, np.ndarray):
        importances = {name: float(val) for name, val in zip(feature_names, importances)}

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([k for k, _ in sorted_items], [v for _, v in sorted_items])
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def _confusion_matrix_plot(y_true, y_pred, class_names: List[str]):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def _residual_plot(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    return fig

# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------

def sidebar_configuration():
    """Render sidebar and return a dict of selected configuration values."""
    st.sidebar.header("Settings")

    # Data
    st.sidebar.subheader("Data Split & Seed")
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_seed = st.sidebar.number_input("Random seed", 1, 999999, 42)

    # Task & optimisation
    st.sidebar.subheader("Task & Optimisation")
    task_type = st.sidebar.selectbox("Task type", ["classification", "regression"], index=0 if st.session_state.task_type == "classification" else 1)
    optimisation_strategy = st.sidebar.selectbox(
        "Optimisation strategy", ["random_search", "grid_search", "bayesian_optimization", "hyperx"],
        index=["random_search", "grid_search", "bayesian_optimization", "hyperx"].index(st.session_state.optimization_strategy),
    )
    optimisation_iterations = st.sidebar.slider("Optimisation iterations", 5, 100, 20)
    cv_folds = st.sidebar.slider("CV folds", 3, 10, 5)

    # Pre‑processing
    st.sidebar.subheader("Pre‑processing")
    handle_nan = st.sidebar.checkbox("Handle missing values", True)
    detect_outliers = st.sidebar.checkbox("Handle outliers", True)
    normalization = st.sidebar.selectbox("Normalization", ["none", "standard", "minmax", "robust"], index=1)

    # Advanced
    with st.sidebar.expander("Advanced"):
        n_jobs = st.slider("n_jobs (-1 = all cores)", -1, 8, -1)
        early_stopping = st.checkbox("Early stopping", True)
        early_rounds = st.slider("Early stopping rounds", 5, 30, 10)
        verbose = st.selectbox("Verbosity", [0, 1, 2], index=1)

    # Update session‑state shortcuts
    st.session_state.task_type = task_type
    st.session_state.optimization_strategy = optimisation_strategy

    return {
        "task_type": task_type,
        "optimization_strategy": optimisation_strategy,
        "optimization_iterations": optimisation_iterations,
        "test_size": test_size,
        "random_seed": random_seed,
        "cv_folds": cv_folds,
        "n_jobs": n_jobs,
        "early_stopping": early_stopping,
        "early_stopping_rounds": early_rounds,
        "verbose": verbose,
        "handle_nan": handle_nan,
        "detect_outliers": detect_outliers,
        "normalization": normalization,
    }


# -----------------------------------------------------------------------------
# Tab helpers
# -----------------------------------------------------------------------------

def tab_data():
    """Data upload, sample datasets and preparation."""
    st.header("Data Upload & Exploration")

    # ---- Sample datasets ----
    with st.expander("📊 Load sample dataset"):
        sample_choice = st.radio("Choose a sample dataset", ["Breast Cancer", "Wine Quality", "Diabetes", "Boston Housing"], horizontal=True)
        if st.button("Load sample"):
            df, descr = load_sample_dataset(sample_choice)
            if df is not None:
                set_dataframe(df, descr)

    # ---- Upload ----
    uploaded = st.file_uploader("Upload data (CSV / Excel / Parquet / JSON)", type=["csv", "xlsx", "xls", "parquet", "json"])
    if uploaded is not None:
        with st.spinner("Loading …"):
            df = load_data(uploaded)
        if df is not None:
            set_dataframe(df)

    # If data available show preview & prep
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("Statistics"):
            st.dataframe(df.describe().T, use_container_width=True)

        # ---- Target & feature selection ----
        st.subheader("Target & Features")
        target_col = st.selectbox("Choose target column", df.columns)
        default_task = "classification" if df[target_col].nunique() < 10 else "regression"
        st.info(f"Inferred task type: **{default_task}**")
        feature_cols = [c for c in df.columns if c != target_col]
        selected_feats = st.multiselect("Select features", feature_cols, default=feature_cols)

        if st.button("Prepare data"):
            prepare_data(df, target_col, selected_feats)


def load_sample_dataset(name: str) -> tuple[pd.DataFrame | None, str]:
    """Return a (df, description) tuple based on sample name."""
    try:
        if name == "Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            desc = "Breast Cancer Wisconsin dataset (binary classification)."
        elif name == "Wine Quality":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=";")
            desc = "Red Wine Quality dataset (score 0‑10)."
        elif name == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            desc = "Diabetes progression dataset (regression)."
        else:  # Boston / California housing fallback
            try:
                from sklearn.datasets import load_boston
                data = load_boston()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                desc = "Boston housing (regression)."
            except Exception:
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                desc = "California housing (regression)."
        return df, desc
    except Exception as e:
        st.error(f"Failed to load sample dataset: {e}")
        return None, ""


def set_dataframe(df: pd.DataFrame, description: str = "") -> None:
    st.session_state.df = df
    st.session_state.sample_data_description = description
    st.session_state.data_uploaded = True


def prepare_data(df: pd.DataFrame, target: str, features: List[str]):
    X = df[features]
    y = df[target]
    # One‑hot encode categorical
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols):
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.feature_names = list(X.columns)
    st.session_state.target_name = target
    st.success("Data prepared — proceed to *Train* tab")


# -----------------------------------------------------------------------------
# Training tab
# -----------------------------------------------------------------------------

def tab_train(config: Dict[str, Any]):
    st.header("Model Training")
    if st.session_state.X_train is None:
        st.warning("Prepare data in the *Data* tab first.")
        return

    # Summary
    st.markdown(f"**Samples:** {len(st.session_state.y_train)}   |   **Features:** {len(st.session_state.feature_names)}   |   **Target:** {st.session_state.target_name}")

    # Model list
    if st.session_state.task_type == "classification":
        model_choices = ["random_forest", "gradient_boosting", "logistic_regression", "svm", "xgboost", "lightgbm"]
    else:
        model_choices = ["random_forest", "gradient_boosting", "linear_regression", "svr", "xgboost", "lightgbm"]
    selected_models = st.multiselect("Choose models to train", model_choices, default=[model_choices[0]])

    # Initialise engine
    if st.button("Initialise engine"):
        if not KOLOSAL_IMPORTS_SUCCESS:
            st.error(f"Kolosal components missing: {import_error}")
        else:
            with st.spinner("Initialising …"):
                st.session_state.engine = create_training_engine({
                    **config,
                    "model_path": "./models",
                    "checkpointing": True,
                    "checkpoint_path": "./checkpoints",
                })
            st.success("Engine initialised")

    # Train
    if st.session_state.engine and selected_models:
        if st.button("Train"):
            for mdl in selected_models:
                with st.spinner(f"Training {mdl} …"):
                    res = st.session_state.engine.train_model(st.session_state.X_train, st.session_state.y_train, model_type=mdl)
                st.session_state.trained_models[res["model_name"]] = res
            st.session_state.current_model = next(iter(st.session_state.trained_models))
            st.success("Training complete")

    # Display metrics
    if st.session_state.trained_models:
        display_model_table()


def display_model_table():
    data = []
    for name, info in st.session_state.trained_models.items():
        row = {"Model": name}
        metrics = info.get("metrics", {})
        row.update(metrics)
        data.append(row)
    st.dataframe(pd.DataFrame(data), use_container_width=True)

    chosen = st.selectbox("Select model", list(st.session_state.trained_models.keys()), key="model_details_select")
    st.session_state.current_model = chosen

    info = st.session_state.trained_models[chosen]
    with st.expander("Details"):
        st.write("**Hyperparameters**")
        st.json(info.get("params", {}))
        if info.get("feature_importance") is not None:
            fig = _feature_importance_plot(info["feature_importance"], st.session_state.feature_names)
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# Evaluation tab
# -----------------------------------------------------------------------------

def tab_evaluate():
    st.header("Evaluate & Test")
    if not st.session_state.trained_models:
        st.warning("Train at least one model first.")
        return

    model_name = st.selectbox("Model to evaluate", list(st.session_state.trained_models.keys()), index=list(st.session_state.trained_models.keys()).index(st.session_state.current_model))
    info = st.session_state.trained_models[model_name]
    mdl = info["model"]
    st.subheader("Training CV Metrics")
    st.json(info.get("metrics", {}))

    # ---- Optional external test ----
    st.subheader("External test data (optional)")
    test_upload = st.file_uploader("Upload test data", type=["csv", "xlsx", "xls", "parquet", "json"], key="test_upload")
    if test_upload is not None:
        df_test = load_data(test_upload)
        if df_test is not None:
            process_and_evaluate_external(df_test, mdl)


def process_and_evaluate_external(df_test: pd.DataFrame, model):
    miss_feats = [f for f in st.session_state.feature_names if f not in df_test.columns]
    if miss_feats:
        st.warning(f"Missing features in test set → {', '.join(miss_feats)}. Using intersection only.")
    X = df_test[[c for c in st.session_state.feature_names if c in df_test.columns]]
    y = df_test.get(st.session_state.target_name)

    preds = model.predict(X)

    res_df = X.copy()
    res_df["prediction"] = preds
    if y is not None:
        res_df["actual"] = y
    st.dataframe(res_df.head(), use_container_width=True)

    if y is not None:
        from sklearn import metrics as skm
        if st.session_state.task_type == "classification":
            acc = skm.accuracy_score(y, preds)
            st.metric("Accuracy", f"{acc:.4f}")
            fig = _confusion_matrix_plot(y, preds, [str(x) for x in np.unique(y)])
            st.pyplot(fig)
        else:
            mse = skm.mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.4f}")
            fig = _residual_plot(y, preds)
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# Export tab
# -----------------------------------------------------------------------------

def tab_export():
    st.header("Export & Deploy")
    if not st.session_state.trained_models:
        st.warning("No models to export.")
        return

    mdl_name = st.selectbox("Model", list(st.session_state.trained_models.keys()), index=list(st.session_state.trained_models.keys()).index(st.session_state.current_model), key="export_select")
    info = st.session_state.trained_models[mdl_name]

    export_fmt = st.radio("Format", ["Pickle", "Joblib", "ONNX"], horizontal=True)
    include_preproc = st.checkbox("Include preprocessor", True)
    compress = st.checkbox("Compress", True)

    if st.button("Export model"):
        with st.spinner("Exporting …"):
            export_model(info, mdl_name, export_fmt, include_preproc, compress)


def export_model(info: Dict[str, Any], name: str, fmt: str, include_preproc: bool, compress: bool):
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("exports", exist_ok=True)
    if fmt == "ONNX":
        try:
            import skl2onnx
            from skl2onnx.common.data_types import FloatTensorType
            initial_type = [("float_input", FloatTensorType([None, len(st.session_state.feature_names)]))]
            onnx_model = skl2onnx.convert_sklearn(info["model"], initial_types=initial_type, options={id(info["model"]): {"zipmap": False}})
            path = f"exports/{name}_{ts}.onnx"
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except Exception as e:
            st.error(f"ONNX export failed: {e}")
            return
    else:
        pkg = {
            "model": info["model"],
            "params": info.get("params", {}),
            "metrics": info.get("metrics", {}),
            "feature_names": st.session_state.feature_names,
            "target_name": st.session_state.target_name,
            "preprocessor": st.session_state.engine.preprocessor if include_preproc else None,
        }
        path = f"exports/{name}_{ts}.pkl" if fmt == "Pickle" else f"exports/{name}_{ts}.joblib"
        if fmt == "Pickle":
            with open(path, "wb") as f:
                pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            import joblib
            joblib.dump(pkg, path, compress=3 if compress else 0)

    st.success(f"Model exported → {path}")
    with open(path, "rb") as f:
        st.download_button("Download", f.read(), os.path.basename(path))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    from PIL import Image

    logo_path = "assets/kolosal-logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=180)

    st.title("Kolosal AutoML")
    st.markdown("### Low‑code ML pipeline builder")

    if not KOLOSAL_IMPORTS_SUCCESS:
        st.error(f"Failed to import Kolosal core components: {import_error}")
        st.stop()

    # Sidebar
    config = sidebar_configuration()

    # Tabs
    tabs = st.tabs(["📊 Data", "🔧 Train", "📈 Evaluate", "💾 Export"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_train(config)
    with tabs[2]:
        tab_evaluate()
    with tabs[3]:
        tab_export()



if __name__ == "__main__":
    main()
