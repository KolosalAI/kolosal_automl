import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import joblib
from io import StringIO
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

# Import the configuration classes
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,
    NormalizationType,
    BatchProcessorConfig,
    BatchProcessingStrategy,
    InferenceEngineConfig,
    QuantizationConfig,
    QuantizationType,
    QuantizationMode,
)
from modules.engine.train_engine import MLTrainingEngine
from modules.device_optimizer import *

# Set page configuration
st.set_page_config(
    page_title="Advanced ML Training Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "data" not in st.session_state:
    st.session_state.data = None
if "target" not in st.session_state:
    st.session_state.target = None
if "features" not in st.session_state:
    st.session_state.features = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if "training_completed" not in st.session_state:
    st.session_state.training_completed = False
if "models" not in st.session_state:
    st.session_state.models = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = []
if "config" not in st.session_state:
    st.session_state.config = None
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "device_optimized" not in st.session_state:
    st.session_state.device_optimized = False
if "optimized_configs" not in st.session_state:
    st.session_state.optimized_configs = {}
if "all_mode_configs" not in st.session_state:
    st.session_state.all_mode_configs = {}
if "device_optimization_mode" not in st.session_state:
    st.session_state.device_optimization_mode = None

# Define model options by task type
MODEL_OPTIONS = {
    TaskType.CLASSIFICATION: [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "SVC",
    ],
    TaskType.REGRESSION: [
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
        "SVR",
    ],
}

# Default hyperparameter grids
DEFAULT_PARAM_GRIDS = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l1", "l2", "elasticnet", None],
        "model__solver": ["lbfgs", "liblinear", "saga"],
    },
    "RandomForestClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
    },
    "XGBClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "XGBRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "LGBMClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "LGBMRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "CatBoostClassifier": {
        "model__iterations": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__depth": [4, 6, 8],
    },
    "CatBoostRegressor": {
        "model__iterations": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__depth": [4, 6, 8],
    },
    "SVC": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto", 0.1, 1.0],
    },
    "SVR": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto", 0.1, 1.0],
    },
    "LinearRegression": {
        "model__fit_intercept": [True, False],
        "model__normalize": [True, False],
    },
}

def import_model_libraries():
    """Import the necessary model libraries based on selections"""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR

    models = {
        "LogisticRegression": LogisticRegression,
        "LinearRegression": LinearRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "SVC": SVC,
        "SVR": SVR,
    }

    # Try to import optional libraries
    try:
        from xgboost import XGBClassifier, XGBRegressor
        models["XGBClassifier"] = XGBClassifier
        models["XGBRegressor"] = XGBRegressor
    except ImportError:
        st.warning("XGBoost is not installed. XGBoost models will not be available.")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        models["LGBMClassifier"] = LGBMClassifier
        models["LGBMRegressor"] = LGBMRegressor
    except ImportError:
        st.warning("LightGBM is not installed. LightGBM models will not be available.")

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        models["CatBoostClassifier"] = CatBoostClassifier
        models["CatBoostRegressor"] = CatBoostRegressor
    except ImportError:
        st.warning("CatBoost is not installed. CatBoost models will not be available.")

    return models

def optimize_for_device():
    """Run device optimization and create optimized configurations"""
    with st.spinner("Optimizing for your device..."):
        # Create output directories if they don't exist
        configs_dir = "./configs"
        checkpoint_dir = "./checkpoints"
        model_registry_dir = "./models"
        
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(model_registry_dir, exist_ok=True)
        
        # Create optimized configurations
        try:
            # Use the create_optimized_configs function from device_optimizer
            master_config = create_optimized_configs(
                config_path=configs_dir,
                checkpoint_path=checkpoint_dir,
                model_registry_path=model_registry_dir,
                optimization_mode=OptimizationMode.BALANCED  # Default balanced mode
            )
            
            # Read and parse the configuration files
            quantization_config_path = master_config.get('quantization_config')
            batch_config_path = master_config.get('batch_processor_config')
            preprocessor_config_path = master_config.get('preprocessor_config')
            inference_config_path = master_config.get('inference_engine_config')
            training_config_path = master_config.get('training_engine_config')
            
            # Load configurations using the load_config function
            quantization_config = load_config(quantization_config_path, QuantizationConfig)
            batch_config = load_config(batch_config_path, BatchProcessorConfig)
            preprocessor_config = load_config(preprocessor_config_path, PreprocessorConfig)
            inference_config = load_config(inference_config_path, InferenceEngineConfig)
            training_config = load_config(training_config_path, MLTrainingEngineConfig)
            
            # Store optimized configurations in session state
            st.session_state.optimized_configs = {
                "quantization": quantization_config,
                "batch": batch_config,
                "preprocessor": preprocessor_config,
                "inference": inference_config,
                "training": training_config
            }
            
            # Display configuration summary
            st.subheader("Device Optimization Summary")
            
            # Show key optimization details
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Optimization Mode", "Balanced")
                st.metric("CPU Cores Used", training_config.n_jobs)
            
            with cols[1]:
                st.metric("Batch Processing", 
                          f"Initial: {batch_config.initial_batch_size}, "
                          f"Max: {batch_config.max_batch_size}")
                st.metric("Quantization", 
                          f"{quantization_config.quantization_type} "
                          f"({quantization_config.quantization_mode})")
            
            with cols[2]:
                st.metric("Memory Optimization", 
                          "Enabled" if training_config.memory_optimization else "Disabled")
                st.metric("Cross-Validation", 
                          f"{training_config.cv_folds} Folds")
            
            # Optional: Show detailed config if needed
            with st.expander("Detailed Optimization Configuration"):
                st.json(master_config)
            
            st.session_state.device_optimized = True
            
            return True, master_config
        
        except Exception as e:
            st.error(f"Error during device optimization: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False, None

def data_upload_and_configuration():
    """Combined data upload and training configuration section"""
    st.title("Data Upload & Training Configuration")
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Data Upload & Exploration")
        
        # File upload
        uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                # Try to determine file type from extension
                if uploaded_file.name.endswith(".csv"):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith((".xls", ".xlsx")):
                    data = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return

                st.session_state.data = data

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())

                # Display basic statistics
                st.subheader("Data Statistics")

                # Show data dimensions
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Rows", f"{data.shape[0]:,}")
                col_b.metric("Columns", data.shape[1])
                col_c.metric("Missing Values", f"{data.isna().sum().sum():,}")
                col_d.metric("Duplicated Rows", f"{data.duplicated().sum():,}")

                # Data exploration tabs
                tab1, tab2, tab3 = st.tabs(["Column Info", "Correlation", "Distribution"])

                with tab1:
                    # Column information
                    column_info = pd.DataFrame(
                        {
                            "Column": data.columns,
                            "Type": data.dtypes,
                            "Non-Null Count": data.count(),
                            "Null Count": data.isna().sum(),
                            "Unique Values": [data[col].nunique() for col in data.columns],
                            "Sample Values": [str(data[col].unique()[:3]) for col in data.columns],
                        }
                    )
                    st.dataframe(column_info)

                with tab2:
                    # Correlation heatmap for numeric columns
                    numeric_data = data.select_dtypes(include=["number"])
                    if not numeric_data.empty:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        corr_matrix = numeric_data.corr()
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        sns.heatmap(
                            corr_matrix,
                            mask=mask,
                            annot=True,
                            fmt=".2f",
                            cmap="coolwarm",
                            square=True,
                            ax=ax,
                        )
                        st.pyplot(fig)
                    else:
                        st.info("No numeric columns found for correlation analysis.")

                with tab3:
                    # Distribution of columns
                    if not numeric_data.empty:
                        selected_column = st.selectbox(
                            "Select column for distribution analysis",
                            options=numeric_data.columns,
                        )

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=data, x=selected_column, kde=True, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.info("No numeric columns found for distribution analysis.")

                # Target column selection
                st.subheader("Target Selection")
                target_col = st.selectbox("Select target column for prediction", options=data.columns)

                if st.button("Set Target"):
                    st.session_state.target = target_col
                    st.session_state.features = data.columns.tolist()
                    st.session_state.features.remove(target_col)
                    st.success(f"Target set to '{target_col}'")

                    # Show target info
                    target_data = data[target_col]

                    st.subheader(f"Target Column: {target_col}")

                    # Detect if classification or regression based on number of unique values
                    unique_values = target_data.nunique()

                    if unique_values < 10:  # Assuming classification
                        st.info(f"Detected a classification problem with {unique_values} classes")

                        # Show distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        value_counts = target_data.value_counts()
                        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                        ax.set_title(f"Distribution of {target_col}")
                        ax.set_ylabel("Count")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:  # Assuming regression
                        st.info("Detected a regression problem")

                        # Show distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=target_data, kde=True, ax=ax)
                        ax.set_title(f"Distribution of {target_col}")
                        st.pyplot(fig)

                        # Show basic stats
                        st.write(
                            {
                                "Mean": target_data.mean(),
                                "Median": target_data.median(),
                                "Min": target_data.min(),
                                "Max": target_data.max(),
                                "Std Dev": target_data.std(),
                            }
                        )

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            # Display sample data option
            if st.button("Load Sample Data"):
                # Create sample data
                import sklearn.datasets

                X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
                data = pd.concat([X, pd.Series(y, name="target")], axis=1)
                st.session_state.data = data
                st.session_state.target = "target"
                st.session_state.features = X.columns.tolist()
                st.success("Sample classification data loaded (Breast Cancer dataset)")
                st.rerun()
    
    with col2:
        st.header("Training Configuration")

        if st.session_state.data is None or st.session_state.target is None:
            st.warning("Please upload data and select a target column first")
            return

        data = st.session_state.data
        target = st.session_state.target

        # Device optimization section
        st.subheader("Device Optimization")
        
        # Optimization mode selection
        optimization_mode_options = [
            ("Balanced (Recommended)", OptimizationMode.BALANCED),
            ("Conservative", OptimizationMode.CONSERVATIVE),
            ("Performance", OptimizationMode.PERFORMANCE),
            ("Full Utilization", OptimizationMode.FULL_UTILIZATION),
            ("Memory Saving", OptimizationMode.MEMORY_SAVING)
        ]

        optimization_mode_labels = [o[0] for o in optimization_mode_options]
        optimization_mode_values = [o[1] for o in optimization_mode_options]

        optimization_mode_index = st.radio(
            "Select Optimization Mode",
            options=range(len(optimization_mode_options)),
            format_func=lambda x: optimization_mode_labels[x],
            index=0  # Default to Balanced
        )
        selected_optimization_mode = optimization_mode_values[optimization_mode_index]

        # Optimization buttons
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            if st.button("Optimize for Your Device", key="device_optimize_btn"):
                try:
                    # Create DeviceOptimizer with selected mode
                    optimizer = DeviceOptimizer(
                        config_path="./configs", 
                        checkpoint_path="./checkpoints", 
                        model_registry_path="./models",
                        optimization_mode=selected_optimization_mode
                    )
                    
                    # Save configurations
                    master_config = optimizer.save_configs()
                    
                    # Store in session state
                    st.session_state.optimized_configs = {
                        "quantization": load_config(master_config['quantization_config'], QuantizationConfig),
                        "batch": load_config(master_config['batch_processor_config'], BatchProcessorConfig),
                        "preprocessor": load_config(master_config['preprocessor_config'], PreprocessorConfig),
                        "inference": load_config(master_config['inference_engine_config'], InferenceEngineConfig),
                        "training": load_config(master_config['training_engine_config'], MLTrainingEngineConfig)
                    }
                    
                    st.session_state.device_optimized = True
                    st.session_state.device_optimization_mode = selected_optimization_mode
                    
                    st.success(f"Device optimization completed in {selected_optimization_mode.value} mode!")
                    st.info("Optimized configurations are ready to be used for training.")
                except Exception as e:
                    st.error(f"Device optimization failed: {str(e)}")

        with col_opt2:
            if st.button("Generate Configs for All Modes", key="all_modes_config_btn"):
                try:
                    # Generate configurations for all optimization modes
                    all_mode_configs = create_configs_for_all_modes(
                        config_path="./configs", 
                        checkpoint_path="./checkpoints", 
                        model_registry_path="./models"
                    )
                    
                    # Store in session state for reference
                    st.session_state.all_mode_configs = all_mode_configs
                    
                    st.success("Configurations generated for all optimization modes!")
                    st.info("You can review and select specific configurations as needed.")
                except Exception as e:
                    st.error(f"Failed to generate all mode configurations: {str(e)}")

        # Show configuration details if optimized
        if st.session_state.device_optimized:
            with st.expander("Current Optimization Configuration", expanded=False):
                st.write(f"**Mode:** {st.session_state.device_optimization_mode.value}")
                
                # Display key configuration highlights
                config = st.session_state.optimized_configs
                
                cols = st.columns(3)
                with cols[0]:
                    st.metric("CPU Threads", config['training'].n_jobs)
                with cols[1]:
                    st.metric("CV Folds", config['training'].cv_folds)
                with cols[2]:
                    st.metric("Batch Size", config['batch'].initial_batch_size)
                
                # Full configuration details
                st.subheader("Detailed Configuration")
                config_tabs = st.tabs([
                    "Training Config", 
                    "Preprocessing Config", 
                    "Batch Processing", 
                    "Inference Config", 
                    "Quantization Config"
                ])
                
                with config_tabs[0]:
                    st.json(safe_dict_serializer(config['training']))
                with config_tabs[1]:
                    st.json(safe_dict_serializer(config['preprocessor']))
                with config_tabs[2]:
                    st.json(safe_dict_serializer(config['batch']))
                with config_tabs[3]:
                    st.json(safe_dict_serializer(config['inference']))
                with config_tabs[4]:
                    st.json(safe_dict_serializer(config['quantization']))
        with st.form("training_config_form"):
            st.subheader("Task Configuration")

            # Task type selection
            task_type_options = [
                ("Classification", TaskType.CLASSIFICATION),
                ("Regression", TaskType.REGRESSION),
            ]

            task_type_labels = [t[0] for t in task_type_options]
            task_type_values = [t[1] for t in task_type_options]

            task_type_index = st.radio(
                "Select task type",
                options=range(len(task_type_options)),
                format_func=lambda x: task_type_labels[x],
            )
            selected_task_type = task_type_values[task_type_index]

            # Feature selection
            st.subheader("Feature Selection")

            enable_feature_selection = st.checkbox("Enable feature selection", value=True)
            feature_selection_method = st.selectbox(
                "Feature selection method",
                options=["f_classif", "mutual_info"],
                disabled=not enable_feature_selection,
            )

            feature_selection_k = st.slider(
                "Number of features to select (k)",
                min_value=1,
                max_value=len(st.session_state.features),
                value=min(10, len(st.session_state.features)),
                disabled=not enable_feature_selection,
            )

            # Model selection
            st.subheader("Model Selection")

            available_models = MODEL_OPTIONS[selected_task_type]
            selected_models = st.multiselect(
                "Select models to train",
                options=available_models,
                default=available_models[:3],  # Default to first 3 models
            )

            # Optimization strategy
            st.subheader("Optimization Strategy")

            optimization_options = [
                ("Grid Search", OptimizationStrategy.GRID_SEARCH),
                ("Random Search", OptimizationStrategy.RANDOM_SEARCH),
                ("Bayesian Optimization", OptimizationStrategy.BAYESIAN_OPTIMIZATION),
                ("ASHT (Adaptive Surrogate-Assisted Hyperparameter Tuning)", OptimizationStrategy.ASHT),
                ("HyperOptX (Multi-Stage Optimization and Meta-Learning)", OptimizationStrategy.HYPERX),
            ]

            optimization_labels = [o[0] for o in optimization_options]
            optimization_values = [o[1] for o in optimization_options]

            optimization_index = st.radio(
                "Select optimization strategy",
                options=range(len(optimization_options)),
                format_func=lambda x: optimization_labels[x],
            )
            selected_optimization = optimization_values[optimization_index]

            optimization_iterations = st.slider(
                "Number of optimization iterations",
                min_value=10,
                max_value=100,
                value=30,
                disabled=selected_optimization == OptimizationStrategy.GRID_SEARCH,
            )

            # Cross-validation configuration
            st.subheader("Cross-Validation")

            cv_folds = st.slider(
                "Number of cross-validation folds",
                min_value=2,
                max_value=10,
                value=5,
            )

            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=40,
                value=20,
            ) / 100

            stratify = st.checkbox(
                "Use stratified sampling",
                value=selected_task_type == TaskType.CLASSIFICATION,
            )

            # Advanced options
            with st.expander("Advanced Options", expanded=False):
                advanced_tabs = st.tabs(["Preprocessing", "Batch Processing", "Quantization", "System"])
                
                with advanced_tabs[0]:
                    # Normalization options
                    normalization_options = [
                        ("None", NormalizationType.NONE),
                        ("Standard Scaling", NormalizationType.STANDARD),
                        ("Min-Max Scaling", NormalizationType.MINMAX),
                        ("Robust Scaling", NormalizationType.ROBUST),
                    ]

                    normalization_index = st.selectbox(
                        "Normalization Method",
                        options=range(len(normalization_options)),
                        format_func=lambda x: normalization_options[x][0],
                    )
                    selected_normalization = normalization_options[normalization_index][1]

                    handle_nan = st.checkbox("Handle missing values", value=True)
                    handle_inf = st.checkbox("Handle infinity values", value=True)

                    nan_strategy = st.selectbox(
                        "Missing value strategy",
                        options=["mean", "median", "most_frequent", "zero"],
                        disabled=not handle_nan,
                    )

                    detect_outliers = st.checkbox("Detect and handle outliers", value=False)
                    
                    outlier_method = st.selectbox(
                        "Outlier detection method",
                        options=["IQR", "Z-score", "Isolation Forest"],
                        disabled=not detect_outliers,
                    )
                    
                    outlier_threshold = st.slider(
                        "Outlier threshold",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1,
                        disabled=not detect_outliers,
                    )
                    
                    categorical_encoding = st.selectbox(
                        "Categorical encoding method",
                        options=["OneHot", "Label", "Target", "Binary", "Frequency"],
                    )
                    
                    enable_feature_engineering = st.checkbox("Enable automatic feature engineering", value=False)
                    
                    if enable_feature_engineering:
                        feature_engineering_methods = st.multiselect(
                            "Feature engineering methods",
                            options=["Polynomial", "Interaction", "Binning", "Log Transform", "Power Transform"],
                            default=["Polynomial", "Interaction"],
                        )
                        
                        polynomial_degree = st.slider(
                            "Polynomial degree",
                            min_value=2,
                            max_value=5,
                            value=2,
                            disabled="Polynomial" not in feature_engineering_methods,
                        )

                with advanced_tabs[1]:
                    # Batch processing options
                    batch_strategy_options = [
                        ("Fixed", BatchProcessingStrategy.FIXED),
                        ("Adaptive", BatchProcessingStrategy.ADAPTIVE),
                        ("Greedy", BatchProcessingStrategy.GREEDY),
                    ]

                    batch_strategy_index = st.selectbox(
                        "Batch Processing Strategy",
                        options=range(len(batch_strategy_options)),
                        format_func=lambda x: batch_strategy_options[x][0],
                    )
                    selected_batch_strategy = batch_strategy_options[batch_strategy_index][1]

                    initial_batch_size = st.slider(
                        "Initial batch size",
                        min_value=1,
                        max_value=200,
                        value=16,
                    )
                    
                    max_batch_size = st.slider(
                        "Maximum batch size",
                        min_value=initial_batch_size,
                        max_value=500,
                        value=min(128, initial_batch_size * 4),
                        disabled=selected_batch_strategy != BatchProcessingStrategy.ADAPTIVE,
                    )
                    
                    batch_growth_factor = st.slider(
                        "Batch growth factor",
                        min_value=1.1,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        disabled=selected_batch_strategy != BatchProcessingStrategy.ADAPTIVE,
                    )
                    
                    enable_batch_caching = st.checkbox("Enable batch caching", value=True)
                    
                    batch_timeout = st.slider(
                        "Batch timeout (seconds)",
                        min_value=1,
                        max_value=60,
                        value=10,
                    )

                with advanced_tabs[2]:
                    # Quantization options
                    enable_quantization = st.checkbox("Enable model quantization", value=False)

                    quantization_type_options = [
                        ("INT8", QuantizationType.INT8),
                        ("UINT8", QuantizationType.UINT8),
                        ("INT16", QuantizationType.INT16),
                    ]

                    quantization_type_index = st.selectbox(
                        "Quantization Type",
                        options=range(len(quantization_type_options)),
                        format_func=lambda x: quantization_type_options[x][0],
                        disabled=not enable_quantization,
                    )
                    selected_quantization_type = quantization_type_options[quantization_type_index][1]

                    quantization_mode_options = [
                        ("Symmetric", QuantizationMode.SYMMETRIC),
                        ("Asymmetric", QuantizationMode.ASYMMETRIC),
                        ("Dynamic Per Batch", QuantizationMode.DYNAMIC_PER_BATCH),
                        ("Dynamic Per Channel", QuantizationMode.DYNAMIC_PER_CHANNEL),
                    ]

                    quantization_mode_index = st.selectbox(
                        "Quantization Mode",
                        options=range(len(quantization_mode_options)),
                        format_func=lambda x: quantization_mode_options[x][0],
                        disabled=not enable_quantization,
                    )
                    selected_quantization_mode = quantization_mode_options[quantization_mode_index][1]
                    
                    calibration_dataset_size = st.slider(
                        "Calibration dataset size",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        disabled=not enable_quantization,
                    )
                    
                    enable_quantization_aware_training = st.checkbox(
                        "Enable quantization-aware training",
                        value=False,
                        disabled=not enable_quantization,
                    )

                with advanced_tabs[3]:
                    # System settings
                    with ThreadPoolExecutor() as executor:
                        max_workers = executor._max_workers
                        n_jobs = st.slider(
                            "Number of parallel jobs",
                            min_value=1,
                            max_value=max_workers,
                            value=max_workers//2,
                        )

                    memory_optimization = st.checkbox("Enable memory optimization", value=True)
                    
                    memory_limit = st.slider(
                        "Memory limit (GB)",
                        min_value=1,
                        max_value=32,
                        value=8,
                        disabled=not memory_optimization,
                    )
                    
                    enable_gpu = st.checkbox("Enable GPU acceleration (if available)", value=True)
                    
                    verbose = st.checkbox("Verbose output", value=True)

                    log_level = st.selectbox(
                        "Log level",
                        options=["DEBUG", "INFO", "WARNING", "ERROR"],
                        index=1,  # Default to INFO
                    )
                    
                    random_state = st.number_input(
                        "Random state (seed)",
                        min_value=0,
                        max_value=9999,
                        value=42,
                    )

                    # Model path
                    model_path = st.text_input("Model save path", value="./models")

            # Feature importance threshold
            feature_importance_threshold = st.slider(
                "Feature importance threshold",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                disabled=not enable_feature_selection,
            )

            # Submit button
            submit_button = st.form_submit_button("Save Configuration")

            if submit_button:
                # Create preprocessor config
                preprocessor_config = PreprocessorConfig(
                    normalization=selected_normalization,
                    handle_nan=handle_nan,
                    handle_inf=handle_inf,
                    nan_strategy=nan_strategy,
                    inf_strategy=nan_strategy,  # Use same strategy for inf
                    detect_outliers=detect_outliers,
                    outlier_method=outlier_method.lower() if detect_outliers else "iqr",
                    outlier_params={
                        "threshold": outlier_threshold if detect_outliers else 1.5,
                        "clip": True,
                        "n_estimators": 100,
                        "contamination": "auto"
                    },
                    clip_values=False,  # You can add UI controls for this if needed
                    enable_input_validation=True,
                    parallel_processing=True,
                    n_jobs=n_jobs,
                    cache_enabled=True,
                    debug_mode=(log_level == "DEBUG"),
                )

                # Create batch processing config
                batch_processing_config = BatchProcessorConfig(
                    min_batch_size=1,
                    max_batch_size=max_batch_size,
                    initial_batch_size=initial_batch_size,
                    max_queue_size=1000,
                    batch_timeout=batch_timeout,
                    processing_strategy=selected_batch_strategy,
                    enable_adaptive_batching=(
                        selected_batch_strategy == BatchProcessingStrategy.ADAPTIVE
                    ),
                    enable_memory_optimization=memory_optimization,
                    max_workers=n_jobs,
                    debug_mode=(log_level == "DEBUG"),
                )

                # Create quantization config
                quantization_config = QuantizationConfig(
                    quantization_type=selected_quantization_type.value,  # Use .value to get the string value
                    quantization_mode=selected_quantization_mode.value,  # Use .value to get the string value
                    enable_cache=True,
                    cache_size=1024,
                    buffer_size=0,  # Default to no buffer
                    use_percentile=False,
                    min_percentile=0.1,
                    max_percentile=99.9,
                    error_on_nan=False,
                    error_on_inf=False,
                    outlier_threshold=outlier_threshold if enable_quantization and detect_outliers else None,
                    num_bits=8,  # Default to 8 bits
                    optimize_memory=memory_optimization
                )

                # Create inference engine config
                inference_engine_config = InferenceEngineConfig(
                    debug_mode=(log_level == "DEBUG"),
                    num_threads=n_jobs,
                    enable_quantization=enable_quantization,
                    enable_model_quantization=enable_quantization,
                    quantization_dtype=selected_quantization_type,
                    quantization_config=quantization_config,
                    enable_batching=True,
                    initial_batch_size=initial_batch_size,
                    enable_memory_optimization=memory_optimization,
                    enable_monitoring=True,
                )

                # Create main config
                config = MLTrainingEngineConfig(
                    task_type=selected_task_type,
                    model_path=model_path,
                    preprocessing_config=preprocessor_config,
                    batch_processing_config=batch_processing_config,
                    inference_config=inference_engine_config,
                    quantization_config=quantization_config,
                    feature_selection=enable_feature_selection,
                    feature_selection_method=feature_selection_method,
                    feature_selection_k=feature_selection_k,
                    feature_importance_threshold=feature_importance_threshold,
                    optimization_strategy=selected_optimization,
                    optimization_iterations=optimization_iterations,
                    cv_folds=cv_folds,
                    test_size=test_size,
                    stratify=stratify,
                    experiment_tracking=True,
                    n_jobs=n_jobs,
                    memory_optimization=memory_optimization,
                    use_intel_optimization=enable_gpu,  # Map enable_gpu to use_intel_optimization
                    verbose=1 if verbose else 0,
                    log_level=log_level,
                    random_state=random_state,
                )

                # Save config in session state
                st.session_state.config = config
                st.session_state.selected_models = selected_models

                st.success("Configuration saved successfully!")

                # Display the config summary
                with st.expander("Configuration Summary", expanded=True):
                    st.json(config.to_dict())

def training_and_evaluation():
    """Combined model training and evaluation section"""
    st.title("Model Training & Evaluation")

    if st.session_state.data is None or st.session_state.target is None:
        st.warning("Please upload data and select a target column first")
        return

    if not hasattr(st.session_state, "config") or st.session_state.config is None:
        st.warning("Please configure training parameters first")
        return

    data = st.session_state.data
    target = st.session_state.target
    config = st.session_state.config
    selected_models = (
        st.session_state.selected_models
        if hasattr(st.session_state, "selected_models")
        else []
    )

    if not selected_models:
        st.error("No models selected for training")
        return

    # Create directory for models if it doesn't exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    # Split data into features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Create tabs for training and evaluation
    train_tab, eval_tab = st.tabs(["Training", "Evaluation"])
    
    with train_tab:
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"Features: {X.shape[1]} columns, {X.shape[0]} samples")
        st.write(f"Target: '{target}'")

        # Display configuration summary
        st.subheader("Training Configuration Summary")
        st.write(f"Task Type: {config.task_type.value}")
        st.write(f"Models to train: {', '.join(selected_models)}")
        st.write(f"Optimization Strategy: {config.optimization_strategy.value}")
        st.write(f"Cross-validation: {config.cv_folds} folds")

        # Advanced training options
        with st.expander("Advanced Training Options", expanded=False):
            early_stopping = st.checkbox("Enable early stopping", value=True)
            
            patience = st.slider(
                "Early stopping patience",
                min_value=1,
                max_value=20,
                value=5,
                disabled=not early_stopping,
            )
            
            train_subset = st.slider(
                "Training data subset (%)",
                min_value=10,
                max_value=100,
                value=100,
            )
            
            custom_scoring = st.text_input(
                "Custom scoring metric (scikit-learn compatible)",
                value="",
                help="Leave empty to use default metrics based on task type"
            )
            
            enable_ensemble = st.checkbox("Enable ensemble of best models", value=False)
            
            ensemble_method = st.selectbox(
                "Ensemble method",
                options=["Voting", "Stacking", "Bagging"],
                disabled=not enable_ensemble,
            )
            
            ensemble_size = st.slider(
                "Number of models in ensemble",
                min_value=2,
                max_value=len(selected_models),
                value=min(3, len(selected_models)),
                disabled=not enable_ensemble,
            )

        # Initialize and run training
        if st.button("Start Training"):
            # Initialize progress bar and status
            progress = st.progress(0)
            status_text = st.empty()

            try:
                # Import model libraries
                models_dict = import_model_libraries()

                # Initialize the training engine
                status_text.text("Initializing ML Training Engine...")

                if st.session_state.engine is None:
                    st.session_state.engine = MLTrainingEngine(config)

                engine = st.session_state.engine

                # Create param grids for selected models
                param_grids = {}
                for model_name in selected_models:
                    if model_name in DEFAULT_PARAM_GRIDS:
                        param_grids[model_name] = DEFAULT_PARAM_GRIDS[model_name]
                    else:
                        # Use empty param grid if no defaults exist
                        param_grids[model_name] = {}

                # Split data for training and testing
                if config.stratify and config.task_type == TaskType.CLASSIFICATION:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y, 
                        test_size=config.test_size,
                        random_state=config.random_state,
                        stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=config.test_size, random_state=config.random_state
                    )
                    
                # Apply training subset if needed
                if train_subset < 100:
                    subset_size = int(len(X_train) * train_subset / 100)
                    if config.stratify and config.task_type == TaskType.CLASSIFICATION:
                        X_train_subset, _, y_train_subset, _ = train_test_split(
                            X_train, y_train,
                            train_size=subset_size,
                            random_state=config.random_state,
                            stratify=y_train
                        )
                    else:
                        X_train_subset, _, y_train_subset, _ = train_test_split(
                            X_train, y_train,
                            train_size=subset_size,
                            random_state=config.random_state
                        )
                    X_train = X_train_subset
                    y_train = y_train_subset
                    status_text.text(f"Using {subset_size} samples ({train_subset}%) for training")

                # Train each model
                results = {}
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training model {i+1}/{len(selected_models)}: {model_name}")
                    progress.progress((i) / len(selected_models))

                    # Instantiate the model
                    if model_name in models_dict:
                        model_class = models_dict[model_name]
                        model_instance = model_class(random_state=config.random_state)

                        # Train the model
                        best_model, metrics = engine.train_model(
                            model=model_instance,
                            model_name=model_name,
                            param_grid=param_grids[model_name],
                            X=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                            y=y_train.values if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) else y_train,
                            X_test=X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
                            y_test=y_test.values if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series) else y_test,
                        )

                        # Save results
                        results[model_name] = {
                            "metrics": metrics,
                            "model_name": model_name,
                        }

                        if engine.save_model(model_name):
                            status_text.text(f"Model {model_name} trained and saved successfully")

                        # Add to session state
                        st.session_state.models[model_name] = best_model
                        st.session_state.model_metrics[model_name] = metrics

                        # Check if this is the best model
                        if engine.best_model and engine.best_model["name"] == model_name:
                            st.session_state.best_model = {
                                "name": model_name,
                                "model": best_model,
                                "metrics": metrics,
                            }
                    else:
                        status_text.text(f"Model {model_name} not found in available models")

                # Create ensemble if enabled
                if enable_ensemble and len(results) >= 2:
                    status_text.text(f"Creating {ensemble_method} ensemble with top {ensemble_size} models...")
                    
                    # This is a placeholder for ensemble creation
                    # In a real implementation, you would create the ensemble here
                    # using the trained models based on the selected ensemble method
                    
                    st.session_state.ensemble_created = True
                    st.session_state.ensemble_method = ensemble_method
                    st.session_state.ensemble_size = ensemble_size

                # Update progress to completion
                progress.progress(1.0)
                status_text.text("Training completed successfully!")

                # Save experiment results
                experiment_result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "models_trained": selected_models,
                    "results": results,
                    "config": config.to_dict(),
                }
                st.session_state.experiment_results.append(experiment_result)

                # Mark training as completed
                st.session_state.training_completed = True

                # Generate report if enabled
                report_path = engine.generate_report()
                if report_path:
                    status_text.text(f"Training completed and report generated at {report_path}")

                # Display success message
                st.success("All models trained successfully! Go to the Evaluation tab to see results.")

            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback

                st.code(traceback.format_exc())
    
    with eval_tab:
        if not st.session_state.training_completed:
            st.warning("Please train models first")
            return

        if not st.session_state.models:
            st.warning("No trained models found")
            return

        # Get model names and metrics
        model_names = list(st.session_state.models.keys())
        metrics = st.session_state.model_metrics

        # Display best model
        st.subheader("Best Performing Model")
        if st.session_state.best_model:
            best_model = st.session_state.best_model
            st.write(f"**{best_model['name']}**")

            # Display metrics
            best_metrics = best_model["metrics"]
            col1, col2, col3, col4 = st.columns(4)

            # Display primary metric
            if "accuracy" in best_metrics:
                col1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
            elif "r2" in best_metrics:
                col1.metric("R²", f"{best_metrics['r2']:.4f}")

            # Display secondary metrics
            if "precision" in best_metrics and "recall" in best_metrics:
                col2.metric("Precision", f"{best_metrics['precision']:.4f}")
                col3.metric("Recall", f"{best_metrics['recall']:.4f}")
                col4.metric("F1 Score", f"{best_metrics['f1']:.4f}")
            elif "mae" in best_metrics and "mse" in best_metrics:
                col2.metric("MAE", f"{best_metrics['mae']:.4f}")
                col3.metric("MSE", f"{best_metrics['mse']:.4f}")
                col4.metric("RMSE", f"{best_metrics['rmse']:.4f}")
        else:
            st.info("No best model identified")

        # Ensemble information if created
        if hasattr(st.session_state, 'ensemble_created') and st.session_state.ensemble_created:
            st.subheader("Ensemble Model")
            st.write(f"Created a {st.session_state.ensemble_method} ensemble with {st.session_state.ensemble_size} models")
            # Display ensemble metrics if available

        # Comparison tabs
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(["Metrics Comparison", "Feature Importance", "Learning Curves", "Prediction Analysis"])

        with eval_tab1:
            # Create comparison dataframe
            comparison_data = []
            for model_name in model_names:
                model_metrics = metrics[model_name]

                # Extract relevant metrics
                model_row = {"Model": model_name}
                for metric_name, metric_value in model_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        model_row[metric_name] = round(metric_value, 4)

                comparison_data.append(model_row)

            comparison_df = pd.DataFrame(comparison_data)

            # Display metrics table
            st.subheader("Metrics Comparison")
            st.dataframe(comparison_df)

            # Plot comparison
            st.subheader("Visual Comparison")
            if comparison_df.shape[0] > 0:
                # Determine if classification or regression
                if "accuracy" in comparison_df.columns:
                    # Classification metrics
                    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
                    metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
                else:
                    # Regression metrics
                    metrics_to_plot = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
                    metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]

                if metrics_to_plot:
                    # Prepare data for plotting
                    plot_data = comparison_df.melt(
                        id_vars=["Model"],
                        value_vars=metrics_to_plot,
                        var_name="Metric",
                        value_name="Value",
                    )

                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=plot_data, x="Model", y="Value", hue="Metric", ax=ax)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

        with eval_tab2:
            st.subheader("Feature Importance")

            selected_model = st.selectbox(
                "Select model for feature importance",
                options=model_names,
            )

            if selected_model in st.session_state.models:
                model = st.session_state.models[selected_model]

                # Check if model has feature_importances_ attribute
                if hasattr(model, "feature_importances_"):
                    # Get feature importances
                    importances = model.feature_importances_
                    feature_names = st.session_state.features

                    # Create dataframe
                    importance_df = pd.DataFrame(
                        {"Feature": feature_names, "Importance": importances}
                    ).sort_values("Importance", ascending=False)

                    # Display table
                    st.dataframe(importance_df)

                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                elif hasattr(model, "coef_"):
                    # For linear models
                    coef = model.coef_
                    if coef.ndim > 1:
                        # For multi-class models, take the mean absolute coefficient
                        coef = np.mean(np.abs(coef), axis=0)
                    
                    feature_names = st.session_state.features
                    
                    # Create dataframe
                    importance_df = pd.DataFrame(
                        {"Feature": feature_names, "Coefficient": coef}
                    ).sort_values("Coefficient", ascending=False)
                    
                    # Display table
                    st.dataframe(importance_df)
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=importance_df.head(15), x="Coefficient", y="Feature", ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("This model doesn't provide feature importance information")

        with eval_tab3:
            st.subheader("Learning Curves")
            
            # This would typically show learning curves from cross-validation
            # For now, we'll just show a placeholder
            st.info("Learning curves visualization will be available in a future update")
            
            # If you have learning curve data from your training process:
            # selected_model_for_curves = st.selectbox(
            #     "Select model for learning curves",
            #     options=model_names,
            #     key="learning_curve_model"
            # )
            # 
            # if selected_model_for_curves in st.session_state.models:
            #     # Plot learning curves if available
            #     pass

        with eval_tab4:
            st.subheader("Prediction Analysis")
            
            # This section would analyze model predictions on test data
            if st.session_state.data is not None and st.session_state.target is not None:
                selected_model_for_analysis = st.selectbox(
                    "Select model for prediction analysis",
                    options=model_names,
                    key="prediction_analysis_model"
                )
                
                if selected_model_for_analysis in st.session_state.models:
                    model = st.session_state.models[selected_model_for_analysis]
                    
                    # Get the engine
                    engine = st.session_state.engine
                    
                    # Get test data
                    data = st.session_state.data
                    target = st.session_state.target
                    X = data.drop(columns=[target])
                    y = data[target]
                    
                    # Split data for analysis
                    if engine.config.stratify and engine.config.task_type == TaskType.CLASSIFICATION:
                        _, X_test, _, y_test = train_test_split(
                            X, y, 
                            test_size=engine.config.test_size,
                            random_state=engine.config.random_state,
                            stratify=y
                        )
                    else:
                        _, X_test, _, y_test = train_test_split(
                            X, y, 
                            test_size=engine.config.test_size,
                            random_state=engine.config.random_state
                        )
                    
                    # Make predictions
                    try:
                        # Convert DataFrame to NumPy array before prediction
                        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                        y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
                        
                        # Use the model directly for prediction to avoid preprocessing issues
                        y_pred = model.predict(X_test_array)
                        
                        # Display prediction analysis based on task type
                        if engine.config.task_type == TaskType.CLASSIFICATION:
                            # For classification
                            from sklearn.metrics import confusion_matrix, classification_report
                            
                            # Confusion matrix
                            st.write("Confusion Matrix")
                            cm = confusion_matrix(y_test_array, y_pred)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            
                            # Classification report
                            st.write("Classification Report")
                            report = classification_report(y_test_array, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # ROC curve for binary classification
                            if len(np.unique(y)) == 2:
                                from sklearn.metrics import roc_curve, auc
                                try:
                                    y_prob = model.predict_proba(X_test_array)[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test_array, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                    ax.plot([0, 1], [0, 1], 'k--')
                                    ax.set_xlim([0.0, 1.0])
                                    ax.set_ylim([0.0, 1.05])
                                    ax.set_xlabel('False Positive Rate')
                                    ax.set_ylabel('True Positive Rate')
                                    ax.set_title('Receiver Operating Characteristic')
                                    ax.legend(loc="lower right")
                                    st.pyplot(fig)
                                except (AttributeError, IndexError):
                                    st.info("ROC curve not available for this model")
                        
                        else:
                            # For regression
                            # Residual plot
                            residuals = y_test_array - y_pred
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_pred, residuals)
                            ax.axhline(y=0, color='r', linestyle='-')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Residuals')
                            ax.set_title('Residual Plot')
                            st.pyplot(fig)
                            
                            # Actual vs Predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_test_array, y_pred)
                            ax.plot([y_test_array.min(), y_test_array.max()], [y_test_array.min(), y_test_array.max()], 'k--')
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted')
                            st.pyplot(fig)
                            
                            # Distribution of residuals
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(residuals, kde=True, ax=ax)
                            ax.set_title('Distribution of Residuals')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error analyzing predictions: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def inference_page():
    """Inference page"""
    st.title("Model Inference")

    if not st.session_state.training_completed:
        st.warning("Please train models first")
        return

    if not st.session_state.models:
        st.warning("No trained models found")
        return

    # Get model names
    model_names = list(st.session_state.models.keys())

    # Select model for prediction
    selected_model = st.selectbox(
        "Select model for prediction",
        options=model_names,
        index=model_names.index(st.session_state.best_model["name"])
        if st.session_state.best_model
        else 0,
    )

    # Input method
    input_method = st.radio("Input method", ["Upload new data", "Manual input", "Batch processing"])

    if input_method == "Upload new data":
        # File upload
        uploaded_file = st.file_uploader(
            "Upload data for prediction (CSV, Excel)", type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                # Try to determine file type from extension
                if uploaded_file.name.endswith(".csv"):
                    pred_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith((".xls", ".xlsx")):
                    pred_data = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(pred_data.head())

                # Check for target column
                if st.session_state.target in pred_data.columns:
                    has_target = st.checkbox("Data contains target column")
                    if not has_target:
                        pred_data = pred_data.drop(columns=[st.session_state.target])

                # Make predictions
                with st.spinner("Making predictions..."):
                    try:
                        # Get model directly from session state
                        model = st.session_state.models[selected_model]
                        
                        # Convert to numpy array
                        pred_array = pred_data.values
                        
                        # Try direct prediction first
                        try:
                            predictions = model.predict(pred_array)
                        except Exception as model_error:
                            st.warning(f"Direct prediction failed: {str(model_error)}. Trying alternative method...")
                            
                            # If direct prediction fails, try to load the model fresh
                            try:
                                import joblib
                                engine = st.session_state.engine
                                model_path = os.path.join(engine.config.model_path, f"{selected_model}.joblib")
                                
                                if os.path.exists(model_path):
                                    fresh_model = joblib.load(model_path)
                                    predictions = fresh_model.predict(pred_array)
                                else:
                                    # If model file doesn't exist, try engine's predict method
                                    predictions = engine.predict(pred_data, selected_model)
                            except Exception as load_error:
                                # If all else fails, raise the error
                                raise load_error

                        # Display predictions
                        st.subheader("Predictions")

                        # Store predictions in session state
                        st.session_state.predictions = predictions

                        # Create dataframe with predictions
                        pred_df = pd.DataFrame({"Prediction": predictions})

                        # Display predictions
                        st.dataframe(pred_df)

                        # Add download button
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


    elif input_method == "Manual input":
        st.subheader("Enter Feature Values")

        # Create a form for feature inputs
        with st.form("prediction_form"):
            # Get feature list
            features = st.session_state.features

            # Create input fields for each feature
            input_values = {}
            for feature in features:
                # Try to determine appropriate input type
                feature_type = st.session_state.data[feature].dtype

                if pd.api.types.is_numeric_dtype(feature_type):
                    # For numeric features
                    min_val = float(st.session_state.data[feature].min())
                    max_val = float(st.session_state.data[feature].max())
                    mean_val = float(st.session_state.data[feature].mean())

                    # Use slider for reasonable ranges, otherwise input
                    if max_val - min_val < 100:
                        input_values[feature] = st.slider(
                            feature,
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100,
                        )
                    else:
                        input_values[feature] = st.number_input(feature, value=mean_val)
                else:
                    # For categorical features
                    unique_values = st.session_state.data[feature].unique().tolist()
                    input_values[feature] = st.selectbox(
                        feature,
                        options=unique_values,
                        index=0,
                    )

            # Submit button
            predict_button = st.form_submit_button("Make Prediction")

        if predict_button:
            try:
                # Create dataframe from input values
                input_df = pd.DataFrame([input_values])
                
                # Convert to numpy array
                input_array = input_df.values
                
                # Get model directly from session state
                model = st.session_state.models[selected_model]
                
                # Make prediction using direct model access
                with st.spinner("Making prediction..."):
                    try:
                        # First try direct prediction with the model
                        prediction = model.predict(input_array)
                    except Exception as model_error:
                        st.warning(f"Direct prediction failed: {str(model_error)}. Trying alternative method...")
                        
                        # If direct prediction fails, try to load the model fresh
                        try:
                            import joblib
                            engine = st.session_state.engine
                            model_path = os.path.join(engine.config.model_path, f"{selected_model}.joblib")
                            
                            if os.path.exists(model_path):
                                fresh_model = joblib.load(model_path)
                                prediction = fresh_model.predict(input_array)
                            else:
                                # If model file doesn't exist, try engine's predict method
                                # but with raw numpy array
                                prediction = engine.predict(input_array, selected_model)
                        except Exception as load_error:
                            # If all else fails, try the engine's predict method with DataFrame
                            prediction = engine.predict(input_df, selected_model)

                # Display prediction
                st.subheader("Prediction Result")

                # Format the prediction based on task type
                if st.session_state.config.task_type == TaskType.CLASSIFICATION:
                    # For classification tasks
                    st.write(f"Predicted class: **{prediction[0]}**")
                    
                    # Try to get probability if available
                    try:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(input_array)
                            
                            # Display probabilities
                            st.write("Class Probabilities:")
                            
                            # Get class names if available
                            if hasattr(model, 'classes_'):
                                classes = model.classes_
                                proba_df = pd.DataFrame({
                                    'Class': classes,
                                    'Probability': proba[0]
                                })
                                
                                # Create bar chart
                                fig, ax = plt.subplots(figsize=(10, 4))
                                sns.barplot(data=proba_df, x='Class', y='Probability', ax=ax)
                                ax.set_ylim(0, 1)
                                st.pyplot(fig)
                            else:
                                # Just show probabilities
                                st.write(proba[0])
                    except Exception as e:
                        st.info("Probability information not available")
                else:
                    # For regression tasks
                    st.write(f"Predicted value: **{prediction[0]:.4f}**")
                    
                    # Try to get prediction intervals if available
                    # This is a placeholder - most models don't provide this directly
                    st.info("Prediction intervals not available for this model")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    
    else:  # Batch processing
        st.subheader("Batch Processing")
        
        # Batch size configuration
        batch_size = st.slider(
            "Batch size",
            min_value=1,
            max_value=1000,
            value=100
        )
        
        # File upload for batch processing
        batch_file = st.file_uploader(
            "Upload data for batch processing (CSV, Excel)",
            type=["csv", "xlsx"],
            key="batch_file_uploader"
        )
        
        if batch_file is not None:
            try:
                # Load data
                if batch_file.name.endswith(".csv"):
                    batch_data = pd.read_csv(batch_file)
                elif batch_file.name.endswith((".xls", ".xlsx")):
                    batch_data = pd.read_excel(batch_file)
                else:
                    st.error("Unsupported file format")
                    return
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(batch_data.head())
                
                # Check for target column
                # Check for target column
                if st.session_state.target in batch_data.columns:
                    has_target = st.checkbox("Data contains target column", key="batch_has_target")
                    if not has_target:
                        batch_data = batch_data.drop(columns=[st.session_state.target])
                
                # Process in batches
                if st.button("Process Batch"):
                    with st.spinner("Processing batch data..."):
                        try:
                            engine = st.session_state.engine
                            
                            # Initialize progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Split data into batches
                            num_samples = len(batch_data)
                            num_batches = (num_samples + batch_size - 1) // batch_size
                            
                            all_predictions = []
                            
                            for i in range(num_batches):
                                # Get batch
                                start_idx = i * batch_size
                                end_idx = min((i + 1) * batch_size, num_samples)
                                batch = batch_data.iloc[start_idx:end_idx]
                                
                                # Update status
                                status_text.text(f"Processing batch {i+1}/{num_batches}...")
                                
                                # Make predictions
                                batch_predictions = engine.predict(batch, selected_model)
                                all_predictions.extend(batch_predictions)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / num_batches)
                            
                            # Create dataframe with predictions
                            pred_df = pd.DataFrame({"Prediction": all_predictions})
                            
                            # Display predictions
                            st.subheader("Batch Predictions")
                            st.dataframe(pred_df)
                            
                            # Add download button
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="Download Batch Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                            )
                            
                            # Store predictions in session state
                            st.session_state.predictions = all_predictions
                            
                            status_text.text("Batch processing completed!")
                            
                        except Exception as e:
                            st.error(f"Error processing batch: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"Error loading batch file: {str(e)}")

def experiment_tracking():
    """Experiment tracking and comparison page"""
    st.title("Experiment Tracking")
    
    if not hasattr(st.session_state, "experiment_results") or not st.session_state.experiment_results:
        st.warning("No experiments have been run yet")
        return
    
    # Display experiment history
    st.subheader("Experiment History")
    
    # Create a summary table of experiments
    experiment_summaries = []
    for i, exp in enumerate(st.session_state.experiment_results):
        # Extract key information
        timestamp = exp.get("timestamp", "Unknown")
        models_trained = exp.get("models_trained", [])
        config = exp.get("config", {})
        
        # Get best model and its performance
        best_model = None
        best_metric = 0
        primary_metric = "accuracy" if config.get("task_type") == "CLASSIFICATION" else "r2"
        
        for model_name, result in exp.get("results", {}).items():
            metrics = result.get("metrics", {})
            if primary_metric in metrics and (best_model is None or metrics[primary_metric] > best_metric):
                best_model = model_name
                best_metric = metrics[primary_metric]
        
        # Create summary
        summary = {
            "Experiment": i + 1,
            "Timestamp": timestamp,
            "Models Trained": len(models_trained),
            "Best Model": best_model,
            f"Best {primary_metric.capitalize()}": f"{best_metric:.4f}" if best_metric else "N/A",
            "Task Type": config.get("task_type", "Unknown"),
        }
        
        experiment_summaries.append(summary)
    
    # Display summary table
    summary_df = pd.DataFrame(experiment_summaries)
    st.dataframe(summary_df)
    
    # Select experiments to compare
    st.subheader("Compare Experiments")
    
    selected_experiments = st.multiselect(
        "Select experiments to compare",
        options=range(1, len(st.session_state.experiment_results) + 1),
        format_func=lambda x: f"Experiment {x}"
    )
    
    if selected_experiments:
        # Get selected experiment data
        selected_data = [st.session_state.experiment_results[i-1] for i in selected_experiments]
        
        # Create comparison dataframe
        comparison_data = []
        
        for exp_idx, exp in zip(selected_experiments, selected_data):
            # Get results for each model
            for model_name, result in exp.get("results", {}).items():
                metrics = result.get("metrics", {})
                
                row = {
                    "Experiment": f"Exp {exp_idx}",
                    "Model": model_name,
                }
                
                # Add metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        row[metric_name] = round(metric_value, 4)
                
                comparison_data.append(row)
        
        # Create dataframe
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(comparison_df)
            
            # Create visualization
            st.subheader("Visual Comparison")
            
            # Determine metrics to plot based on task type
            if "accuracy" in comparison_df.columns:
                # Classification metrics
                metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
                metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
            else:
                # Regression metrics
                metrics_to_plot = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
                metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
            
            if metrics_to_plot:
                # Let user select metric to visualize
                selected_metric = st.selectbox(
                    "Select metric to visualize",
                    options=metrics_to_plot
                )
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=comparison_df, x="Model", y=selected_metric, hue="Experiment", ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("No comparison data available")
    
    # Export experiments
    st.subheader("Export Experiments")
    
    if st.button("Export All Experiments"):
        # Convert to JSON
        import json
        
        # Create a simplified version for export
        export_data = []
        for exp in st.session_state.experiment_results:
            # Create a simplified version that's JSON serializable
            simplified_exp = {
                "timestamp": exp.get("timestamp", ""),
                "models_trained": exp.get("models_trained", []),
                "results": {},
                "config": exp.get("config", {})
            }
            
            # Simplify results
            for model_name, result in exp.get("results", {}).items():
                simplified_exp["results"][model_name] = {
                    "metrics": result.get("metrics", {}),
                    "model_name": result.get("model_name", "")
                }
            
            export_data.append(simplified_exp)
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2)
        
        # Create download button
        st.download_button(
            label="Download Experiments (JSON)",
            data=json_str,
            file_name="ml_experiments.json",
            mime="application/json"
        )

def main():
    """Main function to run the Streamlit app"""
    # Create sidebar
    st.sidebar.title("Navigation")
    
    # Create navigation
    pages = {
        "Data & Configuration": data_upload_and_configuration,
        "Training & Evaluation": training_and_evaluation,
        "Model Inference": inference_page,
        "Experiment Tracking": experiment_tracking,
    }
    
    # Select page
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Display page
    pages[selection]()
    
    # Add info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Session Info")
        
        # Display data info
        if st.session_state.data is not None:
            st.write(f"Data: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        else:
            st.write("Data: None")
        
        # Display target info
        if st.session_state.target is not None:
            st.write(f"Target: {st.session_state.target}")
        else:
            st.write("Target: None")
        
        # Display model info
        if hasattr(st.session_state, "models") and st.session_state.models:
            st.write(f"Trained models: {len(st.session_state.models)}")
            
            if st.session_state.best_model:
                st.write(f"Best model: {st.session_state.best_model['name']}")
        else:
            st.write("Trained models: None")
        
        # Display experiment info
        if hasattr(st.session_state, "experiment_results") and st.session_state.experiment_results:
            st.write(f"Experiments: {len(st.session_state.experiment_results)}")
        else:
            st.write("Experiments: None")
        
        # Add clear session button
        if st.button("Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Add about section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Advanced ML Training Engine")
        st.markdown("Beta Version")
        st.markdown("© Kolosal AI 2025")

if __name__ == "__main__":
    main()
