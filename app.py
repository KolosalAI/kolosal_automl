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
    BatchPriority,
    InferenceEngineConfig,
    QuantizationConfig,
    QuantizationType,
    QuantizationMode,
    ModelSelectionCriteria,
    ExplainabilityConfig,
    MonitoringConfig,
    OptimizationMode,
)
from modules.engine.train_engine import MLTrainingEngine
from modules.engine.inference_engine import InferenceEngine
from modules.engine.data_preprocessor import DataPreprocessor
from modules.engine.batch_processor import BatchProcessor, BatchStats
from modules.device_optimizer import DeviceOptimizer, create_optimized_configs, create_configs_for_all_modes

# Set page configuration
st.set_page_config(
    page_title="Advanced ML Training Engine",
    page_icon="assets\kolosal-logo.png",
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
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["lbfgs", "liblinear", "saga"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "LGBMClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "LGBMRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "CatBoostClassifier": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "depth": [4, 6, 8],
    },
    "CatBoostRegressor": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "depth": [4, 6, 8],
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto", 0.1, 1.0],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto", 0.1, 1.0],
    },
    "LinearRegression": {
        "fit_intercept": [True, False],
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

def load_config(config_path, config_class):
    """Load a configuration from a file path"""
    if not config_path or not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dict to config class instance
        if hasattr(config_class, 'from_dict'):
            return config_class.from_dict(config_dict)
        else:
            return config_class(**config_dict)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return None

def safe_dict_serializer(obj):
    """Convert config object to a JSON-serializable dictionary"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        # Convert object attributes to dictionary
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    result[key] = value
                else:
                    # Try to convert other objects to string representation
                    try:
                        result[key] = str(value)
                    except:
                        result[key] = f"<non-serializable: {type(value).__name__}>"
        return result
    else:
        # Just try to convert to dict directly
        try:
            return dict(obj)
        except:
            return {"error": f"Cannot convert {type(obj).__name__} to dictionary"}

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
            # Use the DeviceOptimizer class for device optimization
            optimizer = DeviceOptimizer(
                config_path=configs_dir,
                checkpoint_path=checkpoint_dir,
                model_registry_path=model_registry_dir,
                optimization_mode=OptimizationMode.BALANCED  # Default balanced mode
            )
            
            # Generate and save configurations
            master_config = optimizer.save_configs()
            
            # Load the saved configurations
            configs = optimizer.load_configs(master_config['config_id'])
            
            # Store optimized configurations in session state
            st.session_state.optimized_configs = {
                "quantization": configs.get("quantization_config"),
                "batch": configs.get("batch_processor_config"),
                "preprocessor": configs.get("preprocessor_config"),
                "inference": configs.get("inference_engine_config"),
                "training": configs.get("training_engine_config"),
                "system_info": configs.get("system_info", {})
            }
            
            # Display configuration summary
            st.subheader("Device Optimization Summary")
            
            # Show key optimization details
            system_info = configs.get("system_info", {})
            cpu_info = system_info.get("cpu", {})
            
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Optimization Mode", "Balanced")
                st.metric("CPU Cores", f"{cpu_info.get('count_physical', 'N/A')} physical, {cpu_info.get('count_logical', 'N/A')} logical")
            
            with cols[1]:
                batch_config = configs.get("batch_processor_config", {})
                st.metric("Batch Processing", 
                          f"Initial: {batch_config.get('initial_batch_size', 'N/A')}, "
                          f"Max: {batch_config.get('max_batch_size', 'N/A')}")
                
                quant_config = configs.get("quantization_config", {})
                st.metric("Quantization", 
                          f"{quant_config.get('quantization_type', 'N/A')} "
                          f"({quant_config.get('quantization_mode', 'N/A')})")
            
            with cols[2]:
                memory_info = system_info.get("memory", {})
                st.metric("Total Memory", f"{memory_info.get('total_gb', 'N/A'):.1f} GB")
                st.metric("Usable Memory", f"{memory_info.get('usable_gb', 'N/A'):.1f} GB")
            
            # Optional: Show detailed config if needed
            with st.expander("System Information"):
                st.json(system_info)
            
            st.session_state.device_optimized = True
            st.session_state.device_optimization_mode = OptimizationMode.BALANCED
            
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
                    
                    # Load configurations
                    configs = optimizer.load_configs(master_config["config_id"])
                    
                    # Store in session state
                    st.session_state.optimized_configs = {
                        "quantization": configs.get("quantization_config"),
                        "batch": configs.get("batch_processor_config"),
                        "preprocessor": configs.get("preprocessor_config"),
                        "inference": configs.get("inference_engine_config"),
                        "training": configs.get("training_engine_config"),
                        "system_info": configs.get("system_info", {})
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
                st.write(f"**Mode:** {st.session_state.device_optimization_mode.value if hasattr(st.session_state.device_optimization_mode, 'value') else str(st.session_state.device_optimization_mode)}")
                
                # Display key configuration highlights
                config = st.session_state.optimized_configs
                
                # Display system info if available
                system_info = config.get("system_info", {})
                if system_info:
                    cpu_info = system_info.get("cpu", {})
                    memory_info = system_info.get("memory", {})
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("CPU Cores", f"{cpu_info.get('count_physical', 'N/A')} physical, {cpu_info.get('count_logical', 'N/A')} logical")
                    with cols[1]:
                        st.metric("Total Memory", f"{memory_info.get('total_gb', 'N/A'):.1f} GB")
                    with cols[2]:
                        st.metric("CPU", f"{cpu_info.get('vendor', {}).get('intel', False) and 'Intel' or (cpu_info.get('vendor', {}).get('amd', False) and 'AMD' or 'Other')}")
                
                # Full configuration details in tabs
                st.subheader("Detailed Configuration")
                config_tabs = st.tabs([
                    "Training Config", 
                    "Preprocessing Config", 
                    "Batch Processing", 
                    "Inference Config", 
                    "Quantization Config"
                ])
                
                with config_tabs[0]:
                    st.json(safe_dict_serializer(config.get('training', {})))
                with config_tabs[1]:
                    st.json(safe_dict_serializer(config.get('preprocessor', {})))
                with config_tabs[2]:
                    st.json(safe_dict_serializer(config.get('batch', {})))
                with config_tabs[3]:
                    st.json(safe_dict_serializer(config.get('inference', {})))
                with config_tabs[4]:
                    st.json(safe_dict_serializer(config.get('quantization', {})))
                    
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
                options=["mutual_info", "f_classif", "chi2"],
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
                ("Optuna", OptimizationStrategy.OPTUNA),
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

            # Model selection criteria
            st.subheader("Model Selection Criteria")
            
            model_selection_criteria_options = []
            if selected_task_type == TaskType.CLASSIFICATION:
                model_selection_criteria_options = [
                    ("Accuracy", ModelSelectionCriteria.ACCURACY),
                    ("F1 Score", ModelSelectionCriteria.F1),
                    ("Precision", ModelSelectionCriteria.PRECISION),
                    ("Recall", ModelSelectionCriteria.RECALL),
                    ("ROC AUC", ModelSelectionCriteria.ROC_AUC),
                    ("Matthews Correlation", ModelSelectionCriteria.Matthews_CORRELATION),
                ]
            else:  # Regression
                model_selection_criteria_options = [
                    ("RMSE", ModelSelectionCriteria.ROOT_MEAN_SQUARED_ERROR),
                    ("MAE", ModelSelectionCriteria.MEAN_ABSOLUTE_ERROR),
                    ("R²", ModelSelectionCriteria.R2),
                    ("Explained Variance", ModelSelectionCriteria.EXPLAINED_VARIANCE),
                ]
                
            selection_criteria_labels = [s[0] for s in model_selection_criteria_options]
            selection_criteria_values = [s[1] for s in model_selection_criteria_options]
            
            selection_criteria_index = st.selectbox(
                "Primary metric for model selection",
                options=range(len(model_selection_criteria_options)),
                format_func=lambda x: selection_criteria_labels[x],
                index=0
            )
            selected_selection_criteria = selection_criteria_values[selection_criteria_index]

            # Early stopping
            early_stopping = st.checkbox("Enable early stopping", value=True)
            early_stopping_rounds = st.slider(
                "Early stopping patience (rounds)",
                min_value=5,
                max_value=50,
                value=10,
                disabled=not early_stopping
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
                        ("Quantile", NormalizationType.QUANTILE),
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
                        options=["mean", "median", "most_frequent", "constant"],
                        disabled=not handle_nan,
                    )

                    detect_outliers = st.checkbox("Detect and handle outliers", value=False)
                    
                    outlier_method = st.selectbox(
                        "Outlier detection method",
                        options=["iqr", "zscore", "isolation_forest", "percentile"],
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
                        options=["one_hot", "label", "target", "binary", "frequency"],
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
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )

                with advanced_tabs[2]:
                    # Quantization options
                    enable_quantization = st.checkbox("Enable model quantization", value=False)

                    quantization_type_options = [
                        ("INT8", QuantizationType.INT8),
                        ("UINT8", QuantizationType.UINT8),
                        ("INT16", QuantizationType.INT16),
                        ("FLOAT16", QuantizationType.FLOAT16),
                        ("MIXED", QuantizationType.MIXED),
                    ]

                    quantization_type_index = st.selectbox(
                        "Quantization Type",
                        options=range(len(quantization_type_options)),
                        format_func=lambda x: quantization_type_options[x][0],
                        disabled=not enable_quantization,
                    )
                    selected_quantization_type = quantization_type_options[quantization_type_index][1]

                    quantization_mode_options = [
                        ("Static", QuantizationMode.STATIC),
                        ("Dynamic", QuantizationMode.DYNAMIC),
                        ("Dynamic Per Batch", QuantizationMode.DYNAMIC_PER_BATCH),
                        ("Dynamic Per Channel", QuantizationMode.DYNAMIC_PER_CHANNEL),
                        ("Symmetric", QuantizationMode.SYMMETRIC),
                        ("Calibrated", QuantizationMode.CALIBRATED),
                    ]

                    quantization_mode_index = st.selectbox(
                        "Quantization Mode",
                        options=range(len(quantization_mode_options)),
                        format_func=lambda x: quantization_mode_options[x][0],
                        disabled=not enable_quantization,
                    )
                    selected_quantization_mode = quantization_mode_options[quantization_mode_index][1]
                    
                    num_bits = st.slider(
                        "Number of bits",
                        min_value=4,
                        max_value=16,
                        value=8,
                        disabled=not enable_quantization,
                    )
                    
                    symmetric_quantization = st.checkbox(
                        "Use symmetric quantization",
                        value=True,
                        disabled=not enable_quantization,
                    )
                    
                    per_channel_quantization = st.checkbox(
                        "Use per-channel quantization",
                        value=False,
                        disabled=not enable_quantization,
                    )
                    
                    calibration_samples = st.slider(
                        "Calibration dataset size",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        disabled=not enable_quantization or selected_quantization_mode != QuantizationMode.CALIBRATED,
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
                    
                    memory_limit_gb = st.slider(
                        "Memory limit (GB)",
                        min_value=1,
                        max_value=32,
                        value=8,
                        disabled=not memory_optimization,
                    )
                    
                    enable_intel_optimization = st.checkbox("Enable Intel optimizations (if available)", value=True)
                    
                    # ONNX/Model compilation options
                    enable_jit = st.checkbox("Enable JIT compilation (if supported)", value=True)
                    enable_onnx = st.checkbox("Enable ONNX export/runtime (if available)", value=False)
                    
                    if enable_onnx:
                        onnx_opset = st.slider(
                            "ONNX opset version",
                            min_value=10,
                            max_value=15,
                            value=13,
                        )
                    
                    # Monitoring options
                    enable_monitoring = st.checkbox("Enable performance monitoring", value=True)
                    monitoring_interval = st.slider(
                        "Monitoring interval (seconds)",
                        min_value=1.0,
                        max_value=60.0,
                        value=10.0,
                        step=1.0,
                        disabled=not enable_monitoring,
                    )

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
                    
                    # Explainability options
                    enable_explainability = st.checkbox("Enable model explainability", value=True)
                    
                    if enable_explainability:
                        explainability_methods = st.multiselect(
                            "Explainability methods",
                            options=["shap", "feature_importance", "partial_dependence", "lime"],
                            default=["shap", "feature_importance"],
                        )
                        
                        default_method = st.selectbox(
                            "Default explainability method",
                            options=explainability_methods,
                            index=0 if "shap" in explainability_methods else 0,
                        )

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
                # Create explainability config
                explainability_config = ExplainabilityConfig(
                    enable_explainability=enable_explainability if 'enable_explainability' in locals() else True,
                    methods=explainability_methods if 'explainability_methods' in locals() else ["shap", "feature_importance"],
                    default_method=default_method if 'default_method' in locals() else "shap",
                    shap_algorithm="auto",
                    shap_samples=100,
                    generate_summary=True,
                    generate_plots=True,
                )
                
                # Create monitoring config
                monitoring_config = MonitoringConfig(
                    enable_monitoring=enable_monitoring if 'enable_monitoring' in locals() else True,
                    drift_detection=True,
                    drift_detection_method="ks_test",
                    monitoring_interval=str(int(monitoring_interval)) + "s" if 'monitoring_interval' in locals() else "10s",
                    performance_tracking=True,
                    alert_on_drift=True
                )
                
                # Create preprocessor config
                preprocessor_config = PreprocessorConfig(
                    normalization=selected_normalization,
                    handle_nan=handle_nan,
                    handle_inf=handle_inf,
                    nan_strategy=nan_strategy,
                    inf_strategy=nan_strategy,  # Use same strategy for inf
                    detect_outliers=detect_outliers,
                    outlier_method=outlier_method if detect_outliers else "iqr",
                    outlier_zscore_threshold=outlier_threshold if detect_outliers and outlier_method == "zscore" else 3.0,
                    outlier_iqr_multiplier=outlier_threshold if detect_outliers and outlier_method == "iqr" else 1.5,
                    categorical_encoding=categorical_encoding,
                    clip_values=False,
                    enable_input_validation=True,
                    parallel_processing=True,
                    n_jobs=n_jobs,
                    cache_enabled=True,
                    debug_mode=(log_level == "DEBUG"),
                )

                # Create batch processing config
                batch_processing_config = BatchProcessorConfig(
                    initial_batch_size=initial_batch_size,
                    min_batch_size=max(1, initial_batch_size // 2),
                    max_batch_size=max_batch_size,
                    max_queue_size=1000,
                    batch_timeout=batch_timeout,
                    processing_strategy=selected_batch_strategy,
                    enable_adaptive_batching=(
                        selected_batch_strategy == BatchProcessingStrategy.ADAPTIVE
                    ),
                    enable_memory_optimization=memory_optimization,
                    num_workers=n_jobs,
                    max_workers=n_jobs,
                    debug_mode=(log_level == "DEBUG"),
                )

                # Create quantization config
                quantization_config = QuantizationConfig(
                    quantization_type=selected_quantization_type,
                    quantization_mode=selected_quantization_mode,
                    per_channel=per_channel_quantization if 'per_channel_quantization' in locals() else False,
                    symmetric=symmetric_quantization if 'symmetric_quantization' in locals() else True,
                    enable_cache=True,
                    cache_size=1024,
                    calibration_samples=calibration_samples if 'calibration_samples' in locals() else 100,
                    num_bits=num_bits if 'num_bits' in locals() else 8,
                    optimize_memory=memory_optimization
                )

                # Create inference engine config
                inference_engine_config = InferenceEngineConfig(
                    enable_intel_optimization=enable_intel_optimization if 'enable_intel_optimization' in locals() else True,
                    debug_mode=(log_level == "DEBUG"),
                    num_threads=n_jobs,
                    set_cpu_affinity=(n_jobs > 1),
                    enable_quantization=enable_quantization,
                    enable_model_quantization=enable_quantization,
                    quantization_dtype=selected_quantization_type.value,
                    quantization_config=quantization_config,
                    enable_jit=enable_jit if 'enable_jit' in locals() else True,
                    enable_onnx=enable_onnx if 'enable_onnx' in locals() else False,
                    onnx_opset=onnx_opset if 'onnx_opset' in locals() else 13,
                    enable_batching=True,
                    initial_batch_size=initial_batch_size,
                    min_batch_size=max(1, initial_batch_size // 2),
                    max_batch_size=max_batch_size,
                    enable_memory_optimization=memory_optimization,
                    memory_limit_gb=memory_limit_gb if 'memory_limit_gb' in locals() else None,
                    enable_monitoring=enable_monitoring if 'enable_monitoring' in locals() else True,
                    monitoring_interval=monitoring_interval if 'monitoring_interval' in locals() else 10.0,
                )

                # Create main config
                config = MLTrainingEngineConfig(
                    task_type=selected_task_type,
                    model_path=model_path,
                    preprocessing_config=preprocessor_config,
                    batch_processing_config=batch_processing_config,
                    inference_config=inference_engine_config,
                    quantization_config=quantization_config,
                    explainability_config=explainability_config,
                    monitoring_config=monitoring_config,
                    feature_selection=enable_feature_selection,
                    feature_selection_method=feature_selection_method,
                    feature_selection_k=feature_selection_k,
                    feature_importance_threshold=feature_importance_threshold,
                    optimization_strategy=selected_optimization,
                    optimization_iterations=optimization_iterations,
                    optimization_metric=selected_selection_criteria,
                    model_selection_criteria=selected_selection_criteria,
                    early_stopping=early_stopping,
                    early_stopping_rounds=early_stopping_rounds,
                    cv_folds=cv_folds,
                    test_size=test_size,
                    stratify=stratify,
                    experiment_tracking=True,
                    experiment_tracking_platform="mlflow",
                    n_jobs=n_jobs,
                    memory_optimization=memory_optimization,
                    use_intel_optimization=enable_intel_optimization if 'enable_intel_optimization' in locals() else True,
                    verbose=1 if verbose else 0,
                    log_level=log_level,
                    random_state=random_state,
                    generate_model_summary=True,
                    compute_permutation_importance=True if enable_explainability else False,
                    generate_feature_importance_report=True if enable_explainability else False,
                    generate_prediction_explanations=True if enable_explainability else False,
                    ensemble_models=False,
                    auto_save=True,
                    auto_save_on_shutdown=True,
                )

                # Save config in session state
                st.session_state.config = config
                st.session_state.selected_models = selected_models

                st.success("Configuration saved successfully!")

                # Display the config summary
                with st.expander("Configuration Summary", expanded=True):
                    st.json(safe_dict_serializer(config))
def model_training_and_evaluation():
    """Model training and evaluation section"""
    st.title("Model Training & Evaluation")
    
    if st.session_state.data is None or st.session_state.target is None:
        st.warning("Please upload data and select a target column first")
        return
    
    if st.session_state.config is None or not st.session_state.selected_models:
        st.warning("Please configure the training settings first")
        return
    
    data = st.session_state.data
    target = st.session_state.target
    features = st.session_state.features
    config = st.session_state.config
    selected_models = st.session_state.selected_models
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Training Setup")
        
        # Show data summary
        st.subheader("Data Summary")
        st.write(f"**Total samples:** {data.shape[0]:,}")
        st.write(f"**Features:** {len(features)}")
        st.write(f"**Target:** {target}")
        
        # Show task type
        st.write(f"**Task type:** {config.task_type.value}")
        
        # Show selected models
        st.subheader("Selected Models")
        st.write(", ".join(selected_models))
        
        # Show optimization strategy
        st.write(f"**Optimization:** {config.optimization_strategy.value}")
        st.write(f"**Iterations:** {config.optimization_iterations}")
        
        # Show cross-validation setup
        st.subheader("Cross-Validation")
        st.write(f"**Folds:** {config.cv_folds}")
        st.write(f"**Test size:** {int(config.test_size * 100)}%")
        
        # Show feature selection
        if config.feature_selection:
            st.subheader("Feature Selection")
            st.write(f"**Method:** {config.feature_selection_method}")
            st.write(f"**Features to select:** {config.feature_selection_k}")
            st.write(f"**Importance threshold:** {config.feature_importance_threshold}")
        
        # Training controls
        st.subheader("Training Controls")
        
        if st.button("Start Training", key="start_training_btn"):
            with st.spinner("Training in progress..."):
                try:
                    # Split data
                    X = data[features]
                    y = data[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=config.test_size,
                        random_state=config.random_state,
                        stratify=y if config.stratify else None
                    )
                    
                    # Initialize training engine
                    training_engine = MLTrainingEngine(
                        config=config,
                        model_names=selected_models,
                        param_grids=DEFAULT_PARAM_GRIDS
                    )
                    
                    # Train models
                    start_time = time.time()
                    training_engine.train(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Get results
                    models = training_engine.get_models()
                    metrics = training_engine.get_metrics()
                    best_model = training_engine.get_best_model()
                    
                    # Store results in session state
                    st.session_state.models = models
                    st.session_state.model_metrics = metrics
                    st.session_state.best_model = best_model
                    st.session_state.training_completed = True
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.training_time = training_time
                    
                    st.success("Training completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.session_state.training_completed:
            st.header("Training Results")
            
            # Show training summary
            st.subheader("Summary")
            st.write(f"**Training time:** {st.session_state.training_time:.2f} seconds")
            st.write(f"**Best model:** {st.session_state.best_model['name']}")
            
            # Show metrics comparison
            st.subheader("Model Performance Comparison")
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame.from_dict(
                st.session_state.model_metrics, 
                orient='index'
            ).reset_index()
            metrics_df.columns = ['Model'] + list(metrics_df.columns[1:])
            
            # Sort by primary metric
            primary_metric = config.model_selection_criteria.value
            metrics_df = metrics_df.sort_values(
                by=primary_metric, 
                ascending=config.task_type == TaskType.REGRESSION
            )
            
            # Display metrics
            st.dataframe(metrics_df.style.highlight_max(
                subset=[c for c in metrics_df.columns if c != 'Model'],
                axis=0
            ))
            
            # Plot metrics comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.plot(
                x='Model', 
                y=primary_metric, 
                kind='bar', 
                ax=ax,
                title=f"Model Comparison by {primary_metric}"
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Show best model details
            st.subheader("Best Model Details")
            best_model_name = st.session_state.best_model['name']
            best_model_metrics = st.session_state.model_metrics[best_model_name]
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Model", best_model_name)
            with cols[1]:
                st.metric(primary_metric, f"{best_model_metrics[primary_metric]:.4f}")
            with cols[2]:
                st.metric("Training Time", f"{best_model_metrics['training_time']:.2f}s")
            
            # Show hyperparameters
            with st.expander("Best Model Hyperparameters"):
                st.json(st.session_state.best_model['params'])
            
            # Feature importance
            if config.explainability_config.generate_feature_importance_report:
                st.subheader("Feature Importance")
                
                try:
                    # Get feature importances
                    feature_importances = training_engine.get_feature_importances()
                    
                    if feature_importances is not None:
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        feature_importances.plot(kind='barh', ax=ax)
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)
                    else:
                        st.info("Feature importance not available for this model")
                except Exception as e:
                    st.warning(f"Could not generate feature importance: {str(e)}")
            
            # SHAP values (if enabled)
            if (config.explainability_config.enable_explainability and 
                'shap' in config.explainability_config.methods):
                st.subheader("SHAP Values")
                
                try:
                    # Get SHAP values
                    shap_values = training_engine.get_shap_values()
                    
                    if shap_values is not None:
                        # Plot SHAP summary
                        fig, ax = plt.subplots(figsize=(10, 6))
                        training_engine.plot_shap_summary(ax=ax)
                        st.pyplot(fig)
                    else:
                        st.info("SHAP values not available for this model")
                except Exception as e:
                    st.warning(f"Could not generate SHAP values: {str(e)}")

def model_inference():
    """Model inference section"""
    st.title("Model Inference")
    
    if not st.session_state.training_completed:
        st.warning("Please complete training first")
        return
    
    if "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.warning("Test data not available")
        return
    
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    best_model = st.session_state.best_model
    config = st.session_state.config
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Inference Setup")
        
        # Show model info
        st.subheader("Model Information")
        st.write(f"**Model:** {best_model['name']}")
        st.write(f"**Task type:** {config.task_type.value}")
        
        # Input options
        st.subheader("Input Options")
        
        inference_mode = st.radio(
            "Inference mode",
            options=["Test set", "Manual input"],
            index=0
        )
        
        if inference_mode == "Test set":
            st.info(f"Using test set with {len(X_test)} samples")
            
            if st.button("Run Inference on Test Set"):
                with st.spinner("Running inference..."):
                    try:
                        # Initialize inference engine
                        inference_engine = InferenceEngine(
                            config=config.inference_config,
                            model=best_model['model']
                        )
                        
                        # Run inference
                        start_time = time.time()
                        predictions = inference_engine.predict(X_test)
                        inference_time = time.time() - start_time
                        
                        # Store predictions
                        st.session_state.predictions = predictions
                        st.session_state.inference_time = inference_time
                        st.session_state.inference_mode = "test_set"
                        
                        st.success(f"Inference completed in {inference_time:.4f} seconds")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Inference failed: {str(e)}")
        
        else:  # Manual input
            st.subheader("Manual Input")
            
            # Create input form based on features
            input_data = {}
            for feature in st.session_state.features:
                # Get sample values for guidance
                sample_value = st.session_state.data[feature].iloc[0]
                dtype = st.session_state.data[feature].dtype
                
                if np.issubdtype(dtype, np.number):
                    min_val = float(st.session_state.data[feature].min())
                    max_val = float(st.session_state.data[feature].max())
                    default_val = float(st.session_state.data[feature].median())
                    
                    input_data[feature] = st.slider(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        help=f"Sample value: {sample_value}"
                    )
                else:
                    unique_vals = st.session_state.data[feature].unique()
                    default_idx = 0 if len(unique_vals) > 0 else None
                    
                    input_data[feature] = st.selectbox(
                        feature,
                        options=unique_vals,
                        index=default_idx,
                        help=f"Sample value: {sample_value}"
                    )
            
            if st.button("Run Inference on Manual Input"):
                with st.spinner("Running inference..."):
                    try:
                        # Convert input to DataFrame
                        input_df = pd.DataFrame([input_data])
                        
                        # Initialize inference engine
                        inference_engine = InferenceEngine(
                            config=config.inference_config,
                            model=best_model['model']
                        )
                        
                        # Run inference
                        start_time = time.time()
                        prediction = inference_engine.predict(input_df)
                        inference_time = time.time() - start_time
                        
                        # Store predictions
                        st.session_state.predictions = prediction
                        st.session_state.inference_time = inference_time
                        st.session_state.inference_mode = "manual"
                        st.session_state.input_data = input_data
                        
                        st.success(f"Inference completed in {inference_time:.4f} seconds")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Inference failed: {str(e)}")
    
    with col2:
        if "predictions" in st.session_state:
            st.header("Inference Results")
            
            # Show inference time
            st.write(f"**Inference time:** {st.session_state.inference_time:.4f} seconds")
            
            if st.session_state.inference_mode == "test_set":
                # Test set evaluation
                st.subheader("Test Set Evaluation")
                
                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score,
                    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
                )
                
                y_true = y_test
                y_pred = st.session_state.predictions
                
                if config.task_type == TaskType.CLASSIFICATION:
                    # Classification metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted')
                    recall = recall_score(y_true, y_pred, average='weighted')
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    
                    try:
                        roc_auc = roc_auc_score(y_true, y_pred)
                    except:
                        roc_auc = "N/A"
                    
                    # Display metrics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with cols[1]:
                        st.metric("Precision", f"{precision:.4f}")
                    with cols[2]:
                        st.metric("Recall", f"{recall:.4f}")
                    with cols[3]:
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    if roc_auc != "N/A":
                        st.metric("ROC AUC", f"{roc_auc:.4f}")
                    
                    # Confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                
                else:  # Regression
                    # Regression metrics
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    # Display metrics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("MSE", f"{mse:.4f}")
                    with cols[1]:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with cols[2]:
                        st.metric("MAE", f"{mae:.4f}")
                    with cols[3]:
                        st.metric("R²", f"{r2:.4f}")
                    
                    # Actual vs Predicted plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_true, y_pred, alpha=0.5)
                    ax.plot([y_true.min(), y_true.max()], 
                            [y_true.min(), y_true.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Actual vs Predicted')
                    st.pyplot(fig)
            
            else:  # Manual input
                # Single prediction display
                st.subheader("Prediction Result")
                
                prediction = st.session_state.predictions[0]
                
                if config.task_type == TaskType.CLASSIFICATION:
                    # For classification, show predicted class and probabilities
                    st.write(f"**Predicted class:** {prediction}")
                    
                    # If model supports predict_proba, show probabilities
                    if hasattr(best_model['model'], 'predict_proba'):
                        proba = best_model['model'].predict_proba(
                            pd.DataFrame([st.session_state.input_data])
                        )[0]
                        
                        classes = best_model['model'].classes_
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=classes, y=proba, ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_title('Class Probabilities')
                        ax.set_ylabel('Probability')
                        st.pyplot(fig)
                else:
                    # For regression, show predicted value
                    st.write(f"**Predicted value:** {prediction:.4f}")
                
                # Show input values
                st.subheader("Input Values")
                st.json(st.session_state.input_data)

def model_management():
    """Model management section"""
    st.title("Model Management")
    
    if not st.session_state.training_completed:
        st.warning("Please complete training first")
        return
    
    best_model = st.session_state.best_model
    config = st.session_state.config
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Model Export")
        
        # Model export options
        export_format = st.selectbox(
            "Export format",
            options=["Pickle", "Joblib", "ONNX", "JSON"]
        )
        
        model_name = st.text_input(
            "Model name",
            value=f"{best_model['name']}_{config.task_type.value.lower()}"
        )
        
        if st.button("Export Model"):
            with st.spinner("Exporting model..."):
                try:
                    # Create export directory if it doesn't exist
                    os.makedirs(config.model_path, exist_ok=True)
                    
                    # Generate filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name}_{timestamp}"
                    
                    if export_format == "Pickle":
                        import pickle
                        filepath = os.path.join(config.model_path, f"{filename}.pkl")
                        with open(filepath, 'wb') as f:
                            pickle.dump(best_model['model'], f)
                    
                    elif export_format == "Joblib":
                        import joblib
                        filepath = os.path.join(config.model_path, f"{filename}.joblib")
                        joblib.dump(best_model['model'], filepath)
                    
                    elif export_format == "ONNX":
                        from skl2onnx import convert_sklearn
                        from skl2onnx.common.data_types import FloatTensorType
                        
                        # Define initial types
                        initial_type = [('float_input', FloatTensorType([None, len(st.session_state.features)]))]
                        
                        # Convert to ONNX
                        onnx_model = convert_sklearn(
                            best_model['model'],
                            initial_types=initial_type,
                            target_opset=config.inference_config.onnx_opset
                        )
                        
                        # Save ONNX model
                        filepath = os.path.join(config.model_path, f"{filename}.onnx")
                        with open(filepath, "wb") as f:
                            f.write(onnx_model.SerializeToString())
                    
                    elif export_format == "JSON":
                        # This will only work for certain model types
                        import json
                        filepath = os.path.join(config.model_path, f"{filename}.json")
                        
                        if hasattr(best_model['model'], 'get_params'):
                            model_params = best_model['model'].get_params()
                            with open(filepath, 'w') as f:
                                json.dump(model_params, f)
                        else:
                            raise Exception("Model does not support JSON export")
                    
                    st.success(f"Model exported successfully to: {filepath}")
                    st.session_state.last_export_path = filepath
                
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        # Show last exported model path if available
        if "last_export_path" in st.session_state:
            st.subheader("Last Exported Model")
            st.code(st.session_state.last_export_path)
            
            # Download button
            with open(st.session_state.last_export_path, 'rb') as f:
                st.download_button(
                    label="Download Model",
                    data=f,
                    file_name=os.path.basename(st.session_state.last_export_path)
                )
    
    with col2:
        st.header("Model Deployment")
        
        deployment_target = st.selectbox(
            "Deployment target",
            options=["REST API", "Batch Processing", "Edge Device", "Cloud Service"]
        )
        
        if deployment_target == "REST API":
            st.info("""
            To deploy as a REST API:
            1. Export the model using one of the formats above
            2. Use a framework like FastAPI or Flask to create an API endpoint
            3. The endpoint should accept input data and return predictions
            """)
            
            # Show sample FastAPI code
            with st.expander("Sample FastAPI Code"):
                st.code("""
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the model
model = joblib.load("path/to/your/model.joblib")

@app.post("/predict")
async def predict(data: dict):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Return prediction
    return {"prediction": prediction.tolist()}
                """, language='python')
        
        elif deployment_target == "Batch Processing":
            st.info("""
            For batch processing:
            1. Export the model
            2. Create a script that loads the model and processes input files
            3. Schedule the script to run periodically or trigger it on new data
            """)
            
            # Show sample batch processing code
            with st.expander("Sample Batch Processing Code"):
                st.code("""
import pandas as pd
import joblib
from pathlib import Path

# Load model
model = joblib.load("path/to/your/model.joblib")

# Process files in input directory
input_dir = Path("input/")
output_dir = Path("output/")

for input_file in input_dir.glob("*.csv"):
    # Read input data
    data = pd.read_csv(input_file)
    
    # Make predictions
    predictions = model.predict(data)
    
    # Save results
    output_file = output_dir / f"predictions_{input_file.name}"
    pd.DataFrame(predictions).to_csv(output_file, index=False)
                """, language='python')
        
        elif deployment_target == "Edge Device":
            st.info("""
            For edge deployment:
            1. Export the model in a format compatible with your edge device
            2. Quantize the model if needed for performance
            3. Package with your edge application
            """)
            
            # Show sample edge deployment code
            with st.expander("Sample Edge Deployment Code"):
                st.code("""
# Example for Raspberry Pi with ONNX runtime
import onnxruntime as rt
import numpy as np

# Create inference session
sess = rt.InferenceSession("path/to/model.onnx")

# Prepare input
input_name = sess.get_inputs()[0].name
input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

# Run inference
prediction = sess.run(None, {input_name: input_data})[0]
                """, language='python')
        
        elif deployment_target == "Cloud Service":
            st.info("""
            For cloud deployment:
            1. Choose a cloud provider (AWS, GCP, Azure, etc.)
            2. Use their ML deployment services (SageMaker, Vertex AI, etc.)
            3. Follow their documentation for model deployment
            """)
            
            # Show sample cloud deployment commands
            with st.expander("Sample AWS SageMaker Deployment"):
                st.code("""
# After exporting your model to a format SageMaker supports
# Create a SageMaker model
aws sagemaker create-model \
    --model-name my-model \
    --execution-role-arn arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole \
    --primary-container Image=<algorithm-image>,ModelDataUrl=s3://my-bucket/path/to/model.tar.gz

# Create an endpoint configuration
aws sagemaker create-endpoint-config \
    --endpoint-config-name my-endpoint-config \
    --production-variants VariantName=variant-1,ModelName=my-model,InitialInstanceCount=1,InstanceType=ml.m5.large

# Create the endpoint
aws sagemaker create-endpoint \
    --endpoint-name my-endpoint \
    --endpoint-config-name my-endpoint-config
                """, language='bash')

def main():
    """Main application flow"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Data & Configuration", "Model Training", "Inference", "Model Management"]
    )
    
    # Display the selected page
    if app_mode == "Data & Configuration":
        data_upload_and_configuration()
    elif app_mode == "Model Training":
        model_training_and_evaluation()
    elif app_mode == "Inference":
        model_inference()
    elif app_mode == "Model Management":
        model_management()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Advanced ML Training Engine\n\n"
        "Version 1.0\n\n"
    )

if __name__ == "__main__":
    main()
