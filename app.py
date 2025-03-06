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

# Import the correct configuration classes from your module
from modules.configs import (
    TaskType,
    OptimizationStrategy,
    MLTrainingEngineConfig,
    PreprocessorConfig,  # Changed from PreprocessingConfig
    NormalizationType,   # Added for normalization types
    BatchProcessorConfig,
    BatchProcessingStrategy,  # Added for batch strategy
    InferenceEngineConfig,    # Changed from InferenceConfig
    QuantizationConfig,
    QuantizationType,
    QuantizationMode
)
from modules.engine.train_engine import MLTrainingEngine

# Set page configuration
st.set_page_config(
    page_title="Kolosal AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = []
    
# Define model options by task type
MODEL_OPTIONS = {
    TaskType.CLASSIFICATION: [
        "LogisticRegression", 
        "RandomForestClassifier", 
        "GradientBoostingClassifier", 
        "XGBClassifier", 
        "LGBMClassifier",
        "CatBoostClassifier",
        "SVC"
    ],
    TaskType.REGRESSION: [
        "LinearRegression", 
        "RandomForestRegressor", 
        "GradientBoostingRegressor", 
        "XGBRegressor", 
        "LGBMRegressor",
        "CatBoostRegressor",
        "SVR"
    ]
}

# Default hyperparameter grids
DEFAULT_PARAM_GRIDS = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l1", "l2", "elasticnet", None],
        "model__solver": ["lbfgs", "liblinear", "saga"]
    },
    "RandomForestClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "RandomForestRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0]
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0]
    },
    "XGBClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    },
    "XGBRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.3],
        "model__max_depth": [3, 5, 8],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }
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
        "SVR": SVR
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

def create_sidebar():
    """Create the sidebar navigation"""
    st.sidebar.title("Kolosal AutoML")
    
    navigation = st.sidebar.radio(
        "Navigation",
        ["Data Upload & Exploration", "Training Configuration", "Model Training", 
         "Model Evaluation", "Prediction", "Export"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Documentation")
    st.sidebar.markdown("""
    Kolosal AutoML is an automated machine learning platform that helps you:
    
    1. Upload and explore your data
    2. Configure and train multiple ML models
    3. Evaluate and compare model performance
    4. Make predictions on new data
    5. Export trained models for deployment
    """)
    
    return navigation

def data_upload_page():
    """Data upload and exploration page"""
    st.title("Data Upload & Exploration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Try to determine file type from extension
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
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
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{data.shape[0]:,}")
            col2.metric("Columns", data.shape[1])
            col3.metric("Missing Values", f"{data.isna().sum().sum():,}")
            col4.metric("Duplicated Rows", f"{data.duplicated().sum():,}")
            
            # Data exploration tabs
            tab1, tab2, tab3 = st.tabs(["Column Info", "Correlation", "Distribution"])
            
            with tab1:
                # Column information
                column_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Null Count': data.isna().sum(),
                    'Unique Values': [data[col].nunique() for col in data.columns],
                    'Sample Values': [str(data[col].unique()[:3]) for col in data.columns]
                })
                st.dataframe(column_info)
            
            with tab2:
                # Correlation heatmap for numeric columns
                numeric_data = data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_matrix = numeric_data.corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                                cmap="coolwarm", square=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found for correlation analysis.")
            
            with tab3:
                # Distribution of columns
                if not numeric_data.empty:
                    selected_column = st.selectbox(
                        "Select column for distribution analysis",
                        options=numeric_data.columns
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=data, x=selected_column, kde=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found for distribution analysis.")
            
            # Target column selection
            st.subheader("Target Selection")
            target_col = st.selectbox(
                "Select target column for prediction",
                options=data.columns
            )
            
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
                    st.write({
                        "Mean": target_data.mean(),
                        "Median": target_data.median(),
                        "Min": target_data.min(),
                        "Max": target_data.max(),
                        "Std Dev": target_data.std()
                    })
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        # Display sample data option
        if st.button("Load Sample Data"):
            # Create sample data
            import sklearn.datasets
            X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)
            data = pd.concat([X, pd.Series(y, name='target')], axis=1)
            st.session_state.data = data
            st.session_state.target = 'target'
            st.session_state.features = X.columns.tolist()
            st.success("Sample classification data loaded (Breast Cancer dataset)")
            st.experimental_rerun()

def training_configuration_page():
    """Configure the model training parameters"""
    st.title("Training Configuration")
    
    if st.session_state.data is None or st.session_state.target is None:
        st.warning("Please upload data and select a target column first")
        return
    
    data = st.session_state.data
    target = st.session_state.target
    
    with st.form("training_config_form"):
        st.subheader("Task Configuration")
        
        # Task type selection
        task_type_options = [("Classification", TaskType.CLASSIFICATION), 
                            ("Regression", TaskType.REGRESSION)]
        
        task_type_labels = [t[0] for t in task_type_options]
        task_type_values = [t[1] for t in task_type_options]
        
        task_type_index = st.radio(
            "Select task type",
            options=range(len(task_type_options)),
            format_func=lambda x: task_type_labels[x]
        )
        selected_task_type = task_type_values[task_type_index]
        
        # Feature selection
        st.subheader("Feature Selection")
        
        enable_feature_selection = st.checkbox("Enable feature selection", value=True)
        feature_selection_method = st.selectbox(
            "Feature selection method",
            options=["f_classif", "mutual_info"],
            disabled=not enable_feature_selection
        )
        
        feature_selection_k = st.slider(
            "Number of features to select (k)",
            min_value=1,
            max_value=len(st.session_state.features),
            value=min(10, len(st.session_state.features)),
            disabled=not enable_feature_selection
        )
        
        # Model selection
        st.subheader("Model Selection")
        
        available_models = MODEL_OPTIONS[selected_task_type]
        selected_models = st.multiselect(
            "Select models to train",
            options=available_models,
            default=available_models[:3]  # Default to first 3 models
        )
        
        # Optimization strategy
        st.subheader("Optimization Strategy")
        
        optimization_options = [
            ("Grid Search", OptimizationStrategy.GRID_SEARCH),
            ("Random Search", OptimizationStrategy.RANDOM_SEARCH),
            ("Bayesian Optimization", OptimizationStrategy.BAYESIAN_OPTIMIZATION)
        ]
        
        optimization_labels = [o[0] for o in optimization_options]
        optimization_values = [o[1] for o in optimization_options]
        
        optimization_index = st.radio(
            "Select optimization strategy",
            options=range(len(optimization_options)),
            format_func=lambda x: optimization_labels[x]
        )
        selected_optimization = optimization_values[optimization_index]
        
        optimization_iterations = st.slider(
            "Number of optimization iterations",
            min_value=10,
            max_value=100,
            value=30,
            disabled=selected_optimization == OptimizationStrategy.GRID_SEARCH
        )
        
        # Cross-validation configuration
        st.subheader("Cross-Validation")
        
        cv_folds = st.slider(
            "Number of cross-validation folds",
            min_value=2,
            max_value=10,
            value=5
        )
        
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=40,
            value=20
        ) / 100
        
        stratify = st.checkbox(
            "Use stratified sampling",
            value=selected_task_type == TaskType.CLASSIFICATION
        )
        
        # Advanced options
        st.subheader("Advanced Options")
        
        with st.expander("Advanced Configuration"):
            # Preprocessing options
            st.write("Preprocessing Settings")
            
            # Use NormalizationType enum for normalization selection
            normalization_options = [
                ("None", NormalizationType.NONE),
                ("Standard Scaling", NormalizationType.STANDARD),
                ("Min-Max Scaling", NormalizationType.MINMAX),
                ("Robust Scaling", NormalizationType.ROBUST)
            ]
            
            normalization_index = st.selectbox(
                "Normalization Method",
                options=range(len(normalization_options)),
                format_func=lambda x: normalization_options[x][0]
            )
            selected_normalization = normalization_options[normalization_index][1]
            
            handle_nan = st.checkbox("Handle missing values", value=True)
            handle_inf = st.checkbox("Handle infinity values", value=True)
            
            nan_strategy = st.selectbox(
                "Missing value strategy",
                options=["mean", "median", "most_frequent", "zero"],
                disabled=not handle_nan
            )
            
            detect_outliers = st.checkbox("Detect and handle outliers", value=False)
            
            # Batch processing options
            st.write("Batch Processing Settings")
            
            batch_strategy_options = [
                ("Fixed", BatchProcessingStrategy.FIXED),
                ("Adaptive", BatchProcessingStrategy.ADAPTIVE),
                ("Greedy", BatchProcessingStrategy.GREEDY)
            ]
            
            batch_strategy_index = st.selectbox(
                "Batch Processing Strategy",
                options=range(len(batch_strategy_options)),
                format_func=lambda x: batch_strategy_options[x][0]
            )
            selected_batch_strategy = batch_strategy_options[batch_strategy_index][1]
            
            initial_batch_size = st.slider(
                "Initial batch size",
                min_value=1,
                max_value=200,
                value=16
            )
            
            # Quantization options
            st.write("Quantization Settings")
            
            enable_quantization = st.checkbox("Enable model quantization", value=False)
            
            quantization_type_options = [
                ("INT8", QuantizationType.INT8.value),
                ("UINT8", QuantizationType.UINT8.value),
                ("INT16", QuantizationType.INT16.value)
            ]
            
            quantization_type_index = st.selectbox(
                "Quantization Type",
                options=range(len(quantization_type_options)),
                format_func=lambda x: quantization_type_options[x][0],
                disabled=not enable_quantization
            )
            selected_quantization_type = quantization_type_options[quantization_type_index][1]
            
            quantization_mode_options = [
                ("Symmetric", QuantizationMode.SYMMETRIC.value),
                ("Asymmetric", QuantizationMode.ASYMMETRIC.value),
                ("Dynamic Per Batch", QuantizationMode.DYNAMIC_PER_BATCH.value),
                ("Dynamic Per Channel", QuantizationMode.DYNAMIC_PER_CHANNEL.value)
            ]
            
            quantization_mode_index = st.selectbox(
                "Quantization Mode",
                options=range(len(quantization_mode_options)),
                format_func=lambda x: quantization_mode_options[x][0],
                disabled=not enable_quantization
            )
            selected_quantization_mode = quantization_mode_options[quantization_mode_index][1]
            
            # Experiment tracking
            experiment_tracking = st.checkbox("Enable experiment tracking", value=True)
            
            # Performance optimization
            n_jobs = st.slider(
                "Number of parallel jobs",
                min_value=1,
                max_value=8,
                value=4
            )
            
            memory_optimization = st.checkbox("Enable memory optimization", value=True)
            
            # Model path
            model_path = st.text_input(
                "Model save path",
                value="./models"
            )
            
            # Verbosity
            verbose = st.checkbox("Verbose output", value=True)
            
            log_level = st.selectbox(
                "Log level",
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1  # Default to INFO
            )
            
            # Feature importance threshold
            feature_importance_threshold = st.slider(
                "Feature importance threshold",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                disabled=not enable_feature_selection
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
                parallel_processing=True,
                n_jobs=n_jobs,
                debug_mode=(log_level == "DEBUG")
            )
            
            # Create batch processing config
            batch_processing_config = BatchProcessorConfig(
                initial_batch_size=initial_batch_size,
                processing_strategy=selected_batch_strategy,
                enable_adaptive_batching=(selected_batch_strategy == BatchProcessingStrategy.ADAPTIVE),
                max_workers=n_jobs,
                debug_mode=(log_level == "DEBUG"),
                enable_memory_optimization=memory_optimization
            )
            
            # Create quantization config
            quantization_config = QuantizationConfig(
                quantization_type=selected_quantization_type,
                quantization_mode=selected_quantization_mode,
                enable_cache=True,
                optimize_memory=memory_optimization,
                error_on_nan=False
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
                enable_monitoring=True
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
                experiment_tracking=experiment_tracking,
                n_jobs=n_jobs,
                memory_optimization=memory_optimization,
                verbose=1 if verbose else 0,
                log_level=log_level,
                random_state=42
            )
            
            # Save config in session state
            st.session_state.config = config
            st.session_state.selected_models = selected_models
            
            st.success("Configuration saved successfully!")
            
            # Display the config summary
            st.json(config.to_dict())

def model_training_page():
    """Model training page"""
    st.title("Model Training")
    
    if st.session_state.data is None or st.session_state.target is None:
        st.warning("Please upload data and select a target column first")
        return
        
    if not hasattr(st.session_state, 'config') or st.session_state.config is None:
        st.warning("Please configure training parameters first")
        return
    
    data = st.session_state.data
    target = st.session_state.target
    config = st.session_state.config
    selected_models = st.session_state.selected_models if hasattr(st.session_state, 'selected_models') else []
    
    if not selected_models:
        st.error("No models selected for training")
        return
    
    # Create directory for models if it doesn't exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    
    # Split data into features and target
    X = data.drop(columns=[target])
    y = data[target]
    
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
                    X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=config.test_size, random_state=config.random_state
                )
            
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
                        X=X_train.values,  # Convert DataFrame to numpy array
                        y=y_train.values if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series) else y_train,
                        X_test=X_test.values,  # Convert DataFrame to numpy array
                        y_test=y_test.values if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series) else y_test
                    )

                    # Save results
                    results[model_name] = {
                        "metrics": metrics,
                        "model_name": model_name
                    }
                    
                    if engine.save_model(model_name):
                        status_text.text(f"Model {model_name} trained and saved successfully")
                    
                    # Add to session state
                    st.session_state.models[model_name] = best_model
                    st.session_state.model_metrics[model_name] = metrics
                    
                    # Check if this is the best model
                    # Check if this is the best model
                    if engine.best_model and engine.best_model["name"] == model_name:
                        st.session_state.best_model = {
                            "name": model_name,
                            "model": best_model,
                            "metrics": metrics
                        }
                else:
                    status_text.text(f"Model {model_name} not found in available models")
            
            # Update progress to completion
            progress.progress(1.0)
            status_text.text("Training completed successfully!")
            
            # Save experiment results
            experiment_result = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models_trained": selected_models,
                "results": results,
                "config": config.to_dict()
            }
            st.session_state.experiment_results.append(experiment_result)
            
            # Mark training as completed
            st.session_state.training_completed = True
            
            # Display success message
            st.success("All models trained successfully! Go to Model Evaluation to see results.")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def model_evaluation_page():
    """Model evaluation page"""
    st.title("Model Evaluation")
    
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
        best_metrics = best_model['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        # Display primary metric
        if 'accuracy' in best_metrics:
            col1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
        elif 'r2' in best_metrics:
            col1.metric("R²", f"{best_metrics['r2']:.4f}")
        
        # Display secondary metrics
        if 'precision' in best_metrics and 'recall' in best_metrics:
            col2.metric("Precision", f"{best_metrics['precision']:.4f}")
            col3.metric("Recall", f"{best_metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{best_metrics['f1']:.4f}")
        elif 'mae' in best_metrics and 'mse' in best_metrics:
            col2.metric("MAE", f"{best_metrics['mae']:.4f}")
            col3.metric("MSE", f"{best_metrics['mse']:.4f}")
            col4.metric("RMSE", f"{best_metrics['rmse']:.4f}")
    else:
        st.info("No best model identified")
    
    # Comparison tabs
    tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "Feature Importance", "Learning Curves"])
    
    with tab1:
        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            model_metrics = metrics[model_name]
            
            # Extract relevant metrics
            model_row = {'Model': model_name}
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
            if 'accuracy' in comparison_df.columns:
                # Classification metrics
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
                metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
            else:
                # Regression metrics
                metrics_to_plot = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
            
            if metrics_to_plot:
                # Prepare data for plotting
                plot_data = comparison_df.melt(
                    id_vars=['Model'], 
                    value_vars=metrics_to_plot,
                    var_name='Metric', 
                    value_name='Value'
                )
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=plot_data, x='Model', y='Value', hue='Metric', ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab2:
        st.subheader("Feature Importance")
        
        selected_model = st.selectbox(
            "Select model for feature importance",
            options=model_names
        )
        
        if selected_model in st.session_state.models:
            model = st.session_state.models[selected_model]
            
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                # Get feature importances
                importances = model.feature_importances_
                feature_names = st.session_state.features
                
                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Display table
                st.dataframe(importance_df)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("This model doesn't provide feature importance information")
    
    with tab3:
        st.subheader("Learning Curves")
        st.info("This feature will be available in a future update")

def prediction_page():
    """Prediction page"""
    st.title("Make Predictions")
    
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
        index=model_names.index(st.session_state.best_model["name"]) if st.session_state.best_model else 0
    )
    
    # Input method
    input_method = st.radio(
        "Input method",
        ["Upload new data", "Manual input"]
    )
    
    if input_method == "Upload new data":
        # File upload
        uploaded_file = st.file_uploader("Upload data for prediction (CSV, Excel)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Try to determine file type from extension
                if uploaded_file.name.endswith('.csv'):
                    pred_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
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
                if st.button("Make Predictions"):
                    model = st.session_state.models[selected_model]
                    
                    # Get engine from session state
                    engine = st.session_state.engine
                    
                    # Make predictions
                    predictions = engine.predict(model, pred_data)
                    
                    # Display predictions
                    st.subheader("Predictions")
                    
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    
                    # Create dataframe with predictions
                    pred_df = pd.DataFrame({
                        'Prediction': predictions
                    })
                    
                    # Display predictions
                    st.dataframe(pred_df)
                    
                    # Add download button
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:  # Manual input
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
                            step=(max_val - min_val) / 100
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature,
                            value=mean_val
                        )
                else:
                    # For categorical features
                    unique_values = st.session_state.data[feature].unique().tolist()
                    input_values[feature] = st.selectbox(
                        feature,
                        options=unique_values,
                        index=0
                    )
            
            # Submit button
            predict_button = st.form_submit_button("Make Prediction")
        
        if predict_button:
            try:
                # Create dataframe from input values
                input_df = pd.DataFrame([input_values])
                
                # Get model
                model = st.session_state.models[selected_model]
                
                # Get engine from session state
                engine = st.session_state.engine
                
                # Make prediction
                prediction = engine.predict(model, input_df)
                
                # Display prediction
                st.subheader("Prediction Result")
                
                # Format the prediction based on task type
                if engine.config.task_type == TaskType.CLASSIFICATION:
                    # For classification, show probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_df)
                        
                        # Display prediction and probability
                        st.write(f"Predicted class: **{prediction[0]}**")
                        
                        # Display class probabilities
                        proba_df = pd.DataFrame(
                            proba[0],
                            index=model.classes_,
                            columns=['Probability']
                        )
                        
                        # Sort by probability
                        proba_df = proba_df.sort_values('Probability', ascending=False)
                        
                        # Display as bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=proba_df.index, y=proba_df['Probability'], ax=ax)
                        plt.title("Class Probabilities")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.write(f"Predicted class: **{prediction[0]}**")
                else:
                    # For regression, just show the value
                    st.write(f"Predicted value: **{prediction[0]:.4f}**")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def export_page():
    """Export model page"""
    st.title("Export Model")
    
    if not st.session_state.training_completed:
        st.warning("Please train models first")
        return
    
    if not st.session_state.models:
        st.warning("No trained models found")
        return
    
    # Get model names
    model_names = list(st.session_state.models.keys())
    
    # Select model for export
    selected_model = st.selectbox(
        "Select model to export",
        options=model_names,
        index=model_names.index(st.session_state.best_model["name"]) if st.session_state.best_model else 0
    )
    
    # Export options
    st.subheader("Export Options")
    
    export_format = st.radio(
        "Export format",
        ["Joblib", "Pickle", "ONNX", "PMML"]
    )
    
    model_filename = st.text_input(
        "Filename",
        value=f"{selected_model.lower().replace(' ', '_')}"
    )
    
    # Export button
    if st.button("Export Model"):
        try:
            # Get model
            model = st.session_state.models[selected_model]
            
            # Create export directory if it doesn't exist
            export_dir = "./exported_models"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Export based on selected format
            if export_format == "Joblib":
                # Export using joblib
                filepath = f"{export_dir}/{model_filename}.joblib"
                joblib.dump(model, filepath)
                st.success(f"Model exported to {filepath}")
                
                # Provide download button
                with open(filepath, "rb") as f:
                    model_bytes = f.read()
                    st.download_button(
                        label="Download Model",
                        data=model_bytes,
                        file_name=f"{model_filename}.joblib",
                        mime="application/octet-stream"
                    )
            
            elif export_format == "Pickle":
                # Export using pickle
                import pickle
                filepath = f"{export_dir}/{model_filename}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)
                st.success(f"Model exported to {filepath}")
                
                # Provide download button
                with open(filepath, "rb") as f:
                    model_bytes = f.read()
                    st.download_button(
                        label="Download Model",
                        data=model_bytes,
                        file_name=f"{model_filename}.pkl",
                        mime="application/octet-stream"
                    )
            
            elif export_format == "ONNX":
                st.info("ONNX export will be available in a future update")
            
            elif export_format == "PMML":
                st.info("PMML export will be available in a future update")
            
            # Generate sample code for model usage
            st.subheader("Sample Code")
            
            if export_format == "Joblib":
                sample_code = f"""
                ```python
                # Sample code to load and use the exported model
                import joblib
                import pandas as pd
                
                # Load the model
                model = joblib.load("{model_filename}.joblib")
                
                # Prepare your data (example)
                # Replace these with your actual feature names and values
                data = {{
                    {", ".join([f"'{feature}': [value]" for feature in st.session_state.features])}
                }}
                
                # Create DataFrame
                X = pd.DataFrame(data)
                
                # Make predictions
                predictions = model.predict(X)
                print(f"Predictions: {{predictions}}")
                ```
                """
                st.markdown(sample_code)
            
            elif export_format == "Pickle":
                sample_code = f"""
                ```python
                # Sample code to load and use the exported model
                import pickle
                import pandas as pd
                
                # Load the model
                with open("{model_filename}.pkl", "rb") as f:
                    model = pickle.load(f)
                
                # Prepare your data (example)
                # Replace these with your actual feature names and values
                data = {{
                    {", ".join([f"'{feature}': [value]" for feature in st.session_state.features])}
                }}
                
                # Create DataFrame
                X = pd.DataFrame(data)
                
                # Make predictions
                predictions = model.predict(X)
                print(f"Predictions: {{predictions}}")
                ```
                """
                st.markdown(sample_code)
        
        except Exception as e:
            st.error(f"Error exporting model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Export experiment results
    st.subheader("Export Experiment Results")
    
    if st.button("Export All Experiment Results"):
        try:
            # Create export directory if it doesn't exist
            export_dir = "./experiment_results"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Export experiment results as JSON
            filepath = f"{export_dir}/experiment_results.json"
            with open(filepath, "w") as f:
                json.dump(st.session_state.experiment_results, f, indent=2)
            
            st.success(f"Experiment results exported to {filepath}")
            
            # Provide download button
            with open(filepath, "r") as f:
                results_json = f.read()
                st.download_button(
                    label="Download Experiment Results",
                    data=results_json,
                    file_name="experiment_results.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"Error exporting experiment results: {str(e)}")

def main():
    """Main function"""
    # Create sidebar navigation
    navigation = create_sidebar()
    
    # Navigate to the selected page
    if navigation == "Data Upload & Exploration":
        data_upload_page()
    elif navigation == "Training Configuration":
        training_configuration_page()
    elif navigation == "Model Training":
        model_training_page()
    elif navigation == "Model Evaluation":
        model_evaluation_page()
    elif navigation == "Prediction":
        prediction_page()
    elif navigation == "Export":
        export_page()

if __name__ == "__main__":
    main()
