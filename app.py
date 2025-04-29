import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import os
from io import StringIO
import base64
from datetime import datetime

# API Base URL - update with your actual API endpoint
API_BASE_URL = "http://localhost:8000/api"

# Configure page
st.set_page_config(
    page_title="kolosal AutoML Dashboard",
    page_icon="assets\kolosal-logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "data" not in st.session_state:
    st.session_state.data = None
if "target" not in st.session_state:
    st.session_state.target = None
if "features" not in st.session_state:
    st.session_state.features = []
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "training_status" not in st.session_state:
    st.session_state.training_status = None
if "training_task_id" not in st.session_state:
    st.session_state.training_task_id = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = []
if "preprocessor_id" not in st.session_state:
    st.session_state.preprocessor_id = None
if "device_optimized" not in st.session_state:
    st.session_state.device_optimized = False
if "api_key" not in st.session_state:
    st.session_state.api_key = "dev_key"  # Default API key

# Function to make API calls
def api_call(endpoint, method="GET", data=None, files=None, params=None):
    """
    Make an API call to the specified endpoint
    
    Args:
        endpoint: API endpoint to call
        method: HTTP method (GET, POST, PUT, DELETE)
        data: Data to send in the request body
        files: Files to send in the request
        params: Query parameters
        
    Returns:
        Response from the API
    """
    url = f"{API_BASE_URL}/{endpoint}"
    headers = {"X-API-Key": st.session_state.api_key}
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, files=files, params=params)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        # Check if response is successful
        response.raise_for_status()
        
        # Return parsed JSON data if present
        if response.content:
            return response.json()
        return {"success": True}
        
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('detail', str(e))
                st.error(f"Error details: {error_detail}")
            except:
                st.error(f"Status code: {e.response.status_code}")
        return {"success": False, "error": str(e)}

# Health Check Function
def check_api_health():
    """Check if the API is healthy"""
    try:
        response = api_call("health")
        if response.get("status") == "healthy":
            return True
        return False
    except:
        return False

# Sidebar API connection and settings
with st.sidebar:
    st.image("https://via.placeholder.com/150x70?text=kolosal.ai", width=150)
    st.title("kolosal AutoML")
    
    # API Connection
    st.header("API Connection")
    api_url = st.text_input("API URL", value=API_BASE_URL)
    API_BASE_URL = api_url
    
    # API Key input
    st.session_state.api_key = st.text_input("API Key", value="dev_key", type="password")
    
    # Check connection
    if st.button("Check Connection"):
        if check_api_health():
            st.success("Connected to API!")
        else:
            st.error("Failed to connect to API!")
    
    # Navigation
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Home", "Data Management", "Device Optimization", "Model Training", "Inference", "Model Management"]
    )
    
    # Show available models if on inference page
    if page == "Inference" and st.session_state.trained_models:
        st.subheader("Available Models")
        for model in st.session_state.trained_models:
            st.write(f"- {model}")
    
    # Settings
    with st.expander("Settings"):
        st.checkbox("Enable dark mode", value=False, key="dark_mode")
        st.checkbox("Auto refresh", value=True, key="auto_refresh")
        refresh_interval = st.slider("Refresh interval (s)", min_value=5, max_value=60, value=10)
    
    # About
    with st.expander("About"):
        st.write("""
        **kolosal AutoML Dashboard**
        
        Version: 1.0.0
        
        A comprehensive UI for interacting with the kolosal AutoML API suite:
        - Data preprocessing
        - Device optimization
        - Model training
        - Inference
        - Model management
        """)

# --- Home Page ---
def home_page():
    st.title("kolosal AutoML Dashboard")
    
    # Check API connection
    api_status = check_api_health()
    
    # Display API status
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("API Status")
        if api_status:
            st.success("API is operational ✅")
            
            # Display API version and components
            health_data = api_call("health")
            if health_data and "components" in health_data:
                st.write(f"API Version: {health_data.get('version', 'Unknown')}")
                st.write(f"Environment: {health_data.get('environment', 'Unknown')}")
                
                # Display component status
                st.subheader("Component Status")
                components = health_data.get("components", {})
                for component, status in components.items():
                    if status == "healthy":
                        st.write(f"✅ {component.capitalize()}")
                    else:
                        st.write(f"❌ {component.capitalize()}")
        else:
            st.error("API is unavailable ❌")
            st.write("Check API connection settings in the sidebar.")
    
    with col2:
        # Quick Stats and Usage
        st.subheader("System Status")
        
        # Only show metrics if API is available
        if api_status:
            try:
                metrics_data = api_call("metrics", method="GET")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Requests", metrics_data.get("total_requests", 0))
                with col_b:
                    st.metric("Active Connections", metrics_data.get("active_connections", 0))
                with col_c:
                    uptime = metrics_data.get("uptime_seconds", 0)
                    uptime_str = f"{int(uptime/3600):d}h {int((uptime%3600)/60):d}m"
                    st.metric("Uptime", uptime_str)
            except:
                st.write("Could not retrieve metrics")
        else:
            st.write("Connect to the API to view system metrics")
    
    # Dashboard Overview
    st.subheader("Dashboard Overview")
    
    # Create four columns for quick links
    cols = st.columns(4)
    
    with cols[0]:
        st.subheader("Data Management")
        if st.session_state.data is not None:
            st.write(f"Data loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        else:
            st.write("No data loaded")
        if st.button("Go to Data Management", key="goto_data"):
            st.session_state.page = "Data Management"
            st.rerun()
    
    with cols[1]:
        st.subheader("Device Optimization")
        if st.session_state.device_optimized:
            st.write("Device optimization completed")
        else:
            st.write("Device not optimized")
        if st.button("Go to Device Optimization", key="goto_device"):
            st.session_state.page = "Device Optimization"
            st.rerun()
    
    with cols[2]:
        st.subheader("Model Training")
        if st.session_state.training_status:
            st.write(f"Training status: {st.session_state.training_status}")
        else:
            st.write("No training in progress")
        if st.button("Go to Model Training", key="goto_training"):
            st.session_state.page = "Model Training"
            st.rerun()
    
    with cols[3]:
        st.subheader("Model Inference")
        if st.session_state.trained_models:
            st.write(f"{len(st.session_state.trained_models)} models available")
        else:
            st.write("No models available")
        if st.button("Go to Inference", key="goto_inference"):
            st.session_state.page = "Inference"
            st.rerun()
    
    # Recent Activity
    st.subheader("Recent Activity")
    
    # Placeholder for recent activity - in a real app, this would be populated from API data
    if api_status:
        st.write("Recent activity will be displayed here")
        # Example placeholder data
        activity_data = [
            {"timestamp": "2025-04-28 10:30:22", "action": "Model trained", "details": "RandomForestClassifier with accuracy 0.92"},
            {"timestamp": "2025-04-28 09:45:11", "action": "Data preprocessed", "details": "StandardScaler applied to 15 features"},
            {"timestamp": "2025-04-27 16:22:05", "action": "Device optimized", "details": "Balanced mode, 8 cores detected"}
        ]
        
        for activity in activity_data:
            st.write(f"**{activity['timestamp']}**: {activity['action']} - {activity['details']}")
    else:
        st.write("Connect to the API to view recent activity")

# --- Data Management Page ---
def data_management_page():
    st.title("Data Management")
    
    # Create tabs for data upload, preprocessing, and exploration
    tab1, tab2, tab3 = st.tabs(["Data Upload", "Data Preprocessing", "Data Exploration"])
    
    with tab1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        # Sample data option
        if st.button("Load Sample Data"):
            # Use sklearn sample dataset
            try:
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer(as_frame=True)
                st.session_state.data = data.data
                st.session_state.data['target'] = data.target
                st.session_state.features = data.feature_names.tolist()
                st.session_state.target = 'target'
                st.success("Sample data loaded! (Breast Cancer dataset)")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        
        # Process uploaded file
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
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                
                # Display data preview
                st.write("Data Preview:")
                st.dataframe(data.head())
                
                # Select target column
                st.subheader("Select Target Column")
                target_col = st.selectbox("Choose target column for predictions", options=data.columns)
                
                if st.button("Set Target"):
                    st.session_state.target = target_col
                    st.session_state.features = [col for col in data.columns if col != target_col]
                    st.success(f"Target column set to: {target_col}")
                    
                    # Analyze target column
                    if pd.api.types.is_numeric_dtype(data[target_col]):
                        st.write("Target is numeric - likely a regression problem")
                    else:
                        unique_values = data[target_col].nunique()
                        if unique_values < 10:
                            st.write(f"Target has {unique_values} unique values - likely a classification problem")
                        else:
                            st.write(f"Target has {unique_values} unique values - might be a multi-class problem")
                
                # Upload data to API (optional)
                if st.button("Upload to API") and st.session_state.target:
                    # Convert data to CSV
                    csv_data = st.session_state.data.to_csv(index=False)
                    files = {"csv_file": ("data.csv", csv_data, "text/csv")}
                    
                    # Upload to API
                    response = api_call("preprocessor/preprocessors", method="POST")
                    if response and "preprocessor_id" in response:
                        preprocessor_id = response["preprocessor_id"]
                        st.session_state.preprocessor_id = preprocessor_id
                        st.success(f"Created preprocessor with ID: {preprocessor_id}")
                        
                        # Upload data for fitting
                        fit_response = api_call(
                            f"preprocessor/preprocessors/{preprocessor_id}/fit",
                            method="POST",
                            files=files,
                            params={"has_header": True}
                        )
                        
                        if fit_response and fit_response.get("success"):
                            st.success("Data uploaded and preprocessor fitted successfully!")
                        else:
                            st.error("Failed to fit preprocessor")
                    else:
                        st.error("Failed to create preprocessor")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.subheader("Data Preprocessing")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        # Check if preprocessor exists
        if st.session_state.preprocessor_id:
            st.success(f"Using preprocessor: {st.session_state.preprocessor_id}")
        else:
            # Create a new preprocessor
            if st.button("Create Preprocessor"):
                response = api_call("preprocessor/preprocessors", method="POST")
                if response and "preprocessor_id" in response:
                    st.session_state.preprocessor_id = response["preprocessor_id"]
                    st.success(f"Created preprocessor with ID: {st.session_state.preprocessor_id}")
                else:
                    st.error("Failed to create preprocessor")
        
        # Create preprocessing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Preprocessing Configuration")
            
            # Normalization options
            normalize = st.checkbox("Apply normalization", value=True)
            
            if normalize:
                normalization_type = st.selectbox(
                    "Normalization type",
                    options=["STANDARD", "MINMAX", "ROBUST", "NONE"]
                )
            
            # Handle missing values
            handle_nan = st.checkbox("Handle missing values", value=True)
            
            if handle_nan:
                nan_strategy = st.selectbox(
                    "Strategy for missing values",
                    options=["mean", "median", "most_frequent", "constant"]
                )
            
            # Outlier detection
            detect_outliers = st.checkbox("Detect outliers", value=False)
            
            if detect_outliers:
                outlier_method = st.selectbox(
                    "Outlier detection method",
                    options=["IQR", "ZSCORE", "PERCENTILE"]
                )
        
        with col2:
            st.subheader("Feature Selection")
            
            # Feature selection options
            enable_feature_selection = st.checkbox("Enable feature selection", value=False)
            
            if enable_feature_selection:
                feature_selection_method = st.selectbox(
                    "Feature selection method",
                    options=["mutual_info", "chi2", "f_classif", "f_regression"]
                )
                
                max_features = len(st.session_state.features)
                k_features = st.slider("Number of features to select", 1, max_features, max_features//2)
            
            # Additional options
            st.subheader("Additional Options")
            
            # Categorical encoding
            categorical_encoding = st.selectbox(
                "Categorical encoding",
                options=["one_hot", "label", "target", "frequency"]
            )
        
        # Update preprocessor config if it exists
        if st.session_state.preprocessor_id and st.button("Apply Preprocessing Configuration"):
            # Build config
            config = {
                "normalization": normalization_type if normalize else "NONE",
                "handle_nan": handle_nan,
                "detect_outliers": detect_outliers
            }
            
            if handle_nan:
                config["nan_strategy"] = nan_strategy
            
            if detect_outliers:
                config["outlier_method"] = outlier_method
            
            # Update config
            response = api_call(
                f"preprocessor/preprocessors/{st.session_state.preprocessor_id}/update-config",
                method="POST",
                data=config
            )
            
            if response and response.get("success"):
                st.success("Preprocessor configuration updated!")
            else:
                st.error("Failed to update preprocessor configuration")
        
        # Apply preprocessing to data
        if st.session_state.preprocessor_id and st.button("Transform Data"):
            # Convert data to CSV
            csv_data = st.session_state.data.to_csv(index=False)
            files = {"csv_file": ("data.csv", csv_data, "text/csv")}
            
            # Send data for transformation
            response = api_call(
                f"preprocessor/preprocessors/{st.session_state.preprocessor_id}/transform",
                method="POST",
                files=files,
                params={"has_header": True, "output_format": "json"}
            )
            
            if response and "transformed_data" in response:
                # Convert transformed data to DataFrame
                transformed_data = pd.DataFrame(
                    response["transformed_data"],
                    columns=st.session_state.features
                )
                
                # Add target column back
                if st.session_state.target:
                    transformed_data[st.session_state.target] = st.session_state.data[st.session_state.target]
                
                # Update session state
                st.session_state.data = transformed_data
                st.success("Data transformed successfully!")
                
                # Show transformed data
                st.write("Transformed Data Preview:")
                st.dataframe(transformed_data.head())
            else:
                st.error("Failed to transform data")
    
    with tab3:
        st.subheader("Data Exploration")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        data = st.session_state.data
        
        # Data summary
        st.write("Data Summary:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isna().sum().sum())
        with col4:
            st.metric("Duplicated Rows", data.duplicated().sum())
        
        # Data types and basic stats
        st.write("Data Types and Basic Statistics:")
        st.dataframe(data.dtypes.rename("Data Type"))
        st.dataframe(data.describe())
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        numeric_cols = data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                square=True,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation analysis")
        
        # Column visualizations
        st.subheader("Column Visualization")
        
        col_to_visualize = st.selectbox("Select column to visualize", options=data.columns)
        
        if col_to_visualize:
            col_data = data[col_to_visualize]
            
            # Choose visualization type based on data type
            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric column - histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=data, x=col_to_visualize, kde=True, ax=ax)
                st.pyplot(fig)
                
                # Show basic stats
                st.write(f"Mean: {col_data.mean():.4f}")
                st.write(f"Median: {col_data.median():.4f}")
                st.write(f"Min: {col_data.min():.4f}")
                st.write(f"Max: {col_data.max():.4f}")
                st.write(f"Standard Deviation: {col_data.std():.4f}")
            else:
                # Categorical column - bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = col_data.value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
                
                # Show counts
                st.write(f"Unique values: {col_data.nunique()}")
                st.dataframe(value_counts)
        
        # Pairplot for selected features
        st.subheader("Feature Pairplot")
        
        # Select columns for pairplot (limit to prevent overload)
        max_pairplot_cols = 5
        available_cols = list(numeric_cols)
        if st.session_state.target and pd.api.types.is_numeric_dtype(data[st.session_state.target]):
            available_cols.append(st.session_state.target)
        
        selected_cols = st.multiselect(
            "Select columns for pairplot (max 5)",
            options=available_cols,
            default=available_cols[:min(3, len(available_cols))]
        )
        
        if len(selected_cols) > 1 and len(selected_cols) <= max_pairplot_cols:
            fig = sns.pairplot(data[selected_cols], diag_kind="kde")
            st.pyplot(fig)
        elif len(selected_cols) > max_pairplot_cols:
            st.warning(f"Please select at most {max_pairplot_cols} columns for pairplot")

# --- Device Optimization Page ---
def device_optimization_page():
    st.title("Device Optimization")
    
    st.write("""
    Device optimization analyzes your system's capabilities and creates optimized configurations 
    for data preprocessing, model training, batch processing, and inference.
    """)
    
    # Create tabs for different optimization aspects
    tab1, tab2, tab3 = st.tabs(["System Information", "Optimization", "Configurations"])
    
    with tab1:
        st.subheader("System Information")
        
        # Fetch system information from the API
        if st.button("Detect System Information"):
            response = api_call("device/system-info")
            
            if response:
                st.success("System information detected!")
                
                # System overview
                st.subheader("System Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("System", response.get("system", "Unknown"))
                    st.metric("Processor", response.get("processor", "Unknown"))
                with col2:
                    st.metric("Environment", response.get("environment", "Unknown"))
                    st.metric("Machine", response.get("machine", "Unknown"))
                with col3:
                    st.metric("Hostname", response.get("hostname", "Unknown"))
                    st.metric("Python Version", response.get("python_version", "Unknown"))
                
                # CPU Information
                st.subheader("CPU Information")
                
                cpu_info = response.get("cpu", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Physical Cores", cpu_info.get("count_physical", "Unknown"))
                    st.metric("Logical Cores", cpu_info.get("count_logical", "Unknown"))
                    
                    # CPU Vendor
                    vendor_info = cpu_info.get("vendor", {})
                    vendor = "Unknown"
                    if vendor_info.get("intel"):
                        vendor = "Intel"
                    elif vendor_info.get("amd"):
                        vendor = "AMD"
                    elif vendor_info.get("arm"):
                        vendor = "ARM"
                    
                    st.metric("CPU Vendor", vendor)
                    
                with col2:
                    # CPU Features
                    features = cpu_info.get("features", {})
                    
                    feature_text = ""
                    for feature, has_feature in features.items():
                        icon = "✅" if has_feature else "❌"
                        feature_text += f"{icon} {feature.upper()}\n"
                    
                    st.text(f"CPU Features:\n{feature_text}")
                    
                    # CPU Frequency
                    freq = cpu_info.get("frequency", {})
                    if freq:
                        st.metric("CPU Frequency", f"{freq.get('current', 0):.2f} MHz")
                
                # Memory Information
                st.subheader("Memory Information")
                
                memory_info = response.get("memory", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Memory", f"{memory_info.get('total_gb', 0):.2f} GB")
                    st.metric("Available Memory", f"{memory_info.get('available_gb', 0):.2f} GB")
                with col2:
                    st.metric("Usable Memory", f"{memory_info.get('usable_gb', 0):.2f} GB")
                    st.metric("Swap Memory", f"{memory_info.get('swap_gb', 0):.2f} GB")
                
                # Disk Information
                st.subheader("Disk Information")
                
                disk_info = response.get("disk", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Disk Space", f"{disk_info.get('total_gb', 0):.2f} GB")
                    st.metric("Free Disk Space", f"{disk_info.get('free_gb', 0):.2f} GB")
                with col2:
                    st.metric("SSD", "Yes" if disk_info.get("is_ssd", False) else "No")
                
                # Hardware Accelerators
                st.subheader("Hardware Accelerators")
                
                accelerators = response.get("accelerators", [])
                
                if accelerators:
                    for acc in accelerators:
                        st.write(f"✅ {acc}")
                else:
                    st.write("❌ No specialized hardware accelerators detected")
            else:
                st.error("Failed to retrieve system information")
    
    with tab2:
        st.subheader("Device Optimization")
        
        # Optimization mode selection
        st.write("Select optimization mode based on your needs:")
        
        optimization_mode = st.radio(
            "Optimization Mode",
            options=[
                "BALANCED", 
                "PERFORMANCE", 
                "MEMORY_SAVING", 
                "FULL_UTILIZATION", 
                "CONSERVATIVE"
            ],
            index=0,
            help="BALANCED: Balances performance and resource usage (recommended for most cases)"
        )
        
        # Additional optimization settings
        col1, col2 = st.columns(2)
        
        with col1:
            workload_type = st.selectbox(
                "Workload Type",
                options=["mixed", "training", "inference"],
                index=0,
                help="Type of workload you plan to run"
            )
            
            environment = st.selectbox(
                "Environment",
                options=["auto", "cloud", "desktop", "edge"],
                index=0,
                help="Computing environment where the system will run"
            )
        
        with col2:
            enable_specialized_accelerators = st.checkbox(
                "Enable Specialized Accelerators",
                value=True,
                help="Use specialized hardware if available"
            )
            
            memory_reservation_percent = st.slider(
                "Memory Reservation (%)",
                min_value=5,
                max_value=50,
                value=10,
                help="Percentage of memory to reserve for the system"
            )
            
            power_efficiency = st.checkbox(
                "Optimize for Power Efficiency",
                value=False,
                help="Prioritize power efficiency over performance"
            )
        
        resilience_level = st.slider(
            "Resilience Level",
            min_value=0,
            max_value=3,
            value=1,
            help="Level of fault tolerance (0-3)"
        )
        
        auto_tune = st.checkbox(
            "Enable Auto-Tuning",
            value=True,
            help="Automatically tune parameters based on system load"
        )
        
        # Run optimization
        if st.button("Run Device Optimization"):
            with st.spinner("Optimizing for your device..."):
                # Prepare optimization request
                optimize_request = {
                    "optimization_mode": optimization_mode,
                    "workload_type": workload_type,
                    "environment": environment,
                    "enable_specialized_accelerators": enable_specialized_accelerators,
                    "memory_reservation_percent": memory_reservation_percent,
                    "power_efficiency": power_efficiency,
                    "resilience_level": resilience_level,
                    "auto_tune": auto_tune
                }
                
                # Call the API
                response = api_call("device/optimize", method="POST", data=optimize_request)
                
                if response and response.get("status") == "success":
                    st.session_state.device_optimized = True
                    st.session_state.config_id = response.get("config_id")
                    st.session_state.master_config = response.get("master_config")
                    
                    st.success("Device optimization completed successfully!")
                    st.write(f"Configuration ID: {st.session_state.config_id}")
                else:
                    st.error("Device optimization failed")
        
        # Generate configurations for all modes
        if st.button("Generate Configs for All Modes"):
            with st.spinner("Generating configurations for all optimization modes..."):
                # Prepare request
                all_modes_request = {
                    "workload_type": workload_type,
                    "environment": environment,
                    "enable_specialized_accelerators": enable_specialized_accelerators,
                    "memory_reservation_percent": memory_reservation_percent,
                    "power_efficiency": power_efficiency,
                    "resilience_level": resilience_level
                }
                
                # Call the API
                response = api_call("device/optimize/all-modes", method="POST", data=all_modes_request)
                
                if response and response.get("status") == "success":
                    st.session_state.all_modes_configs = response.get("configs")
                    st.success("Generated configurations for all optimization modes!")
                else:
                    st.error("Failed to generate configurations for all modes")
    
    with tab3:
        st.subheader("Optimization Configurations")
        
        # Check if device has been optimized
        if not st.session_state.device_optimized:
            st.warning("Please run device optimization first")
            return
        
        # Display available configurations
        if hasattr(st.session_state, "config_id") and hasattr(st.session_state, "master_config"):
            st.write(f"Current configuration ID: {st.session_state.config_id}")
            
            # Create tabs for different configuration components
            config_tabs = st.tabs([
                "Quantization",
                "Batch Processing",
                "Preprocessing",
                "Inference",
                "Training"
            ])
            
            # Load configurations
            config_response = api_call(f"device/configs/load", method="POST", data={
                "config_path": "./configs",
                "config_id": st.session_state.config_id
            })
            
            if config_response and "configs" in config_response:
                configs = config_response.get("configs", {})
                
                # Quantization Config
                with config_tabs[0]:
                    st.subheader("Quantization Configuration")
                    if "quantization_config" in configs:
                        quant_config = configs["quantization_config"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Quantization Type", quant_config.get("quantization_type", "Unknown"))
                            st.metric("Quantization Mode", quant_config.get("quantization_mode", "Unknown"))
                            st.metric("Number of Bits", quant_config.get("num_bits", 8))
                        
                        with col2:
                            st.metric("Symmetric", "Yes" if quant_config.get("symmetric", False) else "No")
                            st.metric("Per Channel", "Yes" if quant_config.get("per_channel", False) else "No")
                            st.metric("Cache Size", quant_config.get("cache_size", 0))
                        
                        # Show full config as JSON
                        with st.expander("View full configuration"):
                            st.json(quant_config)
                    else:
                        st.warning("Quantization configuration not available")
                
                # Batch Processing Config
                with config_tabs[1]:
                    st.subheader("Batch Processing Configuration")
                    if "batch_processor_config" in configs:
                        batch_config = configs["batch_processor_config"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Initial Batch Size", batch_config.get("initial_batch_size", 0))
                            st.metric("Min Batch Size", batch_config.get("min_batch_size", 0))
                            st.metric("Max Batch Size", batch_config.get("max_batch_size", 0))
                        
                        with col2:
                            st.metric("Batch Timeout", f"{batch_config.get('batch_timeout', 0):.2f}s")
                            st.metric("Workers", batch_config.get("num_workers", 0))
                            st.metric("Processing Strategy", batch_config.get("processing_strategy", "Unknown"))
                        
                        # Show full config as JSON
                        with st.expander("View full configuration"):
                            st.json(batch_config)
                    else:
                        st.warning("Batch processing configuration not available")
                
                # Preprocessing Config
                with config_tabs[2]:
                    st.subheader("Preprocessing Configuration")
                    if "preprocessor_config" in configs:
                        preproc_config = configs["preprocessor_config"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Normalization", preproc_config.get("normalization", "Unknown"))
                            st.metric("Handle NaN", "Yes" if preproc_config.get("handle_nan", False) else "No")
                            st.metric("Handle Inf", "Yes" if preproc_config.get("handle_inf", False) else "No")
                        
                        with col2:
                            st.metric("Detect Outliers", "Yes" if preproc_config.get("detect_outliers", False) else "No")
                            st.metric("Parallel Processing", "Yes" if preproc_config.get("parallel_processing", False) else "No")
                            st.metric("NaN Strategy", preproc_config.get("nan_strategy", "Unknown"))
                        
                        # Show full config as JSON
                        with st.expander("View full configuration"):
                            st.json(preproc_config)
                    else:
                        st.warning("Preprocessing configuration not available")
                
                # Inference Config
                with config_tabs[3]:
                    st.subheader("Inference Engine Configuration")
                    if "inference_engine_config" in configs:
                        inference_config = configs["inference_engine_config"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Threads", inference_config.get("num_threads", 0))
                            st.metric("Enable Batching", "Yes" if inference_config.get("enable_batching", False) else "No")
                            st.metric("Max Batch Size", inference_config.get("max_batch_size", 0))
                        
                        with col2:
                            st.metric("Intel Optimization", "Yes" if inference_config.get("enable_intel_optimization", False) else "No")
                            st.metric("Quantization", "Yes" if inference_config.get("enable_quantization", False) else "No")
                            st.metric("Cache Entries", inference_config.get("max_cache_entries", 0))
                        
                        # Show full config as JSON
                        with st.expander("View full configuration"):
                            st.json(inference_config)
                    else:
                        st.warning("Inference configuration not available")
                
                # Training Config
                with config_tabs[4]:
                    st.subheader("Training Engine Configuration")
                    if "training_engine_config" in configs:
                        training_config = configs["training_engine_config"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Optimization Strategy", training_config.get("optimization_strategy", "Unknown"))
                            st.metric("Iterations", training_config.get("optimization_iterations", 0))
                            st.metric("Early Stopping", "Yes" if training_config.get("early_stopping", False) else "No")
                        
                        with col2:
                            st.metric("Feature Selection", "Yes" if training_config.get("feature_selection", False) else "No")
                            st.metric("CV Folds", training_config.get("cv_folds", 0))
                            st.metric("Test Size", f"{training_config.get('test_size', 0):.2f}")
                        
                        # Show full config as JSON
                        with st.expander("View full configuration"):
                            st.json(training_config)
                    else:
                        st.warning("Training configuration not available")
            else:
                st.error("Failed to load configurations")
        
        # Display configs for all modes if available
        if hasattr(st.session_state, "all_modes_configs") and st.session_state.all_modes_configs:
            st.subheader("All Optimization Modes")
            
            # Create expandable sections for each mode
            for mode, config in st.session_state.all_modes_configs.items():
                with st.expander(f"Mode: {mode}"):
                    st.write(f"Configuration ID: {config.get('config_id', 'Unknown')}")
                    st.write(f"Created: {config.get('creation_timestamp', 'Unknown')}")
                    
                    # Show paths to config files
                    st.subheader("Configuration Files")
                    st.write(f"Quantization: {config.get('quantization_config_path', 'N/A')}")
                    st.write(f"Batch Processing: {config.get('batch_processor_config_path', 'N/A')}")
                    st.write(f"Preprocessing: {config.get('preprocessor_config_path', 'N/A')}")
                    st.write(f"Inference: {config.get('inference_engine_config_path', 'N/A')}")
                    st.write(f"Training: {config.get('training_engine_config_path', 'N/A')}")
                    
                    # Button to apply this configuration
                    if st.button(f"Apply {mode} Configuration", key=f"apply_{mode}"):
                        st.session_state.config_id = config.get("config_id")
                        st.session_state.master_config = config
                        st.success(f"Switched to {mode} configuration")

# --- Model Training Page ---
def model_training_page():
    st.title("Model Training")
    
    if st.session_state.data is None or st.session_state.target is None:
        st.warning("Please upload data and select a target column first in the Data Management page")
        return
    
    # Set up tabs for training flow
    tab1, tab2, tab3 = st.tabs(["Training Setup", "Training Execution", "Results & Evaluation"])
    
    with tab1:
        st.subheader("Training Configuration")
        
        # Task type selection
        task_type = st.radio(
            "Task Type",
            options=["classification", "regression"],
            horizontal=True
        )
        
        # Data information
        st.write(f"Data: {st.session_state.data.shape[0]} samples, {len(st.session_state.features)} features")
        st.write(f"Target column: {st.session_state.target}")
        
        # Model selection
        model_options = {
            "classification": [
                "LogisticRegression",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
                "SVC"
            ],
            "regression": [
                "LinearRegression",
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "XGBRegressor",
                "LGBMRegressor",
                "SVR"
            ]
        }
        
        selected_models = st.multiselect(
            "Select Models to Train",
            options=model_options[task_type],
            default=[model_options[task_type][0], model_options[task_type][1]]
        )
        
        # Create two columns for settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic training settings
            st.subheader("Training Settings")
            
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Portion of data to use for testing"
            )
            
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=9999,
                value=42,
                help="Random seed for reproducibility"
            )
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Number of folds for cross-validation"
            )
            
            optimization_strategy = st.selectbox(
                "Optimization Strategy",
                options=[
                    "random_search",
                    "grid_search",
                    "bayesian_optimization",
                    "optuna",
                    "hyperx"
                ],
                index=0,
                help="Strategy for hyperparameter optimization"
            )
            
            optimization_iterations = st.slider(
                "Optimization Iterations",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Number of iterations for optimization"
            )
        
        with col2:
            # Advanced training settings
            st.subheader("Advanced Settings")
            
            enable_feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=True,
                help="Automatically select important features"
            )
            
            if enable_feature_selection:
                feature_selection_method = st.selectbox(
                    "Feature Selection Method",
                    options=["mutual_info", "chi2", "f_classif", "f_regression"],
                    index=0
                )
                
                feature_selection_k = st.slider(
                    "Number of Features to Select",
                    min_value=1,
                    max_value=len(st.session_state.features),
                    value=min(10, len(st.session_state.features))
                )
            
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=True,
                help="Stop training when performance doesn't improve"
            )
            
            if early_stopping:
                early_stopping_rounds = st.slider(
                    "Early Stopping Rounds",
                    min_value=5,
                    max_value=50,
                    value=10
                )
            
            # Model selection criteria
            if task_type == "classification":
                selection_criteria_options = [
                    "accuracy", "f1", "precision", "recall", "roc_auc"
                ]
            else:  # regression
                selection_criteria_options = [
                    "r2", "mean_squared_error", "mean_absolute_error", "explained_variance"
                ]
            
            model_selection_criteria = st.selectbox(
                "Model Selection Criteria",
                options=selection_criteria_options,
                index=0
            )
        
        # Model name
        model_name = st.text_input(
            "Model Name",
            value=f"{selected_models[0] if selected_models else 'model'}_{int(time.time())}",
            help="Name for the trained model"
        )
        
        # Save configuration
        if st.button("Save Training Configuration"):
            st.session_state.training_config = {
                "task_type": task_type,
                "model_name": model_name,
                "selected_models": selected_models,
                "test_size": test_size,
                "random_state": random_state,
                "cv_folds": cv_folds,
                "optimization_strategy": optimization_strategy,
                "optimization_iterations": optimization_iterations,
                "enable_feature_selection": enable_feature_selection,
                "feature_selection_method": feature_selection_method if enable_feature_selection else None,
                "feature_selection_k": feature_selection_k if enable_feature_selection else None,
                "early_stopping": early_stopping,
                "early_stopping_rounds": early_stopping_rounds if early_stopping else None,
                "model_selection_criteria": model_selection_criteria
            }
            
            st.session_state.model_name = model_name
            st.success("Training configuration saved!")
    
    with tab2:
        st.subheader("Training Execution")
        
        # Check if training configuration exists
        if not hasattr(st.session_state, "training_config"):
            st.warning("Please configure training settings first")
            return
        
        # Display training configuration summary
        config = st.session_state.training_config
        
        st.subheader("Training Configuration Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Task Type:** {config['task_type']}")
            st.write(f"**Model Name:** {config['model_name']}")
            st.write(f"**Models:** {', '.join(config['selected_models'])}")
        
        with col2:
            st.write(f"**Test Size:** {config['test_size']}")
            st.write(f"**CV Folds:** {config['cv_folds']}")
            st.write(f"**Optimization:** {config['optimization_strategy']}")
        
        with col3:
            st.write(f"**Feature Selection:** {'Yes' if config['enable_feature_selection'] else 'No'}")
            st.write(f"**Early Stopping:** {'Yes' if config['early_stopping'] else 'No'}")
            st.write(f"**Selection Criteria:** {config['model_selection_criteria']}")
        
        # Start training
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                # Prepare training request
                X = st.session_state.data[st.session_state.features]
                y = st.session_state.data[st.session_state.target]
                
                # Convert data to format for API
                train_data_csv = pd.concat([X, y], axis=1).to_csv(index=False)
                files = {"csv_file": ("train_data.csv", train_data_csv, "text/csv")}
                
                # Initialize training engine if not already done
                if not hasattr(st.session_state, "train_engine_initialized"):
                    # Create training engine initialization request
                    init_request = {
                        "engine_config": {
                            "task_type": config["task_type"],
                            "model_path": "models",
                            "random_state": config["random_state"],
                            "test_size": config["test_size"],
                            "cv_folds": config["cv_folds"],
                            "optimization_strategy": config["optimization_strategy"],
                            "optimization_iterations": config["optimization_iterations"],
                            "model_selection_criteria": config["model_selection_criteria"],
                            "feature_selection": config["enable_feature_selection"],
                            "feature_selection_method": config["feature_selection_method"],
                            "feature_selection_k": config["feature_selection_k"],
                            "early_stopping": config["early_stopping"],
                            "early_stopping_rounds": config["early_stopping_rounds"]
                        }
                    }
                    
                    # Call API to initialize engine
                    init_response = api_call("train/api/initialize", method="POST", data=init_request)
                    
                    if init_response and init_response.get("status") == "success":
                        st.session_state.train_engine_initialized = True
                        st.success("Training engine initialized successfully")
                    else:
                        st.error("Failed to initialize training engine")
                        return
                
                # Start training
                train_request = {
                    "model_type": config["selected_models"][0],  # Use first selected model
                    "model_name": config["model_name"],
                    "target_column": st.session_state.target,
                    "train_data_file": "train_data.csv"  # This will be uploaded as a file
                }
                
                # Call API to train model
                train_response = api_call("train/api/train", method="POST", data=train_request, files=files)
                
                if train_response and "task_id" in train_response:
                    st.session_state.training_task_id = train_response["task_id"]
                    st.session_state.training_status = "running"
                    st.success(f"Training started! Task ID: {train_response['task_id']}")
                else:
                    st.error("Failed to start training")
        
        # Check training status
        if hasattr(st.session_state, "training_task_id") and st.session_state.training_task_id:
            st.subheader("Training Status")
            
            # Add refresh button
            if st.button("Refresh Status"):
                # Call API to get training status
                status_response = api_call(f"train/api/train/status/{st.session_state.training_task_id}")
                
                if status_response:
                    st.session_state.training_status = status_response.get("status")
                    st.session_state.training_progress = status_response.get("progress")
                    st.session_state.training_error = status_response.get("error")
                    
                    if status_response.get("model_name"):
                        st.session_state.model_name = status_response.get("model_name")
                        # Add to trained models list if completed
                        if status_response.get("status") == "completed" and st.session_state.model_name not in st.session_state.trained_models:
                            st.session_state.trained_models.append(st.session_state.model_name)
            
            # Display current status
            if st.session_state.training_status == "running":
                progress = st.session_state.training_progress if hasattr(st.session_state, "training_progress") else 0
                st.progress(progress if progress else 0)
                st.write(f"Training in progress... Progress: {progress:.1%}")
                
                # Display ETA if available
                if hasattr(st.session_state, "training_eta") and st.session_state.training_eta:
                    st.write(f"Estimated time remaining: {st.session_state.training_eta:.1f} seconds")
            
            elif st.session_state.training_status == "completed":
                st.success(f"Training completed! Model name: {st.session_state.model_name}")
                
                # Add button to view results
                if st.button("View Results"):
                    # Switch to results tab
                    pass
            
            elif st.session_state.training_status == "failed":
                st.error("Training failed!")
                
                # Display error if available
                if hasattr(st.session_state, "training_error") and st.session_state.training_error:
                    st.write(f"Error: {st.session_state.training_error}")
    
    with tab3:
        st.subheader("Training Results & Evaluation")
        
        # Check if any models have been trained
        if not st.session_state.trained_models:
            st.warning("No trained models available. Please complete training first.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model to Evaluate",
            options=st.session_state.trained_models
        )
        
        if selected_model:
            # Get model info and metrics
            model_info_response = api_call(f"train/api/models/{selected_model}")
            
            if model_info_response:
                # Display model info
                st.subheader("Model Information")
                
                # Basic info
                st.write(f"**Model Name:** {selected_model}")
                st.write(f"**Model Type:** {model_info_response.get('model_type', 'Unknown')}")
                st.write(f"**Best Model:** {'Yes' if model_info_response.get('is_best', False) else 'No'}")
                
                # Metrics
                if "metrics" in model_info_response:
                    metrics = model_info_response["metrics"]
                    
                    st.subheader("Performance Metrics")
                    
                    # Create columns for metrics
                    metric_cols = st.columns(4)
                    
                    # For classification
                    if "accuracy" in metrics:
                        with metric_cols[0]:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        with metric_cols[1]:
                            st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                        with metric_cols[2]:
                            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        with metric_cols[3]:
                            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                    
                    # For regression
                    elif "mean_squared_error" in metrics:
                        with metric_cols[0]:
                            st.metric("MSE", f"{metrics.get('mean_squared_error', 0):.4f}")
                        with metric_cols[1]:
                            st.metric("RMSE", f"{metrics.get('root_mean_squared_error', 0):.4f}")
                        with metric_cols[2]:
                            st.metric("MAE", f"{metrics.get('mean_absolute_error', 0):.4f}")
                        with metric_cols[3]:
                            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                
                # Feature importance
                if "top_features" in model_info_response:
                    st.subheader("Feature Importance")
                    
                    top_features = model_info_response["top_features"]
                    
                    # Create a DataFrame for visualization
                    feature_df = pd.DataFrame({
                        "Feature": list(top_features.keys()),
                        "Importance": list(top_features.values())
                    }).sort_values("Importance", ascending=False)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax)
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                
                # Evaluate on test data
                st.subheader("Model Evaluation")
                
                # Upload test data option
                test_data_file = st.file_uploader("Upload test data (optional)", type=["csv"])
                use_training_data = st.checkbox("Use training data for evaluation", value=True)
                
                if st.button("Evaluate Model"):
                    with st.spinner("Evaluating model..."):
                        # Prepare evaluation data
                        if test_data_file:
                            # Use uploaded test data
                            test_data = pd.read_csv(test_data_file)
                            test_data_csv = test_data.to_csv(index=False)
                            files = {"csv_file": ("test_data.csv", test_data_csv, "text/csv")}
                            test_target = st.session_state.target
                        elif use_training_data:
                            # Use original training data
                            test_data_csv = st.session_state.data.to_csv(index=False)
                            files = {"csv_file": ("test_data.csv", test_data_csv, "text/csv")}
                            test_target = st.session_state.target
                        else:
                            st.warning("Please upload test data or enable 'Use training data'")
                            return
                        
                        # Call evaluate API
                        eval_request = {
                            "model_name": selected_model,
                            "target_column": test_target,
                            "detailed": True
                        }
                        
                        # Make API call
                        eval_response = api_call(
                            f"train/api/models/{selected_model}/evaluate",
                            method="POST",
                            data=eval_request,
                            files=files
                        )
                        
                        if eval_response and "metrics" in eval_response:
                            st.session_state.evaluation_results = eval_response
                            st.success("Evaluation completed!")
                        else:
                            st.error("Evaluation failed")
                
                # Display evaluation results if available
                if hasattr(st.session_state, "evaluation_results") and st.session_state.evaluation_results:
                    eval_results = st.session_state.evaluation_results
                    metrics = eval_results.get("metrics", {})
                    
                    st.subheader("Evaluation Results")
                    
                    # Create metric columns
                    metric_cols = st.columns(4)
                    
                    # Display metrics based on task type
                    if "accuracy" in metrics:
                        # Classification metrics
                        with metric_cols[0]:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        with metric_cols[1]:
                            st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                        with metric_cols[2]:
                            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                        with metric_cols[3]:
                            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                        
                        # Display confusion matrix if available
                        if "confusion_matrix" in metrics:
                            cm = np.array(metrics["confusion_matrix"])
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                    
                    elif "mean_squared_error" in metrics:
                        # Regression metrics
                        with metric_cols[0]:
                            st.metric("MSE", f"{metrics.get('mean_squared_error', 0):.4f}")
                        with metric_cols[1]:
                            st.metric("RMSE", f"{metrics.get('root_mean_squared_error', 0):.4f}")
                        with metric_cols[2]:
                            st.metric("MAE", f"{metrics.get('mean_absolute_error', 0):.4f}")
                        with metric_cols[3]:
                            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                        
                        # Display actual vs predicted plot if available
                        if "predictions" in eval_results and "actual" in eval_results:
                            preds = eval_results["predictions"]
                            actual = eval_results["actual"]
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(actual, preds, alpha=0.5)
                            ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=2)
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted')
                            st.pyplot(fig)
                
                # Model export options
                st.subheader("Export Model")
                
                export_format = st.selectbox(
                    "Export Format",
                    options=["pickle", "joblib", "onnx", "json"]
                )
                
                if st.button("Export Model"):
                    with st.spinner("Exporting model..."):
                        # Call export API
                        export_response = api_call(
                            f"train/api/models/save/{selected_model}",
                            method="POST"
                        )
                        
                        if export_response and "save_path" in export_response:
                            st.session_state.export_path = export_response["save_path"]
                            st.success(f"Model exported to: {export_response['save_path']}")
                            
                            # Add download option
                            download_response = api_call(
                                f"train/api/reports/download/{os.path.basename(export_response['save_path'])}",
                                method="GET"
                            )
                            
                            if download_response:
                                # Create download link
                                st.download_button(
                                    label="Download Model",
                                    data=json.dumps(download_response),
                                    file_name=f"{selected_model}.{export_format}",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error("Export failed")
            else:
                st.error("Failed to retrieve model information")

# --- Inference Page ---
def inference_page():
    st.title("Model Inference")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please complete training first.")
        return
    
    # Create tabs for different inference options
    tab1, tab2, tab3 = st.tabs(["Real-time Inference", "Batch Inference", "Explainability"])
    
    with tab1:
        st.subheader("Real-time Inference")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Inference",
            options=st.session_state.trained_models
        )
        
        if selected_model:
            # Input method
            input_method = st.radio(
                "Input Method",
                options=["Manual Input", "CSV Upload", "Sample Data"]
            )
            
            inference_data = None
            
            if input_method == "Manual Input":
                # Create form for manual input
                st.subheader("Enter Input Features")
                
                # Determine columns and data types
                feature_data = {}
                
                if st.session_state.features:
                    for feature in st.session_state.features:
                        if feature in st.session_state.data.columns:
                            # Get data type
                            dtype = st.session_state.data[feature].dtype
                            
                            if np.issubdtype(dtype, np.number):
                                # Numeric input
                                min_val = float(st.session_state.data[feature].min())
                                max_val = float(st.session_state.data[feature].max())
                                mean_val = float(st.session_state.data[feature].mean())
                                
                                feature_data[feature] = st.slider(
                                    feature,
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val
                                )
                            else:
                                # Categorical input
                                options = st.session_state.data[feature].unique().tolist()
                                feature_data[feature] = st.selectbox(feature, options=options)
                                
                    # Create DataFrame with single row of input data
                    inference_data = pd.DataFrame([feature_data])
            
            elif input_method == "CSV Upload":
                # File upload
                uploaded_file = st.file_uploader("Upload CSV with input features", type=["csv"])
                
                if uploaded_file:
                    try:
                        inference_data = pd.read_csv(uploaded_file)
                        st.success(f"Loaded {len(inference_data)} samples for inference")
                        
                        # Show data preview
                        st.dataframe(inference_data.head())
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            elif input_method == "Sample Data":
                # Use a few samples from the original data
                if st.session_state.data is not None:
                    sample_size = min(5, len(st.session_state.data))
                    inference_data = st.session_state.data.sample(sample_size).drop(columns=[st.session_state.target])
                    st.success(f"Using {sample_size} random samples for inference")
                    
                    # Show data preview
                    st.dataframe(inference_data)
                else:
                    st.error("No data available for sampling")
            
            # Run inference
            if inference_data is not None and st.button("Run Inference"):
                with st.spinner("Running inference..."):
                    # Convert data to JSON format
                    data_list = inference_data.to_dict(orient="records")
                    
                    # Prepare inference request
                    inference_request = {
                        "model_name": selected_model,
                        "data": data_list,
                        "return_probabilities": True
                    }
                    
                    # Call inference API
                    inference_response = api_call(
                        f"train/api/models/{selected_model}/predict", 
                        method="GET",
                        data=inference_request
                    )
                    
                    if inference_response and "predictions" in inference_response:
                        st.session_state.inference_results = inference_response
                        st.success("Inference completed!")
                    else:
                        st.error("Inference failed")
            
            # Display inference results
            if hasattr(st.session_state, "inference_results") and st.session_state.inference_results:
                st.subheader("Inference Results")
                
                results = st.session_state.inference_results
                predictions = results["predictions"]
                
                # Create a DataFrame with results
                if isinstance(predictions, list) and len(predictions) > 0:
                    if isinstance(predictions[0], list):
                        # Multiple predictions per sample (probabilities)
                        if "probabilities" in results and results["probabilities"]:
                            # Classification with probabilities
                            result_df = pd.DataFrame(predictions, columns=["Class"])
                            
                            # Add input features
                            for col in inference_data.columns:
                                result_df[col] = inference_data[col].values
                            
                            # Add probabilities if available
                            proba_data = results.get("prediction_probabilities", [])
                            if proba_data and len(proba_data) == len(predictions):
                                for i, probs in enumerate(proba_data):
                                    if isinstance(probs, list):
                                        for j, p in enumerate(probs):
                                            result_df[f"Prob_Class_{j}"] = p
                    else:
                        # Single prediction per sample
                        result_df = pd.DataFrame({"Prediction": predictions})
                        
                        # Add input features
                        for col in inference_data.columns:
                            result_df[col] = inference_data[col].values
                    
                    # Display results
                    st.dataframe(result_df)
                    
                    # Visualize predictions
                    if len(predictions) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if isinstance(predictions[0], (int, float)) or (isinstance(predictions[0], list) and len(predictions[0]) == 1):
                            # Numeric predictions - histogram
                            sns.histplot(predictions, kde=True, ax=ax)
                            ax.set_title("Distribution of Predictions")
                        else:
                            # Categorical predictions - bar chart
                            pred_counts = pd.Series(predictions).value_counts()
                            sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax)
                            ax.set_title("Prediction Counts")
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                else:
                    st.write("No predictions available")
    
    with tab2:
        st.subheader("Batch Inference")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Batch Inference",
            options=st.session_state.trained_models,
            key="batch_model_select"
        )
        
        if selected_model:
            # Batch data upload
            batch_file = st.file_uploader("Upload batch data (CSV)", type=["csv"])
            
            if batch_file:
                try:
                    batch_data = pd.read_csv(batch_file)
                    st.success(f"Loaded {len(batch_data)} samples for batch inference")
                    
                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(batch_data.head())
                    
                    # Feature selection
                    if st.session_state.features:
                        # Select features for inference
                        available_cols = [col for col in batch_data.columns if col in st.session_state.features]
                        
                        if not available_cols:
                            st.warning("No matching feature columns found in the uploaded data")
                        else:
                            selected_features = st.multiselect(
                                "Select features for inference",
                                options=available_cols,
                                default=available_cols
                            )
                            
                            if selected_features:
                                # Run batch inference
                                if st.button("Run Batch Inference"):
                                    with st.spinner("Running batch inference..."):
                                        # Prepare batch data
                                        inference_data = batch_data[selected_features]
                                        
                                        # Convert to CSV for API
                                        batch_csv = inference_data.to_csv(index=False)
                                        files = {"data_file": ("batch_data.csv", batch_csv, "text/csv")}
                                        
                                        # Prepare request
                                        batch_request = {
                                            "model_name": selected_model,
                                            "return_probabilities": True
                                        }
                                        
                                        # Call batch inference API
                                        batch_response = api_call(
                                            f"train/api/models/{selected_model}/predict",
                                            method="GET",
                                            data=batch_request,
                                            files=files
                                        )
                                        
                                        if batch_response and "predictions" in batch_response:
                                            st.session_state.batch_results = batch_response
                                            st.success("Batch inference completed!")
                                        else:
                                            st.error("Batch inference failed")
                
                    # Display batch results
                    if hasattr(st.session_state, "batch_results") and st.session_state.batch_results:
                        st.subheader("Batch Inference Results")
                        
                        batch_results = st.session_state.batch_results
                        batch_predictions = batch_results["predictions"]
                        
                        # Create DataFrame with results
                        result_df = batch_data.copy()
                        result_df["Prediction"] = batch_predictions
                        
                        # Display results
                        st.dataframe(result_df)
                        
                        # Download results option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "Download Results as CSV",
                            data=csv,
                            file_name="batch_inference_results.csv",
                            mime="text/csv"
                        )
                        
                        # Visualize batch predictions
                        st.subheader("Prediction Distribution")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if isinstance(batch_predictions[0], (int, float)) or (isinstance(batch_predictions[0], list) and len(batch_predictions[0]) == 1):
                            # Numeric predictions - histogram
                            sns.histplot(batch_predictions, kde=True, ax=ax)
                            ax.set_title("Distribution of Predictions")
                        else:
                            # Categorical predictions - bar chart
                            pred_counts = pd.Series(batch_predictions).value_counts()
                            sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax)
                            ax.set_title("Prediction Counts")
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error processing batch file: {str(e)}")
    
    with tab3:
        st.subheader("Model Explainability")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Explainability",
            options=st.session_state.trained_models,
            key="explain_model_select"
        )
        
        if selected_model:
            # Explainability method
            explain_method = st.selectbox(
                "Explainability Method",
                options=["shap", "feature_importance", "partial_dependence", "lime"],
                index=0
            )
            
            # Data for explanation
            data_option = st.radio(
                "Explanation Data",
                options=["Sample from Training Data", "Upload Data"]
            )
            
            explanation_data = None
            
            if data_option == "Sample from Training Data":
                # Use a few samples from the original data
                if st.session_state.data is not None:
                    sample_size = st.slider("Number of samples", min_value=1, max_value=min(100, len(st.session_state.data)), value=5)
                    explanation_data = st.session_state.data.sample(sample_size)
                    st.success(f"Using {sample_size} random samples for explanation")
                    
                    # Show data preview
                    st.dataframe(explanation_data.head())
                else:
                    st.error("No data available for sampling")
            
            elif data_option == "Upload Data":
                # File upload
                explain_file = st.file_uploader("Upload CSV with data to explain", type=["csv"])
                
                if explain_file:
                    try:
                        explanation_data = pd.read_csv(explain_file)
                        st.success(f"Loaded {len(explanation_data)} samples for explanation")
                        
                        # Show data preview
                        st.dataframe(explanation_data.head())
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            # Generate explanation
            if explanation_data is not None and st.button("Generate Explanation"):
                with st.spinner(f"Generating {explain_method} explanation..."):
                    # Convert to CSV for API
                    explain_csv = explanation_data.to_csv(index=False)
                    files = {"data_file": ("explain_data.csv", explain_csv, "text/csv")}
                    
                    # Prepare request
                    explain_request = {
                        "model_name": selected_model,
                        "method": explain_method
                    }
                    
                    # Call explainability API
                    explain_response = api_call(
                        f"train/api/models/{selected_model}/explain",
                        method="POST",
                        data=explain_request,
                        files=files
                    )
                    
                    if explain_response:
                        st.session_state.explanation_results = explain_response
                        st.success("Explanation generated!")
                    else:
                        st.error("Failed to generate explanation")
            
            # Display explanation
            if hasattr(st.session_state, "explanation_results") and st.session_state.explanation_results:
                st.subheader("Explanation Results")
                
                explanation = st.session_state.explanation_results
                
                # Display based on explanation type
                if "importance" in explanation:
                    # Feature importance explanation
                    importance = explanation["importance"]
                    
                    # Create DataFrame for visualization
                    importance_df = pd.DataFrame({
                        "Feature": list(importance.keys()),
                        "Importance": list(importance.values())
                    }).sort_values("Importance", ascending=False)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                    ax.set_title(f"{explain_method.capitalize()} Feature Importance")
                    st.pyplot(fig)
                
                # Display plot if available
                if "plot_url" in explanation and explanation["plot_url"]:
                    # If the API returns a URL to a plot
                    plot_url = explanation["plot_url"]
                    st.image(f"{API_BASE_URL}/{plot_url.lstrip('/')}")
                
                # Display method details
                st.write(f"Explanation Method: {explanation.get('method', explain_method)}")
                st.write(f"Generated: {explanation.get('timestamp', 'Unknown')}")

# --- Model Management Page ---
def model_management_page():
    st.title("Model Management")
    
    # Create tabs for model management
    tab1, tab2, tab3 = st.tabs(["Model Registry", "Model Deployment", "Model Monitoring"])
    
    with tab1:
        st.subheader("Model Registry")
        
        # Refresh model list
        if st.button("Refresh Model List"):
            try:
                # Call API to get all models
                models_response = api_call("train/api/models")
                
                if models_response and "models" in models_response:
                    st.session_state.model_registry = models_response["models"]
                    st.session_state.best_model = models_response.get("best_model")
                    
                    # Update trained models list
                    st.session_state.trained_models = [model["name"] for model in models_response["models"]]
                    
                    st.success(f"Found {len(st.session_state.model_registry)} models")
                else:
                    st.error("Failed to retrieve models")
            except Exception as e:
                st.error(f"Error retrieving models: {str(e)}")
        
        # Display registered models
        if hasattr(st.session_state, "model_registry") and st.session_state.model_registry:
            st.subheader("Registered Models")
            
            # Create columns for model info
            st.write(f"Total Models: {len(st.session_state.model_registry)}")
            st.write(f"Best Model: {st.session_state.best_model or 'None'}")
            
            # Create a DataFrame for models
            models_data = []
            for model in st.session_state.model_registry:
                model_info = {
                    "Name": model["name"],
                    "Type": model["type"],
                    "Metrics": str(model.get("metrics", {})),
                    "Best Model": "✅" if model["name"] == st.session_state.best_model else "❌",
                    "Loading Source": model.get("loaded_from", "Unknown")
                }
                models_data.append(model_info)
            
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df)
            
            # Model selection for management
            selected_model = st.selectbox(
                "Select Model for Management",
                options=[model["name"] for model in st.session_state.model_registry]
            )
            
            if selected_model:
                st.subheader(f"Manage Model: {selected_model}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model export
                    st.write("**Export Model:**")
                    
                    export_format = st.selectbox(
                        "Export Format",
                        options=["pickle", "joblib", "onnx"]
                    )
                    
                    if st.button("Export Model"):
                        with st.spinner("Exporting model..."):
                            # Call export API
                            export_response = api_call(
                                f"train/api/models/save/{selected_model}",
                                method="POST",
                                params={"include_preprocessor": True}
                            )
                            
                            if export_response and "save_path" in export_response:
                                st.session_state.export_path = export_response["save_path"]
                                st.success(f"Model exported to: {export_response['save_path']}")
                            else:
                                st.error("Export failed")
                
                with col2:
                    # Model deletion and other operations
                    st.write("**Model Operations:**")
                    
                    # Compare models button
                    if st.button("Generate Model Comparison"):
                        with st.spinner("Generating comparison..."):
                            # Call comparison API
                            compare_response = api_call("train/api/models/compare")
                            
                            if compare_response:
                                st.session_state.model_comparison = compare_response
                                st.success("Model comparison generated")
                            else:
                                st.error("Failed to generate comparison")
                    
                    # Generate report button
                    if st.button("Generate Model Report"):
                        with st.spinner("Generating report..."):
                            # Call report API
                            report_response = api_call("train/api/reports/generate", method="POST")
                            
                            if report_response and "report_path" in report_response:
                                st.session_state.report_path = report_response["report_path"]
                                st.session_state.report_download_url = report_response["download_url"]
                                st.success(f"Report generated: {report_response['report_path']}")
                            else:
                                st.error("Failed to generate report")
                
                # Show model comparison if available
                if hasattr(st.session_state, "model_comparison") and st.session_state.model_comparison:
                    st.subheader("Model Comparison")
                    
                    # Display comparison table
                    comparison = st.session_state.model_comparison
                    
                    # Create a DataFrame for comparison
                    if "models" in comparison:
                        models = comparison["models"]
                        metrics = comparison.get("metrics", {})
                        
                        # Build comparison DataFrame
                        comp_data = []
                        for model_name, model_info in models.items():
                            model_metrics = metrics.get(model_name, {})
                            
                            model_row = {
                                "Model": model_name,
                                "Type": model_info.get("type", "Unknown")
                            }
                            
                            # Add metrics
                            for metric, value in model_metrics.items():
                                model_row[metric] = value
                            
                            comp_data.append(model_row)
                        
                        comparison_df = pd.DataFrame(comp_data)
                        st.dataframe(comparison_df)
                    else:
                        st.write("No comparison data available")
                
                # Show report download if available
                if hasattr(st.session_state, "report_download_url") and st.session_state.report_download_url:
                    st.subheader("Model Report")
                    
                    # Get report content
                    report_response = api_call(st.session_state.report_download_url.lstrip("/api"))
                    
                    if report_response:
                        # Create download button
                        report_data = json.dumps(report_response)
                        st.download_button(
                            "Download Report",
                            data=report_data,
                            file_name=f"model_report_{selected_model}.md",
                            mime="text/markdown"
                        )
                        
                        # Display report preview
                        with st.expander("Report Preview"):
                            st.markdown(report_data)
            
            # Model upload
            st.subheader("Upload Model")
            
            model_file = st.file_uploader("Upload model file", type=["pkl", "joblib", "onnx"])
            model_name = st.text_input("Model Name (optional)")
            
            if model_file and st.button("Upload Model"):
                with st.spinner("Uploading model..."):
                    # Create form data
                    files = {"file": (model_file.name, model_file.getvalue(), "application/octet-stream")}
                    
                    # Add custom ID if provided
                    params = {}
                    if model_name:
                        params["custom_id"] = model_name
                    
                    # Call upload API
                    upload_response = api_call(
                        "train/api/models/load",
                        method="POST",
                        files=files,
                        params=params
                    )
                    
                    if upload_response and upload_response.get("status") == "success":
                        st.success(f"Model uploaded successfully: {upload_response.get('model_name')}")
                        
                        # Add to trained models list
                        if upload_response.get("model_name") not in st.session_state.trained_models:
                            st.session_state.trained_models.append(upload_response.get("model_name"))
                    else:
                        st.error("Upload failed")
    
    with tab2:
        st.subheader("Model Deployment")
        
        # Check if models are available
        if not st.session_state.trained_models:
            st.warning("No models available for deployment")
            return
        
        # Model selection for deployment
        deployment_model = st.selectbox(
            "Select Model to Deploy",
            options=st.session_state.trained_models,
            key="deploy_model_select"
        )
        
        # Deployment options
        st.subheader("Deployment Options")
        
        deployment_target = st.selectbox(
            "Deployment Target",
            options=["REST API", "Batch Processing", "Edge Device"]
        )
        
        if deployment_target == "REST API":
            # REST API deployment options
            deployment_name = st.text_input(
                "Deployment Name",
                value=f"{deployment_model}-api"
            )
            
            concurrency = st.slider(
                "Max Concurrent Requests",
                min_value=1,
                max_value=100,
                value=10
            )
            
            enable_batching = st.checkbox("Enable Request Batching", value=True)
            
            if st.button("Deploy as REST API"):
                with st.spinner("Deploying model..."):
                    # In a real application, this would call an API to deploy the model
                    # Here we just simulate a successful deployment
                    
                    # Create mock deployment response
                    deployment_info = {
                        "deployment_id": f"deploy-{int(time.time())}",
                        "model": deployment_model,
                        "target": "REST API",
                        "status": "running",
                        "endpoint": f"https://api.example.com/v1/models/{deployment_model}/predict",
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store deployment info
                    if "deployments" not in st.session_state:
                        st.session_state.deployments = []
                    
                    st.session_state.deployments.append(deployment_info)
                    st.success(f"Model deployed as REST API: {deployment_info['endpoint']}")
        
        elif deployment_target == "Batch Processing":
            # Batch processing deployment options
            schedule_options = ["On-demand", "Hourly", "Daily", "Weekly"]
            schedule = st.selectbox("Schedule", options=schedule_options)
            
            input_location = st.text_input(
                "Input Data Location",
                value="data/input/"
            )
            
            output_location = st.text_input(
                "Output Data Location",
                value="data/output/"
            )
            
            if st.button("Deploy for Batch Processing"):
                with st.spinner("Deploying model for batch processing..."):
                    # In a real application, this would call an API to deploy the model
                    # Here we just simulate a successful deployment
                    
                    # Create mock deployment response
                    deployment_info = {
                        "deployment_id": f"batch-{int(time.time())}",
                        "model": deployment_model,
                        "target": "Batch Processing",
                        "status": "scheduled",
                        "schedule": schedule,
                        "input_location": input_location,
                        "output_location": output_location,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store deployment info
                    if "deployments" not in st.session_state:
                        st.session_state.deployments = []
                    
                    st.session_state.deployments.append(deployment_info)
                    st.success(f"Model deployed for batch processing with {schedule} schedule")
        
        elif deployment_target == "Edge Device":
            # Edge deployment options
            device_type = st.selectbox(
                "Edge Device Type",
                options=["Raspberry Pi", "Jetson Nano", "Arduino", "Custom Device"]
            )
            
            quantize_model = st.checkbox("Quantize Model for Edge", value=True)
            
            optimize_for = st.selectbox(
                "Optimize For",
                options=["Speed", "Memory", "Power Efficiency"]
            )
            
            if st.button("Prepare for Edge Deployment"):
                with st.spinner("Preparing model for edge deployment..."):
                    # In a real application, this would call an API to prepare the model
                    # Here we just simulate a successful preparation
                    
                    # Create mock deployment response
                    deployment_info = {
                        "deployment_id": f"edge-{int(time.time())}",
                        "model": deployment_model,
                        "target": "Edge Device",
                        "device_type": device_type,
                        "quantized": quantize_model,
                        "optimized_for": optimize_for,
                        "status": "ready",
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store deployment info
                    if "deployments" not in st.session_state:
                        st.session_state.deployments = []
                    
                    st.session_state.deployments.append(deployment_info)
                    st.success(f"Model prepared for {device_type} deployment")
        
        # Display current deployments
        if hasattr(st.session_state, "deployments") and st.session_state.deployments:
            st.subheader("Current Deployments")
            
            # Create a DataFrame for deployments
            deploy_data = []
            for deploy in st.session_state.deployments:
                deploy_data.append({
                    "ID": deploy["deployment_id"],
                    "Model": deploy["model"],
                    "Target": deploy["target"],
                    "Status": deploy["status"],
                    "Created": deploy["created_at"]
                })
            
            deploy_df = pd.DataFrame(deploy_data)
            st.dataframe(deploy_df)
            
            # Generate deployment code examples
            selected_deployment_id = st.selectbox(
                "Select Deployment for Code Example",
                options=[deploy["deployment_id"] for deploy in st.session_state.deployments]
            )
            
            if selected_deployment_id:
                # Find selected deployment
                selected_deploy = next(
                    (d for d in st.session_state.deployments if d["deployment_id"] == selected_deployment_id),
                    None
                )
                
                if selected_deploy:
                    st.subheader("Deployment Code Example")
                    
                    if selected_deploy["target"] == "REST API":
                        # REST API example
                        st.code(f"""
import requests
import json

# Define input data
input_data = {{
    "features": [
        [1.2, 3.4, 5.6, 7.8],  # Replace with your actual feature values
        # Add more samples as needed
    ]
}}

# Call the API
response = requests.post(
    "{selected_deploy.get('endpoint', 'https://api.example.com/predict')}",
    headers={{"Content-Type": "application/json"}},
    data=json.dumps(input_data)
)

# Process the results
if response.status_code == 200:
    predictions = response.json()["predictions"]
    print(f"Predictions: {{predictions}}")
else:
    print(f"Error: {{response.status_code}} - {{response.text}}")
""", language="python")
                    
                    elif selected_deploy["target"] == "Batch Processing":
                        # Batch processing example
                        st.code(f"""
import pandas as pd
import os

# Prepare input data
input_dir = "{selected_deploy.get('input_location', 'data/input/')}"
output_dir = "{selected_deploy.get('output_location', 'data/output/')}"

# Example batch processing script
def process_batch():
    # Process all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Read input file
            input_file = os.path.join(input_dir, filename)
            data = pd.read_csv(input_file)
            
            # Process data (in a real scenario, this would use the deployed model)
            # ...
            
            # Save results
            output_file = os.path.join(output_dir, f"results_{{filename}}")
            data.to_csv(output_file, index=False)
            print(f"Processed {{filename}}")

if __name__ == "__main__":
    process_batch()
""", language="python")
                    
                    elif selected_deploy["target"] == "Edge Device":
                        # Edge deployment example
                        st.code(f"""
# Example code for {selected_deploy.get('device_type', 'Edge Device')}

import numpy as np
import time

# In a real scenario, you would load the model using a framework 
# appropriate for your edge device

# Example for a quantized model on edge device
class EdgeModel:
    def __init__(self, model_path):
        # Load the model
        self.model = self._load_model(model_path)
        print(f"Model loaded successfully")
    
    def _load_model(self, model_path):
        # This would use an appropriate framework for your device
        # Example: TFLite, ONNX Runtime, etc.
        return None
    
    def predict(self, input_data):
        # Make a prediction
        start_time = time.time()
        result = np.random.random(1)[0]  # Placeholder
        inference_time = time.time() - start_time
        
        print(f"Inference completed in {{inference_time:.4f}} seconds")
        return result

# Usage
model = EdgeModel("model.tflite")
input_data = np.array([1.2, 3.4, 5.6, 7.8])
prediction = model.predict(input_data)
print(f"Prediction: {{prediction}}")
""", language="python")
    
    with tab3:
        st.subheader("Model Monitoring")
        
        st.subheader("Performance Monitoring")
        
        # Model selection for monitoring
        monitoring_model = st.selectbox(
            "Select Model to Monitor",
            options=st.session_state.trained_models,
            key="monitor_model_select"
        )
        
        if monitoring_model:
            # Time range selection
            time_range = st.selectbox(
                "Time Range",
                options=["Last Hour", "Last Day", "Last Week", "Last Month"]
            )
            
            # Refresh monitoring data
            if st.button("Refresh Monitoring Data"):
                with st.spinner("Fetching monitoring data..."):
                    # In a real application, this would call an API to get monitoring data
                    # Here we just generate some mock data
                    
                    # Generate time series data
                    end_time = datetime.now()
                    if time_range == "Last Hour":
                        start_time = end_time - pd.Timedelta(hours=1)
                        freq = "5min"
                    elif time_range == "Last Day":
                        start_time = end_time - pd.Timedelta(days=1)
                        freq = "1H"
                    elif time_range == "Last Week":
                        start_time = end_time - pd.Timedelta(weeks=1)
                        freq = "12H"
                    else:  # Last Month
                        start_time = end_time - pd.Timedelta(days=30)
                        freq = "1D"
                    
                    # Generate timestamps
                    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
                    
                    # Generate metrics
                    latency_data = np.random.normal(100, 20, size=len(timestamps))  # ms
                    throughput_data = np.random.normal(50, 10, size=len(timestamps))  # req/s
                    error_rate_data = np.random.beta(0.5, 10, size=len(timestamps)) * 100  # percentage
                    
                    # Create DataFrame
                    monitoring_df = pd.DataFrame({
                        "timestamp": timestamps,
                        "latency_ms": latency_data,
                        "throughput": throughput_data,
                        "error_rate": error_rate_data
                    })
                    
                    st.session_state.monitoring_data = monitoring_df
                    st.success("Monitoring data refreshed")
            
            # Display monitoring data
            if hasattr(st.session_state, "monitoring_data") and not st.session_state.monitoring_data.empty:
                data = st.session_state.monitoring_data
                
                # Display metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric(
                        "Avg. Latency",
                        f"{data['latency_ms'].mean():.2f} ms",
                        delta=f"{data['latency_ms'].iloc[-1] - data['latency_ms'].iloc[0]:.2f} ms"
                    )
                
                with metrics_col2:
                    st.metric(
                        "Avg. Throughput",
                        f"{data['throughput'].mean():.2f} req/s",
                        delta=f"{data['throughput'].iloc[-1] - data['throughput'].iloc[0]:.2f} req/s"
                    )
                
                with metrics_col3:
                    st.metric(
                        "Avg. Error Rate",
                        f"{data['error_rate'].mean():.2f}%",
                        delta=f"{data['error_rate'].iloc[-1] - data['error_rate'].iloc[0]:.2f}%",
                        delta_color="inverse"
                    )
                
                # Plot metrics
                tab_lat, tab_tp, tab_err = st.tabs(["Latency", "Throughput", "Error Rate"])
                
                with tab_lat:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(data["timestamp"], data["latency_ms"])
                    ax.set_title("Latency Over Time")
                    ax.set_ylabel("Latency (ms)")
                    ax.set_xlabel("Time")
                    ax.grid(True)
                    st.pyplot(fig)
                
                with tab_tp:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(data["timestamp"], data["throughput"])
                    ax.set_title("Throughput Over Time")
                    ax.set_ylabel("Throughput (req/s)")
                    ax.set_xlabel("Time")
                    ax.grid(True)
                    st.pyplot(fig)
                
                with tab_err:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(data["timestamp"], data["error_rate"])
                    ax.set_title("Error Rate Over Time")
                    ax.set_ylabel("Error Rate (%)")
                    ax.set_xlabel("Time")
                    ax.grid(True)
                    st.pyplot(fig)
            else:
                st.info("No monitoring data available. Click 'Refresh Monitoring Data' to fetch the latest metrics.")
        
        # Data drift monitoring
        st.subheader("Data Drift Monitoring")
        
        if monitoring_model:
            # Generate drift data button
            if st.button("Analyze Data Drift"):
                with st.spinner("Analyzing data drift..."):
                    # In a real application, this would call an API to analyze data drift
                    # Here we just generate some mock data
                    
                    # Generate features
                    features = [f"feature_{i}" for i in range(5)]
                    
                    # Generate drift scores
                    drift_scores = np.random.beta(2, 5, size=len(features)) * 100  # percentage
                    
                    # Create DataFrame
                    drift_df = pd.DataFrame({
                        "feature": features,
                        "drift_score": drift_scores,
                        "status": ["High" if s > 80 else "Medium" if s > 50 else "Low" for s in drift_scores]
                    })
                    
                    st.session_state.drift_data = drift_df
                    st.success("Data drift analysis completed")
            
            # Display drift data
            if hasattr(st.session_state, "drift_data") and not st.session_state.drift_data.empty:
                drift_data = st.session_state.drift_data
                
                # Display drift table
                st.dataframe(drift_data)
                
                # Plot drift scores
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x="feature", y="drift_score", data=drift_data, ax=ax)
                
                # Color bars based on status
                for i, status in enumerate(drift_data["status"]):
                    if status == "High":
                        bars.patches[i].set_facecolor("red")
                    elif status == "Medium":
                        bars.patches[i].set_facecolor("orange")
                    else:
                        bars.patches[i].set_facecolor("green")
                
                ax.set_title("Feature Drift Scores")
                ax.set_ylabel("Drift Score (%)")
                ax.set_xlabel("Feature")
                ax.grid(True, axis="y")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Drift recommendations
                st.subheader("Recommendations")
                if any(drift_data["status"] == "High"):
                    st.warning("High drift detected! Consider retraining the model with more recent data.")
                elif any(drift_data["status"] == "Medium"):
                    st.info("Moderate drift detected. Monitor performance closely.")
                else:
                    st.success("Low drift detected. Model is performing well on current data.")
            else:
                st.info("No drift analysis data available. Click 'Analyze Data Drift' to perform analysis.")

# Main function to manage page navigation
def main():
    # Set page based on sidebar selection
    if page == "Home":
        home_page()
    elif page == "Data Management":
        data_management_page()
    elif page == "Device Optimization":
        device_optimization_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Inference":
        inference_page()
    elif page == "Model Management":
        model_management_page()

if __name__ == "__main__":
    main()