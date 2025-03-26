import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
import requests
import io
import base64
from PIL import Image
import tempfile
import uuid
import plotly.express as px
import plotly.graph_objects as go
from kolosal_client import KolosalAutoML

# Page configuration
st.set_page_config(
    page_title="Kolosal AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .status-online {
        color: green;
        font-weight: bold;
    }
    .status-offline {
        color: red;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .small-font {
        font-size: 0.8rem;
    }
    .medium-font {
        font-size: 1rem;
    }
    .large-font {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4da6ff;
    }
    .warning-box {
        background-color: #fff6e6;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ffaa00;
    }
    .success-box {
        background-color: #e6fff0;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #00cc66;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'client' not in st.session_state:
    st.session_state.client = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "login"
if 'api_status' not in st.session_state:
    st.session_state.api_status = {"status": "unknown"}
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_columns' not in st.session_state:
    st.session_state.data_columns = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'training_result' not in st.session_state:
    st.session_state.training_result = None

def update_model_list():
    """Update the list of available models"""
    try:
        if st.session_state.client:
            models = st.session_state.client.list_models()
            st.session_state.trained_models = models.get("models", [])
    except Exception as e:
        st.error(f"Failed to fetch models: {str(e)}")

def load_data(uploaded_file):
    """Load data from uploaded file into a pandas DataFrame"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        st.session_state.uploaded_data = df
        st.session_state.data_columns = df.columns.tolist()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def show_data_summary(df):
    """Display summary statistics and visualizations for the data"""
    if df is None:
        return
    
    st.subheader("Data Summary")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # Data types summary
    st.write("**Data Types:**")
    dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
    dtypes_df['Count'] = dtypes_df.groupby('Data Type')['Data Type'].transform('count')
    dtypes_summary = dtypes_df.drop_duplicates().reset_index()
    dtypes_summary.columns = ['Column', 'Data Type', 'Count']
    
    # Display as bar chart
    fig = px.bar(
        dtypes_summary, 
        x='Data Type', 
        y='Count',
        color='Data Type',
        title='Column Data Types'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write("**Missing Values:**")
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': (missing_values.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        
        fig = px.bar(
            missing_df, 
            x='Column', 
            y='Percentage',
            title='Missing Values by Column (%)',
            labels={'Percentage': 'Missing (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset")
    
    # Show first few rows
    with st.expander("Preview Data"):
        st.dataframe(df.head(10))
    
    # Data statistics
    with st.expander("Numerical Statistics"):
        st.dataframe(df.describe().T)

def get_plot_for_column(df, column):
    """Generate an appropriate plot for a given column"""
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            # For numeric columns
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[column], name=column))
            fig.update_layout(title=f'Distribution of {column}')
            return fig
        elif pd.api.types.is_string_dtype(df[column]) or df[column].nunique() < 10:
            # For categorical columns or low-cardinality numerics
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']
            fig = px.bar(
                value_counts.head(20), 
                x='Value', 
                y='Count', 
                title=f'Value Counts for {column}'
            )
            return fig
        else:
            # Default to box plot for other types
            fig = px.box(df, y=column, title=f'Box Plot for {column}')
            return fig
    except Exception as e:
        st.error(f"Error generating plot for {column}: {str(e)}")
        return None

def render_login_page():
    """Render the login page"""
    st.markdown("# 🤖 Kolosal AutoML")
    st.markdown("### Connect to AutoML API")
    
    with st.form("login_form"):
        base_url = st.text_input("API Base URL", value="http://localhost:5000")
        
        auth_method = st.radio("Authentication Method", ["Username/Password", "API Key"])
        
        if auth_method == "Username/Password":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            api_key = None
        else:
            username = None
            password = None
            api_key = st.text_input("API Key")
        
        submitted = st.form_submit_button("Connect")
        
        if submitted:
            with st.spinner("Connecting to API..."):
                try:
                    if api_key:
                        client = KolosalAutoML(base_url=base_url, api_key=api_key)
                    else:
                        client = KolosalAutoML(base_url=base_url, username=username, password=password)
                    
                    # Check connection by getting API status
                    status = client.check_status()
                    st.session_state.api_status = status
                    st.session_state.client = client
                    st.session_state.authenticated = True
                    
                    # Get existing models
                    update_model_list()
                    
                    st.success("Connected successfully")
                    st.session_state.current_page = "dashboard"
                    st.experimental_rerun()
                except Exception as e:
                    st.session_state.api_status = {"status": "offline", "error": str(e)}
                    st.error(f"Connection failed: {str(e)}")

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x80?text=Kolosal+AutoML", width=150)
        
        # API Status indicator
        status = st.session_state.api_status.get("status", "unknown")
        if status == "online":
            st.markdown(f"<p>Status: <span class='status-online'>●&nbsp;Online</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p class='small-font'>API Version: {st.session_state.api_status.get('version', 'unknown')}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p>Status: <span class='status-offline'>●&nbsp;Offline</span></p>", unsafe_allow_html=True)
            if "error" in st.session_state.api_status:
                st.markdown(f"<p class='small-font'>Error: {st.session_state.api_status['error']}</p>", unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        pages = {
            "dashboard": "📊 Dashboard",
            "data_upload": "📂 Data Upload",
            "data_exploration": "🔍 Data Exploration",
            "training": "🧠 Model Training",
            "models": "🤖 Models",
            "predictions": "🔮 Predictions",
            "settings": "⚙️ Settings"
        }
        
        # Make buttons look like navigation items
        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", disabled=page_id=="data_exploration" and st.session_state.uploaded_data is None):
                st.session_state.current_page = page_id
                st.experimental_rerun()
        
        st.divider()
        
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.experimental_rerun()

def render_dashboard():
    """Render the dashboard page"""
    st.markdown("# 📊 Dashboard")
    st.markdown("### Kolosal AutoML Platform Overview")
    
    # API Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### API Information")
        st.markdown(f"**Status:** {st.session_state.api_status.get('status', 'unknown')}")
        st.markdown(f"**Version:** {st.session_state.api_status.get('version', 'unknown')}")
        st.markdown(f"**Timestamp:** {st.session_state.api_status.get('timestamp', '')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### System Statistics")
        if st.session_state.client:
            try:
                config = st.session_state.client.get_automl_config()
                st.markdown(f"**Task Type:** {config.get('task_type', 'Unknown')}")
                st.markdown(f"**Optimization Mode:** {config.get('optimization_mode', 'Unknown')}")
                st.markdown(f"**Workers:** {config.get('max_workers', 'Unknown')}")
                st.markdown(f"**Memory Optimization:** {'Enabled' if config.get('memory_optimization', False) else 'Disabled'}")
            except Exception as e:
                st.error(f"Could not fetch system configuration: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Summary
    st.markdown("### Model Summary")
    
    if st.session_state.trained_models:
        # Model count by type
        model_types = {}
        for model in st.session_state.trained_models:
            model_type = model.get("model_type", "Unknown")
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Display as metrics
        cols = st.columns(len(model_types) + 1)
        cols[0].metric("Total Models", len(st.session_state.trained_models))
        
        idx = 1
        for model_type, count in model_types.items():
            if idx < len(cols):
                cols[idx].metric(f"{model_type} Models", count)
                idx += 1
        
        # Recent models table
        st.markdown("#### Recent Models")
        recent_models = sorted(
            st.session_state.trained_models,
            key=lambda x: x.get("modified", ""),
            reverse=True
        )[:5]  # Only show 5 most recent models
        
        model_df = pd.DataFrame([
            {
                "Model Name": model.get("name", ""),
                "Type": model.get("model_type", "Unknown"),
                "Last Modified": model.get("modified", ""),
                "Size": f"{model.get('size', 0) / 1024:.2f} KB"
            }
            for model in recent_models
        ])
        
        st.dataframe(model_df, use_container_width=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("#### No models available")
        st.markdown("You don't have any models yet. Go to the Model Training page to create your first model.")
        if st.button("Go to Training"):
            st.session_state.current_page = "training"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 📂 Upload Data")
        st.markdown("Upload new data for analysis and model training.")
        if st.button("Upload Data", key="dashboard_upload"):
            st.session_state.current_page = "data_upload"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Train Model")
        st.markdown("Create a new machine learning model with your data.")
        if st.button("Train Model", key="dashboard_train"):
            st.session_state.current_page = "training"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🔮 Make Predictions")
        st.markdown("Use your models to make predictions on new data.")
        if st.button("Make Predictions", key="dashboard_predict"):
            st.session_state.current_page = "predictions"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def render_data_upload():
    """Render the data upload page"""
    st.markdown("# 📂 Data Upload")
    st.markdown("### Upload your dataset for analysis and model training")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV, Excel, or JSON file",
        type=["csv", "xlsx", "xls", "json"]
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Quick view of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Explore Data", key="explore_data_btn"):
                    st.session_state.current_page = "data_exploration"
                    st.experimental_rerun()
            with col2:
                if st.button("Train Model", key="train_model_btn"):
                    st.session_state.current_page = "training"
                    st.experimental_rerun()

def render_data_exploration():
    """Render the data exploration page"""
    st.markdown("# 🔍 Data Exploration")
    
    if st.session_state.uploaded_data is None:
        st.warning("No data uploaded. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "data_upload"
            st.experimental_rerun()
        return
    
    df = st.session_state.uploaded_data
    
    # Data summary
    show_data_summary(df)
    
    # Column exploration
    st.markdown("### Column Exploration")
    
    # Let user select columns to explore
    selected_columns = st.multiselect(
        "Select columns to explore",
        options=df.columns.tolist(),
        default=df.select_dtypes(include=['int64', 'float64']).columns.tolist()[:3]
    )
    
    if selected_columns:
        # Create plots for selected columns
        for column in selected_columns:
            st.subheader(f"Analysis of {column}")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                col1.metric("Unique Values", df[column].nunique())
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    col2.metric("Mean", f"{df[column].mean():.2f}")
                    col3.metric("Median", f"{df[column].median():.2f}")
                    col4.metric("Std. Dev", f"{df[column].std():.2f}")
                else:
                    col2.metric("Most Common", df[column].value_counts().index[0] if not df[column].value_counts().empty else "N/A")
                    col3.metric("Most Common (%)", f"{df[column].value_counts(normalize=True).iloc[0]:.2%}" if not df[column].value_counts().empty else "N/A")
                    col4.metric("Missing", f"{df[column].isna().sum()} ({df[column].isna().mean():.2%})")
            except Exception as e:
                st.error(f"Error calculating statistics for {column}: {str(e)}")
            
            # Plot
            fig = get_plot_for_column(df, column)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        with st.expander("Correlation Analysis"):
            corr = df[numeric_cols].corr()
            
            # Plot correlation heatmap
            fig = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show strongest correlations
            st.subheader("Strongest Correlations")
            
            # Get absolute correlations and remove self-correlations
            corr_abs = corr.abs().unstack()
            corr_abs = corr_abs[corr_abs < 1].sort_values(ascending=False)
            
            # Take top 10 correlations
            top_corr = corr_abs.head(10)
            top_corr_df = pd.DataFrame({
                'Feature 1': [i[0] for i in top_corr.index],
                'Feature 2': [i[1] for i in top_corr.index],
                'Correlation': top_corr.values
            })
            
            st.dataframe(top_corr_df, use_container_width=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Data Upload"):
            st.session_state.current_page = "data_upload"
            st.experimental_rerun()
    with col2:
        if st.button("Proceed to Model Training"):
            st.session_state.current_page = "training"
            st.experimental_rerun()

def render_training():
    """Render the model training page"""
    st.markdown("# 🧠 Model Training")
    
    if st.session_state.uploaded_data is None:
        st.warning("No data uploaded. Please upload data first.")
        if st.button("Go to Data Upload"):
            st.session_state.current_page = "data_upload"
            st.experimental_rerun()
        return
    
    df = st.session_state.uploaded_data
    
    with st.form("training_form"):
        st.markdown("### Training Configuration")
        
        # Basic settings
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input("Model Name (optional)", 
                                      placeholder="Leave blank for auto-generated name")
            
            target_column = st.selectbox(
                "Target Column (what to predict)",
                options=df.columns.tolist()
            )
            
            model_type = st.selectbox(
                "Model Type",
                options=["classification", "regression"],
                help="Classification for categorical targets, regression for numerical targets"
            )
        
        with col2:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing"
            )
            
            optimization_mode = st.selectbox(
                "Optimization Mode",
                options=["BALANCED", "CONSERVATIVE", "PERFORMANCE", "FULL_UTILIZATION", "MEMORY_SAVING"],
                help="Controls resource usage during training"
            )
            
            optimization_strategy = st.selectbox(
                "Optimization Strategy",
                options=["RANDOM_SEARCH", "BAYESIAN_OPTIMIZATION", "GRID_SEARCH", "ASHT"],
                help="Method for hyperparameter optimization"
            )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                feature_selection = st.checkbox(
                    "Feature Selection",
                    value=True,
                    help="Automatically select the most important features"
                )
                
                cv_folds = st.number_input(
                    "Cross-Validation Folds",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Number of folds for cross-validation"
                )
            
            with col2:
                optimization_iterations = st.number_input(
                    "Optimization Iterations",
                    min_value=10,
                    max_value=200,
                    value=50,
                    help="Number of iterations for hyperparameter search"
                )
                
                random_state = st.number_input(
                    "Random Seed",
                    value=42,
                    help="Seed for reproducible results"
                )
            
            with col3:
                task_type = st.selectbox(
                    "Task Type (Optional)",
                    options=["", "CLASSIFICATION", "REGRESSION"],
                    help="Explicitly set the task type (usually inferred from model type)"
                )
        
        # Submit button
        submitted = st.form_submit_button("Train Model")
        
        if submitted:
            with st.spinner("Training model... This may take a while."):
                try:
                    # Build training request
                    result = st.session_state.client.train_model(
                        data=df,
                        target_column=target_column,
                        model_type=model_type,
                        model_name=model_name if model_name else None,
                        task_type=task_type if task_type else None,
                        test_size=test_size,
                        optimization_strategy=optimization_strategy,
                        optimization_iterations=optimization_iterations,
                        feature_selection=feature_selection,
                        cv_folds=cv_folds,
                        random_state=random_state,
                        optimization_mode=optimization_mode
                    )
                    
                    st.session_state.training_result = result
                    
                    # Update model list
                    update_model_list()
                    
                    st.success("Model trained successfully!")
                    st.json(result)
                    
                    # Redirect to model view
                    st.session_state.current_page = "models"
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    # Model registry section
    st.markdown("### Existing Models")
    
    if st.session_state.trained_models:
        model_df = pd.DataFrame([
            {
                "Model Name": model.get("name", ""),
                "Type": model.get("model_type", "Unknown"),
                "Last Modified": model.get("modified", ""),
                "Size": f"{model.get('size', 0) / 1024:.2f} KB",
                "Task": model.get("task_type", "Unknown")
            }
            for model in st.session_state.trained_models
        ])
        
        st.dataframe(model_df, use_container_width=True)
        
        cols = st.columns(4)
        with cols[0]:
            if st.button("Refresh Models"):
                update_model_list()
                st.success("Model list refreshed")
    else:
        st.info("No models found. Train your first model using the form above.")

def render_models_page():
    """Render the models management page"""
    st.markdown("# 🤖 Models")
    st.markdown("### Manage your machine learning models")
    
    # Refresh model list
    update_model_list()
    
    if not st.session_state.trained_models:
        st.info("No models found. Go to the Training page to create your first model.")
        if st.button("Go to Training"):
            st.session_state.current_page = "training"
            st.experimental_rerun()
        return
    
    # Model selection
    model_names = [model.get("name", "") for model in st.session_state.trained_models]
    selected_model = st.selectbox("Select a model", model_names)
    
    # Get selected model information
    selected_model_info = next((model for model in st.session_state.trained_models if model.get("name", "") == selected_model), None)
    
    if selected_model_info:
        tabs = st.tabs(["Overview", "Metrics", "Features", "Actions"])
        
        with tabs[0]:  # Overview tab
            st.subheader("Model Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Name:** {selected_model_info.get('name', 'Unknown')}")
                st.markdown(f"**Type:** {selected_model_info.get('model_type', 'Unknown')}")
                st.markdown(f"**Task:** {selected_model_info.get('task_type', 'Unknown')}")
                st.markdown(f"**Size:** {selected_model_info.get('size', 0) / 1024:.2f} KB")
                
                if "modified" in selected_model_info:
                    st.markdown(f"**Last Modified:** {selected_model_info.get('modified', '')}")
            
            with col2:
                try:
                    # Get detailed metrics
                    metrics = st.session_state.client.get_model_metrics(selected_model)
                    if metrics and "metrics" in metrics:
                        st.markdown("**Performance Metrics:**")
                        for metric_name, metric_value in metrics["metrics"].items():
                            if isinstance(metric_value, (int, float)):
                                st.markdown(f"- {metric_name}: {metric_value:.4f}")
                except Exception as e:
                    st.error(f"Error fetching metrics: {str(e)}")
        
        with tabs[1]:  # Metrics tab
            st.subheader("Performance Metrics")
            
            try:
                # Get detailed metrics
                metrics = st.session_state.client.get_model_metrics(selected_model)
                
                if metrics and "metrics" in metrics:
                    # Display metrics as cards
                    metric_values = metrics["metrics"]
                    
                    # Classify metrics
                    classification_metrics = [
                        "accuracy", "precision", "recall", "f1", "auc", "roc_auc"
                    ]
                    regression_metrics = [
                        "mse", "rmse", "mae", "r2", "explained_variance"
                    ]
                    
                    # Filter and display based on type
                    classification_metrics_found = {k: v for k, v in metric_values.items() 
                                                  if k.lower() in classification_metrics and isinstance(v, (int, float))}
                    
                    regression_metrics_found = {k: v for k, v in metric_values.items() 
                                              if k.lower() in regression_metrics and isinstance(v, (int, float))}
                    
                    if classification_metrics_found:
                        st.markdown("#### Classification Metrics")
                        metric_cols = st.columns(len(classification_metrics_found))
                        
                        for i, (metric_name, metric_value) in enumerate(classification_metrics_found.items()):
                            metric_cols[i].metric(
                                label=metric_name.upper(),
                                value=f"{metric_value:.4f}"
                            )
                    
                    if regression_metrics_found:
                        st.markdown("#### Regression Metrics")
                        metric_cols = st.columns(len(regression_metrics_found))
                        
                        for i, (metric_name, metric_value) in enumerate(regression_metrics_found.items()):
                            metric_cols[i].metric(
                                label=metric_name.upper(),
                                value=f"{metric_value:.4f}"
                            )
                    
                    # Other metrics
                    other_metrics = {k: v for k, v in metric_values.items() 
                                   if k.lower() not in classification_metrics and
                                   k.lower() not in regression_metrics and
                                   isinstance(v, (int, float))}
                    
                    if other_metrics:
                        st.markdown("#### Other Metrics")
                        metric_cols = st.columns(min(3, len(other_metrics)))
                        
                        for i, (metric_name, metric_value) in enumerate(other_metrics.items()):
                            col_index = i % len(metric_cols)
                            metric_cols[col_index].metric(
                                label=metric_name.upper(),
                                value=f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value
                            )
                    
                    # If confusion matrix exists, display it
                    if "confusion_matrix" in metric_values:
                        st.markdown("#### Confusion Matrix")
                        conf_matrix = metric_values["confusion_matrix"]
                        
                        # Convert to numpy array if it's a list
                        if isinstance(conf_matrix, list):
                            conf_matrix = np.array(conf_matrix)
                        
                        # Create heatmap
                        fig = px.imshow(
                            conf_matrix,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=[str(i) for i in range(conf_matrix.shape[1])],
                            y=[str(i) for i in range(conf_matrix.shape[0])],
                            text_auto=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show any visualization plots if available
                    for key in metric_values:
                        if key.endswith("_plot") and isinstance(metric_values[key], str):
                            try:
                                # Try to decode base64 image
                                image_data = base64.b64decode(metric_values[key])
                                image = Image.open(io.BytesIO(image_data))
                                st.image(image, caption=key.replace("_plot", "").replace("_", " ").title())
                            except:
                                st.warning(f"Could not display {key} visualization")
                else:
                    st.info("No metrics available for this model.")
            except Exception as e:
                st.error(f"Error fetching metrics: {str(e)}")
        
        with tabs[2]:  # Features tab
            st.subheader("Feature Importance")
            
            try:
                # Get feature importance
                feature_importance = st.session_state.client.feature_importance(selected_model)
                
                if feature_importance and "top_features" in feature_importance:
                    # Get top features
                    top_features = feature_importance["top_features"]
                    
                    # Convert to DataFrame for display
                    if isinstance(top_features, dict):
                        df = pd.DataFrame({
                            "Feature": list(top_features.keys()),
                            "Importance": list(top_features.values())
                        })
                        df = df.sort_values("Importance", ascending=False)
                        
                        # Plot
                        fig = px.bar(
                            df,
                            x="Importance",
                            y="Feature",
                            orientation='h',
                            title="Feature Importance",
                            labels={"Importance": "Importance Score", "Feature": "Feature Name"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # If plot is available, display it
                    if "plot_path" in feature_importance:
                        try:
                            with open(feature_importance["plot_path"], "rb") as file:
                                st.image(file.read(), caption="Feature Importance Visualization")
                        except:
                            st.warning("Could not display feature importance plot")
                else:
                    st.info("No feature importance data available for this model.")
            except Exception as e:
                st.error(f"Error fetching feature importance: {str(e)}")
        
        with tabs[3]:  # Actions tab
            st.subheader("Model Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Model")
                export_format = st.selectbox(
                    "Export Format",
                    options=["sklearn", "onnx", "pmml", "tf", "torchscript"],
                    index=0
                )
                
                include_pipeline = st.checkbox("Include Pipeline", value=True)
                
                if st.button("Export Model"):
                    try:
                        with st.spinner("Exporting model..."):
                            # Create a temporary file path
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format}") as tmp:
                                output_path = tmp.name
                            
                            # Export the model
                            path = st.session_state.client.export_model(
                                model_name=selected_model,
                                format=export_format,
                                include_pipeline=include_pipeline,
                                output_path=output_path
                            )
                            
                            # Read the file
                            with open(path, "rb") as file:
                                file_bytes = file.read()
                            
                            # Create download button
                            st.download_button(
                                label=f"Download {export_format.upper()} Model",
                                data=file_bytes,
                                file_name=f"{selected_model}.{export_format}",
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col2:
                st.markdown("#### Quantize Model")
                
                quantization_type = st.selectbox(
                    "Quantization Type",
                    options=["int8", "float16"],
                    index=0
                )
                
                quantization_mode = st.selectbox(
                    "Quantization Mode",
                    options=["dynamic_per_batch", "dynamic_per_channel"],
                    index=0
                )
                
                if st.button("Quantize Model"):
                    try:
                        with st.spinner("Quantizing model..."):
                            result = st.session_state.client.quantize_model(
                                model_name=selected_model,
                                quantization_type=quantization_type,
                                quantization_mode=quantization_mode
                            )
                            
                            st.success("Model quantized successfully!")
                            st.json(result)
                    except Exception as e:
                        st.error(f"Quantization failed: {str(e)}")
            
            st.divider()
            
            # Danger zone
            st.markdown("#### Danger Zone")
            st.warning("Caution: These actions cannot be undone!")
            
            if st.button("Delete Model"):
                try:
                    confirm = st.text_input("Type the model name to confirm deletion:")
                    
                    if confirm == selected_model:
                        with st.spinner("Deleting model..."):
                            result = st.session_state.client.delete_model(selected_model)
                            
                            st.success("Model deleted successfully!")
                            
                            # Update model list
                            update_model_list()
                            
                            # Redirect to models page
                            st.experimental_rerun()
                    else:
                        st.error("Model name doesn't match. Deletion cancelled.")
                except Exception as e:
                    st.error(f"Deletion failed: {str(e)}")

def render_predictions():
    """Render the predictions page"""
    st.markdown("# 🔮 Predictions")
    st.markdown("### Make predictions with your trained models")
    
    # Update model list
    update_model_list()
    
    if not st.session_state.trained_models:
        st.info("No models found. Go to the Training page to create your first model.")
        if st.button("Go to Training"):
            st.session_state.current_page = "training"
            st.experimental_rerun()
        return
    
    # Tabs for different prediction methods
    pred_tabs = st.tabs(["Upload Data", "Enter Values Manually"])
    
    with pred_tabs[0]:  # Upload Data tab
        st.subheader("Predict with Data File")
        
        # Model selection
        model_names = [model.get("name", "") for model in st.session_state.trained_models]
        selected_model = st.selectbox("Select a model", model_names, key="pred_model_file")
        
        # Upload prediction data
        pred_file = st.file_uploader(
            "Upload prediction data",
            type=["csv", "xlsx", "xls", "json"],
            key="pred_file"
        )
        
        # Options
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=0,
                max_value=1000,
                value=0,
                help="0 means direct prediction, use larger values for big datasets"
            )
        
        with col2:
            return_proba = st.checkbox(
                "Return Probabilities",
                value=False,
                help="For classification models, return class probabilities"
            )
        
        if pred_file is not None:
            if st.button("Make Predictions", key="pred_btn_file"):
                try:
                    with st.spinner("Making predictions..."):
                        # Load data
                        df = load_data(pred_file)
                        
                        if df is None:
                            st.error("Failed to load prediction data")
                            return
                        
                        # Make predictions
                        predictions = st.session_state.client.predict(
                            model=selected_model,
                            data=df,
                            batch_size=batch_size,
                            return_proba=return_proba
                        )
                        
                        if predictions and "predictions" in predictions:
                            # Display predictions
                            st.success(f"Successfully made {len(predictions['predictions'])} predictions")
                            
                            # Convert predictions to DataFrame
                            pred_values = predictions["predictions"]
                            
                            if return_proba and isinstance(pred_values[0], list):
                                # For probability outputs
                                pred_df = pd.DataFrame(
                                    pred_values,
                                    columns=[f"Class {i} Probability" for i in range(len(pred_values[0]))]
                                )
                            else:
                                # For standard predictions
                                pred_df = pd.DataFrame({"Prediction": pred_values})
                            
                            # Add index from original data
                            if hasattr(df, 'index') and len(df.index) == len(pred_df):
                                pred_df.index = df.index
                            
                            # Display predictions
                            st.dataframe(pred_df)
                            
                            # Add download button
                            csv = pred_df.to_csv()
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Show processing time
                            if "processing_time_ms" in predictions:
                                st.info(f"Processing time: {predictions['processing_time_ms']} ms")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    with pred_tabs[1]:  # Manual Input tab
        st.subheader("Predict with Manual Input")
        
        # Model selection
        model_names = [model.get("name", "") for model in st.session_state.trained_models]
        selected_model = st.selectbox("Select a model", model_names, key="pred_model_manual")
        
        # Get model info to show features
        try:
            model_info = next((model for model in st.session_state.trained_models if model.get("name", "") == selected_model), None)
            
            if model_info and "features" in model_info and model_info["features"]:
                features = model_info["features"]
                
                # Create input fields for each feature
                feature_values = {}
                
                st.markdown("### Enter feature values:")
                
                # Create columns for better layout
                num_cols = 3
                features_per_col = len(features) // num_cols + (1 if len(features) % num_cols > 0 else 0)
                
                for i in range(0, len(features), features_per_col):
                    cols = st.columns(num_cols)
                    
                    for col_idx in range(num_cols):
                        feature_idx = i + col_idx
                        
                        if feature_idx < len(features):
                            feature = features[feature_idx]
                            feature_values[feature] = cols[col_idx].number_input(
                                f"{feature}",
                                value=0.0,
                                key=f"feature_{feature_idx}"
                            )
                
                # Prediction button
                if st.button("Make Prediction", key="pred_btn_manual"):
                    try:
                        with st.spinner("Making prediction..."):
                            # Convert values to list
                            feature_list = [feature_values[feature] for feature in features]
                            
                            # Make prediction
                            predictions = st.session_state.client.predict(
                                model=selected_model,
                                data=[feature_list],  # Single sample as a list of lists
                                return_proba=st.checkbox("Return Probabilities", key="manual_proba")
                            )
                            
                            if predictions and "predictions" in predictions:
                                # Display prediction
                                pred_value = predictions["predictions"][0]
                                
                                if isinstance(pred_value, list):
                                    # For probability outputs
                                    st.success("Prediction Successful!")
                                    
                                    # Display as bar chart
                                    pred_df = pd.DataFrame({
                                        "Class": [f"Class {i}" for i in range(len(pred_value))],
                                        "Probability": pred_value
                                    })
                                    
                                    fig = px.bar(
                                        pred_df,
                                        x="Class",
                                        y="Probability",
                                        title="Prediction Probabilities",
                                        labels={"Probability": "Probability", "Class": "Class"}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Also show as text
                                    for i, prob in enumerate(pred_value):
                                        st.write(f"Class {i}: {prob:.4f}")
                                else:
                                    # For standard predictions
                                    st.success(f"Prediction: {pred_value}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            else:
                st.warning("Model information is incomplete. Feature information is not available.")
                
                # Create generic input
                st.markdown("### Enter comma-separated feature values:")
                
                feature_input = st.text_input(
                    "Feature values",
                    placeholder="e.g., 5.1,3.5,1.4,0.2",
                    help="Enter values separated by commas"
                )
                
                if st.button("Make Prediction", key="pred_btn_generic"):
                    try:
                        with st.spinner("Making prediction..."):
                            # Parse input
                            try:
                                feature_list = [float(x.strip()) for x in feature_input.split(",")]
                            except ValueError:
                                st.error("Invalid input format. Please enter numeric values separated by commas.")
                                return
                            
                            # Make prediction
                            predictions = st.session_state.client.predict(
                                model=selected_model,
                                data=[feature_list],  # Single sample as a list of lists
                                return_proba=st.checkbox("Return Probabilities", key="generic_proba")
                            )
                            
                            if predictions and "predictions" in predictions:
                                # Display prediction
                                pred_value = predictions["predictions"][0]
                                
                                if isinstance(pred_value, list):
                                    # For probability outputs
                                    st.success("Prediction Successful!")
                                    
                                    # Display as bar chart
                                    pred_df = pd.DataFrame({
                                        "Class": [f"Class {i}" for i in range(len(pred_value))],
                                        "Probability": pred_value
                                    })
                                    
                                    fig = px.bar(
                                        pred_df,
                                        x="Class",
                                        y="Probability",
                                        title="Prediction Probabilities"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # For standard predictions
                                    st.success(f"Prediction: {pred_value}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
        except Exception as e:
            st.error(f"Error loading model information: {str(e)}")

def render_settings():
    """Render the settings page"""
    st.markdown("# ⚙️ Settings")
    st.markdown("### Configure AutoML Platform Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    # Get current config
    try:
        config = st.session_state.client.get_automl_config()
        
        # Display current settings
        st.markdown("#### Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Task Type:** {config.get('task_type', 'Unknown')}")
            st.markdown(f"**Optimization Mode:** {config.get('optimization_mode', 'Unknown')}")
            st.markdown(f"**Optimization Strategy:** {config.get('optimization_strategy', 'Unknown')}")
            st.markdown(f"**Feature Selection:** {'Enabled' if config.get('feature_selection', False) else 'Disabled'}")
        
        with col2:
            st.markdown(f"**CV Folds:** {config.get('cv_folds', 'Unknown')}")
            st.markdown(f"**Max Workers:** {config.get('max_workers', 'Unknown')}")
            st.markdown(f"**Memory Optimization:** {'Enabled' if config.get('memory_optimization', False) else 'Disabled'}")
            st.markdown(f"**Quantization Enabled:** {'Enabled' if config.get('quantization_enabled', False) else 'Disabled'}")
        
        # Update settings
        st.markdown("#### Update Configuration")
        
        optimization_mode = st.selectbox(
            "Optimization Mode",
            options=["BALANCED", "CONSERVATIVE", "PERFORMANCE", "FULL_UTILIZATION", "MEMORY_SAVING"],
            index=["BALANCED", "CONSERVATIVE", "PERFORMANCE", "FULL_UTILIZATION", "MEMORY_SAVING"].index(
                config.get("optimization_mode", "BALANCED")),
            help="Controls resource usage during training and inference"
        )
        
        if st.button("Update Configuration"):
            try:
                with st.spinner("Updating configuration..."):
                    new_config = st.session_state.client.update_automl_config(optimization_mode=optimization_mode)
                    st.success("Configuration updated successfully")
                    
                    # Update in session state
                    config = new_config
                    
                    # Show new settings
                    st.markdown("#### Updated Configuration")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Task Type:** {config.get('task_type', 'Unknown')}")
                        st.markdown(f"**Optimization Mode:** {config.get('optimization_mode', 'Unknown')}")
                        st.markdown(f"**Optimization Strategy:** {config.get('optimization_strategy', 'Unknown')}")
                        st.markdown(f"**Feature Selection:** {'Enabled' if config.get('feature_selection', False) else 'Disabled'}")
                    
                    with col2:
                        st.markdown(f"**CV Folds:** {config.get('cv_folds', 'Unknown')}")
                        st.markdown(f"**Max Workers:** {config.get('max_workers', 'Unknown')}")
                        st.markdown(f"**Memory Optimization:** {'Enabled' if config.get('memory_optimization', False) else 'Disabled'}")
                        st.markdown(f"**Quantization Enabled:** {'Enabled' if config.get('quantization_enabled', False) else 'Disabled'}")
            except Exception as e:
                st.error(f"Failed to update configuration: {str(e)}")
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")
    
    # API Connection
    st.subheader("API Connection")
    
    # Show current connection info
    st.markdown(f"**Current API URL:** {st.session_state.client.config.base_url}")
    st.markdown(f"**Status:** {'Connected' if st.session_state.api_status.get('status') == 'online' else 'Disconnected'}")
    
    # Reconnect option
    if st.button("Reconnect to API"):
        st.session_state.current_page = "login"
        st.experimental_rerun()

# Main app
def main():
    # Check if authenticated
    if not st.session_state.authenticated:
        render_login_page()
    else:
        # Render sidebar
        render_sidebar()
        
        # Render current page
        if st.session_state.current_page == "dashboard":
            render_dashboard()
        elif st.session_state.current_page == "data_upload":
            render_data_upload()
        elif st.session_state.current_page == "data_exploration":
            render_data_exploration()
        elif st.session_state.current_page == "training":
            render_training()
        elif st.session_state.current_page == "models":
            render_models_page()
        elif st.session_state.current_page == "predictions":
            render_predictions()
        elif st.session_state.current_page == "settings":
            render_settings()
        else:
            # Default to dashboard
            render_dashboard()

if __name__ == "__main__":
    main()
