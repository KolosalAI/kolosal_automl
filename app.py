import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from datetime import datetime
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="ML Platform",
    layout="wide"
)

# API configuration
API_BASE_URL = "http://localhost:5000/api"

# API Functions
def get_models(token):
    try:
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get models: {response.text}")
            return {"models": [], "count": 0}
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {"models": [], "count": 0}

def get_model_info(model_name, token):
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/{model_name}",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get model info: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {}

def get_model_metrics(model_name, token):
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/{model_name}/metrics",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get model metrics: {response.text}")
            return {"model_name": model_name, "metrics": {}}
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {"model_name": model_name, "metrics": {}}

def login(username, password):
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Login failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def make_prediction(model, file, token):
    try:
        files = {"file": file}
        response = requests.post(
            f"{API_BASE_URL}/predict?model={model}",
            files=files,
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def train_model(file, model_type, model_name, target_column, token):
    try:
        files = {"file": file}
        data = {
            "model_type": model_type,
            "model_name": model_name,
            "target_column": target_column
        }
        response = requests.post(
            f"{API_BASE_URL}/train",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def delete_model(model_name, token):
    try:
        response = requests.delete(
            f"{API_BASE_URL}/models/{model_name}",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Model deletion failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def get_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "offline", "error": response.text}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

def decode_plot(base64_str):
    try:
        if base64_str:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        return None
    except Exception as e:
        st.error(f"Error decoding plot: {str(e)}")
        return None

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'token' not in st.session_state:
    st.session_state.token = None

if 'username' not in st.session_state:
    st.session_state.username = None

if 'roles' not in st.session_state:
    st.session_state.roles = []

def login_form():
    st.title("ML Platform")
    
    # Check API status
    api_status = get_api_status()
    if api_status.get("status") == "online":
        st.success(f"API Status: Online (Version: {api_status.get('version', 'unknown')})")
    else:
        st.error(f"API Status: Offline - {api_status.get('error', 'Connection error')}")
        return
    
    # Login form
    st.header("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username and password:
                with st.spinner("Logging in..."):
                    auth_result = login(username, password)
                    if auth_result:
                        st.session_state.authenticated = True
                        st.session_state.token = auth_result["token"]
                        st.session_state.username = auth_result["username"]
                        st.session_state.roles = auth_result["roles"]
                        st.success(f"Welcome, {username}!")
                        st.rerun()
            else:
                st.error("Please enter both username and password")

def dashboard_page():
    st.title("Dashboard")
    
    # Get all models
    with st.spinner("Loading model data..."):
        models_data = get_models(st.session_state.token)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("System Overview")
        
        # API Status
        api_status = get_api_status()
        if api_status.get("status") == "online":
            st.success(f"API Status: Online (Version: {api_status.get('version', 'unknown')})")
        else:
            st.error(f"API Status: Offline - {api_status.get('error', 'Connection error')}")
        
        # Models summary
        st.info(f"Total Models: {models_data.get('count', 0)}")
    
    with col2:
        st.header("Quick Actions")
        
        # Quick actions
        action = st.selectbox("Select Action", [
            "Make a prediction",
            "Train a new model"
        ])
        
        if action == "Make a prediction":
            if st.button("Go to Predictions"):
                st.session_state.page = "Predictions"
                st.rerun()
        elif action == "Train a new model":
            if st.button("Go to Training"):
                st.session_state.page = "Training"
                st.rerun()
    
    # Recent models
    st.header("Recent Models")
    
    if models_data.get("models"):
        # Sort models by modified date (most recent first)
        sorted_models = sorted(
            models_data.get("models", []), 
            key=lambda x: x.get("modified", ""), 
            reverse=True
        )[:5]  # Get 5 most recent
        
        for model in sorted_models:
            modified_date = datetime.fromisoformat(model.get("modified", datetime.now().isoformat()))
            st.write(f"**{model.get('name', 'Unknown Model')}** - Last Modified: {modified_date.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("No models available. Start by training a new model.")

def models_page():
    st.title("Models")
    
    # Get all models
    with st.spinner("Loading models..."):
        models_data = get_models(st.session_state.token)
    
    # Show model list in the sidebar for easier navigation
    model_names = [model.get("name", "Unknown") for model in models_data.get("models", [])]
    
    if not model_names:
        st.info("No models available. Go to the Training page to create a new model.")
        return
    
    selected_model = st.selectbox("Select a model", model_names)
    
    tabs = st.tabs(["Overview", "Metrics", "Actions"])
    
    with tabs[0]:  # Overview tab
        st.header("Model Overview")
        
        # Get detailed model information
        with st.spinner("Loading model details..."):
            model_info = get_model_info(selected_model, st.session_state.token)
        
        if model_info:
            st.write(f"**Name:** {model_info.get('name', 'Unknown')}")
            st.write(f"**Path:** {model_info.get('path', 'Unknown')}")
            st.write(f"**Size:** {model_info.get('size', 0) / 1024:.2f} KB")
            
            if "metadata" in model_info:
                st.subheader("Model Parameters")
                st.write(f"**Framework:** {model_info.get('metadata', {}).get('framework', 'Unknown')}")
                st.write(f"**Type:** {model_info.get('metadata', {}).get('type', 'Unknown')}")
                st.write(f"**Created:** {model_info.get('metadata', {}).get('created', 'Unknown')}")
    
    with tabs[1]:  # Metrics tab
        st.header("Model Metrics")
        
        # Get model metrics
        with st.spinner("Loading model metrics..."):
            metrics_data = get_model_metrics(selected_model, st.session_state.token)
        
        metrics = metrics_data.get("metrics", {})
        
        if metrics:
            # Classification metrics
            if any(m in metrics for m in ["accuracy", "precision", "recall", "f1"]):
                st.subheader("Classification Metrics")
                
                if "accuracy" in metrics:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                
                if "precision" in metrics:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                
                if "recall" in metrics:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                
                if "f1" in metrics:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
            
            # Regression metrics
            if any(m in metrics for m in ["mse", "rmse", "mae", "r2"]):
                st.subheader("Regression Metrics")
                
                if "mse" in metrics:
                    st.metric("Mean Squared Error", f"{metrics.get('mse', 0):.4f}")
                
                if "rmse" in metrics:
                    st.metric("Root Mean Squared Error", f"{metrics.get('rmse', 0):.4f}")
                
                if "mae" in metrics:
                    st.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.4f}")
                
                if "r2" in metrics:
                    st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
            
            # Feature importance
            if "feature_importance" in metrics:
                st.subheader("Feature Importance")
                fi_data = metrics.get("feature_importance", {})
                if isinstance(fi_data, dict):
                    fi_df = pd.DataFrame({
                        'Feature': list(fi_data.keys()),
                        'Importance': list(fi_data.values())
                    })
                    fi_df = fi_df.sort_values('Importance', ascending=False)
                    st.dataframe(fi_df)
        else:
            st.info("No metrics available for this model.")
    
    with tabs[2]:  # Actions tab
        st.header("Model Actions")
        
        # Delete model (admin only)
        if "admin" in st.session_state.roles:
            st.subheader("Delete Model")
            st.warning("Warning: This action is irreversible!")
            delete_confirmation = st.text_input("Type the model name to confirm deletion")
            
            if st.button("Delete Model"):
                if delete_confirmation == selected_model:
                    with st.spinner("Deleting model..."):
                        result = delete_model(selected_model, st.session_state.token)
                        
                        if result:
                            st.success(f"Model deleted successfully: {result.get('message', '')}")
                            st.rerun()
                else:
                    st.error("Confirmation doesn't match model name")

def predictions_page():
    st.title("Make Predictions")
    
    # Get all models
    with st.spinner("Loading models..."):
        models_data = get_models(st.session_state.token)
    
    model_names = [model.get("name", "Unknown") for model in models_data.get("models", [])]
    
    if not model_names:
        st.info("No models available. Go to the Training page to create a new model.")
        return
    
    st.header("Upload Data for Prediction")
    
    selected_model = st.selectbox("Select Model", model_names)
    
    # File upload
    uploaded_file = st.file_uploader("Upload data file (CSV, TXT, JSON, XLSX)", type=["csv", "txt", "json", "xlsx"])
    
    if uploaded_file is not None:
        if st.button("Make Prediction"):
            with st.spinner("Making predictions..."):
                # Reset file cursor
                uploaded_file.seek(0)
                
                # Call prediction API
                result = make_prediction(selected_model, uploaded_file, st.session_state.token)
                
                if result:
                    st.success("Prediction completed successfully!")
                    
                    st.header("Prediction Results")
                    
                    # Show metadata
                    st.write(f"Sample Count: {result.get('sample_count', 0)}")
                    st.write(f"Processing Time: {result.get('processing_time_ms', 0) / 1000:.2f} sec")
                    
                    # Display predictions
                    predictions = result.get("predictions", [])
                    
                    if predictions:
                        # Convert predictions to DataFrame for better display
                        if isinstance(predictions[0], list):
                            # Multi-output predictions
                            df = pd.DataFrame(predictions)
                            df.columns = [f"Output {i+1}" for i in range(df.shape[1])]
                        else:
                            # Single output predictions
                            df = pd.DataFrame({"Prediction": predictions})
                        
                        st.dataframe(df)
                        
                        # Create download link for predictions
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No predictions returned")

def training_page():
    st.title("Model Training")
    
    st.header("Train a New Model")
    
    # Check if user has admin role
    if "admin" not in st.session_state.roles:
        st.warning("You need admin privileges to train models.")
        return
    
    with st.form("training_form"):
        # File upload
        uploaded_file = st.file_uploader("Upload training data (CSV, TXT, JSON, XLSX)", type=["csv", "txt", "json", "xlsx"])
        
        # Training parameters
        model_name = st.text_input("Model Name (optional, will be auto-generated if empty)")
        model_type = st.selectbox("Model Type", ["classification", "regression"])
        target_column = st.text_input("Target Column Name")
        
        # Submit button
        submit = st.form_submit_button("Train Model")
        
        if submit:
            if not uploaded_file:
                st.error("Please upload a training data file")
            elif not target_column:
                st.error("Please enter the target column name")
            else:
                with st.spinner("Training model... This may take a while"):
                    # Reset file cursor
                    uploaded_file.seek(0)
                    
                    # Call training API
                    result = train_model(
                        uploaded_file,
                        model_type,
                        model_name,
                        target_column,
                        st.session_state.token
                    )
                    
                    if result:
                        st.success(f"Model trained successfully! {result.get('message', '')}")
                        
                        # Display model info
                        st.json(result)
                        
                        # Display metrics if available
                        if "metrics" in result:
                            st.header("Model Metrics")
                            
                            metrics = result.get("metrics", {})
                            
                            # Display metrics
                            if model_type == "classification":
                                if "accuracy" in metrics:
                                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                                if "precision" in metrics:
                                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                                if "recall" in metrics:
                                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                                if "f1" in metrics:
                                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                            elif model_type == "regression":
                                if "mse" in metrics:
                                    st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                                if "rmse" in metrics:
                                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                if "mae" in metrics:
                                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                                if "r2" in metrics:
                                    st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")

def main():
    """Main application after user authentication"""
    # Sidebar for navigation
    st.sidebar.title("ML Platform")
    st.sidebar.write(f"User: {st.session_state.username}")
    st.sidebar.write(f"Roles: {', '.join(st.session_state.roles)}")
    
    # Sidebar navigation
    nav_options = ["Dashboard", "Models", "Predictions", "Training"]
    # Show Admin option only for admin users
    if "admin" in st.session_state.roles:
        nav_options.append("Admin")
    
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
        
    page = st.sidebar.radio("Navigation", nav_options, index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0)
    st.session_state.page = page
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.roles = []
        st.rerun()
    
    # Display selected page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Models":
        models_page()
    elif page == "Predictions":
        predictions_page()
    elif page == "Training":
        training_page()
    elif page == "Admin":
        st.title("Admin Panel")
        st.info("Admin functionality simplified for this version")

# Main app flow
if __name__ == "__main__":
    # Run the app
    if not st.session_state.authenticated:
        login_form()
    else:
        main()