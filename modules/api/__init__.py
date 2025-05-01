"""
kolosal AutoML API modules.

This package contains all the API components for the kolosal AutoML platform.
"""

__version__ = "1.0.0"

# This file makes the 'modules/api' directory a Python package
# It allows for imports like:
# from modules.api import data_preprocessor_api

# Define what is exported from this package when using "from modules.api import *"
__all__ = [
    "data_preprocessor_api",
    "device_optimizer_api",
    "inference_engine_api", 
    "model_manager_api",
    "quantizer_api",
    "train_engine_api"
]