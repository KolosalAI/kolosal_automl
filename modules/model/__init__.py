"""
Model Management Module

This module contains model management utilities including:
- Secure model manager for handling trained models
- Model versioning and storage
- Model metadata management
"""

from .model_manager import SecureModelManager

__all__ = [
    "SecureModelManager"
]
