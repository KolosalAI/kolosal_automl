"""
Secure Model Manager

Handles secure storage, loading, and management of trained ML models.
"""

import os
import json
import joblib
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import hashlib
import time
from datetime import datetime
import tempfile

try:
    from modules.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class SecureModelManager:
    """Secure model storage and management system"""
    
    def __init__(self, model_dir: str = "./model_registry", encryption_key: Optional[str] = None):
        """Initialize the secure model manager"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.encryption_key = encryption_key
        self.metadata_file = self.model_dir / "model_registry.json"
        self.models_metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from registry file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save model metadata to registry file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.models_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID"""
        timestamp = str(int(time.time()))
        content = f"{model_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a model with metadata"""
        try:
            model_id = self._generate_model_id(model_name)
            model_path = self.model_dir / f"{model_id}.pkl"
            
            # Save model
            with open(model_path, 'wb') as f:
                joblib.dump(model, f)
            
            # Save metadata
            model_metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "file_path": str(model_path),
                "created_at": datetime.now().isoformat(),
                "file_size": model_path.stat().st_size,
                "file_hash": self._calculate_file_hash(model_path)
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            self.models_metadata[model_id] = model_metadata
            self._save_metadata()
            
            logger.info(f"Model saved: {model_name} -> {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a model by ID"""
        try:
            if model_id not in self.models_metadata:
                raise ValueError(f"Model not found: {model_id}")
            
            metadata = self.models_metadata[model_id]
            model_path = Path(metadata["file_path"])
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Verify file integrity
            current_hash = self._calculate_file_hash(model_path)
            if current_hash != metadata.get("file_hash"):
                logger.warning(f"Model file hash mismatch for {model_id}")
            
            # Load model
            with open(model_path, 'rb') as f:
                model = joblib.load(f)
            
            logger.info(f"Model loaded: {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def load_model_by_name(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load the most recent model by name"""
        try:
            # Find models with matching name
            matching_models = [
                (model_id, meta) for model_id, meta in self.models_metadata.items()
                if meta.get("model_name") == model_name
            ]
            
            if not matching_models:
                raise ValueError(f"No models found with name: {model_name}")
            
            # Sort by creation time and get the most recent
            matching_models.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
            model_id = matching_models[0][0]
            
            return self.load_model(model_id)
            
        except Exception as e:
            logger.error(f"Error loading model by name {model_name}: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            models_list = []
            for model_id, metadata in self.models_metadata.items():
                model_info = {
                    "model_id": model_id,
                    "model_name": metadata.get("model_name", "Unknown"),
                    "created_at": metadata.get("created_at"),
                    "file_size": metadata.get("file_size", 0),
                    "algorithm": metadata.get("algorithm", "Unknown"),
                    "metrics": metadata.get("metrics", {})
                }
                models_list.append(model_info)
            
            # Sort by creation time (most recent first)
            models_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return models_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata"""
        try:
            if model_id not in self.models_metadata:
                raise ValueError(f"Model not found: {model_id}")
            
            metadata = self.models_metadata[model_id]
            model_path = Path(metadata["file_path"])
            
            # Delete model file
            if model_path.exists():
                model_path.unlink()
            
            # Remove from metadata
            del self.models_metadata[model_id]
            self._save_metadata()
            
            logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information without loading the model"""
        try:
            if model_id not in self.models_metadata:
                raise ValueError(f"Model not found: {model_id}")
            
            return self.models_metadata[model_id].copy()
            
        except Exception as e:
            logger.error(f"Error getting model info {model_id}: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def cleanup_old_models(self, max_models: int = 10) -> int:
        """Remove old models to keep only the most recent ones"""
        try:
            models_list = self.list_models()
            
            if len(models_list) <= max_models:
                return 0
            
            # Keep only the most recent models
            models_to_delete = models_list[max_models:]
            deleted_count = 0
            
            for model_info in models_to_delete:
                if self.delete_model(model_info["model_id"]):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old models")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
