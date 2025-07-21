"""
Machine Learning Engine Modules.

This package contains all the core modules for the ML pipeline including:
- Configurations and settings
- Inference and training engines
- Data preprocessing and optimization
- Batch processing and quantization
"""

from .configs import *
from .engine import *
from .model_manager import SecureModelManager

__version__ = "0.1.4"
__author__ = "Kolosal AI Team"

# Package metadata
__all__ = [
    # From configs
    "TaskType",
    "OptimizationStrategy",
    "ModelType",
    "EngineState",
    "QuantizationType",
    "QuantizationMode",
    "NormalizationType",
    "BatchProcessingStrategy",
    "BatchPriority",
    "ModelSelectionCriteria",
    
    # Configuration classes
    "QuantizationConfig",
    "BatchProcessorConfig",
    "PreprocessorConfig",
    "InferenceEngineConfig",
    "MLTrainingEngineConfig",
    "MonitoringConfig",
    "ExplainabilityConfig",
    
    # From engine
    "InferenceEngine",
    "MLTrainingEngine",
    "BatchProcessor",
    "DataPreprocessor",
    "Quantizer",
    
    # From model_manager
    "SecureModelManager",
    
    # Utils
    "LRUTTLCache",
]