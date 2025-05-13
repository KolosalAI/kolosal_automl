"""
Machine Learning Engine Modules.

This package contains all the core modules for the ML pipeline including:
- Configurations and settings
- Inference and training engines
- Data preprocessing and optimization
- Batch processing and quantization
"""

from modules.configs import *
from modules.engine import *

__version__ = "1.0.0"
__author__ = "ML Engine Team"

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
    
    # Utils
    "LRUTTLCache",
]