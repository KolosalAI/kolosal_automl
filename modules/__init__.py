"""
Machine Learning Engine Modules.

This package contains the core modules for the ML pipeline, but to keep
manual imports fast, we avoid importing heavy submodules (pandas, sklearn,
onnx, etc.) at import time. All public symbols are lazily resolved on first
access via PEP 562 (__getattr__).
"""

from __future__ import annotations

import os
import importlib
from typing import Any, Dict, Tuple

# Optional: Compile to bytecode on first import for better performance
_AUTO_COMPILE = os.environ.get("KOLOSAL_AUTO_COMPILE", "false").lower() == "true"
if _AUTO_COMPILE:
    try:
        from .compiler import compile_on_import
        compile_on_import()
    except Exception:
        # Silently continue if compiler module is not available
        pass

__version__ = "0.1.4"
__author__ = "Kolosal AI Team"

# Public API surface (kept for back-compat with star imports)
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

    # From engine - core components
    "InferenceEngine",
    "MLTrainingEngine",
    "BatchProcessor",
    "DataPreprocessor",
    "Quantizer",

    # From model_manager
    "SecureModelManager",

    # From UI
    "DataPreviewGenerator",
    "SampleDataLoader",

    # Utils
    "LRUTTLCache",

    # Availability flag
    "OPTIMIZATION_MODULES_AVAILABLE",
    # Optional optimization symbols (resolved lazily if present)
    "OptimizedDataLoader",
    "DatasetSize",
    "LoadingStrategy",
    "load_data_optimized",
    "AdaptivePreprocessorConfig",
    "PreprocessorConfigOptimizer",
    "MemoryAwareDataProcessor",
    "create_memory_aware_processor",
]

# Map public names to their defining modules and attribute names for lazy import
_LAZY_MAP: Dict[str, Tuple[str, str]] = {
    # Configs
    "TaskType": (".configs", "TaskType"),
    "OptimizationStrategy": (".configs", "OptimizationStrategy"),
    "ModelType": (".configs", "ModelType"),
    "EngineState": (".configs", "EngineState"),
    "QuantizationType": (".configs", "QuantizationType"),
    "QuantizationMode": (".configs", "QuantizationMode"),
    "NormalizationType": (".configs", "NormalizationType"),
    "BatchProcessingStrategy": (".configs", "BatchProcessingStrategy"),
    "BatchPriority": (".configs", "BatchPriority"),
    "ModelSelectionCriteria": (".configs", "ModelSelectionCriteria"),
    "QuantizationConfig": (".configs", "QuantizationConfig"),
    "BatchProcessorConfig": (".configs", "BatchProcessorConfig"),
    "PreprocessorConfig": (".configs", "PreprocessorConfig"),
    "InferenceEngineConfig": (".configs", "InferenceEngineConfig"),
    "MLTrainingEngineConfig": (".configs", "MLTrainingEngineConfig"),
    "MonitoringConfig": (".configs", "MonitoringConfig"),
    "ExplainabilityConfig": (".configs", "ExplainabilityConfig"),

    # Engine core (map directly to implementation modules to avoid importing the whole subpackage)
    "InferenceEngine": (".engine.inference_engine", "InferenceEngine"),
    "MLTrainingEngine": (".engine.train_engine", "MLTrainingEngine"),
    "BatchProcessor": (".engine.batch_processor", "BatchProcessor"),
    "DataPreprocessor": (".engine.data_preprocessor", "DataPreprocessor"),
    "Quantizer": (".engine.quantizer", "Quantizer"),
    "LRUTTLCache": (".engine.lru_ttl_cache", "LRUTTLCache"),

    # Model manager
    "SecureModelManager": (".model_manager", "SecureModelManager"),

    # UI
    "DataPreviewGenerator": (".ui.data_preview_generator", "DataPreviewGenerator"),
    "SampleDataLoader": (".ui.sample_data_loader", "SampleDataLoader"),

    # Optional optimization symbols (only if available)
    "OptimizedDataLoader": (".engine.optimized_data_loader", "OptimizedDataLoader"),
    "DatasetSize": (".engine.optimized_data_loader", "DatasetSize"),
    "LoadingStrategy": (".engine.optimized_data_loader", "LoadingStrategy"),
    "load_data_optimized": (".engine.optimized_data_loader", "load_data_optimized"),
    "AdaptivePreprocessorConfig": (".engine.adaptive_preprocessing", "AdaptivePreprocessorConfig"),
    "PreprocessorConfigOptimizer": (".engine.adaptive_preprocessing", "PreprocessorConfigOptimizer"),
    "MemoryAwareDataProcessor": (".engine.memory_aware_processor", "MemoryAwareDataProcessor"),
    "create_memory_aware_processor": (".engine.memory_aware_processor", "create_memory_aware_processor"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazily import symbols on first access to keep import time fast."""
    if name == "OPTIMIZATION_MODULES_AVAILABLE":
        available = _check_optimization_availability()
        globals()[name] = available
        return available

    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attr = target
    module = importlib.import_module(module_path, package=__name__)
    value = getattr(module, attr)
    # Cache on module for subsequent access
    globals()[name] = value
    return value


def __dir__():  # Keep nice dir() and star-import behavior
    return sorted(list(globals().keys()) + __all__)


def _check_optimization_availability() -> bool:
    """Detect whether optional optimization modules are importable without importing heavy deps eagerly."""
    try:
        importlib.import_module(".engine.optimized_data_loader", package=__name__)
        importlib.import_module(".engine.adaptive_preprocessing", package=__name__)
        importlib.import_module(".engine.memory_aware_processor", package=__name__)
        return True
    except Exception:
        return False