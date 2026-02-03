"""
ML Engine Components (lazy-loaded).

This subpackage exposes core engine components while keeping import time fast
by lazily resolving heavy submodules on first attribute access (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "0.1.4"

# Public API surface
__all__ = [
    # Core engines
    "InferenceEngine",
    "InferenceServer",
    "MLTrainingEngine",

    # Processors
    "BatchProcessor",
    "DataPreprocessor",
    "Quantizer",

    # Utilities
    "LRUTTLCache",

    # Helper functions (from utils)
    "_json_safe",
    "_scrub",
    "_patch_pickle_for_locks",
    "_return_none",

    # Availability flag
    "OPTIMIZATION_MODULES_AVAILABLE",

    # Optional optimization exports
    "OptimizedDataLoader",
    "DatasetSize",
    "LoadingStrategy",
    "load_data_optimized",
    "AdaptivePreprocessorConfig",
    "PreprocessorConfigOptimizer",
    "MemoryAwareDataProcessor",
    "create_memory_aware_processor",
]

_LAZY_MAP: Dict[str, Tuple[str, str]] = {
    # Engines
    "InferenceEngine": (".inference_engine", "InferenceEngine"),
    "InferenceServer": (".inference_engine", "InferenceServer"),
    "MLTrainingEngine": (".train_engine", "MLTrainingEngine"),

    # Processors
    "BatchProcessor": (".batch_processor", "BatchProcessor"),
    "DataPreprocessor": (".data_preprocessor", "DataPreprocessor"),
    "Quantizer": (".quantizer", "Quantizer"),
    "LRUTTLCache": (".lru_ttl_cache", "LRUTTLCache"),

    # Utils
    "_json_safe": (".utils", "_json_safe"),
    "_scrub": (".utils", "_scrub"),
    "_patch_pickle_for_locks": (".utils", "_patch_pickle_for_locks"),
    "_return_none": (".utils", "_return_none"),

    # Optional optimization modules
    "OptimizedDataLoader": (".optimized_data_loader", "OptimizedDataLoader"),
    "DatasetSize": (".optimized_data_loader", "DatasetSize"),
    "LoadingStrategy": (".optimized_data_loader", "LoadingStrategy"),
    "load_data_optimized": (".optimized_data_loader", "load_data_optimized"),
    "AdaptivePreprocessorConfig": (".adaptive_preprocessing", "AdaptivePreprocessorConfig"),
    "PreprocessorConfigOptimizer": (".adaptive_preprocessing", "PreprocessorConfigOptimizer"),
    "MemoryAwareDataProcessor": (".memory_aware_processor", "MemoryAwareDataProcessor"),
    "create_memory_aware_processor": (".memory_aware_processor", "create_memory_aware_processor"),
}


def __getattr__(name: str) -> Any:  # PEP 562
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
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)


def _check_optimization_availability() -> bool:
    try:
        importlib.import_module(".optimized_data_loader", package=__name__)
        importlib.import_module(".adaptive_preprocessing", package=__name__)
        importlib.import_module(".memory_aware_processor", package=__name__)
        return True
    except Exception:
        return False