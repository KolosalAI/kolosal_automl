"""
UI Components Module (lazy-loaded)

Exposes UI-related utilities without importing heavy deps (pandas/sklearn)
until first use.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__all__ = [
    "DataPreviewGenerator",
    "SampleDataLoader",
]

_LAZY_MAP: Dict[str, Tuple[str, str]] = {
    "DataPreviewGenerator": (".data_preview_generator", "DataPreviewGenerator"),
    "SampleDataLoader": (".sample_data_loader", "SampleDataLoader"),
}


def __getattr__(name: str) -> Any:
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
