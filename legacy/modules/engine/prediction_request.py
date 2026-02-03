"""
Prediction request container for batch processing.

This module provides the data structure for representing prediction requests
in the dynamic batching system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PredictionRequest:
    """Container for a prediction request with metadata"""
    id: str
    features: np.ndarray
    priority: int = 0  # Lower value = higher priority
    timestamp: float = 0.0
    future: Optional[Any] = None
    timeout_ms: Optional[float] = None
    
    def __lt__(self, other):
        """Comparison for priority queue (lower value = higher priority)"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


__all__ = ['PredictionRequest']
