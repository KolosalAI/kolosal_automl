"""
Preprocessing exception classes for the data preprocessing module.

This module contains all custom exception classes used throughout
the data preprocessing pipeline.
"""


class PreprocessingError(Exception):
    """Base class for all preprocessing exceptions."""
    pass


class InputValidationError(PreprocessingError):
    """Exception raised for input validation errors."""
    pass


class StatisticsError(PreprocessingError):
    """Exception raised for errors in computing statistics."""
    pass


class SerializationError(PreprocessingError):
    """Exception raised for errors in serialization/deserialization."""
    pass


__all__ = [
    'PreprocessingError',
    'InputValidationError', 
    'StatisticsError',
    'SerializationError'
]
