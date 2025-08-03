"""Logging module for security middleware"""
import logging as _logging

# Re-export all logging functionality
getLogger = _logging.getLogger
basicConfig = _logging.basicConfig
debug = _logging.debug
info = _logging.info
warning = _logging.warning
error = _logging.error
critical = _logging.critical
exception = _logging.exception

# Export log levels
DEBUG = _logging.DEBUG
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL

# Export handler classes
StreamHandler = _logging.StreamHandler
FileHandler = _logging.FileHandler
Handler = _logging.Handler
Formatter = _logging.Formatter

__all__ = [
    'getLogger', 'basicConfig', 'debug', 'info', 'warning', 'error', 'critical', 'exception',
    'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
    'StreamHandler', 'FileHandler', 'Handler', 'Formatter'
]
