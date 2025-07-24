"""Time module for security middleware"""
import time as _time

# Re-export all time functionality
sleep = _time.sleep
time = _time.time
localtime = _time.localtime
strftime = _time.strftime
gmtime = _time.gmtime
mktime = _time.mktime

# Additional exports for compatibility
__all__ = ['sleep', 'time', 'localtime', 'strftime', 'gmtime', 'mktime']
