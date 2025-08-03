"""
Complete pytest logging solution to capture ALL output to tests/test.log
"""
import sys
import time
import os
from pathlib import Path
from io import StringIO


class ComprehensiveTestLogger:
    """Captures ALL pytest output to a log file."""
    
    def __init__(self):
        self.log_file_path = Path("tests/test.log")
        self.log_file = None
        
    def start_session(self):
        """Start the test session logging."""
        # Ensure directory exists
        self.log_file_path.parent.mkdir(exist_ok=True)
        
        # Open log file in write mode (clear existing content)
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)
        
        # Write session header
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.write_line("=" * 100)
        self.write_line(f"PYTEST TEST SESSION STARTED AT {timestamp}")
        self.write_line("=" * 100)
        self.write_line("")
        
    def end_session(self):
        """End the test session logging."""
        if self.log_file:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            self.write_line("")
            self.write_line("=" * 100)
            self.write_line(f"PYTEST TEST SESSION ENDED AT {timestamp}")
            self.write_line("=" * 100)
            self.log_file.close()
            self.log_file = None
    
    def write_line(self, message=""):
        """Write a line to the log file."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
    
    def write_section(self, title, content=""):
        """Write a formatted section to the log."""
        self.write_line(f"\n{'-' * 60}")
        self.write_line(f"{title}")
        self.write_line(f"{'-' * 60}")
        if content:
            self.write_line(content)


# Global logger instance
comprehensive_logger = ComprehensiveTestLogger()


def pytest_configure(config):
    """Called when pytest is configured."""
    comprehensive_logger.start_session()
    
    # Log pytest configuration
    comprehensive_logger.write_section("PYTEST CONFIGURATION")
    comprehensive_logger.write_line(f"Python version: {sys.version}")
    comprehensive_logger.write_line(f"Platform: {sys.platform}")
    comprehensive_logger.write_line(f"Working directory: {os.getcwd()}")
    
    # Log command line arguments
    if hasattr(config, 'invocation_params'):
        comprehensive_logger.write_line(f"Command: {' '.join(config.invocation_params.args)}")


def pytest_unconfigure(config):
    """Called when pytest is unconfigured."""
    comprehensive_logger.end_session()


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    comprehensive_logger.write_section("SESSION START")
    comprehensive_logger.write_line("Test collection and execution starting...")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    comprehensive_logger.write_section("SESSION FINISH")
    comprehensive_logger.write_line(f"Exit status: {exitstatus}")


def pytest_collectreport(report):
    """Called for collection reports."""
    if report.failed:
        comprehensive_logger.write_section("COLLECTION ERROR")
        comprehensive_logger.write_line(f"Location: {report.nodeid}")
        if report.longrepr:
            comprehensive_logger.write_line(f"Error: {report.longrepr}")


def pytest_runtest_logstart(nodeid, location):
    """Called before running a test."""
    comprehensive_logger.write_line("")
    comprehensive_logger.write_line("=" * 80)
    comprehensive_logger.write_line(f"STARTING TEST: {nodeid}")
    comprehensive_logger.write_line(f"Location: {location}")
    comprehensive_logger.write_line("=" * 80)


def pytest_runtest_logfinish(nodeid, location):
    """Called after running a test."""
    comprehensive_logger.write_line(f"FINISHED TEST: {nodeid}")
    comprehensive_logger.write_line("-" * 80)


def pytest_runtest_setup(item):
    """Called to perform setup for a test item."""
    comprehensive_logger.write_section("TEST SETUP", f"Setting up: {item.nodeid}")


def pytest_runtest_call(item):
    """Called to run the test."""
    comprehensive_logger.write_section("TEST EXECUTION", f"Executing: {item.nodeid}")


def pytest_runtest_teardown(item, nextitem):
    """Called to perform teardown for a test item."""
    comprehensive_logger.write_section("TEST TEARDOWN", f"Tearing down: {item.nodeid}")


def pytest_runtest_logreport(report):
    """Called when a test report is available."""
    # Only log the main test call, not setup/teardown
    if report.when == "call":
        outcome = "PASSED" if report.passed else "FAILED" if report.failed else "SKIPPED"
        
        comprehensive_logger.write_section("TEST RESULT")
        comprehensive_logger.write_line(f"Test: {report.nodeid}")
        comprehensive_logger.write_line(f"Outcome: {outcome}")
        
        if hasattr(report, 'duration'):
            comprehensive_logger.write_line(f"Duration: {report.duration:.4f} seconds")
        
        # Log captured stdout
        if hasattr(report, 'capstdout') and report.capstdout:
            comprehensive_logger.write_section("CAPTURED STDOUT")
            comprehensive_logger.write_line(report.capstdout)
        
        # Log captured stderr
        if hasattr(report, 'capstderr') and report.capstderr:
            comprehensive_logger.write_section("CAPTURED STDERR")
            comprehensive_logger.write_line(report.capstderr)
        
        # Log captured logs
        if hasattr(report, 'caplog') and hasattr(report.caplog, 'text') and report.caplog.text:
            comprehensive_logger.write_section("CAPTURED LOGS")
            comprehensive_logger.write_line(report.caplog.text)
        
        # Log failure details
        if report.failed and report.longrepr:
            comprehensive_logger.write_section("FAILURE DETAILS")
            comprehensive_logger.write_line(str(report.longrepr))
        
        # Log skip reason
        if report.skipped and hasattr(report, 'longrepr') and report.longrepr:
            comprehensive_logger.write_section("SKIP REASON")
            comprehensive_logger.write_line(str(report.longrepr))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Called to write the summary report."""
    comprehensive_logger.write_line("")
    comprehensive_logger.write_line("=" * 100)
    comprehensive_logger.write_line("FINAL TEST SUMMARY")
    comprehensive_logger.write_line("=" * 100)
    
    # Get statistics
    if hasattr(terminalreporter, 'stats'):
        for category in ['passed', 'failed', 'skipped', 'error']:
            if category in terminalreporter.stats:
                count = len(terminalreporter.stats[category])
                comprehensive_logger.write_line(f"{category.upper()}: {count}")
    
    # Log final exit status
    comprehensive_logger.write_line(f"\nFINAL EXIT STATUS: {exitstatus}")
    
    # Log total session time
    if hasattr(terminalreporter, '_sessionstarttime'):
        duration = time.time() - terminalreporter._sessionstarttime
        comprehensive_logger.write_line(f"TOTAL SESSION TIME: {duration:.2f} seconds")
    
    # Log any failures
    if hasattr(terminalreporter, 'stats') and 'failed' in terminalreporter.stats:
        comprehensive_logger.write_line("\nFAILED TESTS:")
        for report in terminalreporter.stats['failed']:
            comprehensive_logger.write_line(f"  - {report.nodeid}")


def pytest_warning_recorded(warning_message, when, nodeid, location):
    """Called when a warning is captured."""
    comprehensive_logger.write_section("WARNING")
    comprehensive_logger.write_line(f"When: {when}")
    comprehensive_logger.write_line(f"Node: {nodeid}")
    comprehensive_logger.write_line(f"Location: {location}")
    comprehensive_logger.write_line(f"Message: {warning_message}")


# Monkey patch to capture print statements
original_print = print

def enhanced_print(*args, **kwargs):
    """Enhanced print that also logs to our comprehensive logger."""
    # Call original print function
    original_print(*args, **kwargs)
    
    # Also log to our comprehensive logger
    if comprehensive_logger.log_file:
        message = " ".join(str(arg) for arg in args)
        comprehensive_logger.write_line(f"[PRINT] {message}")

# Replace the built-in print function
print = enhanced_print
