#!/usr/bin/env python3
"""
Genta AutoML API Startup Script

This script provides a convenient way to start the Genta AutoML API server
with proper configuration and environment setup.

Usage:
    python start_api.py
    # or
    uv run python start_api.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to start the API server."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set environment variables if not already set
    env_vars = {
        "API_ENV": "development",
        "API_DEBUG": "True",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "API_WORKERS": "1",
        "REQUIRE_API_KEY": "False",
        "API_KEYS": "dev_key,test_key",
    }
    
    # Update environment
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start the API server
    print("Starting Genta AutoML API...")
    print(f"Environment: {os.environ.get('API_ENV', 'development')}")
    print(f"Debug mode: {os.environ.get('API_DEBUG', 'False')}")
    print(f"Host: {os.environ.get('API_HOST', '0.0.0.0')}")
    print(f"Port: {os.environ.get('API_PORT', '8000')}")
    print(f"API Key required: {os.environ.get('REQUIRE_API_KEY', 'False')}")
    print()
    print("API will be available at:")
    print(f"  - Health check: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/health")
    print(f"  - Documentation: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/docs")
    print(f"  - API root: http://{os.environ.get('API_HOST', '0.0.0.0')}:{os.environ.get('API_PORT', '8000')}/")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run the API server
        subprocess.run([
            sys.executable, 
            "modules/api/app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
