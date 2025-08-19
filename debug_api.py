#!/usr/bin/env python3
"""Debug script to test API key authentication"""
import os

# Set environment variables
os.environ["API_KEYS"] = "test_key"
os.environ["JWT_SECRET"] = "test_secret"

# Import after setting environment
from modules.api.model_manager_api import app, API_KEYS, verify_api_key
from fastapi.testclient import TestClient

print("Environment API_KEYS:", repr(os.environ.get("API_KEYS")))
print("Module API_KEYS:", API_KEYS)

# Test the verification function directly
try:
    result = verify_api_key("test_key")
    print("verify_api_key('test_key') result:", result)
except Exception as e:
    print("verify_api_key('test_key') error:", e)

# Test with TestClient
client = TestClient(app)
response = client.get("/health")
print("Health endpoint (no auth required):", response.status_code, response.json())

# Test authenticated endpoint
response = client.get("/api/managers", headers={"X-API-Key": "test_key"})
print("Managers endpoint status:", response.status_code)
if response.status_code != 200:
    print("Response:", response.json())
