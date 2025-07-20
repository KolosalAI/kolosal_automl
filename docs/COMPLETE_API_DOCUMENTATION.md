# kolosal AutoML - Complete API Documentation

## Overview

kolosal AutoML is a comprehensive machine learning platform that provides advanced batch processing, model optimization, and inference capabilities with enterprise-grade security, monitoring, and deployment features. This documentation covers all APIs, security features, and best practices.

**Version**: 0.1.4  
**Last Updated**: 2025-07-20  
**Production Ready**: ✅

## 🚀 What's New in v0.1.4

- **Real-time Monitoring Dashboard** - Interactive web dashboard with live metrics
- **Advanced Security Framework** - Rate limiting, input validation, audit logging
- **Enhanced Batch Processing** - Dynamic batching with priority queues and analytics
- **Comprehensive Error Handling** - Standardized responses with circuit breakers
- **Production Deployment** - Docker, Docker Compose with full monitoring stack
- **Performance Analytics** - Detailed throughput and resource utilization analysis
- **Alerting System** - Configurable alerts with multiple notification channels

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication & Security](#authentication--security)
3. [API Endpoints](#api-endpoints)
4. [Batch Processing](#batch-processing)
5. [Monitoring & Analytics](#monitoring--analytics)
6. [Security Features](#security-features)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Performance Optimization](#performance-optimization)
10. [Deployment Guide](#deployment-guide)
11. [Client Examples](#client-examples)
12. [Integration Testing](#integration-testing)
13. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+
- FastAPI
- NumPy
- Required dependencies (see `requirements.txt`)

### Installation

```bash
# Clone repository
git clone https://github.com/KolosalAI/kolosal_automl.git
cd kolosal_automl

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m modules.api.app
```

### Basic Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Submit batch processing request
data = {
    "data": [
        {"input": [1.0, 2.0, 3.0]},
        {"input": [4.0, 5.0, 6.0]}
    ],
    "priority": "normal"
}

response = requests.post(
    "http://localhost:8000/batch/submit",
    json=data,
    headers={"X-API-Key": "your-api-key"}
)
```

## Authentication & Security

### API Key Authentication

All protected endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/batch/status
```

### Environment Variables

```bash
# Security Configuration
REQUIRE_API_KEY=true
API_KEYS=key1,key2,key3
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Batch Processing Configuration
BATCH_INITIAL_SIZE=8
BATCH_MAX_SIZE=64
BATCH_TIMEOUT=0.01
```

### Rate Limiting

- Default: 100 requests per 60 seconds per client
- Configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`
- Returns `429 Too Many Requests` when exceeded

### Security Headers

All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`

## API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```

Returns API status and component health.

#### Metrics
```http
GET /metrics
```
**Authentication**: Required  
Returns performance metrics and statistics.

### Batch Processing API

#### Submit Batch
```http
POST /batch/submit
```
**Authentication**: Required  
**Body**:
```json
{
  "data": [
    {"input": [1.0, 2.0, 3.0]},
    {"input": [4.0, 5.0, 6.0]}
  ],
  "priority": "normal",
  "metadata": {"experiment_id": "exp_001"}
}
```

#### Get Batch Status
```http
GET /batch/status/{batch_id}
```
**Authentication**: Required  
Returns the current status of a batch.

#### Get Batch Results
```http
GET /batch/results/{batch_id}
```
**Authentication**: Required  
Returns the results of a completed batch.

#### Process Single Item
```http
POST /batch/process
```
**Authentication**: Required  
**Body**:
```json
{
  "data": [[1.0, 2.0, 3.0]],
  "priority": "normal",
  "timeout": 30.0
}
```

#### Configure Processor
```http
POST /batch/configure
```
**Authentication**: Required  
**Body**:
```json
{
  "initial_batch_size": 8,
  "max_batch_size": 64,
  "batch_timeout": 0.01,
  "enable_adaptive_batching": true,
  "enable_monitoring": true
}
```

#### Start/Stop/Pause Processor
```http
POST /batch/start
POST /batch/stop
POST /batch/pause
POST /batch/resume
```
**Authentication**: Required  
Control batch processor lifecycle.

### Component APIs

#### Data Preprocessor
- **Base Path**: `/preprocess`
- **Features**: Data cleaning, normalization, feature engineering

#### Device Optimizer
- **Base Path**: `/device`
- **Features**: Hardware optimization, GPU/CPU utilization

#### Inference Engine
- **Base Path**: `/inference`
- **Features**: Model inference, prediction serving

#### Model Manager
- **Base Path**: `/models`
- **Features**: Model loading, versioning, management

#### Quantizer
- **Base Path**: `/quantizer`
- **Features**: Model quantization, compression

#### Training Engine
- **Base Path**: `/train`
- **Features**: Model training, hyperparameter optimization

## Batch Processing

### Features

- **Dynamic Batching**: Automatic optimization of batch sizes
- **Priority Queues**: Support for high, normal, and low priority requests
- **Adaptive Sizing**: Batch size adjustment based on system load
- **Memory Management**: Efficient memory usage for large datasets
- **Monitoring**: Real-time performance metrics

### Priority Levels

1. **High**: Critical requests processed first
2. **Normal**: Standard processing priority
3. **Low**: Background processing, lower priority

### Batch Lifecycle

1. **Submit**: Request is queued for processing
2. **Processing**: Batch is formed and processed
3. **Completed**: Results are available
4. **Failed**: Error occurred during processing

### Performance Tuning

```python
# Optimal configuration for different scenarios

# High Throughput
config = {
    "initial_batch_size": 32,
    "max_batch_size": 128,
    "batch_timeout": 0.05,
    "enable_adaptive_batching": True
}

# Low Latency
config = {
    "initial_batch_size": 4,
    "max_batch_size": 16,
    "batch_timeout": 0.001,
    "enable_adaptive_batching": False
}

# Memory Constrained
config = {
    "initial_batch_size": 8,
    "max_batch_size": 32,
    "enable_memory_optimization": True,
    "max_batch_memory_mb": 512
}
```

## Monitoring & Analytics

### Real-time Monitoring Dashboard

Access the interactive monitoring dashboard at `/monitoring/dashboard`:

```
http://localhost:8000/monitoring/dashboard
```

Features:
- **Live System Metrics**: CPU, memory, disk usage
- **API Performance**: Request rates, response times, error rates
- **Active Alerts**: Real-time alert notifications
- **Performance Trends**: Historical performance analysis
- **Resource Utilization**: Detailed system resource tracking

### Monitoring API Endpoints

#### Health Check
```http
GET /monitoring/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-20T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "0.1.4"
}
```

#### System Metrics
```http
GET /monitoring/metrics
Authorization: X-API-Key: your_api_key
```

Response:
```json
{
  "status": "success",
  "data": {
    "current_metrics": {
      "cpu_usage": 45.2,
      "memory_usage": 60.1,
      "request_rate": 12.5,
      "error_rate": 0.8,
      "active_connections": 5
    },
    "performance_analysis": {
      "request_rate_per_second": 12.5,
      "error_rate_percent": 0.8,
      "processing_time_stats": {
        "avg": 0.045,
        "p50": 0.032,
        "p95": 0.125,
        "p99": 0.256
      }
    }
  }
}
```

#### Active Alerts
```http
GET /monitoring/alerts
Authorization: X-API-Key: your_api_key
```

Response:
```json
{
  "status": "success",
  "active_alerts": [
    {
      "name": "high_cpu_usage",
      "level": "warning",
      "message": "CPU usage above 90%",
      "last_triggered": 1642687800.0
    }
  ],
  "alert_count": 1
}
```

#### Performance Analysis
```http
GET /monitoring/performance
Authorization: X-API-Key: your_api_key
```

Response:
```json
{
  "status": "success",
  "performance_analysis": {
    "request_rate_per_second": 15.2,
    "error_rate_percent": 0.5,
    "processing_time_stats": {
      "count": 1000,
      "avg": 0.045,
      "min": 0.001,
      "max": 0.512,
      "p95": 0.125,
      "p99": 0.256
    }
  },
  "resource_analysis": {
    "cpu_utilization": {
      "avg": 45.2,
      "max": 89.5,
      "stress_detected": false
    },
    "memory_utilization": {
      "avg": 60.1,
      "max": 78.9,
      "stress_detected": false
    },
    "recommendations": [
      "System performance is optimal",
      "No optimization needed at current load"
    ]
  }
}
```

### Custom Metrics

You can record custom metrics through the monitoring system:

```python
from modules.api.monitoring import default_monitoring

# Record custom metrics
default_monitoring.metrics_collector.record_counter("custom_events", 1.0)
default_monitoring.metrics_collector.record_gauge("custom_value", 42.5)
default_monitoring.metrics_collector.record_histogram("custom_duration", 0.123)
```

### Alerting System

Configure custom alerts:

```python
from modules.api.monitoring import Alert, AlertLevel

# Create custom alert
alert = Alert(
    name="custom_metric_threshold",
    condition="custom_value > 100",
    threshold=100,
    level=AlertLevel.WARNING,
    message="Custom metric exceeded threshold",
    cooldown_seconds=300
)

default_monitoring.alert_manager.add_alert(alert)
```

### Integration with External Monitoring

The monitoring system exports metrics in formats compatible with:

- **Prometheus**: Native metrics export
- **Grafana**: Dashboard templates included
- **DataDog**: Custom metric forwarding
- **New Relic**: APM integration ready

## Security Features

### Input Validation

All inputs are automatically validated and sanitized:

- XSS protection
- SQL injection prevention
- Path traversal protection
- Input size limits

### Audit Logging

All requests are logged for security auditing:

```json
{
  "timestamp": "2025-07-20T10:30:00Z",
  "client_ip": "192.168.1.100",
  "method": "POST",
  "path": "/batch/submit",
  "response_status": 200,
  "auth_success": true,
  "processing_time_ms": 45.2
}
```

### IP Blocking

Configure blocked IPs via environment:

```bash
BLOCKED_IPS=192.168.1.50,10.0.0.100
```

### Security Headers

All responses include security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`

### API Key Management

API keys support:

- **Multiple keys**: Configure multiple valid API keys
- **Key rotation**: Hot-swap keys without downtime  
- **Environment-based**: Configure via `API_KEYS` environment variable
- **Optional authentication**: Disable for development via `REQUIRE_API_KEY=false`

## Rate Limiting

### Default Limits

- **100 requests per minute** per IP address
- **Sliding window** rate limiting
- **Automatic reset** after window expires

### Configuration

Configure rate limits via environment variables:

```bash
RATE_LIMIT_REQUESTS=100    # Requests per window
RATE_LIMIT_WINDOW=60       # Window size in seconds
```

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642687860
```

### Rate Limit Response

When rate limited, you'll receive:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "status": "error",
  "message": "Rate limit exceeded",
  "retry_after": 60,
  "limit": 100,
  "remaining": 0
}
```

### Bypass Rate Limits

Rate limits can be bypassed for specific scenarios:

- **Health checks**: `/monitoring/health` is exempt
- **Static content**: Static files are exempt
- **Whitelisted IPs**: Configure via `RATE_LIMIT_WHITELIST`

## Error Handling

### Error Response Format

All errors follow a standardized format:

```json
{
  "error": true,
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid input data",
  "detail": "Field 'data' is required",
  "category": "validation",
  "severity": "low",
  "timestamp": "2025-07-20T10:30:00Z",
  "suggestions": ["Check required fields", "Verify data format"]
}
```

### Error Categories

- **validation**: Input validation errors
- **authentication**: Authentication failures
- **authorization**: Permission errors
- **resource_not_found**: Resource not found
- **resource_exhausted**: Rate limits, capacity issues
- **system_error**: Internal server errors
- **processing_error**: Processing failures

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (validation error)
- `401`: Unauthorized (authentication required)
- `403`: Forbidden (access denied)
- `404`: Not Found
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

## Performance Optimization

### Batch Size Optimization

The system automatically optimizes batch sizes based on:

- System load
- Memory usage
- Processing times
- Queue depth

### Memory Management

- Automatic garbage collection
- Memory-aware batching
- Efficient array processing
- Memory leak prevention

### Monitoring Metrics

Available via `/metrics` endpoint:

```json
{
  "processor_stats": {
    "total_processed": 1000,
    "avg_processing_time": 0.05,
    "avg_batch_size": 16.5,
    "throughput": 200.0
  },
  "system_metrics": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 60.1,
    "memory_available_mb": 2048
  },
  "batch_tracking": {
    "total_batches": 100,
    "completed_batches": 95,
    "pending_batches": 5
  }
}
```

## Deployment Guide

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV REQUIRE_API_KEY=true

EXPOSE 8000

CMD ["python", "-m", "modules.api.app"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  kolosal-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REQUIRE_API_KEY=true
      - API_KEYS=prod_key_1,prod_key_2
      - RATE_LIMIT_REQUESTS=1000
      - BATCH_MAX_SIZE=128
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
```

### Production Configuration

```bash
# Environment variables for production
export API_ENV=production
export API_DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export REQUIRE_API_KEY=true
export API_KEYS=secure_prod_key_1,secure_prod_key_2

# Batch processing
export BATCH_MAX_SIZE=128
export BATCH_TIMEOUT=0.02
export ENABLE_MONITORING=true

# Security
export RATE_LIMIT_REQUESTS=1000
export RATE_LIMIT_WINDOW=60
export MAX_REQUEST_SIZE=52428800  # 50MB
```

## Integration Testing

### Running Integration Tests

The platform includes comprehensive integration tests that verify all components work together:

```bash
# Run all integration tests
python tests/integration/test_enhanced_integration.py

# Run specific test category
python -m pytest tests/integration/test_enhanced_integration.py::TestEndToEndWorkflows -v

# Run with coverage
python -m pytest tests/integration/ --cov=modules --cov-report=html
```

### Test Categories

#### End-to-End Workflows
- Complete batch processing pipeline
- Security integration across endpoints
- Monitoring integration with API calls
- Error handling throughout the system

#### Performance Integration
- Concurrent batch submissions
- Rate limiting functionality
- System performance under load
- Resource utilization tracking

#### Multi-Component Integration
- Batch processing with monitoring
- Security with error handling
- Data flow integrity
- Error propagation through pipeline

### Test Configuration

Configure test environment:

```bash
# Test server URL
export TEST_BASE_URL="http://localhost:8000"

# Test API key
export TEST_API_KEY="test_key"

# Test timeout (seconds)
export TEST_TIMEOUT=30
```

### Custom Integration Tests

Create custom integration tests:

```python
from tests.integration.test_enhanced_integration import APITestClient

class TestCustomWorkflow:
    def setup_method(self):
        self.client = APITestClient()
    
    def test_custom_integration(self):
        # Your custom integration test
        response = self.client.get("/your-endpoint")
        assert response.status_code == 200
```

### Continuous Integration

Integration tests are designed for CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Integration Tests
  run: |
    python -m uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 &
    sleep 10  # Wait for server to start
    python tests/integration/test_enhanced_integration.py
```

## Client Examples

### Python Client

```python
import requests
import time
from typing import List, Dict, Any

class KolosalClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def submit_batch(self, data: List[Dict[str, Any]], priority: str = "normal"):
        """Submit a batch for processing"""
        response = requests.post(
            f"{self.base_url}/batch/submit",
            json={"data": data, "priority": priority},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_batch_status(self, batch_id: str):
        """Get batch status"""
        response = requests.get(
            f"{self.base_url}/batch/status/{batch_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, batch_id: str, timeout: int = 300):
        """Wait for batch completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_batch_status(batch_id)
            
            if status["status"] == "completed":
                return self.get_batch_results(batch_id)
            elif status["status"] == "failed":
                raise Exception(f"Batch failed: {status}")
            
            time.sleep(1)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")
    
    def get_batch_results(self, batch_id: str):
        """Get batch results"""
        response = requests.get(
            f"{self.base_url}/batch/results/{batch_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = KolosalClient(api_key="your-api-key")

# Submit batch
batch_data = [
    {"input": [1.0, 2.0, 3.0]},
    {"input": [4.0, 5.0, 6.0]}
]

submission = client.submit_batch(batch_data, priority="high")
batch_id = submission["batch_id"]

# Wait for results
results = client.wait_for_completion(batch_id)
print(f"Results: {results}")
```

### JavaScript Client

```javascript
class KolosalClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = '') {
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }

    async submitBatch(data, priority = 'normal') {
        const response = await fetch(`${this.baseUrl}/batch/submit`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ data, priority })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    async getBatchStatus(batchId) {
        const response = await fetch(`${this.baseUrl}/batch/status/${batchId}`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    async waitForCompletion(batchId, timeout = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const status = await this.getBatchStatus(batchId);
            
            if (status.status === 'completed') {
                return await this.getBatchResults(batchId);
            } else if (status.status === 'failed') {
                throw new Error(`Batch failed: ${JSON.stringify(status)}`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        throw new Error(`Batch ${batchId} did not complete within ${timeout}ms`);
    }
}

// Usage example
const client = new KolosalClient('http://localhost:8000', 'your-api-key');

const batchData = [
    { input: [1.0, 2.0, 3.0] },
    { input: [4.0, 5.0, 6.0] }
];

client.submitBatch(batchData, 'high')
    .then(submission => client.waitForCompletion(submission.batch_id))
    .then(results => console.log('Results:', results))
    .catch(error => console.error('Error:', error));
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Submit batch
curl -X POST http://localhost:8000/batch/submit \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"input": [1.0, 2.0, 3.0]},
      {"input": [4.0, 5.0, 6.0]}
    ],
    "priority": "normal"
  }'

# Get batch status
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/batch/status/batch-id-here

# Get metrics
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/metrics
```

## Troubleshooting

### Common Issues

#### Authentication Errors

**Problem**: `401 Unauthorized`
**Solution**: 
- Ensure `X-API-Key` header is included
- Verify API key is correct
- Check if `REQUIRE_API_KEY` is set properly

#### Rate Limiting

**Problem**: `429 Too Many Requests`
**Solution**:
- Reduce request rate
- Implement exponential backoff
- Check rate limit configuration

#### Batch Processing Errors

**Problem**: Batch fails to process
**Solution**:
- Check input data format
- Verify batch size limits
- Monitor system resources

#### Memory Issues

**Problem**: High memory usage
**Solution**:
- Reduce batch sizes
- Enable memory optimization
- Monitor system metrics

### Debug Mode

Enable debug mode for detailed error information:

```bash
export API_DEBUG=true
```

### Logging

Check log files for detailed information:

- `kolosal_api.log` - General API logs
- `kolosal_errors.log` - Error logs
- `kolosal_security.log` - Security events
- `batch_processor_api.log` - Batch processing logs

### Performance Tuning

#### For High Throughput

```bash
export BATCH_MAX_SIZE=256
export BATCH_TIMEOUT=0.05
export BATCH_NUM_WORKERS=8
export ENABLE_ADAPTIVE_BATCHING=true
```

#### For Low Latency

```bash
export BATCH_MAX_SIZE=16
export BATCH_TIMEOUT=0.001
export ENABLE_ADAPTIVE_BATCHING=false
```

#### For Memory Constrained Environments

```bash
export BATCH_MAX_SIZE=32
export ENABLE_MEMORY_OPTIMIZATION=true
export MAX_BATCH_MEMORY_MB=512
```

### Health Monitoring

Monitor API health using the health endpoint:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed metrics (requires API key)
curl -H "X-API-Key: your-key" http://localhost:8000/metrics
```

### Support

For additional support:

1. Check the [GitHub Issues](https://github.com/KolosalAI/kolosal_automl/issues)
2. Review the [API Documentation](http://localhost:8000/docs)
3. Contact the development team

---

**Last Updated**: 2025-07-20  
**Version**: 0.1.4  
**API Base URL**: `http://localhost:8000`
