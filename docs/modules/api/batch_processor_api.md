# Batch Processor API (`modules/api/batch_processor_api.py`)

## Overview

The Batch Processor API provides a RESTful interface for the BatchProcessor engine, offering advanced batching capabilities for ML workloads. It enables dynamic batch sizing, priority-based processing, and comprehensive monitoring through HTTP endpoints.

## Features

- **RESTful Interface**: HTTP API for batch processing operations
- **Dynamic Batch Sizing**: Automatic optimization of batch sizes
- **Priority-Based Processing**: Multi-level priority queue management
- **Async Processing**: Non-blocking batch processing with WebSocket support
-         "version": "0.1.4",*Performance Monitoring**: Real-time metrics and analytics
- **Security**: API key authentication and request validation
- **Scalability**: Support for high-throughput batch processing

## API Endpoints

### Core Batch Processing

#### Submit Batch Request
```http
POST /api/batch-processor/submit
Content-Type: application/json
X-API-Key: your-api-key

{
    "data": [
        {"input": [1, 2, 3, 4, 5]},
        {"input": [6, 7, 8, 9, 10]}
    ],
    "priority": "normal",
    "batch_id": "optional-batch-id",
    "timeout_ms": 5000,
    "metadata": {
        "model_id": "model-v1.0",
        "user_id": "user123"
    }
}
```

**Response:**
```json
{
    "batch_id": "batch_12345",
    "status": "queued",
    "estimated_completion_time": "2025-01-15T10:30:00Z",
    "queue_position": 3,
    "request_count": 2
}
```

#### Get Batch Status
```http
GET /api/batch-processor/status/{batch_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "batch_id": "batch_12345",
    "status": "processing",
    "progress": 0.75,
    "completed_requests": 150,
    "total_requests": 200,
    "estimated_completion_time": "2025-01-15T10:32:00Z",
    "processing_time_ms": 1500
}
```

#### Get Batch Results
```http
GET /api/batch-processor/results/{batch_id}
X-API-Key: your-api-key
```

**Response:**
```json
{
    "batch_id": "batch_12345",
    "status": "completed",
    "results": [
        {
            "request_id": "req_1",
            "output": [0.1, 0.9, 0.05],
            "confidence": 0.95,
            "processing_time_ms": 15
        },
        {
            "request_id": "req_2",
            "output": [0.8, 0.15, 0.05],
            "confidence": 0.87,
            "processing_time_ms": 12
        }
    ],
    "metadata": {
        "total_processing_time_ms": 1500,
        "average_per_request_ms": 7.5,
        "batch_size": 200
    }
}
```

### Batch Management

#### List Active Batches
```http
GET /api/batch-processor/batches?status=active&limit=10
X-API-Key: your-api-key
```

#### Cancel Batch
```http
DELETE /api/batch-processor/cancel/{batch_id}
X-API-Key: your-api-key
```

#### Batch Configuration
```http
POST /api/batch-processor/configure
Content-Type: application/json
X-API-Key: your-api-key

{
    "max_batch_size": 64,
    "max_wait_time_ms": 50,
    "enable_dynamic_batching": true,
    "priority_weights": {
        "critical": 1.0,
        "high": 0.8,
        "normal": 0.5,
        "low": 0.2
    }
}
```

### Monitoring and Analytics

#### Get System Metrics
```http
GET /api/batch-processor/metrics
X-API-Key: your-api-key
```

**Response:**
```json
{
    "current_stats": {
        "active_batches": 5,
        "queue_length": 150,
        "average_batch_size": 45.2,
        "throughput_per_second": 1250.5,
        "average_latency_ms": 25.8
    },
    "performance_stats": {
        "total_processed": 1000000,
        "success_rate": 0.995,
        "error_rate": 0.005,
        "average_processing_time_ms": 18.5
    },
    "resource_usage": {
        "cpu_usage_percent": 65.2,
        "memory_usage_mb": 2048.5,
        "gpu_usage_percent": 78.9
    }
}
```

#### Get Performance Analytics
```http
GET /api/batch-processor/analytics?period=1h&metrics=throughput,latency
X-API-Key: your-api-key
```

## Usage Examples

### Python Client Usage

```python
import requests
import json
import time

class BatchProcessorClient:
    def __init__(self, base_url="http://localhost:8000", api_key="your-api-key"):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def submit_batch(self, data, priority="normal", timeout_ms=5000, metadata=None):
        """Submit a batch for processing"""
        payload = {
            "data": data,
            "priority": priority,
            "timeout_ms": timeout_ms,
            "metadata": metadata or {}
        }
        
        response = requests.post(
            f"{self.base_url}/api/batch-processor/submit",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to submit batch: {response.text}")
    
    def get_batch_status(self, batch_id):
        """Get the status of a batch"""
        response = requests.get(
            f"{self.base_url}/api/batch-processor/status/{batch_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get batch status: {response.text}")
    
    def get_batch_results(self, batch_id):
        """Get the results of a completed batch"""
        response = requests.get(
            f"{self.base_url}/api/batch-processor/results/{batch_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get batch results: {response.text}")
    
    def wait_for_completion(self, batch_id, check_interval=1.0, timeout=300):
        """Wait for batch completion with polling"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_batch_status(batch_id)
            
            if status["status"] == "completed":
                return self.get_batch_results(batch_id)
            elif status["status"] == "failed":
                raise Exception(f"Batch failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(check_interval)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")

# Example usage
client = BatchProcessorClient()

# Prepare batch data
batch_data = [
    {"input": [1.0, 2.0, 3.0, 4.0]},
    {"input": [5.0, 6.0, 7.0, 8.0]},
    {"input": [9.0, 10.0, 11.0, 12.0]}
]

# Submit batch
response = client.submit_batch(
    data=batch_data,
    priority="high",
    metadata={"experiment_id": "exp_001"}
)

batch_id = response["batch_id"]
print(f"Submitted batch: {batch_id}")

# Wait for completion
try:
    results = client.wait_for_completion(batch_id, timeout=60)
    print(f"Batch completed successfully")
    print(f"Results: {len(results['results'])} items processed")
except Exception as e:
    print(f"Batch processing failed: {e}")
```

### Async Python Client

```python
import aiohttp
import asyncio
import json

class AsyncBatchProcessorClient:
    def __init__(self, base_url="http://localhost:8000", api_key="your-api-key"):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def submit_batch(self, session, data, priority="normal", metadata=None):
        """Submit a batch asynchronously"""
        payload = {
            "data": data,
            "priority": priority,
            "metadata": metadata or {}
        }
        
        async with session.post(
            f"{self.base_url}/api/batch-processor/submit",
            headers=self.headers,
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                raise Exception(f"Failed to submit batch: {text}")
    
    async def get_batch_results(self, session, batch_id):
        """Get batch results asynchronously"""
        async with session.get(
            f"{self.base_url}/api/batch-processor/results/{batch_id}",
            headers=self.headers
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                text = await response.text()
                raise Exception(f"Failed to get results: {text}")
    
    async def process_multiple_batches(self, batch_datasets):
        """Process multiple batches concurrently"""
        async with aiohttp.ClientSession() as session:
            # Submit all batches
            submit_tasks = []
            for i, data in enumerate(batch_datasets):
                task = self.submit_batch(
                    session, 
                    data, 
                    priority="normal",
                    metadata={"batch_index": i}
                )
                submit_tasks.append(task)
            
            # Wait for all submissions
            submissions = await asyncio.gather(*submit_tasks)
            batch_ids = [sub["batch_id"] for sub in submissions]
            
            print(f"Submitted {len(batch_ids)} batches")
            
            # Poll for results
            all_results = []
            for batch_id in batch_ids:
                while True:
                    try:
                        results = await self.get_batch_results(session, batch_id)
                        all_results.append(results)
                        break
                    except:
                        await asyncio.sleep(1)  # Wait and retry
            
            return all_results

# Example async usage
async def main():
    client = AsyncBatchProcessorClient()
    
    # Create multiple batch datasets
    batch_datasets = [
        [{"input": [i, i+1, i+2]} for i in range(j*10, (j+1)*10)]
        for j in range(5)  # 5 batches of 10 items each
    ]
    
    # Process all batches concurrently
    results = await client.process_multiple_batches(batch_datasets)
    
    print(f"Processed {len(results)} batches")
    for i, result in enumerate(results):
        print(f"Batch {i}: {len(result['results'])} items processed")

# Run async example
# asyncio.run(main())
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class BatchProcessorClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = 'your-api-key') {
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }
    
    async submitBatch(data, priority = 'normal', metadata = {}) {
        const payload = {
            data: data,
            priority: priority,
            metadata: metadata
        };
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/batch-processor/submit`,
                payload,
                { headers: this.headers }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Failed to submit batch: ${error.response?.data || error.message}`);
        }
    }
    
    async getBatchStatus(batchId) {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/batch-processor/status/${batchId}`,
                { headers: this.headers }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get batch status: ${error.response?.data || error.message}`);
        }
    }
    
    async getBatchResults(batchId) {
        try {
            const response = await axios.get(
                `${this.baseUrl}/api/batch-processor/results/${batchId}`,
                { headers: this.headers }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get batch results: ${error.response?.data || error.message}`);
        }
    }
    
    async waitForCompletion(batchId, checkInterval = 1000, timeout = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            const status = await this.getBatchStatus(batchId);
            
            if (status.status === 'completed') {
                return await this.getBatchResults(batchId);
            } else if (status.status === 'failed') {
                throw new Error(`Batch failed: ${status.error || 'Unknown error'}`);
            }
            
            await new Promise(resolve => setTimeout(resolve, checkInterval));
        }
        
        throw new Error(`Batch ${batchId} did not complete within ${timeout}ms`);
    }
}

// Example usage
async function exampleUsage() {
    const client = new BatchProcessorClient();
    
    const batchData = [
        { input: [1, 2, 3, 4] },
        { input: [5, 6, 7, 8] },
        { input: [9, 10, 11, 12] }
    ];
    
    try {
        // Submit batch
        const submission = await client.submitBatch(
            batchData,
            'high',
            { experimentId: 'exp_001' }
        );
        
        console.log(`Submitted batch: ${submission.batch_id}`);
        
        // Wait for completion
        const results = await client.waitForCompletion(submission.batch_id);
        console.log(`Batch completed: ${results.results.length} items processed`);
        
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
}

// exampleUsage();
```

## WebSocket Support

### Real-time Batch Updates

```python
import websockets
import json
import asyncio

class BatchProcessorWebSocketClient:
    def __init__(self, ws_url="ws://localhost:8000/ws/batch-processor"):
        self.ws_url = ws_url
        
    async def connect(self, api_key):
        """Connect to WebSocket with authentication"""
        headers = {"X-API-Key": api_key}
        self.websocket = await websockets.connect(self.ws_url, extra_headers=headers)
        
    async def subscribe_to_batch(self, batch_id):
        """Subscribe to real-time updates for a batch"""
        message = {
            "action": "subscribe",
            "batch_id": batch_id
        }
        await self.websocket.send(json.dumps(message))
        
    async def listen_for_updates(self, callback):
        """Listen for real-time batch updates"""
        async for message in self.websocket:
            data = json.loads(message)
            await callback(data)
    
    async def close(self):
        """Close WebSocket connection"""
        await self.websocket.close()

# Example WebSocket usage
async def batch_update_handler(update):
    """Handle real-time batch updates"""
    print(f"Batch {update['batch_id']}: {update['status']} - {update['progress']:.1%}")
    
    if update['status'] == 'completed':
        print(f"Batch completed in {update['total_time_ms']}ms")

async def websocket_example():
    client = BatchProcessorWebSocketClient()
    
    try:
        await client.connect("your-api-key")
        await client.subscribe_to_batch("batch_12345")
        await client.listen_for_updates(batch_update_handler)
    finally:
        await client.close()

# asyncio.run(websocket_example())
```

## Configuration and Deployment

### Docker Deployment

```dockerfile
# Dockerfile for Batch Processor API
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY modules/ modules/
COPY main.py .

# Set environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=4
ENV REQUIRE_API_KEY=true

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "modules.api.batch_processor_api", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```bash
# Environment variables for production deployment
export API_ENV=production
export API_DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4
export REQUIRE_API_KEY=true
export API_KEYS=prod_key_1,prod_key_2,prod_key_3

# Batch processor configuration
export MAX_BATCH_SIZE=128
export MAX_WAIT_TIME_MS=50
export MAX_QUEUE_SIZE=10000
export ENABLE_DYNAMIC_BATCHING=true

# Performance settings
export ENABLE_MONITORING=true
export LOG_LEVEL=INFO
export MAX_CONCURRENT_BATCHES=50
```

### Production Monitoring

```python
# Production monitoring setup
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics definitions
batch_requests_total = Counter('batch_requests_total', 'Total batch requests')
batch_processing_time = Histogram('batch_processing_seconds', 'Batch processing time')
active_batches = Gauge('active_batches', 'Number of active batches')
queue_length = Gauge('queue_length', 'Current queue length')

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Increment request counter
    batch_requests_total.inc()
    
    # Process request
    response = await call_next(request)
    
    # Record processing time
    process_time = time.time() - start_time
    batch_processing_time.observe(process_time)
    
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
```

## Error Handling and Best Practices

### Robust Error Handling

```python
from fastapi import HTTPException
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    error_id = str(uuid.uuid4())
    
    # Log error with trace
    logging.error(f"Error {error_id}: {str(exc)}")
    logging.error(f"Traceback: {traceback.format_exc()}")
    
    # Return user-friendly error
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Input validation
class BatchRequest(BaseModel):
    data: List[Dict[str, Any]]
    priority: str = Field(default="normal", regex="^(critical|high|normal|low)$")
    timeout_ms: int = Field(default=5000, ge=100, le=300000)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 items")
        return v
```

## Related Documentation

- [Batch Processor Engine Documentation](../engine/batch_processor.md)
- [Dynamic Batcher Documentation](../engine/dynamic_batcher.md)
- [Batch Statistics Documentation](../engine/batch_stats.md)
- [API Authentication Documentation](app.md#authentication)
- [Performance Monitoring Documentation](../engine/performance_metrics.md)
