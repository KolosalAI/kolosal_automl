# 🐳 kolosal AutoML Docker Deployment Guide

This comprehensive guide covers deploying kolosal AutoML using Docker and Docker Compose with production-ready configurations, security features, and monitoring.

## 🚀 Quick Start (60 seconds!)

### Prerequisites

- **Docker Engine 20.10+** and **Docker Compose 2.0+**
- **8GB+ RAM** available for containers (4GB minimum)
- **20GB+ free disk space** (models + logs + monitoring data)
- **Python 3.9+** (for local development and setup scripts)

### 🎯 Ultra Quick Start (New!)

```bash
# Automated setup (Windows)
.\setup_venv.bat

# Automated setup (Linux/macOS)  
chmod +x scripts/setup_production.sh
./scripts/setup_production.sh

# Or use Make
make setup && make dev
```

### Development Mode (Recommended for testing)

```bash
# Clone and setup
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl

# Automated setup with virtual environment
setup_venv.bat  # Windows
# OR
./scripts/setup_production.sh  # Linux/macOS

# Quick start with Make
make dev

# Manual setup (if needed)
cp .env.example .env
# Edit .env with your settings
docker-compose -f compose.yaml -f compose.dev.yaml up -d
```

### Production Mode

```bash
# Automated production setup
./scripts/setup_production.sh  # Linux/macOS
# OR 
scripts\setup_production.bat   # Windows

# Quick start with Make
make prod

# Manual production setup
cp .env.example .env
# Configure production settings in .env
docker-compose up -d
```

## 📋 Available Services

| Service | Port | Purpose | Development | Production | Health Check |
|---------|------|---------|-------------|------------|--------------|
| **API Server** | 8000 | Main ML API | ✅ | ✅ | `/health` |
| **Redis** | 6379 | Caching & Sessions | ✅ | ✅ | `redis-cli ping` |
| **Nginx** | 80/443 | Reverse Proxy | ❌ | ✅ | Auto-configured |
| **Prometheus** | 9090 | Metrics Collection | ❌ | ✅ | `/-/healthy` |
| **Grafana** | 3000 | Monitoring Dashboard | ❌ | ✅ | `/api/health` |
| **Loki** | 3100 | Log Aggregation | ❌ | ✅ | `/ready` |
| **Jupyter** | 8888 | Development Notebook | ✅ | ❌ | Token-free access |
| **PostgreSQL** | 5432 | Database (optional) | ✅ | Optional | `pg_isready` |
| **Documentation** | 8002 | Hot-reload docs | ✅ | ❌ | MkDocs server |

### 🌟 New Features (v0.2.0)
- **Multi-stage Docker builds** for optimized production images
- **Advanced security headers** and rate limiting with Nginx
- **Comprehensive monitoring** with Prometheus + Grafana + Loki
- **Automated backup system** with scheduled jobs
- **SSL/TLS support** with auto-generation for development
- **Health checks** for all services with detailed reporting
- **Resource limits** and optimization for production workloads

## 🏗️ Enhanced Architecture

```mermaid
graph TB
    A[User] --> B[Nginx Load Balancer]
    B --> C[kolosal API Cluster]
    C --> D[Redis Cache]
    C --> E[PostgreSQL DB]
    C --> F[Model Storage]
    
    G[Prometheus] --> C
    G --> D
    G --> E
    H[Grafana] --> G
    I[Loki] --> C
    H --> I
    
    subgraph "Security Layer"
        J[SSL/TLS]
        K[Rate Limiting]
        L[API Authentication]
        M[CORS Protection]
    end
    
    B --> J
    B --> K
    C --> L
    C --> M
    
    subgraph "Development Tools"
        N[Jupyter Lab]
        O[Hot Reload Docs]
        P[Debug Tools]
    end
    
    C -.-> N
    C -.-> O
    C -.-> P
```

**Production Architecture Features:**
- **Load-balanced API** with automatic scaling
- **Multi-layer security** (SSL, rate limiting, authentication)
- **Comprehensive monitoring** (metrics, logs, alerts)
- **High availability** with health checks and auto-restart
- **Data persistence** with backup and recovery systems

## ⚙️ Advanced Configuration

### Environment Variables (Enhanced)

**Critical Production Settings:**
```bash
# Security Configuration (Required for Production)
SECURITY_ENV=production
SECURITY_REQUIRE_API_KEY=true
SECURITY_ENABLE_RATE_LIMITING=true
SECURITY_ENABLE_JWT=true
SECURITY_ENABLE_AUDIT_LOGGING=true
SECURITY_ENFORCE_HTTPS=true
SECURITY_HSTS_MAX_AGE=31536000

# API Configuration
API_ENV=production
API_DEBUG=false
API_WORKERS=4
API_PORT=8000
API_MEMORY_LIMIT=4G
API_CPU_LIMIT=2.0

# Batch Processing (New in v0.2.0)
BATCH_MAX_SIZE=128
BATCH_TIMEOUT=0.02
BATCH_NUM_WORKERS=8
ENABLE_ADAPTIVE_BATCHING=true
ENABLE_MEMORY_OPTIMIZATION=true

# Monitoring & Observability
ENABLE_MONITORING=true
ENABLE_AUDIT_LOGGING=true
LOG_LEVEL=INFO
PROMETHEUS_RETENTION=7d
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure-password-123

# SSL/TLS Configuration
TLS_CERT_PATH=./certs/server.crt
TLS_KEY_PATH=./certs/server.key
```

### 🔐 Security Configuration Generator

```bash
# Quick security setup (Windows)
scripts\generate_secure_config.bat

# Quick security setup (Linux/macOS)
./scripts/generate_secure_config.sh

# Manual generation
python generate_ssl.py  # SSL certificates
python -c "import secrets; print('API_KEYS=' + ','.join([f'genta_{secrets.token_urlsafe(32)}' for _ in range(3)]))"
```

### Volume Mounts & Data Persistence

**Production-ready data management:**

```yaml
volumes:
  # Persistent model storage with backup
  kolosal-models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./volumes/models
  
  # Structured logging with rotation
  kolosal-logs:
    driver: local
    driver_opts:
      type: none  
      o: bind
      device: ./volumes/logs
  
  # High-performance temporary data
  kolosal-temp:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=2G,uid=1000
```

**Directory Structure:**
```
volumes/
├── models/          # Trained ML models (persistent)
├── logs/           # Application logs (rotated daily)
├── temp/           # Temporary processing data
├── backups/        # Automated backups
└── monitoring/     # Metrics and monitoring data

configs/            # Configuration files (read-only)
├── default/        # Default configurations
├── production/     # Production overrides
└── security/       # Security policies
```

## 🛠️ Enhanced Development Workflow

### 1. Automated Development Setup

```bash
# Quick development environment (Windows)
setup_venv.bat
scripts\start_development.bat

# Quick development environment (Linux/macOS)
./scripts/setup_production.sh --mode development
./scripts/start_development.sh

# With Make (cross-platform)
make dev
make dev-logs    # Follow logs in real-time
```

### 2. Development Features

**Hot Reload & Live Development:**
- **API hot reload** - Code changes restart automatically
- **Jupyter Lab** - Interactive development environment
- **Live documentation** - MkDocs with hot reload at :8002
- **Database UI** - PostgreSQL admin interface
- **Redis CLI** - Direct cache access

**Enhanced Development Tools:**
```bash
# Development container with full toolchain
make shell                    # Interactive shell in API container
make shell-redis             # Redis CLI access
make jupyter                 # Start Jupyter Lab
make docs                    # Live documentation server

# Code quality tools
make lint                    # Run code linting
make format                  # Auto-format code  
make test                    # Run test suite
make test-coverage           # Test with coverage report
```

### 3. Access Services (Development)

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | No auth required |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Jupyter** | http://localhost:8888 | Token-free access |
| **Documentation** | http://localhost:8002 | Live reload MkDocs |
| **PostgreSQL** | localhost:5432 | kolosal/dev_password_123 |
| **Redis CLI** | `docker exec -it kolosal-redis redis-cli` | No auth |

### 4. Development Commands

```bash
# Container management
make dev-status             # Show container status
make dev-restart            # Restart development containers
make dev-rebuild            # Rebuild and restart
make dev-clean              # Clean development environment

# Development utilities  
make generate-test-data     # Generate sample datasets
make benchmark              # Run performance benchmarks
make profile                # Profile API performance
make debug                  # Start with debugger attached

# Database management
make db-migrate             # Run database migrations
make db-seed               # Seed with sample data
make db-backup             # Backup development database
make db-reset              # Reset to clean state
```

## 🔧 Production Deployment (Complete Guide)

### 1. Automated Production Setup

```bash
# Linux/macOS - Full automated setup
curl -sSL https://get.docker.com | sh  # Install Docker (if needed)
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl
chmod +x scripts/setup_production.sh
./scripts/setup_production.sh

# Windows - PowerShell as Administrator
git clone https://github.com/Genta-Technology/kolosal-automl.git
cd kolosal-automl
.\scripts\setup_production.bat

# With Make (recommended)
make setup-production
make prod
```

### 2. Manual Production Configuration

**Step 1: Environment Setup**
```bash
# Copy and configure environment
cp .env.example .env

# Generate secure credentials
python -c "import secrets; print('API_KEYS=' + ','.join([f'genta_{secrets.token_urlsafe(32)}' for _ in range(3)]))"
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(64))"

# Edit .env with production values
nano .env  # Linux/macOS
notepad .env  # Windows
```

**Step 2: SSL/TLS Setup**
```bash
# Option 1: Let's Encrypt (recommended for production)
sudo certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/server.crt
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/server.key

# Option 2: Self-signed (development only)
python generate_ssl.py

# Option 3: Manual OpenSSL
openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes
```

**Step 3: Security Hardening**
```bash
# Configure firewall (Linux)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Block direct API access
sudo ufw enable

# Windows Firewall
netsh advfirewall firewall add rule name="HTTP" dir=in action=allow protocol=TCP localport=80
netsh advfirewall firewall add rule name="HTTPS" dir=in action=allow protocol=TCP localport=443
netsh advfirewall firewall add rule name="Block Direct API" dir=in action=block protocol=TCP localport=8000
```

### 3. Deploy Production Environment

```bash
# Build and start production services
make prod

# Or manual deployment
docker-compose build --no-cache
docker-compose up -d

# Verify deployment
make health-check
make status
curl -k https://localhost/health
```

### 4. Production Verification Checklist

**✅ System Health:**
```bash
# Run comprehensive health check
python check_production_readiness.py

# Verify all services
make health-check

# Check resource usage  
make system-status

# Test performance
make load-test
```

**✅ Security Verification:**
```bash
# Security scan
make security-scan

# SSL certificate check
openssl x509 -in certs/server.crt -text -noout

# Test authentication
curl -H "X-API-Key: your-key" https://localhost/api/health

# Verify rate limiting
for i in {1..20}; do curl -s https://localhost/health; done
```

## 📊 Advanced Monitoring & Observability

### Comprehensive Monitoring Stack

**Prometheus Metrics Collection:**
```yaml
# Custom metrics available at /metrics
- kolosal_api_requests_total
- kolosal_model_training_duration_seconds
- kolosal_batch_processing_queue_size
- kolosal_memory_usage_bytes
- kolosal_gpu_utilization_percent
- kolosal_model_accuracy_score
```

**Access Monitoring Services:**
| Service | URL | Default Login | Purpose |
|---------|-----|---------------|---------|
| **Prometheus** | http://localhost:9090 | No auth | Metrics & Alerting |
| **Grafana** | http://localhost:3000 | admin/admin123 | Dashboards & Visualization |
| **Loki** | http://localhost:3100 | No auth | Log Aggregation |

### Pre-configured Grafana Dashboards

**1. API Performance Dashboard:**
- Request rates and response times
- Error rates and status codes  
- API endpoint performance breakdown
- Geographic request distribution

**2. ML Model Dashboard:**
- Model training metrics
- Batch processing performance
- Model accuracy trends
- Resource utilization during training

**3. System Resource Dashboard:**
- CPU, Memory, Disk usage
- Container health status
- Network throughput
- Database performance metrics

**4. Security Dashboard:**
- Failed authentication attempts
- Rate limiting events
- Suspicious request patterns
- Security audit logs

### Custom Monitoring Setup

```bash
# Install custom dashboards
make install-dashboards

# Configure alerting rules
make setup-alerts

# Test monitoring stack
make test-monitoring

# Export metrics to external systems
make export-metrics --target prometheus
```

### Log Management with Loki

**Structured Logging Features:**
- **JSON format** for easy parsing
- **Log levels** (DEBUG, INFO, WARN, ERROR, CRITICAL)
- **Correlation IDs** for request tracing
- **Performance metrics** embedded in logs
- **Security events** with context

**Log Query Examples:**
```logql
# Recent errors
{container="kolosal-automl-api"} |= "ERROR" | json

# Performance issues  
{container="kolosal-automl-api"} | json | duration > 1s

# Security events
{container="kolosal-automl-api"} | json | event_type="security_event"

# Model training logs
{container="kolosal-automl-api"} | json | component="model_trainer"
```

## 🧪 Comprehensive Testing & Validation

### Automated Testing Suite

```bash
# Complete test suite with Docker
python test_docker.py --mode production

# Quick development testing
python test_docker.py --mode development --skip-build

# Configuration validation (no Docker required)
python validate_docker_config.py

# Production readiness check
python check_production_readiness.py
```

### Make Commands for Testing

```bash
# Docker-based testing
make test                    # Full test suite
make test-quick             # Skip image rebuild
make test-integration       # Integration tests only
make test-performance       # Performance benchmarks
make test-security          # Security validation

# Load testing
make load-test              # Basic load test
make stress-test            # High-load stress test
make load-test-extended     # Extended duration test
```

### Manual API Testing

**Basic Health Checks:**
```bash
# Health endpoint
curl -f http://localhost:8000/health

# System information
curl -f http://localhost:8000/system/info

# API documentation (browser)
open http://localhost:8000/docs     # macOS
start http://localhost:8000/docs    # Windows
xdg-open http://localhost:8000/docs # Linux
```

**Authenticated API Testing:**
```bash
# Set your API key
export API_KEY="your-generated-api-key"

# Test with authentication
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/models

# Test ML endpoints
curl -X POST "http://localhost:8000/api/train" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2],[3,4]], "target": [0,1], "model_type": "random_forest"}'
```

### Performance Testing

**Built-in Load Testing:**
```bash
# Quick performance check
make benchmark

# Load testing with Locust
pip install locust
locust -f scripts/load_test.py --host http://localhost:8000

# Memory and CPU profiling
make profile
make memory-test
```

**Custom Performance Tests:**
```python
# Example: Custom API performance test
import requests
import time
import statistics

def test_api_performance(url, num_requests=100):
    times = []
    for _ in range(num_requests):
        start = time.time()
        response = requests.get(f"{url}/health")
        end = time.time()
        if response.status_code == 200:
            times.append(end - start)
    
    print(f"Mean: {statistics.mean(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s") 
    print(f"95th percentile: {statistics.quantiles(times, n=20)[18]:.3f}s")

# Run performance test
test_api_performance("http://localhost:8000")
```

## 🔒 Production Security (Enterprise-Grade)

### Multi-layered Security Architecture

**1. Network Security:**
- **TLS 1.3** encryption for all communications
- **HSTS** (HTTP Strict Transport Security) enabled
- **Rate limiting** with configurable thresholds
- **DDoS protection** with request filtering
- **IP whitelisting** for restricted access

**2. Application Security:**
- **JWT-based authentication** with rotation
- **API key authentication** with scoping
- **Input validation** against XSS/injection attacks
- **CSRF protection** for web interfaces
- **Security headers** (CSP, X-Frame-Options, etc.)

**3. Container Security:**
- **Non-root user** execution (UID 10001)
- **Read-only root filesystem** where possible
- **Minimal base images** (distroless/alpine)
- **Security scanning** integrated in CI/CD
- **Resource limits** to prevent resource exhaustion

### Security Configuration

**Quick Security Setup:**
```bash
# Generate secure configuration
python scripts/generate_secure_config.py

# Apply security hardening
make security-harden

# Run security audit
make security-audit
```

**Manual Security Configuration:**
```bash
# Generate strong API keys (32+ characters)
python -c "import secrets; print('API_KEYS=' + ','.join([f'genta_{secrets.token_urlsafe(32)}' for _ in range(3)]))"

# Generate JWT secret (64+ characters)  
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(64))"

# Generate admin password (16+ characters with symbols)
python -c "import secrets, string; chars=string.ascii_letters+string.digits+'!@#$%^&*'; print('ADMIN_PASSWORD=' + ''.join(secrets.choice(chars) for _ in range(16)))"
```

### Security Best Practices Implementation

**Environment Variables for Security:**
```bash
# Authentication & Authorization
SECURITY_REQUIRE_API_KEY=true
SECURITY_ENABLE_JWT=true
SECURITY_JWT_EXPIRY_HOURS=1        # Short expiry for production
SECURITY_ENABLE_SESSION_VALIDATION=true
SECURITY_ENABLE_CSRF_PROTECTION=true

# Network Security
SECURITY_ENFORCE_HTTPS=true
SECURITY_HSTS_MAX_AGE=31536000     # 1 year
SECURITY_ALLOWED_ORIGINS=https://yourdomain.com
SECURITY_ENABLE_RATE_LIMITING=true
SECURITY_RATE_LIMIT_REQUESTS=50    # Adjust based on needs
SECURITY_RATE_LIMIT_WINDOW=60

# Audit & Monitoring
SECURITY_ENABLE_AUDIT_LOGGING=true
SECURITY_LOG_SECURITY_EVENTS=true
SECURITY_ALERT_ON_THREATS=true
ENABLE_INTRUSION_DETECTION=true
```

### Security Monitoring & Alerting

**Real-time Security Monitoring:**
- **Failed authentication attempts** tracking
- **Suspicious request patterns** detection  
- **Rate limit violations** alerting
- **Unusual API usage** monitoring
- **Security event correlation** across services

**Automated Security Responses:**
- **IP blocking** for malicious actors
- **Account lockout** after failed attempts
- **Rate limit escalation** during attacks
- **Alert notifications** via email/Slack/webhooks
- **Automatic backup triggering** during incidents

## 🚀 Performance Optimization & Scaling

### Resource Optimization

**Recommended System Resources:**
```bash
# Minimum (Development)
CPU: 2 cores, RAM: 4GB, Storage: 20GB

# Production (Small)
CPU: 4 cores, RAM: 8GB, Storage: 50GB

# Production (Large)  
CPU: 8 cores, RAM: 16GB, Storage: 100GB

# High Performance (Enterprise)
CPU: 16+ cores, RAM: 32GB+, Storage: 200GB+ NVMe
```

**Container Resource Limits:**
```yaml
# In compose.yaml
services:
  kolosal-api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

**Environment Configuration for Performance:**
```bash
# API Performance
API_WORKERS=4                    # Scale based on CPU cores
API_MEMORY_LIMIT=4G             # Adjust based on available RAM
API_CPU_LIMIT=2.0               # CPU allocation

# Batch Processing Optimization
BATCH_MAX_SIZE=128              # Increase for better throughput
BATCH_NUM_WORKERS=8             # Parallel processing workers
ENABLE_ADAPTIVE_BATCHING=true   # Dynamic batch sizing
ENABLE_MEMORY_OPTIMIZATION=true # Memory-efficient processing
ENABLE_JIT_COMPILATION=true     # Just-in-time compilation

# Cache Optimization
REDIS_MAXMEMORY=512mb           # Cache size
ENABLE_MODEL_CACHING=true       # Cache trained models
ENABLE_DATA_CACHING=true        # Cache preprocessed data
```

### Horizontal Scaling

**1. Load Balancing with Multiple API Instances:**
```bash
# Scale API containers
docker-compose up -d --scale kolosal-api=3

# Nginx automatically load balances across instances
# Check load balancing status
curl http://localhost/health -H "X-Show-Instance: true"
```

**2. Database Scaling:**
```bash
# PostgreSQL with read replicas
make setup-db-replicas

# Redis clustering for high availability
make setup-redis-cluster
```

**3. Container Orchestration (Production):**
```yaml
# docker-compose.prod.yml
version: '3.9'
services:
  kolosal-api:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

### Performance Monitoring

**Built-in Performance Metrics:**
```bash
# Real-time performance monitoring
make performance-dashboard

# Detailed performance analysis
make performance-report

# Memory profiling
make memory-profile

# CPU profiling
make cpu-profile
```

**Performance Testing:**
```bash
# Load testing with different scenarios
make load-test-light        # 100 concurrent users
make load-test-medium       # 500 concurrent users  
make load-test-heavy        # 1000 concurrent users
make load-test-burst        # Burst traffic simulation

# Performance benchmarking
make benchmark-training     # ML training performance
make benchmark-inference    # Model inference speed
make benchmark-api         # API response times
```

### Optimization Best Practices

**1. Model Optimization:**
- **Model quantization** for faster inference
- **Model pruning** to reduce size
- **Batch processing** for multiple predictions
- **Model caching** for frequently used models

**2. Data Pipeline Optimization:**
- **Asynchronous processing** for I/O operations
- **Connection pooling** for database access
- **Memory mapping** for large datasets
- **Streaming processing** for real-time data

**3. Infrastructure Optimization:**
- **SSD storage** for model persistence
- **Redis clustering** for distributed caching
- **CDN integration** for static assets
- **Load balancer optimization** for traffic distribution

## 🛠️ Maintenance & Operations

### Automated Backup System

**Backup Strategy:**
```bash
# Automated daily backups
make setup-backup-cron

# Manual backup
make backup-full            # Complete system backup
make backup-models         # Models only
make backup-data           # Data and configurations
make backup-logs           # Log archives

# Backup verification
make verify-backups

# Restore from backup
make restore BACKUP_DATE=20240815
```

**Backup Configuration:**
```bash
# Environment variables for backup
BACKUP_SCHEDULE="0 2 * * *"        # Daily at 2 AM
BACKUP_RETENTION_DAYS=30           # Keep 30 days
BACKUP_COMPRESSION=true            # Compress backups
BACKUP_ENCRYPTION=true             # Encrypt sensitive data
BACKUP_REMOTE_SYNC=true            # Sync to remote storage
```

### Update & Maintenance Procedures

**System Updates:**
```bash
# Check for updates
make check-updates

# Update Docker images
make update-images

# Update application code
git pull origin main
make rebuild
make restart

# Database migrations
make db-migrate

# Configuration updates  
make update-configs
make validate-configs
```

**Health Monitoring:**
```bash
# Continuous health monitoring
make monitor-health

# System diagnostics
make system-diagnostics

# Performance monitoring
make performance-monitor

# Log analysis
make analyze-logs
```

### Automated Maintenance Tasks

**Daily Tasks (Automated):**
- Health check execution
- Log rotation and cleanup
- Backup verification
- Security scan
- Performance metrics collection

**Weekly Tasks:**
- System resource analysis
- Security audit
- Dependency updates check
- Performance optimization review

**Monthly Tasks:**
- Full security assessment
- Backup restore testing
- Disaster recovery drill
- Configuration review and updates

### Cleanup & Optimization

**Regular Cleanup:**
```bash
# Clean Docker system
make clean-docker

# Clean application data
make clean-logs
make clean-temp
make clean-cache

# Complete cleanup (careful!)
make clean-all
```

**Storage Management:**
```bash
# Monitor disk usage
make disk-usage

# Clean old backups
make clean-old-backups

# Optimize database
make optimize-database

# Compress logs
make compress-logs
```

## 📋 Comprehensive Troubleshooting Guide

### Common Issues & Solutions

**🔧 Container Startup Issues**

*Problem: Containers won't start*
```bash
# Diagnosis
make status
make logs
docker system df    # Check disk space

# Solutions
make clean          # Clean old containers
make build          # Rebuild images
make restart        # Restart services

# Check for port conflicts
netstat -tulpn | grep :8000    # Linux
netstat -an | findstr :8000    # Windows
```

*Problem: Out of memory errors*
```bash
# Check system resources
docker stats
free -h             # Linux
systeminfo | find "Available Physical Memory"  # Windows

# Increase memory limits in .env
API_MEMORY_LIMIT=8G
REDIS_MEMORY_LIMIT=1G

# Restart with new limits
make restart
```

**🌐 API Connection Issues**

*Problem: API not responding*
```bash
# Step-by-step diagnosis
curl -v http://localhost:8000/health    # Test connection
docker logs kolosal-automl-api         # Check API logs
docker exec kolosal-automl-api ps aux  # Check processes

# Check network connectivity
docker network ls
docker network inspect kolosal-network

# Test from inside container
docker exec -it kolosal-automl-api curl localhost:8000/health
```

*Problem: Authentication failures*
```bash
# Verify API key configuration
grep API_KEYS .env
docker exec kolosal-automl-api env | grep API

# Test with correct API key
export API_KEY="your-actual-api-key"
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health

# Check security settings
grep SECURITY .env
```

**🗄️ Database Connection Issues**

*Problem: PostgreSQL connection failed*
```bash
# Check PostgreSQL status
docker logs kolosal-postgres-dev
docker exec -it kolosal-postgres-dev pg_isready

# Test connection manually
docker exec -it kolosal-postgres-dev psql -U kolosal -d kolosal_dev -c "SELECT version();"

# Reset database if needed
make db-reset
```

**📊 Monitoring Issues**

*Problem: Grafana/Prometheus not accessible*
```bash
# Check monitoring services
curl http://localhost:9090/-/healthy    # Prometheus
curl http://localhost:3000/api/health   # Grafana

# Check service logs
docker logs kolosal-prometheus
docker logs kolosal-grafana

# Restart monitoring stack
make monitoring-restart
```

### Performance Troubleshooting

**🚀 Slow API Response Times**

*Diagnosis:*
```bash
# Performance testing
make benchmark
time curl http://localhost:8000/health

# Check resource usage
docker stats kolosal-automl-api
make system-diagnostics

# Profile the application
make profile
```

*Solutions:*
```bash
# Increase workers
API_WORKERS=8

# Enable optimizations
ENABLE_JIT_COMPILATION=true
ENABLE_MEMORY_OPTIMIZATION=true
ENABLE_MODEL_CACHING=true

# Scale horizontally
docker-compose up -d --scale kolosal-api=3
```

**💾 High Memory Usage**

*Diagnosis:*
```bash
# Memory profiling
make memory-profile
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check for memory leaks
make memory-leak-check
```

*Solutions:*
```bash
# Enable memory optimization
ENABLE_MEMORY_OPTIMIZATION=true
BATCH_MAX_SIZE=64

# Increase container limits
API_MEMORY_LIMIT=8G

# Restart services periodically
make restart-daily  # Sets up automatic restart
```

### Security Troubleshooting

**🔒 SSL/TLS Issues**

*Problem: SSL certificate errors*
```bash
# Check certificate validity
openssl x509 -in certs/server.crt -text -noout
openssl x509 -in certs/server.crt -checkend 86400

# Test SSL connection
openssl s_client -connect localhost:443 -servername localhost

# Regenerate certificates if needed
python generate_ssl.py
make restart
```

**🛡️ Security Alerts**

*Problem: Security warnings in logs*
```bash
# Check security logs
make security-logs
grep "SECURITY" volumes/logs/*.log

# Review failed authentication attempts
grep "Authentication failed" volumes/logs/*.log | tail -20

# Check for suspicious patterns
make security-analysis
```

### Advanced Troubleshooting

**🔍 Deep Debugging**

*Enable debug mode:*
```bash
# Development debugging
API_DEBUG=true
LOG_LEVEL=DEBUG
make dev

# Production debugging (temporary)
docker exec -it kolosal-automl-api bash
# Inside container:
tail -f /app/logs/kolosal.log
```

*Performance profiling:*
```bash
# CPU profiling
make cpu-profile

# Memory profiling
make memory-profile

# Network profiling
make network-profile

# Full system analysis
make full-diagnostics > diagnostics_report.txt
```

### Emergency Recovery Procedures

**🚨 Complete System Recovery**

*If everything is broken:*
```bash
# Nuclear option - complete reset
make emergency-reset

# Step-by-step recovery
make stop-all
make clean-all
make build
make setup-production
make deploy
make verify
```

*Data recovery:*
```bash
# Restore from backup
make list-backups
make restore BACKUP_DATE=latest

# Manual data recovery
docker cp kolosal-automl-api:/app/models ./recovery/models
docker cp kolosal-automl-api:/app/configs ./recovery/configs
```

### Getting Help

**📞 Support Channels**

1. **Self-Service:**
   ```bash
   make help                    # Show all available commands
   make diagnostics            # Generate diagnostic report
   python check_production_readiness.py  # Production readiness check
   ```

2. **Documentation:**
   - Read `PRODUCTION_CHECKLIST.md` for production setup
   - Check `SECURITY_SETUP.md` for security configuration
   - Review `docs/` directory for detailed documentation

3. **Community Support:**
   - Create GitHub issues with diagnostic reports
   - Include logs and configuration (remove sensitive data)
   - Use the issue template provided

4. **Enterprise Support:**
   - Contact Genta Technology support team
   - Include full diagnostic report: `make full-diagnostics`
   - Provide system specifications and deployment details

## 🔗 Integration Examples & API Usage

### Python Client Integration

**Basic Client Setup:**
```python
import requests
import json
from typing import Dict, Any, Optional

class KolosalClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        response = self.session.get(f"{self.base_url}/system/info")
        response.raise_for_status()
        return response.json()
    
    def train_model(self, data: list, target: list, model_type: str = "random_forest") -> Dict[str, Any]:
        """Train a machine learning model"""
        payload = {
            "data": data,
            "target": target,
            "model_type": model_type,
            "config": {"n_estimators": 100, "random_state": 42}
        }
        
        response = self.session.post(f"{self.base_url}/api/train", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_id: str, data: list) -> Dict[str, Any]:
        """Make predictions using trained model"""
        payload = {"data": data}
        response = self.session.post(f"{self.base_url}/api/models/{model_id}/predict", json=payload)
        response.raise_for_status()
        return response.json()

# Usage example
client = KolosalClient(api_key="your-api-key")
health = client.health_check()
print(f"API Status: {health['status']}")

# Train a simple model
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
result = client.train_model(X.tolist(), y.tolist(), "random_forest")
print(f"Model trained: {result['model_id']}")
```

### JavaScript/Node.js Integration

```javascript
// kolosal-client.js
const axios = require('axios');

class KolosalClient {
    constructor(baseURL = 'http://localhost:8000', apiKey = null) {
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 30000,
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
    }

    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }

    async getSystemInfo() {
        const response = await this.client.get('/system/info');
        return response.data;
    }

    async trainModel(data, target, modelType = 'random_forest') {
        const response = await this.client.post('/api/train', {
            data: data,
            target: target,
            model_type: modelType,
            config: { n_estimators: 100, random_state: 42 }
        });
        return response.data;
    }

    async predict(modelId, data) {
        const response = await this.client.post(`/api/models/${modelId}/predict`, {
            data: data
        });
        return response.data;
    }
}

// Usage
const client = new KolosalClient('http://localhost:8000', 'your-api-key');

(async () => {
    try {
        const health = await client.healthCheck();
        console.log('API Status:', health.status);
        
        const systemInfo = await client.getSystemInfo();
        console.log('System Info:', systemInfo);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
})();
```

### cURL Command Examples

**Basic Operations:**
```bash
# Health check
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/health

# System information
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/system/info

# List available models
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/api/models

# Get model details
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/api/models/model-id-123
```

**Training and Prediction:**
```bash
# Train a model
curl -X POST \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [[1,2,3],[4,5,6],[7,8,9]],
       "target": [0,1,1],
       "model_type": "random_forest",
       "config": {"n_estimators": 100}
     }' \
     http://localhost:8000/api/train

# Make predictions
curl -X POST \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [[2,3,4],[5,6,7]]
     }' \
     http://localhost:8000/api/models/your-model-id/predict

# Batch prediction
curl -X POST \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [[1,2],[3,4],[5,6],[7,8]],
       "batch_size": 2
     }' \
     http://localhost:8000/api/batch/predict
```

### Docker Compose Integration

**External Service Integration:**
```yaml
# docker-compose.override.yml - Add external services
version: '3.9'

services:
  # Add your external database
  external-db:
    image: postgres:15
    environment:
      POSTGRES_DB: external_data
      POSTGRES_USER: external_user
      POSTGRES_PASSWORD: external_pass
    volumes:
      - external_db_data:/var/lib/postgresql/data
    networks:
      - kolosal-network

  # Connect kolosal to external services
  kolosal-api:
    environment:
      EXTERNAL_DB_URL: postgresql://external_user:external_pass@external-db:5432/external_data
      EXTERNAL_API_ENDPOINT: https://api.external-service.com
      EXTERNAL_API_KEY: your-external-api-key
    depends_on:
      - external-db

volumes:
  external_db_data:

# Start with override
# docker-compose -f compose.yaml -f docker-compose.override.yml up -d
```

### Monitoring Integration

**Prometheus Metrics Scraping:**
```yaml
# prometheus.yml - Add custom scraping
scrape_configs:
  - job_name: 'kolosal-custom-metrics'
    static_configs:
      - targets: ['kolosal-api:8000']
    metrics_path: '/custom/metrics'
    scrape_interval: 15s
    
  - job_name: 'external-service'
    static_configs:
      - targets: ['your-service:9090']
```

**Grafana Dashboard Integration:**
```json
{
  "dashboard": {
    "title": "Custom kolosal Integration",
    "panels": [
      {
        "title": "API Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kolosal_api_request_duration_seconds_sum[5m]) / rate(kolosal_api_request_duration_seconds_count[5m])",
            "legendFormat": "Average Response Time"
          }
        ]
      }
    ]
  }
}
```

## 📚 Additional Resources & Documentation

### 📖 Complete Documentation Suite

**Core Documentation:**
- **[PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)** - Complete production deployment guide
- **[SECURITY_SETUP.md](SECURITY_SETUP.md)** - Comprehensive security configuration
- **[VENV_SETUP.md](VENV_SETUP.md)** - Virtual environment setup guide
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI (when running)
- **[docs/](docs/)** - Detailed technical documentation

**Setup Scripts & Utilities:**
- **[setup_venv.bat](setup_venv.bat)** - Windows automated setup
- **[scripts/setup_production.sh](scripts/setup_production.sh)** - Linux/macOS setup
- **[check_production_readiness.py](check_production_readiness.py)** - Readiness validator
- **[test_docker.py](test_docker.py)** - Docker testing suite
- **[validate_docker_config.py](validate_docker_config.py)** - Configuration validator

### 🛠️ Development Tools

**Docker Tools:**
- **[Dockerfile](Dockerfile)** - Multi-stage production Docker image
- **[compose.yaml](compose.yaml)** - Production Docker Compose
- **[compose.dev.yaml](compose.dev.yaml)** - Development overrides
- **[Makefile](Makefile)** - Build and deployment automation
- **[nginx.conf](nginx.conf)** - Production Nginx configuration

**Monitoring & Observability:**
- **[monitoring/prometheus.yml](monitoring/prometheus.yml)** - Prometheus configuration
- **Grafana Dashboards** - Pre-built monitoring dashboards
- **Log aggregation** - Loki for centralized logging
- **Health checks** - Comprehensive service monitoring

### 🌐 External Resources

**Docker & Container Technologies:**
- [Docker Documentation](https://docs.docker.com/) - Official Docker docs
- [Docker Compose Reference](https://docs.docker.com/compose/) - Compose file format
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/) - Production guidelines
- [Container Security](https://docs.docker.com/engine/security/) - Security best practices

**FastAPI & Python:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - FastAPI framework
- [Uvicorn Documentation](https://www.uvicorn.org/) - ASGI server
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/) - Data validation

**Monitoring & Observability:**
- [Prometheus Documentation](https://prometheus.io/docs/) - Metrics collection
- [Grafana Documentation](https://grafana.com/docs/) - Visualization
- [Loki Documentation](https://grafana.com/docs/loki/) - Log aggregation
- [Nginx Documentation](https://nginx.org/en/docs/) - Reverse proxy

### 🎓 Learning Resources

**Docker & Containerization:**
- [Docker Tutorial](https://docker-curriculum.com/) - Beginner-friendly guide
- [Docker Compose Tutorial](https://docs.docker.com/compose/gettingstarted/) - Multi-container apps
- [Production Docker](https://dockertldr.com/) - Production deployment patterns

**Machine Learning Operations (MLOps):**
- [MLOps Practices](https://ml-ops.org/) - MLOps community guidelines
- [Model Deployment](https://neptune.ai/blog/ml-model-deployment) - Deployment strategies
- [ML Monitoring](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/) - Model monitoring

**API Development:**
- [REST API Design](https://restfulapi.net/) - API design principles
- [API Security](https://owasp.org/www-project-api-security/) - Security best practices
- [Rate Limiting](https://blog.bytebytego.com/p/rate-limiting-fundamentals) - Traffic control

### 🔧 Advanced Configuration Examples

**Custom Environment Configurations:**
```bash
# .env.staging - Staging environment
API_ENV=staging
SECURITY_ENV=staging
ENABLE_DEBUG_ENDPOINTS=true
LOG_LEVEL=INFO
API_WORKERS=2

# .env.production - Production environment  
API_ENV=production
SECURITY_ENV=production
ENABLE_DEBUG_ENDPOINTS=false
LOG_LEVEL=WARNING
API_WORKERS=8
```

**Multi-Environment Docker Compose:**
```bash
# Development
docker-compose -f compose.yaml -f compose.dev.yaml up

# Staging
docker-compose -f compose.yaml -f compose.staging.yaml up

# Production
docker-compose -f compose.yaml up
```

**Custom Makefile Targets:**
```makefile
# Add to existing Makefile
staging: ## Deploy to staging environment
	docker-compose -f compose.yaml -f compose.staging.yaml up -d

performance-test: ## Run performance tests
	python test_docker.py --mode production --performance-only

security-audit: ## Run security audit
	docker run --rm -v $(PWD):/app securecodewarrior/docker-security-scan /app

compliance-check: ## Check compliance requirements
	python scripts/compliance_checker.py --standard SOC2
```

## 🤝 Community & Support

### 💬 Getting Help

**Community Support:**
1. **GitHub Issues** - Report bugs and request features
2. **Discussions** - Ask questions and share experiences  
3. **Wiki** - Community-maintained documentation
4. **Discord/Slack** - Real-time community chat

**Professional Support:**
1. **Enterprise Support** - Dedicated support for production deployments
2. **Consulting Services** - Custom implementation and optimization
3. **Training Programs** - Team training on kolosal AutoML

### 🤝 Contributing

**Ways to Contribute:**
- **Bug Reports** - Help us improve by reporting issues
- **Feature Requests** - Suggest new functionality
- **Documentation** - Improve guides and examples
- **Code Contributions** - Submit pull requests
- **Testing** - Help test new releases

**Development Setup for Contributors:**
```bash
# Fork and clone repository
git clone https://github.com/YOUR-USERNAME/kolosal-automl.git
cd kolosal-automl

# Set up development environment
make dev-setup
make dev

# Run tests before submitting PRs
make test-all
make lint
make security-check
```

---

## ✨ What's New in v0.2.0

### 🚀 Major Improvements

- **🏗️ Multi-stage Docker builds** - Optimized production images
- **🔒 Enhanced security** - JWT, CSRF protection, audit logging  
- **📊 Advanced monitoring** - Prometheus + Grafana + Loki integration
- **🧪 Comprehensive testing** - Automated Docker testing suite
- **📦 Automated setup** - One-command production deployment
- **⚡ Performance optimization** - JIT compilation, memory management
- **🛡️ Production hardening** - Security headers, rate limiting, SSL/TLS

### 🔄 Migration from v0.1.x

```bash
# Backup existing data
make backup-full

# Update configuration
cp .env .env.backup
cp .env.example .env
# Migrate settings from .env.backup to new .env

# Update and restart
git pull origin main
make clean
make build
make prod

# Verify migration
make health-check
make test-integration
```

## 🎉 Success! You're Ready for Production

Once you've completed this setup:

✅ **Scalable Architecture** - Load-balanced API with auto-scaling  
✅ **Enterprise Security** - Multi-layer security with audit logging  
✅ **Production Monitoring** - Comprehensive observability stack  
✅ **Automated Operations** - Backup, updates, and maintenance  
✅ **High Availability** - Health checks and auto-recovery  
✅ **Performance Optimized** - Caching, JIT compilation, resource management

**Quick Start Commands:**
```bash
# 🚀 Ultra-fast production setup
./setup_venv.bat                    # Windows
./scripts/setup_production.sh       # Linux/macOS
make prod                           # Start production environment
make health-check                   # Verify everything works
```

**Access Your Deployment:**
- 🌐 **Web Interface**: https://localhost:7860
- 🔌 **API Server**: https://localhost:8000  
- 📊 **Monitoring**: http://localhost:3000
- 📚 **Documentation**: http://localhost:8000/docs

---

**🔐 Remember:** Always secure your API keys, use HTTPS in production, and regularly update your deployment!

**📞 Need Help?** Check [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) for detailed guidance or create a GitHub issue.
