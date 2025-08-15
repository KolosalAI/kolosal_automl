# 🚀 kolosal AutoML - Quick Start Guide

This guide will get you up and running with kolosal AutoML in Docker within 5 minutes.

## Prerequisites

- Docker Engine 20.10+ with Docker Compose 2.0+
- At least 4GB RAM and 10GB disk space
- Ports 80, 443, 3000, 8000, 9090 available

## 1. Quick Setup (2 minutes)

```bash
# Clone and enter directory
git clone <repository-url>
cd kolosal-automl

# Generate secure configuration (recommended)
make setup-secure

# Or use development defaults
make setup-dev
```

**Windows users:**
```cmd
# Generate secure configuration
scripts\generate_secure_config.bat

# Or copy example
copy .env.example .env
```

## 2. Start Services (2 minutes)

```bash
# Start all services
make start

# Or manually with docker compose
docker compose up -d
```

**Windows users:**
```cmd
docker compose up -d
```

## 3. Verify Installation (1 minute)

```bash
# Run health check
make health-check

# Or manually
curl http://localhost:8000/health
```

**Windows users:**
```cmd
scripts\health_check.bat
```

## 4. Access Your AutoML Platform

🌐 **Main Endpoints:**
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090

## 5. First API Call

```bash
# Get API key from .env file
API_KEY=$(grep "^API_KEYS=" .env | cut -d'=' -f2 | cut -d',' -f1)

# Test API health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health

# Get system info
curl -H "X-API-Key: $API_KEY" http://localhost:8000/system/info
```

**PowerShell:**
```powershell
# Get API key
$API_KEY = (Get-Content .env | Where-Object {$_ -match "^API_KEYS="}) -split "=" | Select-Object -Last 1 | Split-String "," | Select-Object -First 1

# Test API
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health
```

## 6. Your First ML Training

```python
import requests

API_KEY = "your-api-key-here"
headers = {"X-API-Key": API_KEY}

# Upload training data
with open('your_data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/data/upload",
        files={"file": f},
        headers=headers
    )

# Start training
training_config = {
    "dataset_id": response.json()["dataset_id"],
    "target_column": "your_target_column",
    "task_type": "classification",  # or "regression"
    "algorithms": ["random_forest", "xgboost", "lightgbm"]
}

response = requests.post(
    "http://localhost:8000/training/start",
    json=training_config,
    headers=headers
)

print(f"Training job started: {response.json()['job_id']}")
```

## Quick Commands Reference

```bash
# Essential commands
make start          # Start all services
make stop           # Stop all services
make restart        # Restart services
make health-check   # Check system health
make logs           # View logs
make clean          # Clean up resources

# Development
make setup-dev      # Setup development environment
make test           # Run tests
make shell          # Enter API container

# Production
make setup-secure   # Generate secure configuration
make ssl-certs      # Generate SSL certificates
make backup         # Backup data
```

## Troubleshooting Quick Fixes

### API not responding
```bash
# Check container status
docker ps
docker compose logs kolosal-automl-api

# Restart API service
docker compose restart kolosal-automl-api
```

### Memory issues
```bash
# Check resource usage
docker stats

# Increase memory limits in docker-compose.yml
# Then restart services
```

### Permission errors
```bash
# Fix file permissions
chmod +x scripts/*.sh
sudo chown -R $USER:$USER volumes/
```

## Configuration Files

| File | Purpose | Required |
|------|---------|----------|
| `.env` | Main configuration | ✅ Yes |
| `docker-compose.yml` | Service definitions | ✅ Yes |
| `nginx.conf` | Reverse proxy config | ✅ Yes |
| `certs/server.crt` | SSL certificate | 🟡 Production |
| `monitoring/` | Monitoring configs | 🟡 Optional |

## Environment Variables Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | Generated | API authentication keys |
| `SECURITY_ENV` | `development` | Security mode |
| `ML_MAX_WORKERS` | `4` | ML processing threads |
| `REDIS_PASSWORD` | Generated | Redis authentication |
| `POSTGRES_PASSWORD` | Generated | Database password |

## Next Steps

1. 📚 **Read Documentation**: [Full API Documentation](http://localhost:8000/docs)
2. 🔧 **Customize Configuration**: Edit `.env` and `docker-compose.yml`
3. 📊 **Monitor Performance**: Access Grafana dashboard
4. 🔒 **Security Hardening**: Follow [SECURITY_SETUP.md](SECURITY_SETUP.md)
5. 🚀 **Scale Deployment**: See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)

## Support

- 🐛 **Issues**: Check container logs and health status
- 📖 **Documentation**: See `docs/` directory
- 🔧 **Configuration**: Review `DOCKER_README.md`

---

**🎉 You're ready to start building ML models with kolosal AutoML!**

For detailed configuration and advanced features, see the [Complete Docker Documentation](DOCKER_README.md).
