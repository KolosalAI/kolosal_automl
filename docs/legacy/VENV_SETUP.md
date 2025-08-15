# 🚀 Kolosal AutoML - Virtual Environment Production Setup

## Quick Setup (5 Minutes) ⚡

### 1. **Run the Automated Setup**
```batch
# Run the automated setup script
setup_venv.bat
```

This script will:
- ✅ Create and activate virtual environment (`.venv`)
- ✅ Install all required dependencies
- ✅ Generate secure API keys
- ✅ Create directory structure
- ✅ Set up monitoring and backup systems
- ✅ Configure production settings

### 2. **Start the System**
```batch
# Development mode (recommended for first run)
scripts\start_development.bat

# Production mode
scripts\start_production.bat
```

### 3. **Verify Everything Works**
```batch
# Run health check
scripts\health_check.bat

# Test API
curl http://localhost:8000/health
```

---

## Manual Setup (If You Prefer Control) 🛠️

### Step 1: Create Virtual Environment
```batch
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate.bat

# Verify activation
echo %VIRTUAL_ENV%
```

### Step 2: Install Dependencies
```batch
# Upgrade pip first
python -m pip install --upgrade pip

# Install core dependencies
pip install pydantic>=2.0.0 fastapi>=0.100.0 uvicorn[standard]>=0.20.0

# Install project with all dependencies
pip install -e .[all]

# Or if that fails, use requirements file
pip install -r requirements.txt
```

### Step 3: Fix Pydantic Compatibility
```batch
# If you get Pydantic errors, upgrade explicitly
pip install --upgrade pydantic>=2.0.0 fastapi>=0.100.0

# Verify installation
python -c "import pydantic; print(f'Pydantic version: {pydantic.version.VERSION}')"
```

### Step 4: Configure Environment
```batch
# Copy template and configure
copy .env.template .env

# Edit .env file and set:
# SECURITY_LEVEL=production
# REQUIRE_API_KEY=true
# API_KEYS=your_secure_api_key_here
```

### Step 5: Create Required Directories
```batch
mkdir logs models temp_data certs backups monitoring configs
```

### Step 6: Test the Setup
```batch
# Run production readiness check
python check_production_readiness.py

# Start the system
python main.py --mode api
```

---

## Production Deployment Checklist 📋

### ✅ **Environment Setup**
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] `.env` file configured for production
- [ ] Required directories created with proper permissions

### ✅ **Security Configuration**
- [ ] Strong API keys generated (32+ characters)
- [ ] HTTPS enforced in production
- [ ] Rate limiting enabled
- [ ] Input validation enabled
- [ ] Security headers configured

### ✅ **SSL/TLS Setup**
```batch
# For production, get real certificates from Let's Encrypt:
# certbot certonly --standalone -d yourdomain.com

# For testing, generate self-signed:
openssl req -x509 -newkey rsa:4096 -keyout certs\server.key -out certs\server.crt -days 365 -nodes
```

### ✅ **Docker Deployment** (Optional but Recommended)
```batch
# Build production image
docker build -t kolosal-automl:latest .

# Start full stack
docker-compose up -d
```

### ✅ **Monitoring Setup**
```batch
# Start monitoring stack (if using Docker)
docker-compose up -d prometheus grafana

# Access dashboards:
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

---

## Virtual Environment Best Practices 🎯

### **Activation Scripts**
Create convenient activation scripts:

**`activate_dev.bat`:**
```batch
@echo off
call .venv\Scripts\activate.bat
echo 🚀 Kolosal AutoML Development Environment Activated
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
python --version
echo.
echo Available commands:
echo   python main.py                    # Interactive mode
echo   python main.py --mode gui         # Web interface
echo   python main.py --mode api         # API server
echo   python check_production_readiness.py  # System check
```

### **Development Workflow**
```batch
# 1. Always activate venv first
.venv\Scripts\activate.bat

# 2. Install new dependencies
pip install new-package

# 3. Update requirements if needed
pip freeze > requirements_freeze.txt

# 4. Run your development tasks
python main.py --mode gui

# 5. Deactivate when done
deactivate
```

### **Production Deployment**
```batch
# 1. Activate production environment
.venv\Scripts\activate.bat

# 2. Set production variables
set API_ENV=production
set SECURITY_LEVEL=production

# 3. Start production server
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Troubleshooting Common Issues 🔧

### **Issue: Pydantic Import Error**
```batch
# Solution 1: Upgrade Pydantic
pip install --upgrade pydantic>=2.0.0

# Solution 2: Reinstall FastAPI stack
pip uninstall fastapi pydantic uvicorn
pip install fastapi>=0.100.0 pydantic>=2.0.0 uvicorn[standard]>=0.20.0
```

### **Issue: Virtual Environment Not Working**
```batch
# Delete and recreate
rmdir /S /Q .venv
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -e .[all]
```

### **Issue: Dependencies Conflict**
```batch
# Clean install approach
pip freeze > old_requirements.txt
pip uninstall -r old_requirements.txt -y
pip install -e .[all]
```

### **Issue: Port Already in Use**
```batch
# Find what's using the port
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /F /PID <PID>

# Or use different port
set API_PORT=8001
python main.py --mode api
```

---

## Production-Ready Validation ✅

Run this comprehensive check:

```batch
# 1. Activate environment
.venv\Scripts\activate.bat

# 2. Check Python and dependencies
python --version
python -c "import pydantic, fastapi, uvicorn; print('✅ Core dependencies OK')"

# 3. Check project imports
python -c "from modules.engine.train_engine import MLTrainingEngine; print('✅ Core modules OK')"

# 4. Run production readiness check
python check_production_readiness.py

# 5. Test API startup
timeout 5 >nul && python -c "
import requests
import time
import subprocess
import sys

# Start API in background
proc = subprocess.Popen([sys.executable, 'start_api.py'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)
time.sleep(3)

try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    print('✅ API health check passed')
    proc.terminate()
except:
    print('❌ API not responding')
    proc.terminate()
"
```

---

## Success Metrics 🎯

Your system is production-ready when:

- [ ] **Virtual Environment**: `.venv` activated and working
- [ ] **Dependencies**: All imports successful, no version conflicts
- [ ] **Security**: API keys generated, HTTPS configured
- [ ] **Health Checks**: All endpoints responding correctly
- [ ] **Performance**: System responds within acceptable timeframes
- [ ] **Monitoring**: Logging and metrics collection working
- [ ] **Backup**: Automated backup system configured
- [ ] **Documentation**: Team knows how to operate the system

---

## Quick Reference Commands 📚

```batch
# Environment Management
.venv\Scripts\activate.bat              # Activate virtual environment
deactivate                              # Deactivate virtual environment
pip list                                # Show installed packages
pip freeze > requirements.txt          # Export requirements

# System Operations
python main.py --mode gui              # Start web interface
python main.py --mode api              # Start API server
python check_production_readiness.py   # Check system status
scripts\health_check.bat               # Quick health check
scripts\backup.bat                     # Create system backup

# Development
python main.py --compile               # Compile for performance
python -m pytest tests/ -v             # Run test suite
docker-compose up -d                   # Start full stack

# Troubleshooting
python -c "import sys; print(sys.executable)"  # Find Python path
pip check                              # Check for dependency conflicts
python -c "import fastapi; print(fastapi.__version__)"  # Check versions
```

---

## 🎉 You're All Set!

Once you complete this setup:
1. Your system will be in a clean virtual environment
2. All dependencies will be properly isolated
3. Production configuration will be ready
4. Security measures will be in place
5. Monitoring and backup systems will be configured

**Next step**: Run `setup_venv.bat` and follow the prompts!
