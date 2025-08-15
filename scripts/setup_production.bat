@echo off
REM Kolosal AutoML - Production Setup Script for Windows
REM This script automates the production setup process

echo 🚀 Kolosal AutoML - Production Setup (Windows)
echo ===============================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Desktop with Compose.
    pause
    exit /b 1
)

echo [INFO] Docker and Docker Compose are installed ✓

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.9+ first.
    pause
    exit /b 1
)

echo [INFO] Python is installed ✓

REM Create necessary directories
echo === Setting up Directory Structure ===
mkdir logs 2>nul
mkdir models 2>nul
mkdir temp_data 2>nul
mkdir certs 2>nul
mkdir backups 2>nul
mkdir monitoring 2>nul
mkdir monitoring\prometheus 2>nul
mkdir monitoring\grafana 2>nul
mkdir monitoring\grafana\dashboards 2>nul
mkdir scripts 2>nul

echo [INFO] Created directory structure ✓

REM Generate .env file if not exists
echo === Configuring Environment ===
if not exist .env (
    if exist .env.template (
        copy .env.template .env >nul
        echo [INFO] Created .env from template
        
        REM Generate secure credentials (basic version for Windows)
        python -c "import secrets; print('Generated API key: genta_' + secrets.token_urlsafe(32))"
        echo [WARNING] Please manually update the .env file with production values
        echo [WARNING] Change SECURITY_LEVEL=development to SECURITY_LEVEL=production
        echo [WARNING] Change REQUIRE_API_KEY=false to REQUIRE_API_KEY=true
        echo [WARNING] Update API_KEYS with the generated key above
    ) else (
        echo [ERROR] .env.template not found. Please ensure you're in the correct directory.
        pause
        exit /b 1
    )
) else (
    echo [INFO] .env file already exists ✓
)

REM Generate SSL certificates for development/testing
echo === Setting up SSL Certificates ===
if not exist certs\server.crt (
    echo [WARNING] Generating self-signed SSL certificates for testing...
    echo [WARNING] For production, replace these with valid certificates from a CA!
    
    REM Check if OpenSSL is available
    openssl version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        openssl req -x509 -newkey rsa:4096 -keyout certs\server.key -out certs\server.crt -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        echo [INFO] Generated self-signed SSL certificates ✓
    ) else (
        echo [WARNING] OpenSSL not found. Please generate SSL certificates manually.
        echo [WARNING] Place server.crt and server.key in the certs\ directory
    )
) else (
    echo [INFO] SSL certificates already exist ✓
)

REM Create monitoring configuration
echo === Setting up Monitoring ===
if not exist monitoring\prometheus.yml (
    (
        echo global:
        echo   scrape_interval: 15s
        echo   evaluation_interval: 15s
        echo:
        echo scrape_configs:
        echo   - job_name: 'kolosal-automl'
        echo     static_configs:
        echo       - targets: ['kolosal-api:8000']
        echo     metrics_path: '/metrics'
        echo     scrape_interval: 30s
        echo:
        echo   - job_name: 'prometheus'
        echo     static_configs:
        echo       - targets: ['localhost:9090']
    ) > monitoring\prometheus.yml
    echo [INFO] Created Prometheus configuration ✓
)

REM Create backup script for Windows
echo === Setting up Backup System ===
if not exist scripts\backup.bat (
    (
        echo @echo off
        echo REM Kolosal AutoML Backup Script for Windows
        echo set DATE=%%date:~10,4%%%%date:~4,2%%%%date:~7,2%%_%%time:~0,2%%%%time:~3,2%%%%time:~6,2%%
        echo set DATE=%%DATE: =0%%
        echo set BACKUP_DIR=backups\%%DATE%%
        echo:
        echo echo Starting backup at %%date%% %%time%%
        echo:
        echo REM Create backup directory
        echo mkdir %%BACKUP_DIR%% 2^>nul
        echo:
        echo REM Backup critical directories
        echo xcopy models %%BACKUP_DIR%%\models\ /E /I /Q 2^>nul
        echo xcopy configs %%BACKUP_DIR%%\configs\ /E /I /Q 2^>nul
        echo xcopy logs %%BACKUP_DIR%%\logs\ /E /I /Q 2^>nul
        echo copy .env %%BACKUP_DIR%%\ 2^>nul
        echo:
        echo REM Create zip archive ^(requires PowerShell^)
        echo powershell Compress-Archive -Path %%BACKUP_DIR%% -DestinationPath backups\kolosal_backup_%%DATE%%.zip
        echo:
        echo REM Remove temporary directory
        echo rmdir /S /Q %%BACKUP_DIR%% 2^>nul
        echo:
        echo echo Backup completed: kolosal_backup_%%DATE%%.zip
    ) > scripts\backup.bat
    echo [INFO] Created backup script ✓
)

REM Create health check script for Windows
if not exist scripts\health_check.bat (
    (
        echo @echo off
        echo echo 🏥 Kolosal AutoML Health Check
        echo echo ==============================
        echo:
        echo REM Check Docker containers
        echo echo Docker Containers:
        echo docker-compose ps
        echo:
        echo REM Check API health
        echo echo API Health Check:
        echo curl -s -f http://localhost:8000/health ^>nul 2^>^&1
        echo if %%ERRORLEVEL%% equ 0 ^(
        echo     echo ✅ API is healthy
        echo ^) else ^(
        echo     echo ❌ API health check failed
        echo ^)
        echo:
        echo REM Check system resources
        echo echo System Resources:
        echo echo Memory Usage:
        echo wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value
        echo:
        echo echo Disk Usage:
        echo dir /s /-c backups logs models
    ) > scripts\health_check.bat
    echo [INFO] Created health check script ✓
)

REM Install Python dependencies
echo === Installing Dependencies ===
echo [INFO] Installing Python dependencies...
pip install -e .[all] --quiet
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies may have failed to install, but core functionality should work
)

REM Compile Python bytecode for performance
echo === Optimizing Performance ===
echo [INFO] Compiling Python bytecode for better performance...
python main.py --compile >nul 2>&1 || (
    echo [WARNING] Bytecode compilation skipped - this is normal
)

REM Build Docker images
echo === Building Docker Images ===
echo [INFO] Building production Docker image...
docker build -t kolosal-automl:latest . --quiet
if %ERRORLEVEL% equ 0 (
    echo [INFO] Docker image built successfully ✓
) else (
    echo [WARNING] Docker build had issues, but may still work
)

REM Final setup verification
echo === Verifying Setup ===
if exist .env echo ✅ .env file exists
if exist certs\server.crt (echo ✅ SSL certificates exist) else (echo ❌ SSL certificates missing)
if exist scripts\backup.bat echo ✅ Backup script exists
if exist monitoring\prometheus.yml echo ✅ Monitoring config exists

echo:
echo === Setup Complete! ===
echo 🎉 Kolosal AutoML production setup is complete!
echo:
echo Next steps:
echo 1. Review and customize the .env file with your specific settings
echo 2. Replace self-signed SSL certificates with valid ones for production
echo 3. Start the services: docker-compose up -d
echo 4. Run health checks: scripts\health_check.bat
echo 5. Access the web interface: https://localhost:7860
echo 6. Access the API: https://localhost:8000
echo 7. Access Grafana: http://localhost:3000 (admin/admin123)
echo:
echo 🔐 Important: Change default passwords and secure your API keys!
echo 📚 Read PRODUCTION_CHECKLIST.md for detailed production guidelines
echo:
echo [INFO] Setup script completed successfully!
echo:
pause
