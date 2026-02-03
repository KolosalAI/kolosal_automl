@echo off
setlocal enabledelayedexpansion
REM Comprehensive health check script for kolosal AutoML (Windows)

echo 🏥 kolosal AutoML - Comprehensive Health Check
echo ==============================================

REM Check if Docker is running
echo.
echo 🐳 Docker Environment Check
echo ----------------------------

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [❌] Docker daemon is not running
    pause
    exit /b 1
)

echo [✅] Docker is running

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    echo [✅] Docker Compose v2 is available
    set COMPOSE_CMD=docker compose
) else (
    docker-compose --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✅] Docker Compose is available
        set COMPOSE_CMD=docker-compose
    ) else (
        echo [❌] Docker Compose is not available
        pause
        exit /b 1
    )
)

REM Check container status
echo.
echo 📦 Container Status Check
echo -------------------------

set containers=kolosal-automl-api kolosal-redis kolosal-nginx kolosal-prometheus kolosal-grafana

for %%c in (%containers%) do (
    docker ps --format "{{.Names}}" | findstr /C:"%%c" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [✅] %%c is running
    ) else (
        echo [⚠️] %%c is not running
    )
)

REM Check API health endpoint
echo.
echo 🔌 API Health Check
echo -------------------

REM Try to get API key from .env file
if exist .env (
    for /f "tokens=2 delims==" %%i in ('findstr "^API_KEYS=" .env 2^>nul') do (
        set API_KEY=%%i
        for /f "delims=," %%j in ("!API_KEY!") do set API_KEY=%%j
    )
)

REM Test API endpoints
set "endpoint_found=false"

REM Test localhost:8000
if defined API_KEY (
    curl -s -w "%%{http_code}" -H "X-API-Key: !API_KEY!" "http://localhost:8000/health" -o nul --connect-timeout 5 --max-time 10 2>nul | findstr "200" >nul
) else (
    curl -s -w "%%{http_code}" "http://localhost:8000/health" -o nul --connect-timeout 5 --max-time 10 2>nul | findstr "200" >nul
)

if !errorlevel! equ 0 (
    echo [✅] API is responding at http://localhost:8000/health
    set "endpoint_found=true"
    set "API_ENDPOINT=http://localhost:8000/health"
) else (
    REM Test HTTPS endpoints
    if defined API_KEY (
        curl -s -w "%%{http_code}" -H "X-API-Key: !API_KEY!" -k "https://localhost/health" -o nul --connect-timeout 5 --max-time 10 2>nul | findstr "200" >nul
    ) else (
        curl -s -w "%%{http_code}" -k "https://localhost/health" -o nul --connect-timeout 5 --max-time 10 2>nul | findstr "200" >nul
    )
    
    if !errorlevel! equ 0 (
        echo [✅] API is responding at https://localhost/health
        set "endpoint_found=true"
        set "API_ENDPOINT=https://localhost/health"
    ) else (
        echo [❌] API is not responding on any endpoint
    )
)

REM Check monitoring services
echo.
echo 📊 Monitoring Services Check
echo -----------------------------

REM Prometheus
curl -s http://localhost:9090/-/healthy >nul 2>&1
if %errorlevel% equ 0 (
    echo [✅] Prometheus is healthy
) else (
    echo [⚠️] Prometheus health check failed
)

REM Grafana
curl -s http://localhost:3000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [✅] Grafana is healthy
) else (
    echo [⚠️] Grafana health check failed
)

REM Check Redis
echo.
echo 💾 Database Services Check
echo ---------------------------

docker exec kolosal-redis redis-cli ping 2>nul | findstr "PONG" >nul
if %errorlevel% equ 0 (
    echo [✅] Redis is responding
) else (
    echo [⚠️] Redis health check failed
)

REM Check PostgreSQL (if running)
docker ps --format "{{.Names}}" | findstr "postgres" >nul 2>&1
if %errorlevel% equ 0 (
    for /f %%i in ('docker ps --format "{{.Names}}" ^| findstr postgres') do (
        docker exec %%i pg_isready -U kolosal 2>nul | findstr "accepting connections" >nul
        if !errorlevel! equ 0 (
            echo [✅] PostgreSQL is accepting connections
        ) else (
            echo [⚠️] PostgreSQL health check failed
        )
    )
)

REM Check system resources
echo.
echo 💻 System Resources Check
echo --------------------------

REM Check disk space (current drive)
for /f "tokens=3" %%i in ('dir /-c ^| findstr /C:"bytes free"') do set freespace=%%i
echo [ℹ️] Free disk space available

REM Security check
echo.
echo 🔐 Security Configuration Check
echo --------------------------------

if exist .env (
    findstr "SECURITY_ENV=production" .env >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✅] Security environment set to production
    ) else (
        echo [⚠️] Security environment not set to production
    )
    
    findstr "SECURITY_REQUIRE_API_KEY=true" .env >nul 2>&1
    if %errorlevel% equ 0 (
        echo [✅] API key authentication enabled
    ) else (
        echo [⚠️] API key authentication not enabled
    )
    
    findstr "admin123 password123 default" .env >nul 2>&1
    if %errorlevel% equ 0 (
        echo [❌] Default passwords detected - change them!
    ) else (
        echo [✅] No default passwords detected
    )
) else (
    echo [⚠️] .env file not found
)

REM SSL Certificate check
if exist "certs\server.crt" (
    if exist "certs\server.key" (
        echo [✅] SSL certificates present
    ) else (
        echo [⚠️] SSL private key not found
    )
) else (
    echo [⚠️] SSL certificates not found
)

REM Final summary
echo.
echo 📋 Health Check Summary
echo =======================

if "%endpoint_found%"=="true" (
    echo [✅] ✅ Core system is operational
    echo.
    echo 🌐 Access URLs:
    echo    API: %API_ENDPOINT%
    echo    API Documentation: %API_ENDPOINT:health=docs%
    echo    Grafana: http://localhost:3000
    echo    Prometheus: http://localhost:9090
    
    if defined API_KEY (
        echo.
        echo 🔑 Test command:
        echo    curl -H "X-API-Key: !API_KEY!" %API_ENDPOINT%
    )
) else (
    echo [❌] ❌ System needs attention
    echo.
    echo 🔧 Troubleshooting steps:
    echo    1. Check container logs: %COMPOSE_CMD% logs
    echo    2. Restart services: %COMPOSE_CMD% restart
    echo    3. Check configuration: type .env
    echo    4. Verify ports: netstat -an ^| findstr :8000
)

echo.
echo ==============================================
echo Health check completed at %date% %time%
echo ==============================================
