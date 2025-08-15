@echo off
REM Generate secure configuration for kolosal AutoML (Windows)

echo 🔐 Generating secure configuration for kolosal AutoML...
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run the configuration generator
python scripts\generate_secure_config.py %*

REM Check if generation was successful
if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ Secure configuration generated successfully!
    echo 📚 Next steps:
    echo    1. Review the .env file and customize as needed
    echo    2. Start the services: docker-compose up -d
    echo    3. Test the API: scripts\health_check.bat
    echo    4. Access monitoring: http://localhost:3000
) else (
    echo.
    echo ❌ Configuration generation failed
    echo Please check the error messages above
)

pause
