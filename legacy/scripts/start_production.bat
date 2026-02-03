@echo off
echo 🚀 Starting Kolosal AutoML Production Server
echo ============================================
call .venv\Scripts\activate.bat
echo Starting production API server...
set API_ENV=production
set SECURITY_LEVEL=production
uvicorn modules.api.app:app --host 0.0.0.0 --port 8000 --workers 4
