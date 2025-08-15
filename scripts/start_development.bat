@echo off
echo 🚀 Starting Kolosal AutoML Development Server
echo =============================================
call .venv\Scripts\activate.bat
echo Starting API server...
python start_api.py
