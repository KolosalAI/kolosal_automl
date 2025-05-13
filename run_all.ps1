# Script name: run_all.ps1
# Purpose: Set up environment and run Streamlit app only

# Change to the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir
Write-Host "Working directory set to: $(Get-Location)"

# Define the virtual environment directory
$VenvDir = "venv"
Write-Host "=== Starting Streamlit App ===" -ForegroundColor Yellow

# Check if virtual environment exists
if (-not (Test-Path $VenvDir)) {
    Write-Host "Virtual environment not found. Please create one manually." -ForegroundColor Yellow
} else {
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    if ($IsWindows -or $env:OS -match "Windows") {
        & "$VenvDir\Scripts\Activate.ps1"
    } else {
        & "$VenvDir/bin/Activate.ps1"
    }
    Write-Host "Virtual environment activated." -ForegroundColor Green
    
    # Run Streamlit app directly
    Write-Host "Starting Streamlit app..." -ForegroundColor Yellow
    streamlit run app.py
}