# Script name: run_all.ps1
# Purpose: Set up environment and run Streamlit app and API server

# Change to the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir
Write-Host "Working directory set to: $(Get-Location)"

# Define the virtual environment directory
$VenvDir = "venv"
$RequirementsFile = "requirements.txt"

Write-Host "=== Starting Setup and Launch Script ===" -ForegroundColor Yellow

# Check if virtual environment exists
if (-not (Test-Path $VenvDir)) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv $VenvDir
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
if ($IsWindows -or $env:OS -match "Windows") {
    & "$VenvDir\Scripts\Activate.ps1"
} else {
    & "$VenvDir/bin/Activate.ps1"
}
Write-Host "Virtual environment activated." -ForegroundColor Green

# Check if requirements file exists
if (Test-Path $RequirementsFile) {
    Write-Host "Checking and installing dependencies..." -ForegroundColor Yellow
    pip install -r $RequirementsFile
    Write-Host "Dependencies installed/verified." -ForegroundColor Green
} else {
    Write-Host "Warning: $RequirementsFile not found. Installing basic requirements..." -ForegroundColor Red
    pip install streamlit pytest
    Write-Host "Basic dependencies installed." -ForegroundColor Green
}

# Run tests
Write-Host "Running tests with pytest..." -ForegroundColor Yellow
python -m pytest -vv
Write-Host "Tests completed." -ForegroundColor Green

# Function to run the Streamlit app
function Start-Streamlit {
    Write-Host "Starting Streamlit app..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "-m", "streamlit", "run", "app.py" -NoNewWindow
}

# Function to run the API
function Start-API {
    Write-Host "Starting API server..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "modules/api/app.py" -NoNewWindow
}

# Run both applications
Write-Host "Starting both applications..." -ForegroundColor Yellow
$streamlitJob = Start-Job -ScriptBlock { 
    Set-Location $using:ScriptDir
    & "$using:VenvDir\Scripts\python.exe" -m streamlit run app.py 
}
$apiJob = Start-Job -ScriptBlock { 
    Set-Location $using:ScriptDir
    & "$using:VenvDir\Scripts\python.exe" "modules/api/app.py" 
}

Write-Host "Both applications are now running." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop all servers." -ForegroundColor Yellow

try {
    # Keep the script running
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if jobs are still running
        $streamlitStatus = Receive-Job -Job $streamlitJob -Keep
        $apiStatus = Receive-Job -Job $apiJob -Keep
        
        if ($streamlitJob.State -ne "Running" -or $apiJob.State -ne "Running") {
            Write-Host "One of the applications has stopped." -ForegroundColor Red
            break
        }
    }
}
finally {
    # Clean up when script is interrupted
    Write-Host "Shutting down servers..." -ForegroundColor Red
    Stop-Job -Job $streamlitJob
    Stop-Job -Job $apiJob
    Remove-Job -Job $streamlitJob
    Remove-Job -Job $apiJob
}