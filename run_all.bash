#!/bin/bash

# Script name: run_all.sh
# Purpose: Set up environment and run Streamlit app and API server

set -e  # Exit immediately if a command exits with a non-zero status

# Change to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo "Working directory set to: $(pwd)"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define the virtual environment directory
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

echo -e "${YELLOW}=== Starting Setup and Launch Script ===${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python -m venv $VENV_DIR
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate || source $VENV_DIR/Scripts/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Check if requirements file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Checking and installing dependencies...${NC}"
    pip install -r $REQUIREMENTS_FILE
    echo -e "${GREEN}Dependencies installed/verified.${NC}"
else
    echo -e "${RED}Warning: $REQUIREMENTS_FILE not found. Installing basic requirements...${NC}"
    pip install streamlit pytest
    echo -e "${GREEN}Basic dependencies installed.${NC}"
fi

# Run tests
echo -e "${YELLOW}Running tests with pytest...${NC}"
python -m pytest -vv
echo -e "${GREEN}Tests completed.${NC}"

# Function to run the Streamlit app
run_streamlit() {
    echo -e "${YELLOW}Starting Streamlit app...${NC}"
    streamlit run app.py
}

# Function to run the API
run_api() {
    echo -e "${YELLOW}Starting API server...${NC}"
    python modules/api/app.py
}

# Use trap to handle script termination
trap "echo -e '${RED}Shutting down servers...${NC}'; exit" INT TERM

# Run both applications in background
echo -e "${YELLOW}Starting both applications...${NC}"
run_streamlit & STREAMLIT_PID=$!
run_api & API_PID=$!

echo -e "${GREEN}Both applications are now running.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all servers.${NC}"

# Wait for both processes
wait $STREAMLIT_PID $API_PID