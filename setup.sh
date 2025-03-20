#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models uploads

# Generate a secure secret key and update .env file
echo "Generating secure secret key..."
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
sed -i.bak "s/your_generated_secret_key_here_please_change_in_production/$SECRET_KEY/g" .env

echo "Setup complete! Run the application with: uvicorn main:app --reload"