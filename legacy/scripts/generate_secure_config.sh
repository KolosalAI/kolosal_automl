#!/bin/bash
# Generate secure configuration for kolosal AutoML (Linux/macOS)

echo "🔐 Generating secure configuration for kolosal AutoML..."
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    exit 1
fi

# Run the configuration generator
python3 scripts/generate_secure_config.py "$@"

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Secure configuration generated successfully!"
    echo "📚 Next steps:"
    echo "   1. Review the .env file and customize as needed"
    echo "   2. Start the services: docker-compose up -d"
    echo "   3. Test the API: ./scripts/health_check.sh"
    echo "   4. Access monitoring: http://localhost:3000"
else
    echo ""
    echo "❌ Configuration generation failed"
    echo "Please check the error messages above"
fi
