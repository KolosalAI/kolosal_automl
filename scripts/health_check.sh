#!/bin/bash
# Comprehensive health check script for kolosal AutoML

echo "🏥 kolosal AutoML - Comprehensive Health Check"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️]${NC} $1"
}

# Check if Docker is running
echo ""
echo "🐳 Docker Environment Check"
echo "----------------------------"

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

print_status "Docker is running"

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose is available"
elif command -v docker compose &> /dev/null; then
    print_status "Docker Compose (v2) is available"
    alias docker-compose="docker compose"
else
    print_error "Docker Compose is not available"
    exit 1
fi

# Check container status
echo ""
echo "📦 Container Status Check"
echo "-------------------------"

# Check if containers are running
CONTAINERS=(
    "kolosal-automl-api"
    "kolosal-redis"
    "kolosal-nginx"
    "kolosal-prometheus"
    "kolosal-grafana"
)

for container in "${CONTAINERS[@]}"; do
    if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
        health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [ "$health" = "healthy" ]; then
            print_status "$container is running and healthy"
        elif [ "$health" = "starting" ]; then
            print_warning "$container is starting up"
        elif [ "$health" = "unhealthy" ]; then
            print_error "$container is unhealthy"
        else
            print_status "$container is running (no health check)"
        fi
    else
        print_warning "$container is not running"
    fi
done

# Check API health endpoint
echo ""
echo "🔌 API Health Check"
echo "-------------------"

API_KEY=$(grep "^API_KEYS=" .env 2>/dev/null | cut -d'=' -f2 | cut -d',' -f1)

# Try different API endpoints
endpoints=(
    "http://localhost:8000/health"
    "https://localhost/health"
    "https://localhost:443/health"
)

api_responding=false
for endpoint in "${endpoints[@]}"; do
    if [ -n "$API_KEY" ]; then
        response=$(curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" "$endpoint" -o /dev/null --connect-timeout 5 --max-time 10 2>/dev/null)
    else
        response=$(curl -s -w "%{http_code}" "$endpoint" -o /dev/null --connect-timeout 5 --max-time 10 2>/dev/null)
    fi
    
    if [ "$response" = "200" ]; then
        print_status "API is responding at $endpoint"
        api_responding=true
        API_ENDPOINT="$endpoint"
        break
    fi
done

if [ "$api_responding" = false ]; then
    print_error "API is not responding on any endpoint"
    print_info "Tried endpoints: ${endpoints[*]}"
else
    # Test API endpoints in detail
    echo ""
    echo "🧪 Detailed API Testing"
    echo "-----------------------"
    
    # Health endpoint
    if [ -n "$API_KEY" ]; then
        health_response=$(curl -s -H "X-API-Key: $API_KEY" "$API_ENDPOINT" 2>/dev/null)
    else
        health_response=$(curl -s "$API_ENDPOINT" 2>/dev/null)
    fi
    
    if echo "$health_response" | grep -q "status"; then
        print_status "Health endpoint returning valid response"
    else
        print_warning "Health endpoint response format unexpected"
    fi
    
    # System info endpoint
    system_endpoint="${API_ENDPOINT/health/system/info}"
    if [ -n "$API_KEY" ]; then
        system_response=$(curl -s -w "%{http_code}" -H "X-API-Key: $API_KEY" "$system_endpoint" -o /dev/null 2>/dev/null)
    else
        system_response=$(curl -s -w "%{http_code}" "$system_endpoint" -o /dev/null 2>/dev/null)
    fi
    
    if [ "$system_response" = "200" ]; then
        print_status "System info endpoint accessible"
    else
        print_warning "System info endpoint not accessible (status: $system_response)"
    fi
fi

# Check monitoring services
echo ""
echo "📊 Monitoring Services Check"
echo "-----------------------------"

# Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    print_status "Prometheus is healthy"
else
    print_warning "Prometheus health check failed"
fi

# Grafana
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    print_status "Grafana is healthy"
else
    print_warning "Grafana health check failed"
fi

# Check Redis
echo ""
echo "💾 Database Services Check"
echo "---------------------------"

if docker exec kolosal-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    print_status "Redis is responding"
else
    print_warning "Redis health check failed"
fi

# Check PostgreSQL (if running)
if docker ps --format "{{.Names}}" | grep -q "postgres"; then
    if docker exec -it "$(docker ps --format "{{.Names}}" | grep postgres)" pg_isready -U kolosal 2>/dev/null | grep -q "accepting connections"; then
        print_status "PostgreSQL is accepting connections"
    else
        print_warning "PostgreSQL health check failed"
    fi
fi

# Check system resources
echo ""
echo "💻 System Resources Check"
echo "--------------------------"

# Check memory usage
if command -v free &> /dev/null; then
    memory_usage=$(free | grep '^Mem:' | awk '{print int($3/$2 * 100)}')
    if [ "$memory_usage" -lt 80 ]; then
        print_status "Memory usage: ${memory_usage}%"
    elif [ "$memory_usage" -lt 90 ]; then
        print_warning "Memory usage: ${memory_usage}% (getting high)"
    else
        print_error "Memory usage: ${memory_usage}% (critical)"
    fi
fi

# Check disk space
disk_usage=$(df . | tail -1 | awk '{print int($3/$2 * 100)}')
if [ "$disk_usage" -lt 80 ]; then
    print_status "Disk usage: ${disk_usage}%"
elif [ "$disk_usage" -lt 90 ]; then
    print_warning "Disk usage: ${disk_usage}% (getting full)"
else
    print_error "Disk usage: ${disk_usage}% (critical)"
fi

# Check load average
if command -v uptime &> /dev/null; then
    load_avg=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | sed 's/,//')
    print_info "System load average: $load_avg"
fi

# Check log files for errors
echo ""
echo "📝 Log Analysis"
echo "---------------"

if [ -d "volumes/logs" ]; then
    error_count=$(find volumes/logs -name "*.log" -exec grep -i "error\|critical\|fatal" {} \; 2>/dev/null | wc -l)
    if [ "$error_count" -eq 0 ]; then
        print_status "No critical errors found in logs"
    elif [ "$error_count" -lt 10 ]; then
        print_warning "Found $error_count error entries in logs"
    else
        print_error "Found $error_count error entries in logs (needs attention)"
    fi
else
    print_info "Log directory not found (may be normal for new installation)"
fi

# Security check
echo ""
echo "🔐 Security Configuration Check"
echo "--------------------------------"

if [ -f ".env" ]; then
    # Check if production settings are enabled
    if grep -q "SECURITY_ENV=production" .env; then
        print_status "Security environment set to production"
    else
        print_warning "Security environment not set to production"
    fi
    
    if grep -q "SECURITY_REQUIRE_API_KEY=true" .env; then
        print_status "API key authentication enabled"
    else
        print_warning "API key authentication not enabled"
    fi
    
    if grep -q "SECURITY_ENABLE_RATE_LIMITING=true" .env; then
        print_status "Rate limiting enabled"
    else
        print_warning "Rate limiting not enabled"
    fi
    
    # Check for default passwords
    if grep -q "admin123\|password123\|default" .env; then
        print_error "Default passwords detected - change them!"
    else
        print_status "No default passwords detected"
    fi
else
    print_warning ".env file not found"
fi

# SSL Certificate check
if [ -f "certs/server.crt" ] && [ -f "certs/server.key" ]; then
    # Check certificate expiry
    if command -v openssl &> /dev/null; then
        if openssl x509 -in certs/server.crt -checkend 2592000 -noout 2>/dev/null; then
            expiry_date=$(openssl x509 -in certs/server.crt -noout -enddate 2>/dev/null | cut -d= -f2)
            print_status "SSL certificate valid until: $expiry_date"
        else
            print_warning "SSL certificate expires within 30 days"
        fi
    else
        print_status "SSL certificates present"
    fi
else
    print_warning "SSL certificates not found"
fi

# Final summary
echo ""
echo "📋 Health Check Summary"
echo "======================="

if [ "$api_responding" = true ]; then
    print_status "✅ Core system is operational"
    echo ""
    echo "🌐 Access URLs:"
    echo "   API: $API_ENDPOINT"
    echo "   API Documentation: ${API_ENDPOINT/health/docs}"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    
    if [ -n "$API_KEY" ]; then
        echo ""
        echo "🔑 Test command:"
        echo "   curl -H \"X-API-Key: $API_KEY\" $API_ENDPOINT"
    fi
else
    print_error "❌ System needs attention"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Check container logs: docker-compose logs"
    echo "   2. Restart services: docker-compose restart"
    echo "   3. Check configuration: cat .env"
    echo "   4. Verify ports: netstat -tulpn | grep :8000"
fi

echo ""
echo "=============================================="
echo "Health check completed at $(date)"
echo "=============================================="
