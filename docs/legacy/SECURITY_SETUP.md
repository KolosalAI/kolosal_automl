# 🔒 Security Configuration Guide

## Production Security Setup

### 1. Environment Variables Configuration

Create a production `.env` file with these essential security settings:

```bash
# Copy template and modify
cp .env.template .env
```

#### Critical Production Settings
```bash
# Security Level
SECURITY_LEVEL=production
API_ENV=production
API_DEBUG=false

# Authentication
REQUIRE_API_KEY=true
REQUIRE_GRADIO_AUTH=true

# HTTPS/TLS
ENFORCE_HTTPS=true
ENABLE_HSTS=true
SECURITY_HSTS_MAX_AGE=31536000  # 1 year

# Rate Limiting
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=50  # Adjust based on your needs
RATE_LIMIT_WINDOW=60

# Security Headers
ENABLE_SECURITY_HEADERS=true
ENABLE_INPUT_VALIDATION=true
MAX_REQUEST_SIZE=10485760  # 10MB (more restrictive)

# Audit Logging
ENABLE_AUDIT_LOGGING=true
LOG_SECURITY_EVENTS=true
ALERT_ON_THREATS=true
```

#### Generate Secure Secrets

**API Keys:**
```bash
python -c "
import secrets
keys = [f'genta_{secrets.token_urlsafe(32)}' for _ in range(3)]
print('API_KEYS=' + ','.join(keys))
"
```

**JWT Secret:**
```bash
python -c "
import secrets
print('JWT_SECRET=' + secrets.token_urlsafe(64))
"
```

**Admin Password:**
```bash
python -c "
import secrets, string
chars = string.ascii_letters + string.digits + '!@#$%^&*'
password = ''.join(secrets.choice(chars) for _ in range(16))
print('ADMIN_PASSWORD=' + password)
"
```

### 2. SSL/TLS Certificate Setup

#### Production Certificates (Recommended)
```bash
# Using Let's Encrypt (certbot)
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates to your app
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/server.crt
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/server.key
```

#### Self-Signed Certificates (Development Only)
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=YourOrg/CN=yourdomain.com"
```

### 3. Firewall Configuration

#### Linux (UFW)
```bash
# Enable firewall
sudo ufw enable

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct API access (use reverse proxy)
sudo ufw deny 8000/tcp

# Allow SSH (if needed)
sudo ufw allow 22/tcp

# Check status
sudo ufw status
```

#### Windows Firewall
```powershell
# Allow HTTP/HTTPS through Windows Firewall
New-NetFirewallRule -DisplayName "HTTP" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
New-NetFirewallRule -DisplayName "HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow

# Block direct API port
New-NetFirewallRule -DisplayName "Block Direct API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Block
```

### 4. Nginx Reverse Proxy Configuration

Create `nginx/nginx.conf`:
```nginx
upstream kolosal_api {
    server kolosal-api:8000;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # API routes
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://kolosal_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Static files
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Main application
    location / {
        proxy_pass http://kolosal_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 5. Database Security

#### Model Encryption
```python
from modules.optimizer.model_manager import ModelManager

# Configure encrypted model storage
config = {
    "enable_encryption": True,
    "encryption_key": "your-32-byte-base64-encoded-key",
    "storage_backend": "local",
    "secure_delete": True
}

model_manager = ModelManager(config)
```

#### Generate Encryption Key
```python
import secrets
import base64

# Generate a 32-byte key for AES-256
key = secrets.token_bytes(32)
encoded_key = base64.b64encode(key).decode()
print(f"ENCRYPTION_KEY={encoded_key}")
```

### 6. Access Control Lists

#### IP Whitelisting
```bash
# In .env file
IP_WHITELIST=192.168.1.0/24,10.0.0.0/8,203.0.113.10

# Or in code
from modules.security import EnhancedSecurityConfig

config = EnhancedSecurityConfig(
    enable_ip_whitelist=True,
    ip_whitelist=[
        "192.168.1.0/24",  # Local network
        "10.0.0.0/8",      # Private network
        "203.0.113.10"     # Specific trusted IP
    ]
)
```

#### IP Blocking
```bash
# Block malicious IPs
BLOCKED_IPS=192.168.1.100,10.0.0.50,203.0.113.0/24
```

### 7. Authentication & Authorization

#### JWT Configuration
```python
from modules.security import EnhancedSecurityManager

security_manager = EnhancedSecurityManager({
    "enable_jwt_auth": True,
    "jwt_secret": "your-64-byte-jwt-secret",
    "jwt_expiry_hours": 1,  # Short expiry for production
    "jwt_algorithm": "HS256"
})

# Create token with user roles
token = security_manager.create_jwt_token({
    "user_id": "admin_user",
    "roles": ["admin", "api_user"],
    "permissions": ["read", "write", "admin"]
})
```

#### Multi-Factor Authentication Setup
```python
# Enable additional security layers
config = {
    "require_api_key": True,
    "enable_jwt_auth": True,
    "enable_session_validation": True,
    "enable_csrf_protection": True
}
```

### 8. Logging & Monitoring

#### Security Event Logging
```python
from modules.security import SecurityAuditor

auditor = SecurityAuditor()

# Log security events
auditor.log_security_event(
    event_type="AUTHENTICATION_FAILURE",
    severity="HIGH",
    details={"ip": "192.168.1.100", "attempts": 5},
    source_ip="192.168.1.100"
)
```

#### Log Monitoring with fail2ban
```bash
# Install fail2ban
sudo apt-get install fail2ban

# Configure for Kolosal AutoML
sudo tee /etc/fail2ban/filter.d/kolosal.conf << 'EOF'
[Definition]
failregex = .*SECURITY THREAT.*from <HOST>.*
            .*Authentication failed.*client_ip.*<HOST>.*
            .*Rate limit exceeded.*<HOST>.*
ignoreregex =
EOF

# Create jail configuration
sudo tee /etc/fail2ban/jail.d/kolosal.conf << 'EOF'
[kolosal]
enabled = true
filter = kolosal
logpath = /path/to/kolosal-automl/logs/kolosal_security.log
maxretry = 5
bantime = 3600
findtime = 600
action = iptables-multiport[name=kolosal, port="80,443,8000"]
EOF

# Restart fail2ban
sudo systemctl restart fail2ban
```

### 9. Security Hardening Checklist

#### Application Level
- [ ] All default passwords changed
- [ ] API keys rotated and secured
- [ ] JWT secrets are cryptographically strong
- [ ] Input validation enabled
- [ ] Rate limiting configured appropriately
- [ ] HTTPS enforced with HSTS
- [ ] Security headers properly configured
- [ ] Error messages don't leak sensitive information

#### Infrastructure Level
- [ ] Firewall configured and enabled
- [ ] SSH keys used instead of passwords
- [ ] Non-root user for application
- [ ] File permissions properly set
- [ ] Unnecessary services disabled
- [ ] Security updates applied

#### Monitoring & Response
- [ ] Security event logging enabled
- [ ] Log monitoring configured
- [ ] Automated alerting set up
- [ ] Incident response plan documented
- [ ] Regular security audits scheduled

### 10. Security Testing

#### Automated Security Testing
```bash
# Install security testing tools
pip install bandit safety

# Run security scan on code
bandit -r modules/

# Check for known vulnerabilities
safety check

# Test API security
curl -X POST "http://localhost:8000/api/test" \
  -H "Content-Type: application/json" \
  -d '{"test": "<script>alert(\"xss\")</script>"}'
```

#### Penetration Testing
```bash
# Install OWASP ZAP for web app security testing
# Or use online tools like:
# - SSL Labs SSL Test: https://www.ssllabs.com/ssltest/
# - Mozilla Observatory: https://observatory.mozilla.org/
# - Security Headers: https://securityheaders.com/
```

### 11. Emergency Response

#### Incident Response Script
```bash
#!/bin/bash
# emergency_lockdown.sh

echo "🚨 Emergency Security Lockdown"

# Stop all services
docker-compose down

# Block all traffic temporarily
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw enable

# Rotate API keys (backup current ones first)
cp .env .env.backup.$(date +%s)

# Generate new API keys
python3 -c "
import secrets
keys = [f'genta_{secrets.token_urlsafe(32)}' for _ in range(3)]
print('New API keys:', keys)
"

echo "Services stopped and keys rotated."
echo "Review logs and restart when safe."
```

### 12. Security Compliance

#### GDPR Compliance
```python
# Data retention configuration
config = {
    "data_retention_days": 30,
    "enable_data_anonymization": True,
    "audit_log_retention_days": 365,
    "enable_right_to_deletion": True
}
```

#### SOC 2 Compliance
- Implement access controls
- Maintain audit trails  
- Regular security assessments
- Incident response procedures
- Data encryption at rest and in transit

---

## 🔐 Security Best Practices Summary

1. **Never use default credentials** in production
2. **Rotate secrets regularly** (API keys, certificates, passwords)
3. **Use principle of least privilege** for access control
4. **Monitor and log security events** continuously
5. **Keep dependencies updated** and scan for vulnerabilities
6. **Implement defense in depth** with multiple security layers
7. **Test security configurations** regularly
8. **Have an incident response plan** ready
9. **Encrypt sensitive data** at rest and in transit
10. **Regular security audits** and penetration testing

Remember: Security is an ongoing process, not a one-time setup!
