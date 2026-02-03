#!/usr/bin/env python3
"""
Generate secure configuration for kolosal AutoML production deployment
"""

import secrets
import string
import os
from pathlib import Path
import argparse

def generate_api_keys(count=3):
    """Generate secure API keys"""
    keys = []
    for i in range(count):
        key = f"genta_{secrets.token_urlsafe(32)}"
        keys.append(key)
    return keys

def generate_jwt_secret():
    """Generate JWT secret"""
    return secrets.token_urlsafe(64)

def generate_admin_password(length=16):
    """Generate secure admin password"""
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(chars) for _ in range(length))

def generate_encryption_key():
    """Generate encryption key for model storage"""
    import base64
    key = secrets.token_bytes(32)  # 32 bytes for AES-256
    return base64.b64encode(key).decode()

def update_env_file(config_values, env_file=".env"):
    """Update .env file with secure values"""
    env_path = Path(env_file)
    
    if not env_path.exists():
        if Path(".env.example").exists():
            # Copy from template
            import shutil
            shutil.copy(".env.example", env_file)
            print(f"Created {env_file} from .env.example")
        else:
            print(f"Error: {env_file} and .env.example not found")
            return False
    
    # Read current content
    content = env_path.read_text()
    
    # Update values
    for key, value in config_values.items():
        if f"{key}=" in content:
            # Replace existing value
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    break
            content = '\n'.join(lines)
        else:
            # Add new value
            content += f"\n{key}={value}"
    
    # Write updated content
    env_path.write_text(content)
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate secure configuration for kolosal AutoML")
    parser.add_argument("--output", default=".env", help="Output .env file path")
    parser.add_argument("--api-keys", type=int, default=3, help="Number of API keys to generate")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without updating files")
    
    args = parser.parse_args()
    
    print("🔐 Generating secure configuration for kolosal AutoML...")
    print("=" * 60)
    
    # Generate secure values
    api_keys = generate_api_keys(args.api_keys)
    jwt_secret = generate_jwt_secret()
    admin_password = generate_admin_password()
    encryption_key = generate_encryption_key()
    
    config_values = {
        "API_ENV": "production",
        "SECURITY_ENV": "production", 
        "SECURITY_REQUIRE_API_KEY": "true",
        "SECURITY_ENABLE_JWT": "true",
        "SECURITY_ENABLE_RATE_LIMITING": "true",
        "SECURITY_ENABLE_AUDIT_LOGGING": "true",
        "SECURITY_ENFORCE_HTTPS": "true",
        "API_KEYS": ",".join(api_keys),
        "JWT_SECRET": jwt_secret,
        "GRAFANA_ADMIN_PASSWORD": admin_password,
        "ENCRYPTION_KEY": encryption_key,
    }
    
    print("Generated secure configuration:")
    print(f"API Keys ({len(api_keys)}):")
    for i, key in enumerate(api_keys, 1):
        print(f"  {i}. {key}")
    
    print(f"\nJWT Secret: {jwt_secret[:20]}...{jwt_secret[-10:]}")
    print(f"Admin Password: {admin_password}")
    print(f"Encryption Key: {encryption_key[:20]}...{encryption_key[-10:]}")
    
    if args.dry_run:
        print("\n⚠️  Dry run mode - no files updated")
        print("\nTo apply these settings, run without --dry-run flag")
        return
    
    # Update .env file
    if update_env_file(config_values, args.output):
        print(f"\n✅ Configuration updated in {args.output}")
        print(f"🔐 IMPORTANT: Save these credentials securely!")
        print(f"📝 Review {args.output} and customize other settings as needed")
        
        # Generate commands for easy copy-paste
        print("\n📋 Quick start commands:")
        print("docker-compose up -d")
        print("make health-check")
        
        print("\n🔑 API Key for testing:")
        print(f'curl -H "X-API-Key: {api_keys[0]}" http://localhost:8000/health')
        
    else:
        print(f"❌ Failed to update {args.output}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
