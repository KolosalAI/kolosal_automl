use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{Utc, DateTime};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub api_keys: Vec<String>,
    pub jwt_secret: Option<String>,
    pub jwt_expiry_secs: u64,
    pub enable_api_key_auth: bool,
    pub enable_jwt_auth: bool,
    pub blocked_ips: Vec<String>,
    pub audit_log_enabled: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            jwt_secret: None,
            jwt_expiry_secs: 3600,
            enable_api_key_auth: true,
            enable_jwt_auth: false,
            blocked_ips: Vec::new(),
            audit_log_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiKeyVerifier {
    keys: Arc<RwLock<Vec<String>>>,
}

impl ApiKeyVerifier {
    pub fn new(keys: Vec<String>) -> Self {
        Self {
            keys: Arc::new(RwLock::new(keys)),
        }
    }

    pub fn verify(&self, key: &str) -> bool {
        self.keys.read().iter().any(|k| k == key)
    }

    pub fn add_key(&self, key: &str) {
        self.keys.write().push(key.to_string());
    }

    pub fn remove_key(&self, key: &str) {
        self.keys.write().retain(|k| k != key);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JwtClaims {
    sub: String,
    exp: usize,
    iat: usize,
    #[serde(flatten)]
    extra: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct JwtVerifier {
    secret: String,
    expiry_secs: u64,
}

impl JwtVerifier {
    pub fn new(secret: String, expiry_secs: u64) -> Self {
        Self { secret, expiry_secs }
    }

    pub fn create_token(&self, claims: HashMap<String, String>) -> Result<String, String> {
        let now = Utc::now().timestamp() as usize;
        let jwt_claims = JwtClaims {
            sub: claims.get("sub").cloned().unwrap_or_default(),
            exp: now + self.expiry_secs as usize,
            iat: now,
            extra: claims,
        };
        encode(
            &Header::default(),
            &jwt_claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
        .map_err(|e| format!("Failed to create token: {}", e))
    }

    pub fn verify_token(&self, token: &str) -> Result<HashMap<String, String>, String> {
        let token_data = decode::<JwtClaims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| format!("Failed to verify token: {}", e))?;

        let mut result = token_data.claims.extra;
        result.insert("sub".to_string(), token_data.claims.sub);
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub client_id: String,
    pub action: String,
    pub resource: String,
    pub status_code: u16,
    pub ip_address: String,
}

#[derive(Debug, Clone)]
pub struct SecurityManager {
    config: SecurityConfig,
    api_key_verifier: ApiKeyVerifier,
    jwt_verifier: Option<JwtVerifier>,
    audit_log: Arc<RwLock<Vec<AuditEntry>>>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Self {
        let api_key_verifier = ApiKeyVerifier::new(config.api_keys.clone());
        let jwt_verifier = config.jwt_secret.as_ref().map(|secret| {
            JwtVerifier::new(secret.clone(), config.jwt_expiry_secs)
        });
        Self {
            config,
            api_key_verifier,
            jwt_verifier,
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn verify_api_key(&self, key: &str) -> bool {
        if !self.config.enable_api_key_auth {
            return true;
        }
        self.api_key_verifier.verify(key)
    }

    pub fn verify_jwt(&self, token: &str) -> Result<HashMap<String, String>, String> {
        if !self.config.enable_jwt_auth {
            return Ok(HashMap::new());
        }
        match &self.jwt_verifier {
            Some(verifier) => verifier.verify_token(token),
            None => Err("JWT authentication not configured".to_string()),
        }
    }

    pub fn check_ip_blocked(&self, ip: &str) -> bool {
        self.config.blocked_ips.iter().any(|blocked| blocked == ip)
    }

    pub fn sanitize_input(input: &str) -> String {
        input
            .chars()
            .filter(|c| {
                c.is_alphanumeric()
                    || c.is_whitespace()
                    || matches!(c, '.' | '-' | '_' | '/' | ':' | '@' | ',')
            })
            .collect()
    }

    pub fn validate_input(input: &str) -> Result<(), String> {
        let patterns = [
            "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ",
            "<script", "javascript:", "onclick=", "onerror=",
            "../", "..\\", "; rm ", "| rm ", "&& rm ",
        ];
        let lower = input.to_lowercase();
        for pattern in &patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return Err(format!("Potentially dangerous pattern detected: {}", pattern));
            }
        }
        Ok(())
    }

    pub fn log_request(&self, entry: AuditEntry) {
        if self.config.audit_log_enabled {
            self.audit_log.write().push(entry);
        }
    }

    pub fn get_audit_log(&self, limit: usize) -> Vec<AuditEntry> {
        let log = self.audit_log.read();
        log.iter().rev().take(limit).cloned().collect()
    }

    pub fn generate_api_key() -> String {
        let mut rng = rand::thread_rng();
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            .chars()
            .collect();
        (0..32).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
    }

    pub fn get_security_headers() -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Strict-Transport-Security".to_string(),
            "max-age=31536000; includeSubDomains".to_string(),
        );
        headers.insert(
            "X-Content-Type-Options".to_string(),
            "nosniff".to_string(),
        );
        headers.insert(
            "X-Frame-Options".to_string(),
            "DENY".to_string(),
        );
        headers.insert(
            "X-XSS-Protection".to_string(),
            "1; mode=block".to_string(),
        );
        headers.insert(
            "Content-Security-Policy".to_string(),
            "default-src 'self'".to_string(),
        );
        headers.insert(
            "Referrer-Policy".to_string(),
            "strict-origin-when-cross-origin".to_string(),
        );
        headers
    }
}
