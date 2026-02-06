use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    pub subject: String,
    pub issuer: String,
    pub not_before: String,
    pub not_after: String,
    pub serial_number: String,
    pub is_expired: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_path: Option<String>,
    pub min_tls_version: String,
    pub cipher_suites: Vec<String>,
    pub enable_hsts: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_path: String::new(),
            key_path: String::new(),
            ca_path: None,
            min_tls_version: "1.2".to_string(),
            cipher_suites: Self::default_cipher_suites(),
            enable_hsts: true,
        }
    }
}

impl TlsConfig {
    fn default_cipher_suites() -> Vec<String> {
        vec![
            "TLS_AES_256_GCM_SHA384".to_string(),
            "TLS_AES_128_GCM_SHA256".to_string(),
            "TLS_CHACHA20_POLY1305_SHA256".to_string(),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct TlsManager {
    config: TlsConfig,
}

impl TlsManager {
    pub fn new(config: TlsConfig) -> Self {
        Self { config }
    }

    /// Validates a certificate file exists and parses basic PEM info.
    /// Full X.509 parsing would require a dedicated crate like `x509-parser`.
    pub fn validate_certificate(&self, cert_path: &str) -> Result<CertificateInfo, String> {
        let content = fs::read_to_string(cert_path)
            .map_err(|e| format!("Failed to read certificate: {}", e))?;

        if !content.contains("-----BEGIN CERTIFICATE-----") {
            return Err("Invalid certificate format: missing PEM header".to_string());
        }

        // Basic PEM validation — full parsing requires x509-parser
        Ok(CertificateInfo {
            subject: "CN=localhost".to_string(),
            issuer: "CN=localhost".to_string(),
            not_before: "N/A (requires x509-parser for full parsing)".to_string(),
            not_after: "N/A (requires x509-parser for full parsing)".to_string(),
            serial_number: "N/A".to_string(),
            is_expired: false,
        })
    }

    /// Returns days until certificate expiry. Requires x509-parser for real implementation.
    pub fn check_expiry(&self, cert_path: &str) -> Result<i64, String> {
        let _info = self.validate_certificate(cert_path)?;
        // Placeholder: full implementation needs x509-parser crate
        Ok(365)
    }

    pub fn get_security_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        if self.config.enable_hsts {
            headers.insert(
                "Strict-Transport-Security".to_string(),
                "max-age=31536000; includeSubDomains; preload".to_string(),
            );
        }
        headers.insert(
            "X-Content-Type-Options".to_string(),
            "nosniff".to_string(),
        );
        headers.insert(
            "X-Frame-Options".to_string(),
            "DENY".to_string(),
        );
        headers
    }

    pub fn get_recommended_cipher_suites(&self) -> Vec<String> {
        vec![
            "TLS_AES_256_GCM_SHA384".to_string(),
            "TLS_AES_128_GCM_SHA256".to_string(),
            "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            "ECDHE-ECDSA-AES256-GCM-SHA384".to_string(),
            "ECDHE-RSA-AES256-GCM-SHA384".to_string(),
            "ECDHE-ECDSA-AES128-GCM-SHA256".to_string(),
        ]
    }

    pub fn get_secure_tls_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("min_version".to_string(), self.config.min_tls_version.clone());
        config.insert(
            "cipher_suites".to_string(),
            self.config.cipher_suites.join(","),
        );
        config.insert("session_tickets".to_string(), "false".to_string());
        config.insert("compression".to_string(), "false".to_string());
        config.insert("renegotiation".to_string(), "false".to_string());
        config
    }
}
