use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{Utc, DateTime};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use rand::Rng;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecretType {
    ApiKey,
    Password,
    Token,
    Certificate,
    EncryptionKey,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub secret_id: String,
    pub secret_type: SecretType,
    pub created_at: DateTime<Utc>,
    pub rotated_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSecret {
    value: String,
    hmac: String,
    metadata: SecretMetadata,
}

#[derive(Debug, Clone)]
pub struct SecretsManager {
    storage_path: String,
    secrets: Arc<RwLock<HashMap<String, StoredSecret>>>,
    hmac_key: Vec<u8>,
}

impl SecretsManager {
    pub fn new(storage_path: &str) -> Self {
        let hmac_key: Vec<u8> = {
            let mut rng = rand::thread_rng();
            (0..32).map(|_| rng.gen::<u8>()).collect()
        };
        Self {
            storage_path: storage_path.to_string(),
            secrets: Arc::new(RwLock::new(HashMap::new())),
            hmac_key,
        }
    }

    pub fn store_secret(
        &self,
        id: &str,
        secret: &str,
        secret_type: SecretType,
    ) -> Result<(), String> {
        let hmac_value = self.compute_hmac(secret)?;
        let encoded = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            secret.as_bytes(),
        );
        let metadata = SecretMetadata {
            secret_id: id.to_string(),
            secret_type,
            created_at: Utc::now(),
            rotated_at: None,
            expires_at: None,
            version: 1,
        };
        let stored = StoredSecret {
            value: encoded,
            hmac: hmac_value,
            metadata,
        };
        self.secrets.write().insert(id.to_string(), stored);
        Ok(())
    }

    pub fn get_secret(&self, id: &str) -> Result<String, String> {
        let secrets = self.secrets.read();
        let stored = secrets
            .get(id)
            .ok_or_else(|| format!("Secret '{}' not found", id))?;

        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &stored.value,
        )
        .map_err(|e| format!("Failed to decode secret: {}", e))?;

        let secret = String::from_utf8(decoded)
            .map_err(|e| format!("Invalid UTF-8 in secret: {}", e))?;

        // Verify HMAC integrity
        let expected_hmac = self.compute_hmac(&secret)?;
        if expected_hmac != stored.hmac {
            return Err("Secret integrity check failed".to_string());
        }

        Ok(secret)
    }

    pub fn rotate_secret(&self, id: &str) -> Result<String, String> {
        let secret_type = {
            let secrets = self.secrets.read();
            let stored = secrets
                .get(id)
                .ok_or_else(|| format!("Secret '{}' not found", id))?;
            stored.metadata.secret_type.clone()
        };

        let new_secret = self.generate_secret(secret_type.clone(), 32);
        let hmac_value = self.compute_hmac(&new_secret)?;
        let encoded = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            new_secret.as_bytes(),
        );

        let mut secrets = self.secrets.write();
        let stored = secrets
            .get_mut(id)
            .ok_or_else(|| format!("Secret '{}' not found", id))?;

        stored.value = encoded;
        stored.hmac = hmac_value;
        stored.metadata.rotated_at = Some(Utc::now());
        stored.metadata.version += 1;

        Ok(new_secret)
    }

    pub fn delete_secret(&self, id: &str) -> Result<(), String> {
        self.secrets
            .write()
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| format!("Secret '{}' not found", id))
    }

    pub fn list_secrets(&self) -> Vec<SecretMetadata> {
        self.secrets
            .read()
            .values()
            .map(|s| s.metadata.clone())
            .collect()
    }

    pub fn check_rotations_needed(&self) -> Vec<String> {
        let secrets = self.secrets.read();
        let now = Utc::now();
        secrets
            .iter()
            .filter(|(_, stored)| {
                if let Some(expires_at) = stored.metadata.expires_at {
                    let days_until = (expires_at - now).num_days();
                    return days_until <= 30;
                }
                // Flag secrets never rotated that are older than 90 days
                if stored.metadata.rotated_at.is_none() {
                    let age_days = (now - stored.metadata.created_at).num_days();
                    return age_days > 90;
                }
                false
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn generate_secret(&self, secret_type: SecretType, length: usize) -> String {
        let mut rng = rand::thread_rng();
        match secret_type {
            SecretType::ApiKey => {
                let chars: Vec<char> =
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        .chars()
                        .collect();
                (0..length).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
            }
            SecretType::Password => {
                let chars: Vec<char> =
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+"
                        .chars()
                        .collect();
                (0..length).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
            }
            SecretType::EncryptionKey => {
                let bytes: Vec<u8> = (0..length).map(|_| rng.gen::<u8>()).collect();
                hex::encode(bytes)
            }
            _ => {
                let bytes: Vec<u8> = (0..length).map(|_| rng.gen::<u8>()).collect();
                base64::Engine::encode(
                    &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                    &bytes,
                )
            }
        }
    }

    pub fn assess_strength(&self, secret: &str) -> u32 {
        let mut score: u32 = 0;
        let len = secret.len();

        // Length scoring
        score += match len {
            0..=7 => 0,
            8..=11 => 15,
            12..=15 => 25,
            16..=23 => 35,
            _ => 40,
        };

        if secret.chars().any(|c| c.is_lowercase()) {
            score += 10;
        }
        if secret.chars().any(|c| c.is_uppercase()) {
            score += 10;
        }
        if secret.chars().any(|c| c.is_ascii_digit()) {
            score += 10;
        }
        if secret.chars().any(|c| !c.is_alphanumeric()) {
            score += 15;
        }

        // Uniqueness of characters
        let unique: std::collections::HashSet<char> = secret.chars().collect();
        let ratio = unique.len() as f64 / len.max(1) as f64;
        if ratio > 0.7 {
            score += 15;
        }

        score.min(100)
    }

    fn compute_hmac(&self, data: &str) -> Result<String, String> {
        let mut mac = HmacSha256::new_from_slice(&self.hmac_key)
            .map_err(|e| format!("HMAC initialization failed: {}", e))?;
        mac.update(data.as_bytes());
        Ok(hex::encode(mac.finalize().into_bytes()))
    }
}
