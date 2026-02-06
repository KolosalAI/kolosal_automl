// Security module - authentication, rate limiting, TLS, secrets management
pub mod auth;
pub mod rate_limiter;
pub mod tls;
pub mod secrets;
pub mod middleware;

pub use auth::{SecurityManager, SecurityConfig, ApiKeyVerifier, JwtVerifier};
pub use rate_limiter::{RateLimiter, RateLimitConfig, RateLimitAlgorithm};
pub use tls::{TlsManager, TlsConfig, CertificateInfo};
pub use secrets::{SecretsManager, SecretType, SecretMetadata};
pub use middleware::SecurityMiddleware;
