use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use chrono::Utc;

use super::auth::{AuditEntry, SecurityManager};
use super::rate_limiter::RateLimiter;

#[derive(Debug, Clone)]
pub struct SecurityMiddleware {
    pub security_manager: Arc<SecurityManager>,
    pub rate_limiter: Arc<RateLimiter>,
}

impl SecurityMiddleware {
    pub fn new(security_manager: Arc<SecurityManager>, rate_limiter: Arc<RateLimiter>) -> Self {
        Self {
            security_manager,
            rate_limiter,
        }
    }
}

pub async fn security_layer(
    axum::extract::State(middleware): axum::extract::State<SecurityMiddleware>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP
    let ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // Check IP blocked
    if middleware.security_manager.check_ip_blocked(&ip) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Extract client identifier for rate limiting
    let client_id = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&ip)
        .to_string();

    // Check rate limit
    if !middleware.rate_limiter.is_allowed(&client_id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Verify API key if present
    if let Some(api_key) = request.headers().get("x-api-key").and_then(|v| v.to_str().ok()) {
        if !middleware.security_manager.verify_api_key(api_key) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    let method = request.method().to_string();
    let uri = request.uri().path().to_string();

    // Process request
    let response = next.run(request).await;
    let status = response.status().as_u16();

    // Log audit entry
    middleware.security_manager.log_request(AuditEntry {
        timestamp: Utc::now(),
        client_id,
        action: method,
        resource: uri,
        status_code: status,
        ip_address: ip,
    });

    // Add security headers
    let mut response = response;
    let headers = response.headers_mut();
    for (key, value) in SecurityManager::get_security_headers() {
        if let (Ok(name), Ok(val)) = (
            axum::http::HeaderName::try_from(key),
            axum::http::HeaderValue::from_str(&value),
        ) {
            headers.insert(name, val);
        }
    }

    Ok(response)
}
