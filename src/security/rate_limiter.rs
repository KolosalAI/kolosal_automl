use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;
use parking_lot::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    SlidingWindow,
    TokenBucket,
    FixedWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_window: u32,
    pub window_seconds: u64,
    pub algorithm: RateLimitAlgorithm,
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 100,
            window_seconds: 60,
            algorithm: RateLimitAlgorithm::SlidingWindow,
            burst_size: 10,
        }
    }
}

#[derive(Debug)]
struct ClientState {
    request_times: Vec<Instant>,
    tokens: f64,
    last_refill: Instant,
    window_start: Instant,
    window_count: u32,
    total_requests: usize,
    blocked_requests: usize,
}

impl ClientState {
    fn new(burst_size: u32) -> Self {
        let now = Instant::now();
        Self {
            request_times: Vec::new(),
            tokens: burst_size as f64,
            last_refill: now,
            window_start: now,
            window_count: 0,
            total_requests: 0,
            blocked_requests: 0,
        }
    }
}

#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: Mutex<HashMap<String, ClientState>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_allowed(&self, client_id: &str) -> bool {
        let mut clients = self.clients.lock();
        let state = clients
            .entry(client_id.to_string())
            .or_insert_with(|| ClientState::new(self.config.burst_size));
        state.total_requests += 1;

        let allowed = match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => {
                self.check_sliding_window(state)
            }
            RateLimitAlgorithm::TokenBucket => {
                self.check_token_bucket(state)
            }
            RateLimitAlgorithm::FixedWindow => {
                self.check_fixed_window(state)
            }
        };

        if !allowed {
            state.blocked_requests += 1;
        }
        allowed
    }

    pub fn get_remaining(&self, client_id: &str) -> u32 {
        let mut clients = self.clients.lock();
        let state = match clients.get_mut(client_id) {
            Some(s) => s,
            None => return self.config.requests_per_window,
        };

        match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => {
                let now = Instant::now();
                let window = std::time::Duration::from_secs(self.config.window_seconds);
                let count = state.request_times.iter().filter(|t| now.duration_since(**t) < window).count() as u32;
                self.config.requests_per_window.saturating_sub(count)
            }
            RateLimitAlgorithm::TokenBucket => {
                self.refill_tokens(state);
                state.tokens as u32
            }
            RateLimitAlgorithm::FixedWindow => {
                self.config.requests_per_window.saturating_sub(state.window_count)
            }
        }
    }

    pub fn reset_client(&self, client_id: &str) {
        self.clients.lock().remove(client_id);
    }

    pub fn get_metrics(&self) -> HashMap<String, usize> {
        let clients = self.clients.lock();
        let mut metrics = HashMap::new();
        let mut total_requests: usize = 0;
        let mut total_blocked: usize = 0;

        for (id, state) in clients.iter() {
            total_requests += state.total_requests;
            total_blocked += state.blocked_requests;
            metrics.insert(format!("client_{}_requests", id), state.total_requests);
            metrics.insert(format!("client_{}_blocked", id), state.blocked_requests);
        }

        metrics.insert("total_requests".to_string(), total_requests);
        metrics.insert("total_blocked".to_string(), total_blocked);
        metrics.insert("active_clients".to_string(), clients.len());
        metrics
    }

    fn check_sliding_window(&self, state: &mut ClientState) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.config.window_seconds);
        state.request_times.retain(|t| now.duration_since(*t) < window);

        if (state.request_times.len() as u32) < self.config.requests_per_window {
            state.request_times.push(now);
            true
        } else {
            false
        }
    }

    fn check_token_bucket(&self, state: &mut ClientState) -> bool {
        self.refill_tokens(state);
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn check_fixed_window(&self, state: &mut ClientState) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.config.window_seconds);

        if now.duration_since(state.window_start) >= window {
            state.window_start = now;
            state.window_count = 0;
        }

        if state.window_count < self.config.requests_per_window {
            state.window_count += 1;
            true
        } else {
            false
        }
    }

    fn refill_tokens(&self, state: &mut ClientState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let rate = self.config.requests_per_window as f64 / self.config.window_seconds as f64;
        state.tokens = (state.tokens + elapsed * rate).min(self.config.burst_size as f64);
        state.last_refill = now;
    }
}
