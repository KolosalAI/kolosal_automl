//! Role-Based Access Control (RBAC) for ISO 27001 compliance.
//!
//! Provides role definitions, permission checking, and authorization
//! for API endpoints.
//!
//! # ISO Standards Coverage
//! - ISO/IEC 27001:2022 Annex A.5.15-5.18: Access control
//! - ISO/IEC 27002:2022 Control 8.3: Information access restriction

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User roles with different permission levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Full access to all operations
    Admin,
    /// Can upload and manage data, run preprocessing
    DataOwner,
    /// Can train models, run evaluation, export
    Trainer,
    /// Can make predictions only
    Consumer,
    /// Read-only access to audit logs, metrics, reports
    Auditor,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::Admin => write!(f, "admin"),
            Role::DataOwner => write!(f, "data_owner"),
            Role::Trainer => write!(f, "trainer"),
            Role::Consumer => write!(f, "consumer"),
            Role::Auditor => write!(f, "auditor"),
        }
    }
}

impl Role {
    /// Parse role from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "admin" => Some(Role::Admin),
            "data_owner" | "dataowner" => Some(Role::DataOwner),
            "trainer" => Some(Role::Trainer),
            "consumer" => Some(Role::Consumer),
            "auditor" => Some(Role::Auditor),
            _ => None,
        }
    }
}

/// Resource categories that can be protected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    Data,
    Model,
    Training,
    Prediction,
    Audit,
    System,
    Fairness,
    Privacy,
    Compliance,
    Export,
}

/// Actions that can be performed on resources
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Create,
    Read,
    Update,
    Delete,
}

/// A specific permission (resource + action)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permission {
    pub resource: Resource,
    pub action: Action,
}

impl Permission {
    pub fn new(resource: Resource, action: Action) -> Self {
        Self { resource, action }
    }
}

/// RBAC policy manager
#[derive(Debug)]
pub struct RbacManager {
    /// Role to permissions mapping
    role_permissions: HashMap<Role, Vec<Permission>>,
}

impl RbacManager {
    /// Create a new RBAC manager with default role permissions
    pub fn new() -> Self {
        let mut role_permissions = HashMap::new();

        // Admin: full access
        let all_resources = vec![
            Resource::Data, Resource::Model, Resource::Training,
            Resource::Prediction, Resource::Audit, Resource::System,
            Resource::Fairness, Resource::Privacy, Resource::Compliance,
            Resource::Export,
        ];
        let all_actions = vec![Action::Create, Action::Read, Action::Update, Action::Delete];
        let admin_perms: Vec<Permission> = all_resources.iter()
            .flat_map(|r| all_actions.iter().map(move |a| Permission::new(r.clone(), a.clone())))
            .collect();
        role_permissions.insert(Role::Admin, admin_perms);

        // DataOwner: data management + read predictions
        role_permissions.insert(Role::DataOwner, vec![
            Permission::new(Resource::Data, Action::Create),
            Permission::new(Resource::Data, Action::Read),
            Permission::new(Resource::Data, Action::Update),
            Permission::new(Resource::Data, Action::Delete),
            Permission::new(Resource::Prediction, Action::Read),
            Permission::new(Resource::Privacy, Action::Read),
            Permission::new(Resource::Fairness, Action::Read),
        ]);

        // Trainer: training + model management + read data
        role_permissions.insert(Role::Trainer, vec![
            Permission::new(Resource::Data, Action::Read),
            Permission::new(Resource::Model, Action::Create),
            Permission::new(Resource::Model, Action::Read),
            Permission::new(Resource::Model, Action::Update),
            Permission::new(Resource::Training, Action::Create),
            Permission::new(Resource::Training, Action::Read),
            Permission::new(Resource::Prediction, Action::Create),
            Permission::new(Resource::Prediction, Action::Read),
            Permission::new(Resource::Export, Action::Create),
            Permission::new(Resource::Export, Action::Read),
            Permission::new(Resource::Fairness, Action::Read),
        ]);

        // Consumer: predictions only
        role_permissions.insert(Role::Consumer, vec![
            Permission::new(Resource::Prediction, Action::Create),
            Permission::new(Resource::Prediction, Action::Read),
            Permission::new(Resource::Model, Action::Read),
        ]);

        // Auditor: read-only access to everything
        let auditor_perms: Vec<Permission> = all_resources.iter()
            .map(|r| Permission::new(r.clone(), Action::Read))
            .collect();
        role_permissions.insert(Role::Auditor, auditor_perms);

        Self { role_permissions }
    }

    /// Check if a role has a specific permission
    pub fn has_permission(&self, role: &Role, resource: &Resource, action: &Action) -> bool {
        if let Some(perms) = self.role_permissions.get(role) {
            perms.iter().any(|p| &p.resource == resource && &p.action == action)
        } else {
            false
        }
    }

    /// Get all permissions for a role
    pub fn get_permissions(&self, role: &Role) -> Vec<Permission> {
        self.role_permissions.get(role).cloned().unwrap_or_default()
    }

    /// Check permission for an API path and method
    pub fn check_api_access(&self, role: &Role, path: &str, method: &str) -> bool {
        let (resource, action) = Self::path_to_resource_action(path, method);
        self.has_permission(role, &resource, &action)
    }

    /// Map an API path and HTTP method to a resource and action
    fn path_to_resource_action(path: &str, method: &str) -> (Resource, Action) {
        let action = match method.to_uppercase().as_str() {
            "GET" => Action::Read,
            "POST" => Action::Create,
            "PUT" | "PATCH" => Action::Update,
            "DELETE" => Action::Delete,
            _ => Action::Read,
        };

        let resource = if path.starts_with("/api/data") {
            Resource::Data
        } else if path.starts_with("/api/models") {
            Resource::Model
        } else if path.starts_with("/api/train") {
            Resource::Training
        } else if path.starts_with("/api/predict") {
            Resource::Prediction
        } else if path.starts_with("/api/audit") || path.starts_with("/api/security") {
            Resource::Audit
        } else if path.starts_with("/api/system") || path.starts_with("/api/health")
            || path.starts_with("/api/monitoring") || path.starts_with("/api/device") {
            Resource::System
        } else if path.starts_with("/api/fairness") {
            Resource::Fairness
        } else if path.starts_with("/api/privacy") {
            Resource::Privacy
        } else if path.starts_with("/api/compliance") {
            Resource::Compliance
        } else if path.starts_with("/api/export") {
            Resource::Export
        } else {
            // Default: training-related for most ML endpoints
            Resource::Training
        };

        (resource, action)
    }
}

impl Default for RbacManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_has_all_permissions() {
        let rbac = RbacManager::new();
        assert!(rbac.has_permission(&Role::Admin, &Resource::Data, &Action::Create));
        assert!(rbac.has_permission(&Role::Admin, &Resource::Model, &Action::Delete));
        assert!(rbac.has_permission(&Role::Admin, &Resource::Audit, &Action::Read));
    }

    #[test]
    fn test_consumer_limited_access() {
        let rbac = RbacManager::new();
        assert!(rbac.has_permission(&Role::Consumer, &Resource::Prediction, &Action::Create));
        assert!(rbac.has_permission(&Role::Consumer, &Resource::Model, &Action::Read));
        assert!(!rbac.has_permission(&Role::Consumer, &Resource::Data, &Action::Create));
        assert!(!rbac.has_permission(&Role::Consumer, &Resource::Training, &Action::Create));
    }

    #[test]
    fn test_auditor_read_only() {
        let rbac = RbacManager::new();
        assert!(rbac.has_permission(&Role::Auditor, &Resource::Audit, &Action::Read));
        assert!(rbac.has_permission(&Role::Auditor, &Resource::Data, &Action::Read));
        assert!(!rbac.has_permission(&Role::Auditor, &Resource::Data, &Action::Create));
        assert!(!rbac.has_permission(&Role::Auditor, &Resource::Model, &Action::Delete));
    }

    #[test]
    fn test_api_path_mapping() {
        let rbac = RbacManager::new();
        assert!(rbac.check_api_access(&Role::Consumer, "/api/predict", "POST"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/train", "POST"));
        assert!(rbac.check_api_access(&Role::Trainer, "/api/train", "POST"));
        assert!(rbac.check_api_access(&Role::DataOwner, "/api/data/upload", "POST"));
    }

    #[test]
    fn test_role_from_str() {
        assert_eq!(Role::from_str("admin"), Some(Role::Admin));
        assert_eq!(Role::from_str("consumer"), Some(Role::Consumer));
        assert_eq!(Role::from_str("data_owner"), Some(Role::DataOwner));
        assert_eq!(Role::from_str("unknown"), None);
    }

    #[test]
    fn test_role_from_str_case_insensitive() {
        assert_eq!(Role::from_str("ADMIN"), Some(Role::Admin));
        assert_eq!(Role::from_str("Trainer"), Some(Role::Trainer));
        assert_eq!(Role::from_str("AUDITOR"), Some(Role::Auditor));
        assert_eq!(Role::from_str("DataOwner"), Some(Role::DataOwner));
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::Admin.to_string(), "admin");
        assert_eq!(Role::DataOwner.to_string(), "data_owner");
        assert_eq!(Role::Trainer.to_string(), "trainer");
        assert_eq!(Role::Consumer.to_string(), "consumer");
        assert_eq!(Role::Auditor.to_string(), "auditor");
    }

    #[test]
    fn test_data_owner_permissions() {
        let rbac = RbacManager::new();
        // DataOwner can CRUD data
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Data, &Action::Create));
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Data, &Action::Read));
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Data, &Action::Update));
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Data, &Action::Delete));
        // DataOwner can read predictions and privacy
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Prediction, &Action::Read));
        assert!(rbac.has_permission(&Role::DataOwner, &Resource::Privacy, &Action::Read));
        // DataOwner cannot train or manage models
        assert!(!rbac.has_permission(&Role::DataOwner, &Resource::Training, &Action::Create));
        assert!(!rbac.has_permission(&Role::DataOwner, &Resource::Model, &Action::Create));
    }

    #[test]
    fn test_trainer_permissions() {
        let rbac = RbacManager::new();
        // Trainer can read data, create/read models, train, predict, export
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Data, &Action::Read));
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Model, &Action::Create));
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Model, &Action::Read));
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Training, &Action::Create));
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Prediction, &Action::Create));
        assert!(rbac.has_permission(&Role::Trainer, &Resource::Export, &Action::Create));
        // Trainer cannot delete data or access audit
        assert!(!rbac.has_permission(&Role::Trainer, &Resource::Data, &Action::Delete));
        assert!(!rbac.has_permission(&Role::Trainer, &Resource::Audit, &Action::Read));
    }

    #[test]
    fn test_get_permissions_all_roles() {
        let rbac = RbacManager::new();
        // Admin should have the most permissions
        let admin_perms = rbac.get_permissions(&Role::Admin);
        let consumer_perms = rbac.get_permissions(&Role::Consumer);
        assert!(admin_perms.len() > consumer_perms.len());

        // Consumer should have exactly 3 permissions
        assert_eq!(consumer_perms.len(), 3);

        // Auditor should have exactly 10 permissions (1 read per resource)
        let auditor_perms = rbac.get_permissions(&Role::Auditor);
        assert_eq!(auditor_perms.len(), 10);
    }

    #[test]
    fn test_api_path_mapping_all_resources() {
        let rbac = RbacManager::new();
        // Data paths
        assert!(rbac.check_api_access(&Role::Admin, "/api/data/upload", "POST"));
        assert!(rbac.check_api_access(&Role::Admin, "/api/data/preview", "GET"));
        // Model paths
        assert!(rbac.check_api_access(&Role::Admin, "/api/models/abc123", "GET"));
        assert!(rbac.check_api_access(&Role::Admin, "/api/models/abc123", "DELETE"));
        // Training paths
        assert!(rbac.check_api_access(&Role::Trainer, "/api/train", "POST"));
        assert!(rbac.check_api_access(&Role::Trainer, "/api/train/status/j1", "GET"));
        // Prediction paths
        assert!(rbac.check_api_access(&Role::Consumer, "/api/predict", "POST"));
        assert!(rbac.check_api_access(&Role::Consumer, "/api/predict/batch", "POST"));
        // Audit paths
        assert!(rbac.check_api_access(&Role::Auditor, "/api/audit/events", "GET"));
        assert!(rbac.check_api_access(&Role::Auditor, "/api/security/status", "GET"));
        // System paths
        assert!(rbac.check_api_access(&Role::Auditor, "/api/system/status", "GET"));
        assert!(rbac.check_api_access(&Role::Auditor, "/api/health", "GET"));
        assert!(rbac.check_api_access(&Role::Auditor, "/api/monitoring/dashboard", "GET"));
        assert!(rbac.check_api_access(&Role::Auditor, "/api/device/info", "GET"));
        // Fairness
        assert!(rbac.check_api_access(&Role::Auditor, "/api/fairness/evaluate", "GET"));
        // Privacy
        assert!(rbac.check_api_access(&Role::Auditor, "/api/privacy/scan", "GET"));
        // Compliance
        assert!(rbac.check_api_access(&Role::Auditor, "/api/compliance/report", "GET"));
        // Export
        assert!(rbac.check_api_access(&Role::Trainer, "/api/export/model", "POST"));
    }

    #[test]
    fn test_api_method_to_action_mapping() {
        let rbac = RbacManager::new();
        // PUT/PATCH map to Update
        assert!(rbac.check_api_access(&Role::Admin, "/api/data/something", "PUT"));
        assert!(rbac.check_api_access(&Role::Admin, "/api/data/something", "PATCH"));
        // DELETE maps to Delete
        assert!(rbac.check_api_access(&Role::Admin, "/api/models/abc", "DELETE"));
        // Unknown method maps to Read
        assert!(rbac.check_api_access(&Role::Admin, "/api/data/x", "OPTIONS"));
    }

    #[test]
    fn test_consumer_blocked_from_sensitive_paths() {
        let rbac = RbacManager::new();
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/data/upload", "POST"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/train", "POST"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/audit/events", "GET"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/privacy/scan", "POST"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/compliance/report", "GET"));
    }

    #[test]
    fn test_unknown_path_defaults_to_training() {
        let rbac = RbacManager::new();
        // Unknown path defaults to Training resource
        assert!(rbac.check_api_access(&Role::Trainer, "/api/unknown/endpoint", "POST"));
        assert!(!rbac.check_api_access(&Role::Consumer, "/api/unknown/endpoint", "POST"));
    }
}
