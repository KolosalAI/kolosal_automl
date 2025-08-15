# Makefile for kolosal AutoML Docker Management
# Provides convenient commands for building, running, and managing Docker containers

# Variables
VERSION ?= 0.2.0
MODE ?= development
COMPOSE_FILES_DEV = -f compose.yaml -f compose.dev.yaml
COMPOSE_FILES_PROD = -f compose.yaml
BUILD_DATE = $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF = $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Help target (default)
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)kolosal AutoML Docker Management$(NC)"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment variables:"
	@echo "  VERSION=$(VERSION)"
	@echo "  MODE=$(MODE)"
	@echo "  BUILD_DATE=$(BUILD_DATE)"
	@echo "  VCS_REF=$(VCS_REF)"

# Prerequisites and setup
.PHONY: check-prereqs
check-prereqs: ## Check if Docker and docker-compose are installed
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is not installed$(NC)" >&2; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)docker-compose is not installed$(NC)" >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "$(RED)Docker daemon is not running$(NC)" >&2; exit 1; }
	@echo "$(GREEN)✓ All prerequisites are met$(NC)"

.PHONY: setup
setup: check-prereqs ## Create required directories and files
	@echo "$(BLUE)Setting up directories...$(NC)"
	@mkdir -p volumes/{models,logs,temp} certs monitoring/grafana/{dashboards,provisioning}
	@test -f .env || cp .env.example .env
	@echo "$(GREEN)✓ Setup complete$(NC)"

# Build targets
.PHONY: build
build: setup ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build \
		--tag kolosal-automl:$(VERSION) \
		--tag kolosal-automl:latest \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg VERSION="$(VERSION)" \
		.
	@echo "$(GREEN)✓ Build complete$(NC)"

.PHONY: build-no-cache
build-no-cache: setup ## Build Docker image without cache
	@echo "$(BLUE)Building Docker image (no cache)...$(NC)"
	docker build --no-cache \
		--tag kolosal-automl:$(VERSION) \
		--tag kolosal-automl:latest \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VCS_REF="$(VCS_REF)" \
		--build-arg VERSION="$(VERSION)" \
		.
	@echo "$(GREEN)✓ Build complete$(NC)"

# Development targets
.PHONY: dev
dev: build ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker-compose $(COMPOSE_FILES_DEV) up -d
	@echo "$(GREEN)✓ Development environment started$(NC)"
	@echo "$(YELLOW)API available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Jupyter available at: http://localhost:8888$(NC)"
	@echo "$(YELLOW)Documentation available at: http://localhost:8002$(NC)"

.PHONY: dev-logs
dev-logs: ## Show logs for development environment
	docker-compose $(COMPOSE_FILES_DEV) logs -f

.PHONY: dev-stop
dev-stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	docker-compose $(COMPOSE_FILES_DEV) down
	@echo "$(GREEN)✓ Development environment stopped$(NC)"

# Production targets
.PHONY: prod
prod: build ## Start production environment
	@echo "$(BLUE)Starting production environment...$(NC)"
	docker-compose $(COMPOSE_FILES_PROD) up -d
	@echo "$(GREEN)✓ Production environment started$(NC)"
	@echo "$(YELLOW)API available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Monitoring available at: http://localhost:9090 (Prometheus), http://localhost:3000 (Grafana)$(NC)"

.PHONY: prod-logs
prod-logs: ## Show logs for production environment
	docker-compose $(COMPOSE_FILES_PROD) logs -f

.PHONY: prod-stop
prod-stop: ## Stop production environment
	@echo "$(BLUE)Stopping production environment...$(NC)"
	docker-compose $(COMPOSE_FILES_PROD) down
	@echo "$(GREEN)✓ Production environment stopped$(NC)"

# General management
.PHONY: up
up: ## Start containers (based on MODE variable)
ifeq ($(MODE),production)
	$(MAKE) prod
else
	$(MAKE) dev
endif

.PHONY: down
down: ## Stop containers (based on MODE variable)
ifeq ($(MODE),production)
	$(MAKE) prod-stop
else
	$(MAKE) dev-stop
endif

.PHONY: restart
restart: down up ## Restart containers

.PHONY: logs
logs: ## Show container logs
ifeq ($(MODE),production)
	$(MAKE) prod-logs
else
	$(MAKE) dev-logs
endif

.PHONY: status
status: ## Show container status
	@echo "$(BLUE)Container status:$(NC)"
	docker-compose $(COMPOSE_FILES_DEV) ps 2>/dev/null || docker-compose $(COMPOSE_FILES_PROD) ps

# Testing and validation
.PHONY: test
test: ## Run comprehensive Docker tests
	@echo "$(BLUE)Running Docker tests...$(NC)"
	python test_docker.py --mode $(MODE)

.PHONY: test-quick
test-quick: ## Run quick Docker tests (skip build)
	@echo "$(BLUE)Running quick Docker tests...$(NC)"
	python test_docker.py --mode $(MODE) --skip-build

.PHONY: health-check
health-check: ## Check if services are healthy
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -f http://localhost:8000/health >/dev/null 2>&1 && echo "$(GREEN)✓ API is healthy$(NC)" || echo "$(RED)✗ API is not responding$(NC)"
	@docker exec kolosal-redis redis-cli ping >/dev/null 2>&1 && echo "$(GREEN)✓ Redis is healthy$(NC)" || echo "$(RED)✗ Redis is not responding$(NC)"

# Maintenance and cleanup
.PHONY: clean
clean: ## Stop containers and remove volumes
	@echo "$(YELLOW)Stopping containers and removing volumes...$(NC)"
	docker-compose $(COMPOSE_FILES_DEV) down -v --remove-orphans 2>/dev/null || true
	docker-compose $(COMPOSE_FILES_PROD) down -v --remove-orphans 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

.PHONY: clean-images
clean-images: clean ## Remove Docker images
	@echo "$(YELLOW)Removing Docker images...$(NC)"
	docker rmi kolosal-automl:$(VERSION) kolosal-automl:latest 2>/dev/null || true
	@echo "$(GREEN)✓ Images removed$(NC)"

.PHONY: clean-all
clean-all: clean-images ## Complete cleanup (containers, volumes, images, networks)
	@echo "$(YELLOW)Performing complete cleanup...$(NC)"
	docker system prune -f
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

# Utility targets
.PHONY: shell
shell: ## Open shell in API container
	docker exec -it kolosal-automl-api bash

.PHONY: shell-redis
shell-redis: ## Open Redis CLI
	docker exec -it kolosal-redis redis-cli

.PHONY: backup-models
backup-models: ## Backup trained models
	@echo "$(BLUE)Backing up models...$(NC)"
	@mkdir -p backups
	docker cp kolosal-automl-api:/app/models ./backups/models-$(shell date +%Y%m%d-%H%M%S)
	@echo "$(GREEN)✓ Models backed up$(NC)"

.PHONY: restore-models
restore-models: ## Restore models from backup (requires BACKUP_PATH variable)
ifndef BACKUP_PATH
	@echo "$(RED)Error: BACKUP_PATH variable is required$(NC)"
	@echo "Usage: make restore-models BACKUP_PATH=./backups/models-20240101-120000"
	@exit 1
endif
	@echo "$(BLUE)Restoring models from $(BACKUP_PATH)...$(NC)"
	docker cp $(BACKUP_PATH) kolosal-automl-api:/app/models
	@echo "$(GREEN)✓ Models restored$(NC)"

.PHONY: update-image
update-image: build down up ## Update Docker image and restart services

# Security and monitoring
.PHONY: security-scan
security-scan: ## Run security scan on Docker image
	@echo "$(BLUE)Running security scan...$(NC)"
	@command -v docker-scan >/dev/null 2>&1 && docker scan kolosal-automl:$(VERSION) || echo "$(YELLOW)docker scan not available$(NC)"
	@command -v trivy >/dev/null 2>&1 && trivy image kolosal-automl:$(VERSION) || echo "$(YELLOW)trivy not available$(NC)"

.PHONY: monitoring
monitoring: ## Start only monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose $(COMPOSE_FILES_PROD) up -d prometheus grafana
	@echo "$(GREEN)✓ Monitoring stack started$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin123)$(NC)"

# Documentation and info
.PHONY: info
info: ## Show system information
	@echo "$(BLUE)System Information:$(NC)"
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker-compose --version)"
	@echo "Current mode: $(MODE)"
	@echo "Image version: $(VERSION)"
	@echo "Build date: $(BUILD_DATE)"
	@echo "VCS ref: $(VCS_REF)"
	@echo ""
	@echo "$(BLUE)Available endpoints:$(NC)"
	@echo "- API: http://localhost:8000"
	@echo "- Health: http://localhost:8000/health"
	@echo "- Docs: http://localhost:8000/docs"
	@echo "- Prometheus: http://localhost:9090"
	@echo "- Grafana: http://localhost:3000"

# Development utilities
.PHONY: format-code
format-code: ## Format code in container
	docker run --rm -v "$(PWD):/app" kolosal-automl:$(VERSION) python -m black /app/modules

.PHONY: lint-code
lint-code: ## Lint code in container
	docker run --rm -v "$(PWD):/app" kolosal-automl:$(VERSION) python -m ruff check /app/modules

.PHONY: test-unit
test-unit: ## Run unit tests in container
	docker run --rm -v "$(PWD):/app" kolosal-automl:$(VERSION) python -m pytest tests/

# Docker registry operations (uncomment and modify for your registry)
# .PHONY: push
# push: build ## Push image to registry
# 	docker tag kolosal-automl:$(VERSION) your-registry/kolosal-automl:$(VERSION)
# 	docker push your-registry/kolosal-automl:$(VERSION)

# .PHONY: pull
# pull: ## Pull image from registry
# 	docker pull your-registry/kolosal-automl:$(VERSION)
# 	docker tag your-registry/kolosal-automl:$(VERSION) kolosal-automl:$(VERSION)

# Export/import for offline deployment
.PHONY: export-image
export-image: build ## Export Docker image to tar file
	@echo "$(BLUE)Exporting Docker image...$(NC)"
	docker save kolosal-automl:$(VERSION) | gzip > kolosal-automl-$(VERSION).tar.gz
	@echo "$(GREEN)✓ Image exported to kolosal-automl-$(VERSION).tar.gz$(NC)"

.PHONY: import-image
import-image: ## Import Docker image from tar file (requires IMAGE_FILE variable)
ifndef IMAGE_FILE
	@echo "$(RED)Error: IMAGE_FILE variable is required$(NC)"
	@echo "Usage: make import-image IMAGE_FILE=kolosal-automl-0.2.0.tar.gz"
	@exit 1
endif
	@echo "$(BLUE)Importing Docker image from $(IMAGE_FILE)...$(NC)"
	gunzip -c $(IMAGE_FILE) | docker load
	@echo "$(GREEN)✓ Image imported$(NC)"
