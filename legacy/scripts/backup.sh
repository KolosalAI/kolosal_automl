#!/bin/bash
# Comprehensive backup script for kolosal AutoML

set -e  # Exit on any error

# Configuration
BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="kolosal_backup_$TIMESTAMP"
RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️]${NC} $1"
}

show_help() {
    echo "kolosal AutoML Backup Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -f, --full        Full backup (default)"
    echo "  -d, --data        Data only backup"
    echo "  -c, --config      Configuration only backup"
    echo "  -m, --models      Models only backup"
    echo "  -r, --restore     Restore from backup"
    echo "  --list            List available backups"
    echo "  --cleanup         Clean up old backups"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full backup"
    echo "  $0 --data              # Backup only data"
    echo "  $0 --restore backup.tar.gz  # Restore from backup"
    echo "  $0 --cleanup           # Remove backups older than $RETENTION_DAYS days"
}

create_backup_dir() {
    mkdir -p "$BACKUP_DIR"
    print_status "Created backup directory: $BACKUP_DIR"
}

backup_data() {
    print_info "Backing up data volumes..."
    
    # Create temporary directory for this backup
    local temp_dir="$BACKUP_DIR/temp_$TIMESTAMP"
    mkdir -p "$temp_dir"
    
    # Backup volumes directory
    if [ -d "volumes" ]; then
        print_info "Backing up volumes directory..."
        cp -r volumes "$temp_dir/"
        print_status "Volumes backed up"
    else
        print_warning "Volumes directory not found"
    fi
    
    # Backup model registry
    if [ -d "model_registry" ]; then
        print_info "Backing up model registry..."
        cp -r model_registry "$temp_dir/"
        print_status "Model registry backed up"
    fi
    
    # Backup models directory
    if [ -d "models" ]; then
        print_info "Backing up models directory..."
        cp -r models "$temp_dir/"
        print_status "Models directory backed up"
    fi
    
    # Backup experiments
    if [ -d "experiments" ]; then
        print_info "Backing up experiments..."
        cp -r experiments "$temp_dir/"
        print_status "Experiments backed up"
    fi
    
    # Backup MLflow runs
    if [ -d "mlruns" ]; then
        print_info "Backing up MLflow runs..."
        cp -r mlruns "$temp_dir/"
        print_status "MLflow runs backed up"
    fi
    
    # Backup logs
    if [ -d "logs" ]; then
        print_info "Backing up logs..."
        cp -r logs "$temp_dir/"
        print_status "Logs backed up"
    fi
    
    return 0
}

backup_config() {
    print_info "Backing up configuration files..."
    
    local temp_dir="$BACKUP_DIR/temp_$TIMESTAMP"
    mkdir -p "$temp_dir/config"
    
    # Environment file
    if [ -f ".env" ]; then
        cp .env "$temp_dir/config/"
        print_status ".env backed up"
    fi
    
    # Docker compose files
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml "$temp_dir/config/"
        print_status "docker-compose.yml backed up"
    fi
    
    if [ -f "compose.dev.yaml" ]; then
        cp compose.dev.yaml "$temp_dir/config/"
        print_status "compose.dev.yaml backed up"
    fi
    
    # Nginx configuration
    if [ -f "nginx.conf" ]; then
        cp nginx.conf "$temp_dir/config/"
        print_status "nginx.conf backed up"
    fi
    
    # SSL certificates
    if [ -d "certs" ]; then
        cp -r certs "$temp_dir/config/"
        print_status "SSL certificates backed up"
    fi
    
    # Monitoring configurations
    if [ -d "monitoring" ]; then
        cp -r monitoring "$temp_dir/config/"
        print_status "Monitoring configs backed up"
    fi
    
    # Configuration files
    if [ -d "configs" ]; then
        cp -r configs "$temp_dir/config/"
        print_status "Application configs backed up"
    fi
    
    return 0
}

backup_database() {
    print_info "Backing up databases..."
    
    local temp_dir="$BACKUP_DIR/temp_$TIMESTAMP"
    mkdir -p "$temp_dir/database"
    
    # Check if PostgreSQL container is running
    if docker ps --format "{{.Names}}" | grep -q "postgres"; then
        postgres_container=$(docker ps --format "{{.Names}}" | grep postgres)
        print_info "Found PostgreSQL container: $postgres_container"
        
        # Create database dump
        docker exec "$postgres_container" pg_dump -U kolosal -d kolosal_automl > "$temp_dir/database/postgresql_dump.sql"
        if [ $? -eq 0 ]; then
            print_status "PostgreSQL database backed up"
        else
            print_warning "PostgreSQL backup failed"
        fi
    else
        print_info "PostgreSQL container not running, skipping database backup"
    fi
    
    # Backup Redis data (if container is running)
    if docker ps --format "{{.Names}}" | grep -q "kolosal-redis"; then
        print_info "Creating Redis backup..."
        docker exec kolosal-redis redis-cli BGSAVE
        
        # Wait for background save to complete
        sleep 5
        
        # Copy Redis dump
        docker cp kolosal-redis:/data/dump.rdb "$temp_dir/database/" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_status "Redis data backed up"
        else
            print_warning "Redis backup failed"
        fi
    else
        print_info "Redis container not running, skipping Redis backup"
    fi
    
    return 0
}

create_archive() {
    local temp_dir="$BACKUP_DIR/temp_$TIMESTAMP"
    local archive_path="$BACKUP_DIR/$BACKUP_NAME.tar.gz"
    
    print_info "Creating compressed archive..."
    
    # Create tar.gz archive
    cd "$BACKUP_DIR"
    tar -czf "$BACKUP_NAME.tar.gz" -C "temp_$TIMESTAMP" . 2>/dev/null
    cd ..
    
    # Calculate archive size
    local archive_size=$(du -h "$archive_path" | cut -f1)
    
    # Remove temporary directory
    rm -rf "$temp_dir"
    
    print_status "Backup archive created: $archive_path ($archive_size)"
    
    # Create backup metadata
    cat > "$BACKUP_DIR/$BACKUP_NAME.info" << EOF
Backup Information
==================
Name: $BACKUP_NAME
Date: $(date)
Size: $archive_size
Type: Full Backup
Components:
- Configuration files
- Data volumes
- Model registry
- Experiments
- Logs
- Database dumps

Restore command:
./scripts/backup.sh --restore $BACKUP_NAME.tar.gz
EOF
    
    print_status "Backup metadata saved: $BACKUP_DIR/$BACKUP_NAME.info"
}

list_backups() {
    print_info "Available backups:"
    echo ""
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR/*.tar.gz 2>/dev/null)" ]; then
        print_warning "No backups found"
        return
    fi
    
    echo "Name                          Date                    Size"
    echo "-------------------------------------------------------------"
    
    for backup in "$BACKUP_DIR"/*.tar.gz; do
        if [ -f "$backup" ]; then
            backup_name=$(basename "$backup")
            backup_size=$(du -h "$backup" | cut -f1)
            backup_date=$(stat -c %y "$backup" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
            
            printf "%-30s %-20s %s\n" "$backup_name" "$backup_date" "$backup_size"
        fi
    done
    
    echo ""
    print_info "To restore a backup: $0 --restore <backup_name>"
}

restore_backup() {
    local backup_file="$1"
    
    if [ -z "$backup_file" ]; then
        print_error "Please specify backup file to restore"
        echo "Usage: $0 --restore <backup_file>"
        exit 1
    fi
    
    # Check if backup file exists
    if [ ! -f "$backup_file" ] && [ ! -f "$BACKUP_DIR/$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Use full path if file exists in backup directory
    if [ -f "$BACKUP_DIR/$backup_file" ]; then
        backup_file="$BACKUP_DIR/$backup_file"
    fi
    
    print_warning "This will overwrite existing data. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Restore cancelled"
        exit 0
    fi
    
    print_info "Stopping services before restore..."
    docker compose down
    
    print_info "Restoring from backup: $backup_file"
    
    # Create temporary restore directory
    local restore_dir="restore_temp_$$"
    mkdir -p "$restore_dir"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$restore_dir"
    
    # Restore configuration files
    if [ -d "$restore_dir/config" ]; then
        print_info "Restoring configuration files..."
        cp -r "$restore_dir/config/"* ./
        print_status "Configuration restored"
    fi
    
    # Restore data
    if [ -d "$restore_dir/volumes" ]; then
        print_info "Restoring volumes..."
        rm -rf volumes
        cp -r "$restore_dir/volumes" ./
        print_status "Volumes restored"
    fi
    
    if [ -d "$restore_dir/model_registry" ]; then
        print_info "Restoring model registry..."
        rm -rf model_registry
        cp -r "$restore_dir/model_registry" ./
        print_status "Model registry restored"
    fi
    
    if [ -d "$restore_dir/models" ]; then
        print_info "Restoring models..."
        rm -rf models
        cp -r "$restore_dir/models" ./
        print_status "Models restored"
    fi
    
    if [ -d "$restore_dir/experiments" ]; then
        print_info "Restoring experiments..."
        rm -rf experiments
        cp -r "$restore_dir/experiments" ./
        print_status "Experiments restored"
    fi
    
    if [ -d "$restore_dir/mlruns" ]; then
        print_info "Restoring MLflow runs..."
        rm -rf mlruns
        cp -r "$restore_dir/mlruns" ./
        print_status "MLflow runs restored"
    fi
    
    # Clean up
    rm -rf "$restore_dir"
    
    print_info "Starting services..."
    docker compose up -d
    
    print_status "Restore completed successfully!"
    print_info "Please verify the system using: $0 --health-check"
}

cleanup_old_backups() {
    print_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        print_warning "Backup directory does not exist"
        return
    fi
    
    local deleted_count=0
    
    # Find and remove old backup files
    find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -exec rm -f {} \; -exec echo "Removed: {}" \; | while read line; do
        ((deleted_count++))
        print_info "$line"
    done
    
    # Remove corresponding .info files
    find "$BACKUP_DIR" -name "*.info" -type f -mtime +$RETENTION_DAYS -exec rm -f {} \;
    
    if [ $deleted_count -eq 0 ]; then
        print_info "No old backups to clean up"
    else
        print_status "Cleaned up $deleted_count old backup(s)"
    fi
}

# Main execution
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --list)
        list_backups
        exit 0
        ;;
    --cleanup)
        cleanup_old_backups
        exit 0
        ;;
    -r|--restore)
        restore_backup "$2"
        exit 0
        ;;
    -d|--data)
        BACKUP_TYPE="data"
        ;;
    -c|--config)
        BACKUP_TYPE="config"
        ;;
    -m|--models)
        BACKUP_TYPE="models"
        ;;
    -f|--full|"")
        BACKUP_TYPE="full"
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

# Start backup process
echo "🔄 kolosal AutoML Backup Script"
echo "==============================="
print_info "Starting $BACKUP_TYPE backup at $(date)"
print_info "Backup will be saved to: $BACKUP_DIR/$BACKUP_NAME.tar.gz"

create_backup_dir

case "$BACKUP_TYPE" in
    "full")
        backup_config
        backup_data
        backup_database
        ;;
    "data")
        backup_data
        backup_database
        ;;
    "config")
        backup_config
        ;;
    "models")
        backup_data  # Models are part of data backup
        ;;
esac

create_archive

# Cleanup old backups
cleanup_old_backups

print_status "✅ Backup completed successfully!"
print_info "Backup location: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
print_info "To restore: $0 --restore $BACKUP_NAME.tar.gz"

echo ""
echo "==============================="
echo "Backup completed at $(date)"
echo "==============================="
