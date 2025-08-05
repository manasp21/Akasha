#!/bin/bash
# Akasha Production Management Script
# Provides commands for managing the production deployment

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env.production"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"

# Functions
log() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"; }
success() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"; }

# Docker Compose wrapper
dc() {
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
    fi
}

# Command functions
cmd_status() {
    log "Checking Akasha status..."
    echo
    dc ps
    echo
    
    # Check service health
    if curl -f http://localhost:8000/health &> /dev/null; then
        success "API is healthy"
    else
        error "API is not responding"
    fi
    
    if curl -f http://localhost:3000 &> /dev/null; then
        success "Frontend is healthy"
    else
        error "Frontend is not responding"
    fi
}

cmd_start() {
    log "Starting Akasha services..."
    dc up -d
    success "Services started"
    cmd_status
}

cmd_stop() {
    log "Stopping Akasha services..."
    dc stop
    success "Services stopped"
}

cmd_restart() {
    log "Restarting Akasha services..."
    dc restart
    success "Services restarted"
    cmd_status
}

cmd_logs() {
    local service="${2:-}"
    if [ -n "$service" ]; then
        log "Showing logs for $service..."
        dc logs -f "$service"
    else
        log "Showing logs for all services..."
        dc logs -f
    fi
}

cmd_update() {
    log "Updating Akasha deployment..."
    
    # Pull latest code (if in git repo)
    if [ -d "$PROJECT_ROOT/.git" ]; then
        log "Pulling latest code..."
        cd "$PROJECT_ROOT"
        git pull
    fi
    
    # Rebuild images
    log "Rebuilding images..."
    docker build -f Dockerfile.prod -t akasha-backend:prod . --platform linux/arm64
    docker build -f frontend/Dockerfile.prod -t akasha-frontend:prod ./frontend --platform linux/arm64
    
    # Update services
    log "Updating services..."
    dc up -d --force-recreate
    
    success "Update completed"
    cmd_status
}

cmd_backup() {
    log "Creating backup..."
    
    # Create backup directory
    backup_dir="$PROJECT_ROOT/backups"
    mkdir -p "$backup_dir"
    
    # Create backup with timestamp
    backup_name="akasha-backup-$(date +%Y%m%d-%H%M%S)"
    backup_file="$backup_dir/$backup_name.tar.gz"
    
    # Stop services temporarily for consistent backup
    warn "Stopping services for backup..."
    dc stop
    
    # Create backup
    log "Creating backup archive..."
    tar -czf "$backup_file" -C "$PROJECT_ROOT" \
        data logs config \
        --exclude="data/temp/*" \
        --exclude="logs/*.log.*" 2>/dev/null || true
    
    # Restart services
    log "Restarting services..."
    dc start
    
    success "Backup created: $backup_name.tar.gz"
    ls -lh "$backup_file"
}

cmd_restore() {
    local backup_file="$2"
    
    if [ -z "$backup_file" ]; then
        error "Please specify backup file to restore"
        echo "Usage: $0 restore <backup-file>"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    warn "This will replace current data. Are you sure?"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Restore cancelled"
        exit 0
    fi
    
    # Stop services
    log "Stopping services..."
    dc stop
    
    # Restore backup
    log "Restoring from backup..."
    tar -xzf "$backup_file" -C "$PROJECT_ROOT"
    
    # Start services
    log "Starting services..."
    dc start
    
    success "Restore completed"
}

cmd_scale() {
    local service="$2"
    local replicas="$3"
    
    if [ -z "$service" ] || [ -z "$replicas" ]; then
        error "Usage: $0 scale <service> <replicas>"
        exit 1
    fi
    
    log "Scaling $service to $replicas replicas..."
    dc up -d --scale "$service=$replicas"
    success "Scaling completed"
}

cmd_exec() {
    local service="$2"
    shift 2
    local command="$*"
    
    if [ -z "$service" ]; then
        error "Usage: $0 exec <service> <command>"
        exit 1
    fi
    
    log "Executing command in $service: $command"
    dc exec "$service" $command
}

cmd_shell() {
    local service="${2:-akasha-api}"
    log "Opening shell in $service..."
    dc exec "$service" /bin/bash
}

cmd_monitor() {
    log "Starting monitoring dashboard..."
    
    # Check if monitoring services are available
    if dc ps prometheus | grep -q "Up"; then
        echo "Prometheus: http://localhost:9091"
    fi
    
    if dc ps grafana | grep -q "Up"; then
        echo "Grafana: http://localhost:3001"
    fi
    
    # Show resource usage
    echo
    log "Current resource usage:"
    dc top
}

cmd_cleanup() {
    log "Cleaning up unused Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful!)
    read -p "Remove unused volumes? This may delete data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

cmd_config() {
    log "Current configuration:"
    echo
    dc config
}

cmd_help() {
    echo -e "${PURPLE}Akasha Production Management${NC}"
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  status              Show service status"
    echo "  start               Start all services"
    echo "  stop                Stop all services"
    echo "  restart             Restart all services"
    echo "  logs [service]      Show logs (all services or specific service)"
    echo "  update              Update deployment (rebuild and restart)"
    echo "  backup              Create backup of data"
    echo "  restore <file>      Restore from backup file"
    echo "  scale <service> <n> Scale service to n replicas"
    echo "  exec <service> <cmd> Execute command in service"
    echo "  shell [service]     Open shell in service (default: akasha-api)"
    echo "  monitor             Show monitoring information"
    echo "  cleanup             Clean up unused Docker resources"
    echo "  config              Show Docker Compose configuration"
    echo "  help                Show this help message"
    echo
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 logs akasha-api"
    echo "  $0 exec akasha-api python -c 'print(\"Hello\")'"
    echo "  $0 backup"
    echo "  $0 restore backups/akasha-backup-20240101-120000.tar.gz"
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    
    case "$command" in
        "status"|"st")
            cmd_status
            ;;
        "start")
            cmd_start
            ;;
        "stop")
            cmd_stop
            ;;
        "restart"|"rs")
            cmd_restart
            ;;
        "logs"|"log")
            cmd_logs "$@"
            ;;
        "update"|"up")
            cmd_update
            ;;
        "backup"|"bak")
            cmd_backup
            ;;
        "restore"|"res")
            cmd_restore "$@"
            ;;
        "scale")
            cmd_scale "$@"
            ;;
        "exec")
            cmd_exec "$@"
            ;;
        "shell"|"sh")
            cmd_shell "$@"
            ;;
        "monitor"|"mon")
            cmd_monitor
            ;;
        "cleanup"|"clean")
            cmd_cleanup
            ;;
        "config"|"cfg")
            cmd_config
            ;;
        "help"|"-h"|"--help")
            cmd_help
            ;;
        *)
            error "Unknown command: $command"
            echo
            cmd_help
            exit 1
            ;;
    esac
}

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    error "Docker Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Run main function
main "$@"