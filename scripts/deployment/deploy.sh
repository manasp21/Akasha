#!/bin/bash
# Akasha Production Deployment Script
# Handles complete production deployment with safety checks

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env.production"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

banner() {
    echo -e "${PURPLE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    🚀 AKASHA DEPLOYMENT                       ║"
    echo "║                Production Deployment Script                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available memory
    available_memory=$(python3 -c "
import psutil
print(int(psutil.virtual_memory().available / (1024**3)))
" 2>/dev/null || echo "0")
    
    if [ "$available_memory" -lt 8 ]; then
        warn "Available memory is less than 8GB. Deployment may fail."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    success "Prerequisites check passed"
}

check_environment() {
    log "Checking environment configuration..."
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        warn "Production environment file not found: $ENV_FILE"
        
        if [ -f "$ENV_FILE.template" ]; then
            log "Creating environment file from template..."
            cp "$ENV_FILE.template" "$ENV_FILE"
            warn "Please edit $ENV_FILE with your production values"
            warn "Especially set secure values for JWT_SECRET_KEY and admin credentials"
            read -p "Continue with template values? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            error "No environment template found"
            exit 1
        fi
    fi
    
    # Check for critical environment variables
    source "$ENV_FILE"
    
    if [ "$JWT_SECRET_KEY" = "your-super-secure-jwt-secret-key-change-in-production-please-make-it-very-long-and-random" ]; then
        error "JWT_SECRET_KEY is still using template value. Please set a secure key."
        exit 1
    fi
    
    if [ ${#JWT_SECRET_KEY} -lt 32 ]; then
        error "JWT_SECRET_KEY should be at least 32 characters long"
        exit 1
    fi
    
    success "Environment configuration is valid"
}

create_directories() {
    log "Creating required directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data/vectors"
    mkdir -p "$PROJECT_ROOT/data/cache"
    mkdir -p "$PROJECT_ROOT/data/temp"
    mkdir -p "$PROJECT_ROOT/uploads"
    
    success "Directories created"
}

backup_existing() {
    log "Creating backup of existing data..."
    
    if [ -d "$PROJECT_ROOT/data" ]; then
        backup_name="akasha-backup-$(date +%Y%m%d-%H%M%S)"
        tar -czf "$BACKUP_DIR/$backup_name.tar.gz" -C "$PROJECT_ROOT" data logs 2>/dev/null || true
        success "Backup created: $backup_name.tar.gz"
    else
        log "No existing data to backup"
    fi
}

build_images() {
    log "Building Docker images..."
    
    # Build backend image
    log "Building backend image..."
    docker build -f Dockerfile.prod -t akasha-backend:prod . --platform linux/arm64
    
    # Build frontend image
    log "Building frontend image..."
    docker build -f frontend/Dockerfile.prod -t akasha-frontend:prod ./frontend --platform linux/arm64
    
    success "Docker images built successfully"
}

deploy_services() {
    log "Deploying services..."
    
    # Set environment variables for docker-compose
    export COMPOSE_FILE="$COMPOSE_FILE"
    export COMPOSE_PROJECT_NAME="akasha-prod"
    
    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    fi
    
    success "Services deployed"
}

wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for API to be healthy
    log "Waiting for API service..."
    timeout=300  # 5 minutes
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            success "API service is ready"
            break
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
        
        if [ $elapsed -ge $timeout ]; then
            error "API service failed to start within $timeout seconds"
            return 1
        fi
    done
    
    # Wait for frontend
    log "Waiting for frontend service..."
    timeout=120  # 2 minutes
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:3000 &> /dev/null; then
            success "Frontend service is ready"
            break
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
        
        if [ $elapsed -ge $timeout ]; then
            error "Frontend service failed to start within $timeout seconds"
            return 1
        fi
    done
    
    success "All services are ready"
}

show_status() {
    log "Deployment Status:"
    echo
    
    # Show running containers
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE" ps
    else
        docker compose -f "$COMPOSE_FILE" ps
    fi
    
    echo
    success "🎉 Akasha is now running in production mode!"
    echo
    echo -e "${GREEN}Access URLs:${NC}"
    echo -e "  Frontend: ${BLUE}http://localhost:3000${NC}"
    echo -e "  API:      ${BLUE}http://localhost:8000${NC}"
    echo -e "  API Docs: ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  Health:   ${BLUE}http://localhost:8000/health${NC}"
    echo
    echo -e "${YELLOW}Default Admin Credentials:${NC}"
    echo -e "  Email:    ${ADMIN_EMAIL:-admin@example.com}"
    echo -e "  Password: ${ADMIN_PASSWORD:-admin123}"
    echo -e "  ${RED}⚠️  Please change these credentials after first login!${NC}"
    echo
}

# Main deployment process
main() {
    banner
    
    log "Starting Akasha production deployment..."
    
    check_prerequisites
    check_environment
    create_directories
    backup_existing
    build_images
    deploy_services
    wait_for_services
    show_status
    
    success "Deployment completed successfully! 🚀"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "check")
        check_prerequisites
        check_environment
        ;;
    "build")
        build_images
        ;;
    "backup")
        backup_existing
        ;;
    *)
        echo "Usage: $0 [deploy|check|build|backup]"
        exit 1
        ;;
esac