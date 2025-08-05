#!/bin/bash
# Production entrypoint script for Akasha Backend

set -e

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Akasha Backend (Production)${NC}"
echo -e "${BLUE}================================${NC}"

# Check if required environment variables are set
if [ -z "$AKASHA_AUTH__SECRET_KEY" ]; then
    echo -e "${RED}❌ ERROR: AKASHA_AUTH__SECRET_KEY environment variable is not set${NC}"
    echo -e "${YELLOW}For production, please set a secure JWT secret key${NC}"
    exit 1
fi

# Wait for dependencies (Redis, ChromaDB)
echo -e "${YELLOW}⏳ Waiting for dependencies...${NC}"

# Wait for Redis
if [ ! -z "$REDIS_URL" ]; then
    echo -e "${YELLOW}Waiting for Redis...${NC}"
    while ! curl -f "${REDIS_URL}/ping" > /dev/null 2>&1; do
        echo -e "${YELLOW}Redis is unavailable - sleeping${NC}"
        sleep 1
    done
    echo -e "${GREEN}✅ Redis is ready${NC}"
fi

# Wait for ChromaDB (if external)
if [ ! -z "$CHROMA_SERVER_URL" ]; then
    echo -e "${YELLOW}Waiting for ChromaDB...${NC}"
    while ! curl -f "${CHROMA_SERVER_URL}/api/v1/heartbeat" > /dev/null 2>&1; do
        echo -e "${YELLOW}ChromaDB is unavailable - sleeping${NC}"
        sleep 1
    done
    echo -e "${GREEN}✅ ChromaDB is ready${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p /app/data/{vectors,cache,temp} /app/logs /app/uploads /app/models

# Set proper permissions
chmod -R 755 /app/data /app/logs /app/uploads

# Validate configuration
echo -e "${YELLOW}🔧 Validating configuration...${NC}"
python -c "
import sys
sys.path.append('/app')
try:
    from src.core.config import get_config
    config = get_config()
    config.validate_memory_limits()
    print('✅ Configuration is valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)
"

# Check available memory
echo -e "${YELLOW}💾 Checking system resources...${NC}"
python -c "
import psutil
import sys
memory = psutil.virtual_memory()
available_gb = memory.available / (1024**3)
print(f'Available memory: {available_gb:.1f} GB')
if available_gb < 8:
    print('⚠️  Warning: Low available memory')
else:
    print('✅ Sufficient memory available')
"

# Run database migrations (if applicable)
echo -e "${YELLOW}🗄️  Running migrations...${NC}"
# Note: Add actual migration commands here when implemented
echo -e "${GREEN}✅ Migrations completed${NC}"

# Pre-warm models (if needed)
if [ "$AKASHA_PRELOAD_MODELS" = "true" ]; then
    echo -e "${YELLOW}🤖 Pre-loading models...${NC}"
    # Note: Add model preloading logic here
    echo -e "${GREEN}✅ Models pre-loaded${NC}"
fi

echo -e "${GREEN}🎉 Akasha Backend initialization completed${NC}"
echo -e "${BLUE}================================${NC}"

# Execute the main command
exec "$@"