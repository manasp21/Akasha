# Akasha Deployment Guide

## Table of Contents
1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Quick Start](#3-quick-start)
4. [Docker Deployment](#4-docker-deployment)
5. [Manual Installation](#5-manual-installation)
6. [Configuration](#6-configuration)
7. [Production Deployment](#7-production-deployment)
8. [Cloud Deployment](#8-cloud-deployment)
9. [Monitoring and Logging](#9-monitoring-and-logging)
10. [Backup and Recovery](#10-backup-and-recovery)
11. [Security Hardening](#11-security-hardening)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

This guide provides comprehensive instructions for deploying the Akasha multimodal RAG system in various environments, from development setups to production deployments.

### 1.1 Deployment Options

- **Docker Compose**: Recommended for development and small-scale deployments
- **Kubernetes**: Recommended for production and large-scale deployments
- **Manual Installation**: For custom environments and development
- **Cloud Platforms**: AWS, GCP, Azure with managed services

### 1.2 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  API Gateway                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Akasha    │  │   Akasha    │  │   Akasha    │        │
│  │  Service 1  │  │  Service 2  │  │  Service N  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Vector    │  │    Cache    │  │  File       │        │
│  │   Store     │  │   (Redis)   │  │  Storage    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Requirements

### 2.1 Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **CPU** | 4 cores (x86_64 or ARM64) |
| **RAM** | 8 GB |
| **Storage** | 50 GB SSD |
| **GPU** | Optional (improves performance) |
| **OS** | Linux, macOS, Windows |

### 2.2 Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| **CPU** | 8+ cores (x86_64 or ARM64) |
| **RAM** | 16+ GB |
| **Storage** | 200+ GB NVMe SSD |
| **GPU** | NVIDIA GPU with 8+ GB VRAM |
| **OS** | Ubuntu 20.04+, macOS 12+, Windows 11 |

### 2.3 Software Dependencies

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.8+ (for manual installation)
- **Node.js**: 16+ (for UI development)
- **Git**: 2.30+

### 2.4 Hardware Recommendations by Use Case

#### Development Environment
- **CPU**: 4-8 cores
- **RAM**: 8-16 GB
- **Storage**: 50-100 GB
- **GPU**: Optional

#### Small Production (<1000 documents)
- **CPU**: 8-16 cores
- **RAM**: 16-32 GB
- **Storage**: 200-500 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM

#### Large Production (>10,000 documents)
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **Storage**: 1+ TB NVMe SSD
- **GPU**: Multiple GPUs or high-end GPU
- **Network**: 10 Gbps+

---

## 3. Quick Start

### 3.1 Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/akasha-ai/akasha.git
cd akasha

# Copy and customize configuration
cp docker-compose.example.yml docker-compose.yml
cp .env.example .env

# Edit configuration files
nano .env
nano akasha.yaml

# Start the system
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f akasha
```

### 3.2 Environment Variables (.env)

```bash
# Basic configuration
AKASHA_ENVIRONMENT=production
AKASHA_LOG_LEVEL=INFO
AKASHA_API_PORT=8000
AKASHA_UI_PORT=3000

# Security
AKASHA_SECRET_KEY=your-secret-key-here
AKASHA_API_KEYS=api-key-1,api-key-2

# Storage
AKASHA_DATA_DIR=./data
AKASHA_MODELS_DIR=./models
AKASHA_CACHE_DIR=./cache

# Models
AKASHA_LLM_BACKEND=mlx
AKASHA_LLM_MODEL=gemma-3-27b
AKASHA_EMBEDDING_MODEL=jinaai/jina-embeddings-v4

# Vector Store
AKASHA_VECTOR_STORE=chroma
AKASHA_VECTOR_STORE_PATH=./data/vector_store

# Cache
AKASHA_CACHE_L1_ENABLED=true
AKASHA_CACHE_L1_SIZE_MB=512
AKASHA_CACHE_L2_ENABLED=false
AKASHA_CACHE_L3_ENABLED=true

# Plugins
AKASHA_PLUGINS_DIR=./plugins
AKASHA_PLUGINS_ENABLED=mineru-enhanced,jina-embeddings
```

### 3.3 Basic Configuration (akasha.yaml)

```yaml
system:
  name: "akasha"
  version: "1.0.0"
  environment: "production"
  log_level: "INFO"

ingestion:
  backend: "mineru2"
  batch_size: 10
  max_file_size_mb: 100

embedding:
  model: "jinaai/jina-embeddings-v4"
  device: "auto"
  batch_size: 32

vector_store:
  backend: "chroma"
  collection_name: "akasha_documents"

llm:
  backend: "mlx"
  model_name: "gemma-3-27b"
  model_path: "./models"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

---

## 4. Docker Deployment

### 4.1 Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  akasha:
    image: akasha:latest
    container_name: akasha-main
    ports:
      - "${AKASHA_API_PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
      - ./plugins:/app/plugins
    environment:
      - AKASHA_CONFIG=/app/config/akasha.yaml
      - AKASHA_ENVIRONMENT=${AKASHA_ENVIRONMENT:-production}
    env_file:
      - .env
    depends_on:
      - redis
      - vector-store
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  akasha-ui:
    image: akasha-ui:latest
    container_name: akasha-ui
    ports:
      - "${AKASHA_UI_PORT:-3000}:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:${AKASHA_API_PORT:-8000}
    depends_on:
      - akasha
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: akasha-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  vector-store:
    image: qdrant/qdrant:latest
    container_name: akasha-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: akasha-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - akasha
      - akasha-ui
    restart: unless-stopped

volumes:
  redis_data:
  qdrant_data:

networks:
  default:
    name: akasha-network
```

### 4.2 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/cache /app/plugins

# Set permissions
RUN chmod +x ./scripts/*.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "src.main"]
```

### 4.3 Build and Deployment Scripts

```bash
#!/bin/bash
# build.sh

set -e

echo "Building Akasha Docker images..."

# Build main application
docker build -t akasha:latest .

# Build UI
docker build -t akasha-ui:latest -f ui/Dockerfile ui/

# Build with GPU support (optional)
if [ "$1" == "gpu" ]; then
    docker build -t akasha:gpu -f Dockerfile.gpu .
fi

echo "Build completed successfully!"
```

```bash
#!/bin/bash
# deploy.sh

set -e

echo "Deploying Akasha system..."

# Pull latest images
docker-compose pull

# Stop existing containers
docker-compose down

# Start services
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Run health checks
docker-compose exec akasha curl -f http://localhost:8000/health

echo "Deployment completed successfully!"
echo "Access the system at:"
echo "  API: http://localhost:${AKASHA_API_PORT:-8000}"
echo "  UI:  http://localhost:${AKASHA_UI_PORT:-3000}"
```

---

## 5. Manual Installation

### 5.1 Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git curl build-essential

# macOS
brew install python@3.11 git curl

# Install Node.js (for UI)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 5.2 Python Environment Setup

```bash
# Create virtual environment
python3.11 -m venv akasha-env
source akasha-env/bin/activate  # On Windows: akasha-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Akasha
pip install akasha[all]

# Or install from source
git clone https://github.com/akasha-ai/akasha.git
cd akasha
pip install -e .[all]
```

### 5.3 Model Download

```bash
# Create models directory
mkdir -p ./models

# Download Gemma 3 27B model
akasha download-model gemma-3-27b --output ./models/

# Download JINA v4 embeddings
akasha download-model jina-embeddings-v4 --output ./models/

# Alternative: Download via Hugging Face
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='google/gemma-2-27b-it', local_dir='./models/gemma-3-27b')
snapshot_download(repo_id='jinaai/jina-embeddings-v4', local_dir='./models/jina-embeddings-v4')
"
```

### 5.4 Database Setup

```bash
# Install and configure ChromaDB (default)
pip install chromadb

# Or install Qdrant for production
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Install Redis for caching
docker run -d --name redis -p 6379:6379 redis:alpine
```

### 5.5 Configuration

```bash
# Create configuration directory
mkdir -p ./config

# Generate default configuration
akasha init-config --output ./config/akasha.yaml

# Edit configuration
nano ./config/akasha.yaml
```

### 5.6 Start Services

```bash
# Start the main application
akasha start --config ./config/akasha.yaml

# Or run individual components
akasha run api --port 8000 &
akasha run ui --port 3000 &
```

---

## 6. Configuration

### 6.1 Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Command Line Arguments**
3. **Local Configuration File** (`./akasha.yaml`)
4. **User Configuration** (`~/.akasha/akasha.yaml`)
5. **System Configuration** (`/etc/akasha/akasha.yaml`)
6. **Default Values** (lowest priority)

### 6.2 Production Configuration

```yaml
# config/production.yaml
system:
  name: "akasha-production"
  environment: "production"
  log_level: "WARNING"
  debug: false

api:
  host: "0.0.0.0"
  port: 8000
  workers: 8
  cors:
    allowed_origins: ["https://your-domain.com"]
  authentication:
    enabled: true
    api_keys:
      - "secure-api-key-1"
      - "secure-api-key-2"
  rate_limiting:
    enabled: true
    requests_per_minute: 100

ingestion:
  batch_size: 20
  max_file_size_mb: 200
  timeout_seconds: 600

embedding:
  model: "jinaai/jina-embeddings-v4"
  device: "cuda"
  batch_size: 64
  cache_embeddings: true

vector_store:
  backend: "qdrant"
  host: "qdrant"
  port: 6333
  collection_name: "akasha_production"

llm:
  backend: "mlx"
  model_name: "gemma-3-27b"
  model_path: "/app/models"
  max_tokens: 4096
  temperature: 0.7

cache:
  l1_enabled: true
  l1_max_size_mb: 2048
  l2_enabled: true
  l2_backend: "redis"
  l2_host: "redis"
  l2_port: 6379

plugins:
  enabled:
    - "mineru-enhanced"
    - "jina-embeddings"
  plugin_paths:
    - "/app/plugins"
```

### 6.3 Environment-Specific Overrides

```bash
# Development
export AKASHA_ENVIRONMENT=development
export AKASHA_LOG_LEVEL=DEBUG
export AKASHA_DEBUG=true

# Staging
export AKASHA_ENVIRONMENT=staging
export AKASHA_LOG_LEVEL=INFO
export AKASHA_API_WORKERS=4

# Production
export AKASHA_ENVIRONMENT=production
export AKASHA_LOG_LEVEL=WARNING
export AKASHA_API_WORKERS=8
```

---

## 7. Production Deployment

### 7.1 High Availability Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    restart: unless-stopped

  # API instances
  akasha-api-1:
    image: akasha:latest
    environment:
      - AKASHA_API_PORT=8001
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  akasha-api-2:
    image: akasha:latest
    environment:
      - AKASHA_API_PORT=8002
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped

  # Database cluster
  qdrant-node-1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data_1:/qdrant/storage

  qdrant-node-2:
    image: qdrant/qdrant:latest
    ports:
      - "6334:6333"
    volumes:
      - qdrant_data_2:/qdrant/storage

  # Redis cluster
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_master_data:/data

  redis-slave:
    image: redis:7-alpine
    command: redis-server --slaveof redis-master 6379
    volumes:
      - redis_slave_data:/data

volumes:
  qdrant_data_1:
  qdrant_data_2:
  redis_master_data:
  redis_slave_data:
```

### 7.2 Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream akasha_api {
        server akasha-api-1:8001;
        server akasha-api-2:8002;
    }

    upstream qdrant_cluster {
        server qdrant-node-1:6333;
        server qdrant-node-2:6333;
    }

    # API proxy
    server {
        listen 80;
        server_name api.your-domain.com;

        location / {
            proxy_pass http://akasha_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }

    # UI proxy
    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://akasha-ui:3000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        location /api/ {
            proxy_pass http://akasha_api/;
        }
    }

    # SSL configuration
    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://akasha_api;
        }
    }
}
```

### 7.3 Process Management

```bash
#!/bin/bash
# scripts/manage.sh

COMMAND=$1
ENVIRONMENT=${2:-production}

case $COMMAND in
    start)
        echo "Starting Akasha ($ENVIRONMENT)..."
        docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
        ;;
    stop)
        echo "Stopping Akasha ($ENVIRONMENT)..."
        docker-compose -f docker-compose.$ENVIRONMENT.yml down
        ;;
    restart)
        echo "Restarting Akasha ($ENVIRONMENT)..."
        docker-compose -f docker-compose.$ENVIRONMENT.yml restart
        ;;
    status)
        docker-compose -f docker-compose.$ENVIRONMENT.yml ps
        ;;
    logs)
        docker-compose -f docker-compose.$ENVIRONMENT.yml logs -f
        ;;
    update)
        echo "Updating Akasha ($ENVIRONMENT)..."
        docker-compose -f docker-compose.$ENVIRONMENT.yml pull
        docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
        ;;
    backup)
        echo "Creating backup..."
        ./scripts/backup.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|update|backup} [environment]"
        exit 1
        ;;
esac
```

---

## 8. Cloud Deployment

### 8.1 AWS Deployment

#### 8.1.1 ECS with Fargate

```yaml
# aws/ecs-task-definition.json
{
  "family": "akasha",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "akasha",
      "image": "your-account.dkr.ecr.region.amazonaws.com/akasha:latest",
      "cpu": 1536,
      "memory": 6144,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AKASHA_ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "AKASHA_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:akasha-secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/akasha",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 8.1.2 CloudFormation Template

```yaml
# aws/cloudformation.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Akasha RAG System Infrastructure'

Parameters:
  VpcCIDR:
    Type: String
    Default: '10.0.0.0/16'
  
  ImageURI:
    Type: String
    Description: 'ECR image URI for Akasha'

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: akasha-cluster

  # Application Load Balancer
  ALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: akasha-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: akasha-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref ECSSecurityGroup
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2

  # RDS for metadata
  RDSInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: akasha-db
      DBInstanceClass: db.t3.medium
      Engine: postgres
      MasterUsername: akasha
      MasterUserPassword: !Ref DBPassword
      AllocatedStorage: 100

  # ElastiCache for caching
  ElastiCacheCluster:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheNodeType: cache.t3.medium
      Engine: redis
      NumCacheNodes: 1

Outputs:
  LoadBalancerDNS:
    Description: 'DNS name of the load balancer'
    Value: !GetAtt ALB.DNSName
```

### 8.2 Google Cloud Platform

#### 8.2.1 GKE Deployment

```yaml
# gcp/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: akasha-api
  labels:
    app: akasha-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: akasha-api
  template:
    metadata:
      labels:
        app: akasha-api
    spec:
      containers:
      - name: akasha
        image: gcr.io/project-id/akasha:latest
        ports:
        - containerPort: 8000
        env:
        - name: AKASHA_ENVIRONMENT
          value: "production"
        - name: AKASHA_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: akasha-secrets
              key: secret-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: akasha-service
spec:
  selector:
    app: akasha-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 8.3 Azure Deployment

#### 8.3.1 Container Instances

```yaml
# azure/container-instance.yml
apiVersion: 2021-09-01
location: West US 2
name: akasha-instance
properties:
  containers:
  - name: akasha
    properties:
      image: your-registry.azurecr.io/akasha:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: AKASHA_ENVIRONMENT
        value: production
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
tags:
  Environment: Production
  Application: Akasha
```

---

## 9. Monitoring and Logging

### 9.1 Prometheus Monitoring

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'akasha'
    static_configs:
      - targets: ['akasha:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### 9.2 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Akasha System Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(akasha_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(akasha_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"akasha\"}",
            "legendFormat": "Memory (bytes)"
          }
        ]
      }
    ]
  }
}
```

### 9.3 Logging Configuration

```yaml
# config/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /app/logs/akasha.log
    maxBytes: 100MB
    backupCount: 5

  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: json
    address: ['syslog-server', 514]

loggers:
  akasha:
    level: INFO
    handlers: [console, file]
    propagate: no

  uvicorn:
    level: INFO
    handlers: [console]
    propagate: no

root:
  level: WARNING
  handlers: [console, syslog]
```

---

## 10. Backup and Recovery

### 10.1 Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

set -e

BACKUP_DIR="/backup/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup at $BACKUP_DIR..."

# Backup configuration
echo "Backing up configuration..."
cp -r ./config "$BACKUP_DIR/"

# Backup data
echo "Backing up application data..."
cp -r ./data "$BACKUP_DIR/"

# Backup vector store
echo "Backing up vector store..."
if [ "$VECTOR_STORE" == "qdrant" ]; then
    docker exec qdrant-node-1 qdrant-backup --output /backup/qdrant
    cp -r /var/lib/docker/volumes/qdrant_data/_data "$BACKUP_DIR/qdrant"
elif [ "$VECTOR_STORE" == "chroma" ]; then
    cp -r ./data/vector_store "$BACKUP_DIR/"
fi

# Backup Redis
echo "Backing up Redis..."
docker exec redis redis-cli BGSAVE
sleep 5
cp /var/lib/docker/volumes/redis_data/_data/dump.rdb "$BACKUP_DIR/"

# Create archive
echo "Creating archive..."
cd /backup
tar -czf "akasha-backup-$(date +%Y-%m-%d_%H-%M-%S).tar.gz" "$(basename $BACKUP_DIR)"

# Upload to S3 (optional)
if [ ! -z "$AWS_S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "akasha-backup-$(date +%Y-%m-%d_%H-%M-%S).tar.gz" "s3://$AWS_S3_BUCKET/backups/"
fi

echo "Backup completed successfully!"
```

### 10.2 Recovery Procedure

```bash
#!/bin/bash
# scripts/restore.sh

set -e

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file>"
    exit 1
fi

echo "Restoring from backup: $BACKUP_FILE"

# Stop services
docker-compose down

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

BACKUP_DIR="/tmp/$(basename $BACKUP_FILE .tar.gz)"

# Restore configuration
echo "Restoring configuration..."
cp -r "$BACKUP_DIR/config" ./

# Restore data
echo "Restoring application data..."
cp -r "$BACKUP_DIR/data" ./

# Restore vector store
echo "Restoring vector store..."
if [ -d "$BACKUP_DIR/qdrant" ]; then
    cp -r "$BACKUP_DIR/qdrant" /var/lib/docker/volumes/qdrant_data/_data/
elif [ -d "$BACKUP_DIR/vector_store" ]; then
    cp -r "$BACKUP_DIR/vector_store" ./data/
fi

# Restore Redis
echo "Restoring Redis..."
if [ -f "$BACKUP_DIR/dump.rdb" ]; then
    cp "$BACKUP_DIR/dump.rdb" /var/lib/docker/volumes/redis_data/_data/
fi

# Start services
docker-compose up -d

echo "Restore completed successfully!"
```

---

## 11. Security Hardening

### 11.1 Network Security

```bash
# Firewall configuration (UFW)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Docker network isolation
docker network create --driver bridge --internal akasha-internal
```

### 11.2 Container Security

```dockerfile
# Secure Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r akasha && useradd -r -g akasha akasha

# Set working directory
WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=akasha:akasha src/ ./src/

# Switch to non-root user
USER akasha

# Security settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "-m", "src.main"]
```

### 11.3 TLS/SSL Configuration

```nginx
# SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_stapling on;
ssl_stapling_verify on;

# Security headers
add_header Strict-Transport-Security "max-age=63072000" always;
add_header X-Content-Type-Options nosniff;
add_header X-Frame-Options DENY;
add_header X-XSS-Protection "1; mode=block";
```

---

## 12. Troubleshooting

### 12.1 Common Issues

#### 12.1.1 Service Won't Start

```bash
# Check logs
docker-compose logs akasha

# Check configuration
akasha validate-config --config ./config/akasha.yaml

# Check ports
netstat -tulpn | grep :8000

# Check disk space
df -h

# Check memory
free -h
```

#### 12.1.2 Performance Issues

```bash
# Monitor resource usage
docker stats

# Check system metrics
top
htop
iotop

# Profile application
akasha profile --duration 60

# Check database performance
docker exec qdrant-node-1 curl http://localhost:6333/metrics
```

#### 12.1.3 Memory Issues

```bash
# Check memory usage
docker exec akasha ps aux --sort=-%mem | head

# Restart with more memory
docker-compose up -d --scale akasha=0
docker-compose run --rm -e AKASHA_MAX_MEMORY=16G akasha

# Clear caches
docker exec redis redis-cli FLUSHALL
```

### 12.2 Debug Mode

```bash
# Enable debug mode
export AKASHA_DEBUG=true
export AKASHA_LOG_LEVEL=DEBUG

# Run with profiling
akasha start --profile --debug

# Interactive debugging
akasha shell
```

### 12.3 Health Checks

```bash
# API health check
curl -f http://localhost:8000/health

# Component status
curl http://localhost:8000/status | jq

# Database connection
curl http://localhost:6333/ | jq

# Cache status
redis-cli ping
```

This comprehensive deployment guide provides everything needed to successfully deploy and operate the Akasha system in various environments, from development to large-scale production deployments.