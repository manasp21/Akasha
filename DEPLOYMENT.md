# 🚀 Akasha Production Deployment Guide

This guide covers deploying Akasha in production with Docker containers, optimized for Apple Silicon M4 Pro 48GB systems.

## 📋 Prerequisites

### System Requirements
- **Memory**: 32GB+ (48GB recommended for M4 Pro)
- **Storage**: 100GB+ available space
- **Platform**: Apple Silicon (ARM64) or x86_64
- **OS**: macOS, Linux, or Windows with WSL2

### Software Requirements
- Docker 20.10+ with Docker Compose
- Git (for updates)
- curl (for health checks)
- Python 3.11+ (for validation scripts)

## 🔧 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd akasha
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.production.template .env.production

# Edit with your secure values
nano .env.production
```

**⚠️ CRITICAL**: Change these values in `.env.production`:
- `JWT_SECRET_KEY`: Generate a secure 64+ character random string
- `ADMIN_EMAIL` and `ADMIN_PASSWORD`: Set secure admin credentials

### 3. Deploy
```bash
# Run deployment script
./scripts/deployment/deploy.sh

# Or deploy manually
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
```

### 4. Verify Deployment
```bash
# Check status
./scripts/deployment/manage.sh status

# View services
curl http://localhost:8000/health
curl http://localhost:3000
```

## 📂 Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Vector DB     │
│   (React/Nginx) │◄──►│   (FastAPI)     │◄──►│   (ChromaDB)    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Cache/Queue   │              │
         └──────────────┤     (Redis)     │◄─────────────┘
                        │   Port: 6379    │
                        └─────────────────┘
```

## 🐳 Docker Services

### Core Services
- **akasha-api**: FastAPI backend with authentication, rate limiting, security
- **akasha-frontend**: React app served with Nginx
- **akasha-redis**: Cache and job queue
- **akasha-chromadb**: Vector database for embeddings

### Optional Services
- **nginx**: Reverse proxy (with `--profile nginx`)
- **prometheus**: Metrics collection (with `--profile monitoring`)

## 🔐 Security Features

### Authentication & Authorization
- JWT tokens (15min access, 30-day refresh)
- Role-based access control (Admin, User, Viewer)
- Bcrypt password hashing (12 rounds)
- Rate limiting (5 login attempts/min, 100 API requests/min)

### Security Headers
- Content Security Policy (CSP)
- X-Frame-Options, X-XSS-Protection
- Strict Transport Security (HSTS)
- Content-Type-Options

### Input Validation
- Request size limits (100MB)
- File type validation
- Suspicious pattern detection
- SQL injection prevention

## 📊 Resource Allocation (M4 Pro 48GB)

```yaml
Backend (API):     20GB  # LLM + embeddings + processing
Vector Store:      8GB   # ChromaDB with cache
Cache (Redis):     1GB   # Session + queue data
Frontend:          512MB # Nginx + static files
System Reserve:    8GB   # macOS + other processes
Available:         ~10GB # Buffer for operations
```

## 🛠️ Management Commands

### Service Management
```bash
# Check status
./scripts/deployment/manage.sh status

# Start/stop services
./scripts/deployment/manage.sh start
./scripts/deployment/manage.sh stop
./scripts/deployment/manage.sh restart

# View logs
./scripts/deployment/manage.sh logs
./scripts/deployment/manage.sh logs akasha-api
```

### Updates and Maintenance
```bash
# Update deployment
./scripts/deployment/manage.sh update

# Create backup
./scripts/deployment/manage.sh backup

# Restore from backup
./scripts/deployment/manage.sh restore backups/akasha-backup-20240101-120000.tar.gz

# Clean up unused resources
./scripts/deployment/manage.sh cleanup
```

### Debugging
```bash
# Open shell in container
./scripts/deployment/manage.sh shell akasha-api

# Execute command
./scripts/deployment/manage.sh exec akasha-api python -c "print('Hello')"

# Show configuration
./scripts/deployment/manage.sh config
```

## 🔍 Monitoring & Health Checks

### Health Endpoints
- API Health: `http://localhost:8000/health`
- System Status: `http://localhost:8000/status`
- Frontend: `http://localhost:3000`

### Logs
```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# Service-specific logs
docker-compose -f docker-compose.prod.yml logs -f akasha-api
```

### Metrics (with monitoring profile)
```bash
# Start with monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Access Prometheus
open http://localhost:9091
```

## 🚦 Environment Configuration

### Production Environment Variables
```bash
# Security (REQUIRED)
JWT_SECRET_KEY=your-64-character-secure-random-string
ADMIN_EMAIL=admin@yourdomain.com
ADMIN_PASSWORD=your-secure-password

# System Resources
AKASHA_SYSTEM__MAX_MEMORY_GB=40
AKASHA_LLM__MEMORY_LIMIT_GB=16
AKASHA_VECTOR_STORE__MEMORY_LIMIT_GB=8

# API Configuration
AKASHA_API__CORS_ORIGINS=["https://yourdomain.com"]
AKASHA_AUTH__ENABLE_RATE_LIMITING=true

# Enable monitoring
AKASHA_MONITORING__ENABLE_METRICS=true
```

## 🔄 Backup and Recovery

### Automated Backups
```bash
# Create backup
./scripts/deployment/manage.sh backup

# Schedule with cron
0 2 * * * /path/to/akasha/scripts/deployment/manage.sh backup
```

### Data Locations
- **Application Data**: `akasha-prod-data` volume
- **Logs**: `akasha-prod-logs` volume
- **Models**: `akasha-prod-models` volume
- **Vector DB**: `akasha-prod-chroma` volume

## 🐛 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
./scripts/deployment/manage.sh logs akasha-api

# Check resources
docker stats

# Restart service
docker-compose -f docker-compose.prod.yml restart akasha-api
```

#### Authentication Issues
```bash
# Check JWT secret is set
grep JWT_SECRET_KEY .env.production

# Reset admin user (exec in container)
./scripts/deployment/manage.sh exec akasha-api python -c "
from src.api.auth import create_user, users_db
users_db.clear()
# Admin user will be recreated on restart
"
```

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Reduce memory limits in docker-compose.prod.yml
# Restart with new limits
./scripts/deployment/manage.sh restart
```

### Performance Tuning

#### For Systems with Less RAM
```yaml
# In docker-compose.prod.yml, adjust:
AKASHA_SYSTEM__MAX_MEMORY_GB=24
AKASHA_LLM__MEMORY_LIMIT_GB=12
AKASHA_VECTOR_STORE__MEMORY_LIMIT_GB=4
```

#### For Better Performance
```yaml
# Enable model preloading
AKASHA_PRELOAD_MODELS=true

# Increase API workers (if sufficient CPU)
AKASHA_API__WORKERS=2
```

## 🔐 SSL/HTTPS Setup

### With Nginx Profile
```bash
# 1. Get SSL certificates (Let's Encrypt, etc.)
# 2. Place in docker/nginx/ssl/
# 3. Update docker/nginx/default.conf
# 4. Start with nginx profile
docker-compose -f docker-compose.prod.yml --profile nginx up -d
```

### Environment Variables for SSL
```bash
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
AKASHA_SECURITY__ENABLE_HTTPS_ONLY=true
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale API instances
./scripts/deployment/manage.sh scale akasha-api 2

# With load balancer
docker-compose -f docker-compose.prod.yml --profile nginx up -d
```

### Resource Scaling
```bash
# Increase memory limits
# Edit docker-compose.prod.yml
# Restart services
./scripts/deployment/manage.sh restart
```

## 📞 Support

### Validation
```bash
# Run system validation
python test_phases_final_validation.py

# Check deployment
./scripts/deployment/deploy.sh check
```

### Getting Help
1. Check logs: `./scripts/deployment/manage.sh logs`
2. Verify health: `curl http://localhost:8000/health`
3. Check system resources: `docker stats`
4. Review configuration: `./scripts/deployment/manage.sh config`

## 🎯 Next Steps

After successful deployment:

1. **Change Default Credentials**: Update admin email/password
2. **Configure Domain**: Set up proper domain and SSL
3. **Setup Monitoring**: Enable Prometheus/Grafana
4. **Schedule Backups**: Automate regular backups
5. **Performance Tune**: Optimize for your specific workload

---

**🎉 Your Akasha production deployment is ready!**

Access your system at:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs