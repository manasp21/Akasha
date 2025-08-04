# Akasha

A state-of-the-art, modular, local-first multimodal RAG system optimized for Apple Silicon and research workflows.

## Features

- 🚀 **Local-First**: Complete privacy with local LLM and embedding models
- 🍎 **Apple Silicon Optimized**: Leverages MLX for efficient inference on M-series chips
- 🔧 **Modular Architecture**: Plugin-based system for extensibility
- 📊 **Multimodal**: Support for text, images, and mixed content
- ⚡ **High Performance**: Optimized for M4 Pro 48GB RAM systems
- 🛡️ **Secure**: Sandboxed plugin execution and comprehensive security
- 📈 **Observable**: Structured logging and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM (48GB recommended for optimal performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/akasha.git
   cd akasha
   ```

2. **Run the setup script**
   ```bash
   python scripts/setup.py
   ```

3. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

4. **Start the development server**
   ```bash
   python scripts/dev.py
   ```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Start server
python -m src.api.main
```

## Project Structure

```
akasha/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── core/              # Core utilities (config, logging, exceptions)
│   └── plugins/           # Plugin system
├── tests/                 # Test suite
├── scripts/               # Development scripts
├── config/                # Configuration files
├── docker/                # Docker configuration
└── docs/                  # Documentation
```

## Configuration

Akasha uses YAML configuration files. The main configuration is in `config/akasha.yaml`:

```yaml
system:
  name: akasha
  environment: development
  max_memory_gb: 40

api:
  host: "127.0.0.1"
  port: 8000

llm:
  backend: mlx
  model_name: gemma-3-27b
  memory_limit_gb: 16

logging:
  level: INFO
  output: console
```

Environment variables can override any configuration:
- `AKASHA_CONFIG`: Path to configuration file
- `AKASHA_SYSTEM__DEBUG`: Enable debug mode
- `AKASHA_API__PORT`: API server port

## Development

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src

# Specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Architecture Overview

**Phase 1: Foundation** ✅ (Complete)
- ✅ Core infrastructure (config, logging, API)
- ✅ Plugin architecture
- ✅ Testing framework
- ✅ Docker environment

**Phase 2: Core RAG** (Next)
- Document ingestion and processing
- Embedding generation and storage
- Vector search and retrieval
- LLM integration with MLX

**Phase 3: Advanced Features**
- Multimodal processing
- Graph RAG capabilities
- Advanced plugin ecosystem
- Performance optimizations

## API Endpoints

- `GET /`: System information
- `GET /health`: Health check
- `GET /status`: Detailed system status
- `GET /config`: Configuration (non-sensitive)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Roadmap

- [ ] Document processing pipeline
- [ ] MLX-based LLM integration
- [ ] Vector database integration
- [ ] Multimodal embedding support
- [ ] Web interface
- [ ] Plugin marketplace

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-org/akasha/issues)
- 💬 [Discussions](https://github.com/your-org/akasha/discussions)

