# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Akasha is a state-of-the-art, modular, local-first multimodal RAG (Retrieval-Augmented Generation) system designed for research and academic workflows. The system enables users to ingest, process, and query complex documents (PDFs, research papers) containing both text and visual elements, providing contextually relevant responses while maintaining complete privacy and local control.

## Current State: Specification Phase Complete

The project currently contains comprehensive technical specifications in `docs/`:

- **SYSTEM_ARCHITECTURE.md**: Complete system design with 10 core components
- **COMPONENT_SPECIFICATIONS.md**: Detailed implementation specs for each component  
- **API_SPECIFICATION.md**: Full REST and WebSocket API documentation
- **PLUGIN_ARCHITECTURE.md**: Comprehensive plugin system design
- **DEPLOYMENT_GUIDE.md**: Production deployment strategies
- **DEVELOPMENT_ROADMAP.md**: 24-week development timeline in 6 phases

## Technology Stack (M4 Pro 48GB Optimized)

- **Backend**: Python 3.11+ with FastAPI
- **Frontend**: React + TypeScript  
- **Document Processing**: MinerU 2 for PDF extraction
- **Embeddings**: JINA v4 for multimodal text/image embeddings
- **Vector Storage**: ChromaDB (development) → Qdrant (production)
- **LLM**: Gemma 3 27B via MLX (Apple Silicon optimized, 4-bit quantized)
- **Containerization**: Docker + Compose (ARM64 for Apple Silicon)
- **Memory Requirements**: 32GB minimum, 48GB+ recommended

## Development Phases

The project follows a structured 6-phase approach (4 weeks each):

1. **Foundation** (Weeks 1-4): Core architecture, plugin system, basic API
2. **Core Processing** (Weeks 5-8): Document ingestion, embeddings, vector storage  
3. **Advanced RAG** (Weeks 9-12): Multi-stage retrieval, LLM integration, GraphRAG
4. **User Interface** (Weeks 13-16): React frontend, document management, chat
5. **Production Ready** (Weeks 17-20): Security, deployment, monitoring
6. **Advanced Features** (Weeks 21-24): Plugin marketplace, collaboration tools

## Architecture Overview

The system consists of 10 core components in a layered architecture:

**Presentation Layer**: Web UI (React), REST API (FastAPI), WebSocket API
**Service Layer**: RAG Engine, LLM Service, Plugin Manager
**Processing Layer**: Embedding Service, Vector Store, Knowledge Graph  
**Data Layer**: Ingestion Engine, Document Processor, Cache Manager

## Key Design Principles

- **Modularity**: Every component is independently replaceable
- **Privacy-First**: 100% local operation, no external data transmission
- **Research-Oriented**: Optimized for academic and technical document workflows
- **Multimodal**: Native support for text, images, tables, and visual elements
- **Performance**: Sub-3 second query response times target

## Next Steps for Implementation

When ready to begin implementation, start with **Phase 1 Foundation**:

1. Set up Python project structure with FastAPI
2. Implement configuration system with Pydantic validation
3. Create plugin architecture framework
4. Build basic API endpoints with authentication
5. Set up Docker development environment
6. Establish CI/CD pipeline

Refer to `docs/DEVELOPMENT_ROADMAP.md` for detailed task breakdown and success criteria for each phase.

## Apple Silicon M4 Pro Compatibility

The system is specifically optimized for Apple Silicon M4 Pro with 48GB unified memory:

- **MLX Backend**: Use MLX for optimal Apple Silicon performance
- **4-bit Quantization**: Essential for fitting Gemma 3 27B in available memory
- **Memory Distribution**: 
  - Gemma 3 27B (4-bit): ~13.5GB
  - JINA v4 embeddings: ~3GB
  - Vector storage: ~5-10GB
  - System overhead: ~8-12GB (macOS)
  - Application processes: ~3-5GB
  - **Total**: ~32-43GB (fits comfortably in 48GB)

## Important Notes

- All components must support the modular plugin architecture
- Security and privacy are paramount - never implement external data transmission
- Follow the specifications exactly as defined in the docs/ directory
- Maintain comprehensive test coverage (>80% target)
- Performance optimization is critical for research workflows
- **Always use MLX backend and 4-bit quantization for Apple Silicon deployment**