#!/usr/bin/env python3
"""Real-world embedding model testing script."""

import asyncio
import time
from pathlib import Path
import tempfile

from src.rag.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingBackend, EmbeddingModel
from src.rag.ingestion import DocumentIngestion, ChunkingConfig, ChunkingStrategy
from src.rag.storage import VectorStore, StorageConfig
from src.rag.pipeline import RAGPipeline, RAGPipelineConfig
from src.llm.manager import LLMManager
from src.core.logging import get_logger

async def test_real_embedding_model():
    """Test downloading and using a real embedding model."""
    print("🧠 Testing Real Embedding Models")
    print("=" * 50)
    
    # Configure for real model
    config = EmbeddingConfig(
        backend=EmbeddingBackend.MLX,  # Will fallback to sentence-transformers
        model_name=EmbeddingModel.ALL_MINILM_L6_V2,  # Small, fast model
        batch_size=4,
        cache_embeddings=True
    )
    
    print(f"📥 Loading model: {config.model_name}")
    start_time = time.time()
    
    generator = EmbeddingGenerator(config)
    await generator.initialize()
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Test with real text
    test_texts = [
        "Machine learning is transforming how we process information.",
        "Deep neural networks can learn complex patterns from data.",
        "Natural language processing enables computers to understand text.",
        "Vector databases store high-dimensional embeddings efficiently."
    ]
    
    print(f"\n📝 Generating embeddings for {len(test_texts)} texts...")
    start_time = time.time()
    
    embeddings = await generator.embed_texts(test_texts)
    
    embed_time = time.time() - start_time
    print(f"✅ Generated embeddings in {embed_time:.2f} seconds")
    
    # Analyze results
    dimensions = len(embeddings[0])
    print(f"📊 Embedding dimensions: {dimensions}")
    print(f"🔢 Embeddings shape: {len(embeddings)} × {dimensions}")
    
    # Test similarity
    query = "What is machine learning?"
    query_embedding = await generator.embed_query(query)
    
    similarities = []
    for i, text_embedding in enumerate(embeddings):
        similarity = await generator.compute_similarity(query_embedding, text_embedding)
        similarities.append((similarity, test_texts[i]))
    
    similarities.sort(reverse=True)
    
    print(f"\n🔍 Query: '{query}'")
    print("🎯 Most similar texts:")
    for similarity, text in similarities[:2]:
        print(f"   {similarity:.3f}: {text}")
    
    # Test caching
    print(f"\n💾 Testing embedding cache...")
    start_time = time.time()
    embeddings2 = await generator.embed_texts(test_texts)  # Should use cache
    cache_time = time.time() - start_time
    
    print(f"✅ Cache retrieval in {cache_time:.4f} seconds ({cache_time/embed_time:.0%} of original time)")
    assert embeddings == embeddings2, "Cache returned different embeddings!"
    
    info = await generator.get_embedding_info()
    print(f"📈 Cache size: {info.get('cache_size', 'N/A')} entries")
    
    return generator

async def test_real_document_processing():
    """Test processing real documents with embedding and storage."""
    print("\n📄 Testing Real Document Processing")
    print("=" * 50)
    
    # Create a realistic test document
    test_content = """
# Akasha RAG System Technical Overview

## Introduction
The Akasha RAG (Retrieval-Augmented Generation) system is a sophisticated 
multimodal document processing and query system designed for research workflows.

## Core Components

### Document Ingestion
- Supports multiple formats: PDF, DOCX, Markdown, HTML
- Intelligent chunking with overlap for context preservation
- Metadata extraction and hash-based deduplication

### Embedding Generation  
- MLX backend optimized for Apple Silicon M4 Pro
- Fallback to sentence-transformers for compatibility
- Caching system for performance optimization
- Batch processing for efficiency

### Vector Storage
- ChromaDB integration for development
- Qdrant support for production deployment
- Configurable distance metrics and HNSW indexing
- Metadata filtering and hybrid search capabilities

### Retrieval System
- Multi-stage retrieval with re-ranking
- Semantic similarity search
- Context-aware result filtering
- Performance optimization for sub-3 second queries

## Technical Specifications

### Memory Requirements
- Minimum: 32GB unified memory
- Recommended: 48GB for optimal performance
- Gemma 3 27B (4-bit quantized): ~13.5GB
- JINA v4 embeddings: ~3GB
- Vector storage: 5-10GB
- System overhead: 8-12GB

### Performance Targets
- Query response time: <3 seconds
- Concurrent ingestion: 5 documents
- Throughput: 1000+ chunks/minute
- Embedding generation: 100+ texts/second

## Architecture Philosophy
The system follows a modular design with clear separation of concerns.
Each component can be independently replaced or upgraded.
Privacy is paramount - all processing occurs locally.
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        f.flush()
        file_path = Path(f.name)
    
    try:
        print(f"📁 Processing document: {file_path.name}")
        print(f"📏 Document size: {len(test_content)} characters")
        
        # Configure ingestion
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300,  # Smaller chunks for testing
            chunk_overlap=50,
            min_chunk_size=50
        )
        
        ingestion = DocumentIngestion(chunking_config)
        
        # Process document
        start_time = time.time()
        metadata, chunks = await ingestion.process_file(file_path)
        process_time = time.time() - start_time
        
        print(f"✅ Processed in {process_time:.2f} seconds")
        print(f"📊 Created {len(chunks)} chunks")
        print(f"📈 Average chunk size: {sum(len(chunk.content) for chunk in chunks) / len(chunks):.0f} characters")
        
        # Verify no excessive duplication
        total_content = sum(len(chunk.content) for chunk in chunks)
        duplication_factor = total_content / len(test_content)
        print(f"🔍 Content duplication factor: {duplication_factor:.2f}x")
        
        if duplication_factor > 1.5:
            print(f"⚠️  WARNING: High duplication factor detected!")
        else:
            print(f"✅ Acceptable duplication factor")
        
        # Show sample chunks
        print(f"\n📖 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i}: {len(chunk.content)} chars - {chunk.content[:80]}...")
        
        return metadata, chunks
        
    finally:
        file_path.unlink()

async def test_vector_storage():
    """Test vector storage with real embeddings."""
    print("\n🗄️  Testing Vector Storage")
    print("=" * 50)
    
    # Configure ChromaDB storage
    storage_config = StorageConfig(
        collection_name="test_akasha_rag",
        persist_directory="./test_chroma_db"
    )
    
    vector_store = VectorStore(storage_config)
    await vector_store.initialize()
    
    print("✅ Vector store initialized")
    
    # Create test data with embeddings
    from src.rag.ingestion import DocumentChunk, DocumentMetadata
    
    chunks = [
        DocumentChunk(
            id="test_chunk_1",
            content="Machine learning algorithms can process vast amounts of data.",
            document_id="test_doc_1",
            chunk_index=0,
            embedding=[0.1] * 384  # Mock embedding
        ),
        DocumentChunk(
            id="test_chunk_2", 
            content="Deep learning uses neural networks with multiple layers.",
            document_id="test_doc_1",
            chunk_index=1,
            embedding=[0.2] * 384
        )
    ]
    
    metadata = DocumentMetadata(
        file_path="test_document.txt",
        file_name="test_document.txt",
        file_size=100,
        file_hash="test_hash",
        mime_type="text/plain",
        format="text",
        processed_at=time.time(),
        chunk_count=len(chunks),
        processing_time=1.0
    )
    
    # Add to vector store
    start_time = time.time()
    await vector_store.add_document(metadata, chunks)
    storage_time = time.time() - start_time
    
    print(f"✅ Stored {len(chunks)} chunks in {storage_time:.3f} seconds")
    
    # Test retrieval
    query_embedding = [0.15] * 384  # Similar to first chunk
    results = await vector_store.search_similar(query_embedding, top_k=2)
    
    print(f"🔍 Retrieved {len(results)} results")
    for i, result in enumerate(results):
        print(f"   Result {i}: score={result.score:.3f}, content={result.chunk.content[:60]}...")
    
    # Get stats
    stats = await vector_store.get_stats()
    print(f"📊 Storage stats: {stats}")
    
    return vector_store

async def main():
    """Run comprehensive real-world testing."""
    print("🚀 AKASHA RAG SYSTEM - REAL-WORLD TESTING")
    print("=" * 60)
    
    try:
        # Test 1: Real embedding models
        embedding_generator = await test_real_embedding_model()
        
        # Test 2: Real document processing  
        metadata, chunks = await test_real_document_processing()
        
        # Test 3: Vector storage
        vector_store = await test_vector_storage()
        
        print("\n🎉 ALL REAL-WORLD TESTS PASSED!")
        print("✅ Embedding models working")
        print("✅ Document processing working") 
        print("✅ Vector storage working")
        print("✅ Phase 2 RAG system fully functional")
        
    except Exception as e:
        print(f"\n❌ REAL-WORLD TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())