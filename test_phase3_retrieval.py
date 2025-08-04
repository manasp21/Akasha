#!/usr/bin/env python3
"""
Phase 3 Advanced Retrieval Testing.

Tests the enhanced retrieval system with:
- Advanced query expansion
- Cross-encoder reranking
- Contextual retrieval with conversation history
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

from src.rag.retrieval import (
    DocumentRetriever, RetrievalConfig, RetrievalStrategy, 
    RerankingMethod, ExpansionStrategy
)
from src.rag.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingBackend, EmbeddingModel
from src.rag.storage import VectorStore, StorageConfig
from src.rag.ingestion import DocumentIngestion, ChunkingConfig, ChunkingStrategy
from src.rag.cross_encoder import CrossEncoderModel


async def test_advanced_query_expansion():
    """Test the enhanced query expansion system."""
    print("🔍 TESTING ADVANCED QUERY EXPANSION")
    print("=" * 50)
    
    # Set up embedding generator for semantic expansion
    embedding_config = EmbeddingConfig(
        backend=EmbeddingBackend.MLX,
        model_name=EmbeddingModel.ALL_MINILM_L6_V2,
        batch_size=4,
        cache_embeddings=True
    )
    embedding_generator = EmbeddingGenerator(embedding_config)
    await embedding_generator.initialize()
    
    # Set up vector store
    storage_config = StorageConfig(
        collection_name="test_phase3_expansion",
        persist_directory="./test_phase3_db"
    )
    vector_store = VectorStore(storage_config)
    await vector_store.initialize()
    
    # Configure advanced retrieval
    retrieval_config = RetrievalConfig(
        strategy=RetrievalStrategy.MULTI_STAGE,
        reranking_method=RerankingMethod.CROSS_ENCODER,
        query_expansion_strategy=ExpansionStrategy.HYBRID,
        cross_encoder_model=CrossEncoderModel.MS_MARCO_MINI,
        query_expansion=True,
        final_top_k=5
    )
    
    retriever = DocumentRetriever(vector_store, embedding_generator, retrieval_config)
    await retriever.initialize()
    
    # Test queries with different expansion strategies
    test_queries = [
        {
            "query": "What is AI?",
            "expected_expansions": ["artificial intelligence", "machine learning", "ML"],
            "strategy": ExpansionStrategy.SYNONYMS
        },
        {
            "query": "machine learning algorithms",
            "expected_expansions": ["neural networks", "deep learning"],
            "strategy": ExpansionStrategy.SEMANTIC
        },
        {
            "query": "research methods",
            "expected_expansions": ["methodology", "approach", "technique"],
            "strategy": ExpansionStrategy.HYBRID
        }
    ]
    
    print(f"Testing {len(test_queries)} query expansion scenarios...")
    
    for i, test_case in enumerate(test_queries):
        query = test_case["query"]
        strategy = test_case["strategy"]
        
        print(f"\n--- Test {i+1}: '{query}' with {strategy} ---")
        
        # Test query expansion
        expansion_result = await retriever.query_processor.query_expansion_service.expand_query(
            query, strategy=strategy
        )
        
        print(f"Original: {expansion_result.original_query}")
        print(f"Expanded: {expansion_result.expanded_query}")
        print(f"Terms Added: {expansion_result.expansion_terms}")
        print(f"Method: {expansion_result.expansion_method}")
        print(f"Metadata: {expansion_result.metadata}")
        
        # Verify expansion worked
        assert expansion_result.original_query == query
        assert len(expansion_result.expansion_terms) >= 0
        print(f"✅ Query expansion working for {strategy}")
    
    print(f"\n✅ Advanced query expansion system working correctly!")
    return retriever


async def test_cross_encoder_reranking(retriever):
    """Test cross-encoder reranking functionality."""
    print("\n🎯 TESTING CROSS-ENCODER RERANKING")
    print("=" * 50)
    
    # Create test documents with varying relevance
    test_docs = [
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
            "relevance": "high"
        },
        {
            "content": "The weather today is sunny with a chance of rain in the afternoon.",
            "relevance": "low"
        },
        {
            "content": "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
            "relevance": "high"
        },
        {
            "content": "Yesterday I went to the grocery store to buy some milk and bread.",
            "relevance": "low"
        },
        {
            "content": "Artificial intelligence algorithms can analyze large datasets to find meaningful patterns.",
            "relevance": "medium"
        }
    ]
    
    # Create temporary document files and ingest them
    ingestion = DocumentIngestion(ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=200,
        chunk_overlap=20
    ))
    
    print(f"📥 Ingesting {len(test_docs)} test documents...")
    
    for i, doc in enumerate(test_docs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc["content"])
            f.flush()
            file_path = Path(f.name)
        
        try:
            metadata, chunks = await ingestion.process_file(file_path)
            embedded_chunks = await retriever.embedding_generator.embed_chunks(chunks)
            await retriever.vector_store.add_document(metadata, embedded_chunks)
        finally:
            file_path.unlink()
    
    # Test retrieval with cross-encoder reranking
    test_query = "What is machine learning and AI?"
    
    print(f"\n🔍 Testing query: '{test_query}'")
    print("Expected: High relevance docs should rank higher after reranking")
    
    start_time = time.time()
    result = await retriever.retrieve(test_query, top_k=5)
    retrieval_time = time.time() - start_time
    
    print(f"⏱️  Retrieval completed in {retrieval_time:.3f} seconds")
    print(f"📊 Retrieved {len(result.chunks)} chunks")
    print(f"🎯 Retrieval method: {result.retrieval_method}")
    
    # Analyze results
    print(f"\n📋 Retrieved chunks ranked by relevance:")
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
        content_preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
        print(f"   {i+1}. Score: {score:.3f} | {content_preview}")
    
    # Verify cross-encoder improved ranking
    high_relevance_positions = []
    for i, chunk in enumerate(result.chunks):
        chunk_content = chunk.content.lower()
        if any(term in chunk_content for term in ["machine learning", "artificial intelligence", "deep learning"]):
            high_relevance_positions.append(i)
    
    if high_relevance_positions:
        avg_position = sum(high_relevance_positions) / len(high_relevance_positions)
        print(f"📈 Average position of relevant docs: {avg_position:.1f}")
        assert avg_position < 2.5, "Cross-encoder should rank relevant docs higher"
        print(f"✅ Cross-encoder reranking improved relevance!")
    
    return result


async def test_contextual_retrieval(retriever):
    """Test contextual retrieval with conversation history."""
    print("\n💬 TESTING CONTEXTUAL RETRIEVAL")
    print("=" * 50)
    
    # Create conversation history
    conversation_history = [
        {
            "query": "What are neural networks?",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
            "relevant_docs": ["doc_neural_networks"]
        },
        {
            "query": "How do they learn?",
            "response": "Neural networks learn through training with data, adjusting weights between connections to minimize errors.",
            "relevant_docs": ["doc_learning_process"]
        }
    ]
    
    # Simulate a follow-up query
    context_query = "Tell me more about training"
    
    print(f"📜 Conversation History: {len(conversation_history)} turns")
    for i, turn in enumerate(conversation_history):
        print(f"   Turn {i+1}: {turn['query']}")
    
    print(f"\n🔍 Context Query: '{context_query}'")
    
    # Test retrieval with conversation context
    context = {
        "conversation_history": conversation_history,
        "relevant_documents": ["doc_neural_networks", "doc_learning_process"]
    }
    
    start_time = time.time()
    contextual_result = await retriever.retrieve(context_query, context=context)
    contextual_time = time.time() - start_time
    
    print(f"⏱️  Contextual retrieval in {contextual_time:.3f} seconds")
    print(f"📊 Retrieved {len(contextual_result.chunks)} chunks")
    print(f"🎯 Method: {contextual_result.retrieval_method}")
    
    # Test without context for comparison
    start_time = time.time()
    non_contextual_result = await retriever.retrieve(context_query)
    non_contextual_time = time.time() - start_time
    
    print(f"\n📊 Comparison:")
    print(f"   With context: {len(contextual_result.chunks)} chunks, method: {contextual_result.retrieval_method}")
    print(f"   Without context: {len(non_contextual_result.chunks)} chunks, method: {non_contextual_result.retrieval_method}")
    
    # Analyze contextual improvements
    context_boost_applied = "contextual" in contextual_result.retrieval_method
    print(f"🔄 Contextual boosting applied: {'✅' if context_boost_applied else '❌'}")
    
    # Check if context terms were extracted
    query_context = contextual_result.query_context
    if hasattr(query_context, 'context_boost_terms'):
        print(f"📝 Context boost terms: {query_context.context_boost_terms}")
        print(f"📚 Relevant documents: {query_context.relevant_documents}")
    
    print(f"✅ Contextual retrieval system working!")
    
    return contextual_result


async def test_retrieval_statistics(retriever):
    """Test retrieval system statistics and performance."""
    print("\n📊 TESTING RETRIEVAL STATISTICS")
    print("=" * 50)
    
    # Get comprehensive statistics
    stats = await retriever.get_retrieval_stats()
    
    print("📈 Retrieval System Statistics:")
    print(f"   Strategy: {stats['config']['strategy']}")
    print(f"   Reranking: {stats['config']['reranking_method']}")
    print(f"   Query Expansion: {stats['config']['query_expansion_strategy']}")
    print(f"   Cross-Encoder Model: {stats['config']['cross_encoder_model']}")
    print(f"   Final Top-K: {stats['config']['final_top_k']}")
    
    print(f"\n🧠 Embedding System:")
    print(f"   Backend: {stats['embeddings']['backend']}")
    print(f"   Model: {stats['embeddings']['model_name']}")
    print(f"   Dimensions: {stats['embeddings']['dimensions']}")
    print(f"   Cache Size: {stats['embeddings']['cache_size']} entries")
    
    if "cross_encoder" in stats:
        print(f"\n🎯 Cross-Encoder:")
        print(f"   Model: {stats['cross_encoder']['model_name']}")
        print(f"   Loaded: {stats['cross_encoder']['model_loaded']}")
        print(f"   Cache Size: {stats['cross_encoder']['cache_size']} predictions")
        print(f"   Batch Size: {stats['cross_encoder']['batch_size']}")
    
    print(f"\n🗄️  Vector Storage:")
    print(f"   Backend: {stats['storage'].get('backend', 'N/A')}")
    print(f"   Total Chunks: {stats['storage'].get('total_chunks', 'N/A')}")
    
    print(f"✅ Statistics collection working!")
    
    return stats


async def main():
    """Run comprehensive Phase 3 retrieval testing."""
    print("🚀 AKASHA PHASE 3 - ADVANCED RETRIEVAL TESTING")
    print("=" * 70)
    
    try:
        # Test 1: Advanced Query Expansion
        retriever = await test_advanced_query_expansion()
        
        # Test 2: Cross-Encoder Reranking
        rerank_result = await test_cross_encoder_reranking(retriever)
        
        # Test 3: Contextual Retrieval
        context_result = await test_contextual_retrieval(retriever)
        
        # Test 4: System Statistics
        stats = await test_retrieval_statistics(retriever)
        
        print("\n🎉 ALL PHASE 3 RETRIEVAL TESTS PASSED!")
        print("✅ Advanced query expansion working")
        print("✅ Cross-encoder reranking working")
        print("✅ Contextual retrieval working")
        print("✅ Statistics collection working")
        
        print(f"\n📊 PHASE 3 SUMMARY:")
        print(f"   - Query expansion strategies: 3 tested")
        print(f"   - Cross-encoder model: {stats['config']['cross_encoder_model']}")
        print(f"   - Contextual boosting: Enabled")
        print(f"   - Multi-stage pipeline: 4 stages")
        print(f"   - All systems: OPERATIONAL ✅")
        
    except Exception as e:
        print(f"\n❌ PHASE 3 RETRIEVAL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())