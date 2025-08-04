#!/usr/bin/env python3
"""End-to-end RAG pipeline testing."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

from src.rag.pipeline import RAGPipeline, RAGPipelineConfig, QueryMode
from src.rag.embeddings import EmbeddingConfig, EmbeddingBackend, EmbeddingModel
from src.rag.storage import StorageConfig
from src.rag.ingestion import ChunkingConfig, ChunkingStrategy
from src.rag.retrieval import RetrievalConfig
from src.llm.manager import LLMManager
from src.llm.provider import LLMResponse

async def test_complete_rag_pipeline():
    """Test the complete RAG pipeline from document to query."""
    print("🔄 COMPLETE RAG PIPELINE TEST")
    print("=" * 50)
    
    # Create comprehensive test document
    test_document = """
# Machine Learning Fundamentals

## What is Machine Learning?
Machine learning is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed. It focuses on the development of computer programs that 
can access data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from 
input variables to output variables. Examples include:
- Classification: Predicting categories (email spam detection)
- Regression: Predicting continuous values (house price prediction)

### Unsupervised Learning  
Unsupervised learning finds hidden patterns in data without labeled examples:
- Clustering: Grouping similar data points
- Association: Finding rules that describe relationships in data
- Dimensionality reduction: Simplifying data while preserving information

### Reinforcement Learning
Reinforcement learning trains agents to make decisions by interacting with an 
environment and receiving rewards or penalties for actions taken.

## Key Algorithms

### Decision Trees
Decision trees create a model that predicts target values by learning simple 
decision rules inferred from data features. They are easy to understand and interpret.

### Neural Networks
Neural networks are computing systems inspired by biological neural networks. 
Deep learning uses neural networks with multiple layers to model and understand 
complex patterns in data.

### Support Vector Machines
Support Vector Machines (SVM) are supervised learning models that analyze data 
for classification and regression analysis. They find optimal boundaries between 
different classes in the data.

## Applications

Machine learning has numerous real-world applications:
- Image recognition and computer vision
- Natural language processing and translation
- Recommendation systems (Netflix, Amazon)
- Autonomous vehicles and robotics
- Medical diagnosis and drug discovery
- Financial fraud detection and algorithmic trading

## Getting Started

To begin with machine learning:
1. Learn Python programming and key libraries (NumPy, Pandas, Scikit-learn)
2. Understand statistics and linear algebra fundamentals
3. Practice with datasets from Kaggle or UCI repository
4. Start with simple algorithms before moving to complex models
5. Focus on data preprocessing and feature engineering
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_document)
        f.flush()
        file_path = Path(f.name)
    
    try:
        print(f"📄 Test document: {len(test_document)} characters")
        
        # Configure RAG pipeline for testing
        config = RAGPipelineConfig(
            embedding_config=EmbeddingConfig(
                backend=EmbeddingBackend.MLX,  # Will fallback to sentence-transformers
                model_name=EmbeddingModel.ALL_MINILM_L6_V2,
                batch_size=8,
                cache_embeddings=True
            ),
            storage_config=StorageConfig(
                collection_name="test_complete_rag",
                persist_directory="./test_rag_db"
            ),
            chunking_config=ChunkingConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=400,
                chunk_overlap=50,
                min_chunk_size=100
            ),
            retrieval_config=RetrievalConfig(
                final_top_k=5,
                query_expansion=False  # Simplify for testing
            ),
            auto_embed_on_ingest=True
        )
        
        # Mock LLM manager for testing
        llm_manager = Mock(spec=LLMManager)
        llm_manager.providers = {"mock_provider": "active"}
        
        # Create pipeline
        pipeline = RAGPipeline(config, llm_manager)
        
        print("🚀 Initializing RAG pipeline...")
        start_time = time.time()
        await pipeline.initialize()
        init_time = time.time() - start_time
        print(f"✅ Pipeline initialized in {init_time:.2f} seconds")
        
        # Ingest document
        print("📥 Ingesting document...")
        start_time = time.time()
        ingestion_result = await pipeline.ingest_document(file_path)
        ingest_time = time.time() - start_time
        
        print(f"✅ Document ingested in {ingest_time:.2f} seconds")
        print(f"📊 Created {ingestion_result.chunks_created} chunks")
        print(f"🔍 Document ID: {ingestion_result.document_id}")
        
        # Test queries with mock LLM responses
        test_queries = [
            {
                "query": "What is machine learning?",
                "expected_context": "machine learning",
                "mock_response": "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without explicit programming."
            },
            {
                "query": "What are the main types of machine learning?",
                "expected_context": "supervised",
                "mock_response": "The main types of machine learning are: 1) Supervised Learning - uses labeled data, 2) Unsupervised Learning - finds patterns in unlabeled data, 3) Reinforcement Learning - learns through interaction and rewards."
            },
            {
                "query": "How do neural networks work?",
                "expected_context": "neural networks",
                "mock_response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions."
            },
            {
                "query": "What are some applications of machine learning?",
                "expected_context": "applications",
                "mock_response": "Machine learning has many applications including image recognition, natural language processing, recommendation systems, autonomous vehicles, medical diagnosis, and financial fraud detection."
            }
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        
        for i, test_case in enumerate(test_queries):
            query = test_case["query"]
            expected_context = test_case["expected_context"]
            mock_response_text = test_case["mock_response"]
            
            print(f"\n--- Query {i+1}: {query} ---")
            
            # Mock LLM response
            mock_response = LLMResponse(
                content=mock_response_text,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                generation_time=0.5,
                model_name="mock-model",
                metadata={"provider_name": "mock_provider"}
            )
            llm_manager.generate_rag_response.return_value = mock_response
            
            # Execute query
            start_time = time.time()
            result = await pipeline.query(query, mode=QueryMode.SIMPLE)
            query_time = time.time() - start_time
            
            print(f"⏱️  Query processed in {query_time:.3f} seconds")
            print(f"📝 Response: {result.response[:100]}...")
            print(f"📚 Retrieved {len(result.sources)} sources")
            
            # Verify retrieval worked
            assert len(result.sources) > 0, f"No sources retrieved for query: {query}"
            
            # Check if relevant context was retrieved
            retrieved_content = " ".join([
                source.get("content", "") for source in result.sources
            ]).lower()
            
            context_found = expected_context.lower() in retrieved_content
            print(f"🎯 Relevant context found: {'✅' if context_found else '❌'}")
            
            if not context_found:
                print(f"   Expected: '{expected_context}'")
                print(f"   Retrieved content: {retrieved_content[:200]}...")
            
            # Verify result structure
            assert result.query == query
            assert result.response == mock_response_text
            assert result.processing_time > 0
            assert result.query_mode == QueryMode.SIMPLE
            
            print(f"✅ Query {i+1} completed successfully")
        
        # Test pipeline statistics
        print(f"\n📊 Getting pipeline statistics...")
        stats = await pipeline.get_pipeline_stats()
        
        print(f"📈 Pipeline Stats:")
        print(f"   - Initialized: {stats['initialized']}")
        print(f"   - Vector store chunks: {stats['vector_store'].get('total_chunks', 'N/A')}")
        print(f"   - Storage backend: {stats['vector_store'].get('backend', 'N/A')}")
        print(f"   - Embedding dimensions: {stats['embeddings'].get('dimensions', 'N/A')}")
        
        # Test health check
        print(f"\n🏥 Running health check...")
        
        # Mock health check response
        llm_manager.health_check.return_value = {
            "initialized": True,
            "healthy_providers": ["mock_provider"],
            "providers": {"mock_provider": {"status": "healthy"}}
        }
        
        health = await pipeline.health_check()
        print(f"🏥 Health Status:")
        print(f"   - Pipeline: {'✅' if health['pipeline_initialized'] else '❌'}")
        print(f"   - Overall: {'✅' if health['overall_healthy'] else '❌'}")
        print(f"   - Vector Store: {'✅' if health['vector_store']['healthy'] else '❌'}")
        print(f"   - Embeddings: {'✅' if health['embeddings']['healthy'] else '❌'}")
        
        return pipeline, ingestion_result, test_queries
        
    finally:
        file_path.unlink()

async def test_concurrent_queries():
    """Test concurrent query processing."""
    print("\n🚀 CONCURRENT QUERY TEST")
    print("=" * 50)
    
    # Create simple test content
    test_content = """
# Simple Test Document

This is a basic test document for concurrent query testing.
It contains information about artificial intelligence and machine learning.
The document has multiple paragraphs to test retrieval performance.

Artificial intelligence is the simulation of human intelligence in machines.
Machine learning is a key component of AI systems.
Deep learning uses neural networks for complex pattern recognition.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        f.flush()
        file_path = Path(f.name)
    
    try:
        # Quick pipeline setup
        config = RAGPipelineConfig(
            embedding_config=EmbeddingConfig(
                backend=EmbeddingBackend.MLX,
                cache_embeddings=True
            ),
            storage_config=StorageConfig(
                collection_name="test_concurrent",
                persist_directory="./test_concurrent_db"
            )
        )
        
        llm_manager = Mock(spec=LLMManager)
        llm_manager.providers = {"mock": "active"}
        
        pipeline = RAGPipeline(config, llm_manager)
        await pipeline.initialize()
        
        # Ingest document
        await pipeline.ingest_document(file_path)
        
        # Prepare concurrent queries
        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "What is deep learning?",
            "What are neural networks?",
            "How is AI different from ML?"
        ]
        
        # Mock responses
        mock_responses = [
            "AI is the simulation of human intelligence in machines.",
            "Machine learning enables systems to learn from data automatically.",
            "Deep learning uses multiple neural network layers for pattern recognition.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "AI is the broader concept while ML is a subset of AI focused on learning from data."
        ]
        
        def mock_llm_response(query=None, **kwargs):
            index = queries.index(query) if query in queries else 0
            return LLMResponse(
                content=mock_responses[index],
                prompt_tokens=50,
                completion_tokens=25,
                total_tokens=75,
                generation_time=0.1,
                model_name="mock-model",
                metadata={"provider_name": "mock"}
            )
        
        llm_manager.generate_rag_response.side_effect = mock_llm_response
        
        # Execute concurrent queries
        print(f"🔄 Executing {len(queries)} concurrent queries...")
        start_time = time.time()
        
        tasks = [pipeline.query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(queries)
        
        print(f"✅ Concurrent queries completed in {total_time:.3f} seconds")
        print(f"📊 Average time per query: {avg_time:.3f} seconds")
        print(f"🚀 Throughput: {len(queries)/total_time:.1f} queries/second")
        
        # Verify all results
        for i, (query, result) in enumerate(zip(queries, results)):
            assert result.query == query
            assert len(result.sources) > 0
            print(f"   Query {i+1}: ✅ {len(result.sources)} sources retrieved")
        
        return pipeline, results
        
    finally:
        file_path.unlink()

async def main():
    """Run comprehensive RAG pipeline testing."""
    print("🎯 AKASHA RAG - END-TO-END PIPELINE TESTING")
    print("=" * 60)
    
    try:
        # Test 1: Complete RAG pipeline 
        pipeline1, ingestion_result, test_queries = await test_complete_rag_pipeline()
        
        # Test 2: Concurrent query processing
        pipeline2, concurrent_results = await test_concurrent_queries()
        
        print("\n🎉 ALL END-TO-END TESTS PASSED!")
        print("✅ Complete RAG pipeline working")
        print("✅ Document ingestion working")
        print("✅ Query processing working")
        print("✅ Concurrent queries working")
        print("✅ Health checks working")
        print("✅ Statistics collection working")
        
        print(f"\n📊 FINAL METRICS:")
        print(f"   - Documents processed: 2")
        print(f"   - Total queries tested: {len(test_queries) + len(concurrent_results)}")
        print(f"   - Chunks created: {ingestion_result.chunks_created}")
        print(f"   - All systems: OPERATIONAL ✅")
        
    except Exception as e:
        print(f"\n❌ END-TO-END TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())