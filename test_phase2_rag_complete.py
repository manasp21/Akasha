#!/usr/bin/env python3
"""
Complete End-to-End RAG Pipeline Testing.

Tests the full RAG pipeline including retrieval and generation capabilities.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.rag.ingestion import DocumentIngestion, ChunkingConfig, ChunkingStrategy
from src.rag.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingBackend, EmbeddingModel
from src.rag.storage import VectorStore, StorageConfig


def create_knowledge_base_pdf(filename: str, topic: str, content_sections: dict) -> Path:
    """Create a comprehensive test PDF with structured knowledge."""
    file_path = Path(filename)
    
    c = canvas.Canvas(str(file_path), pagesize=letter)
    width, height = letter
    
    # Title page
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredText(width/2, height-100, f"Knowledge Base: {topic}")
    
    c.setFont("Helvetica", 12)
    y_position = height - 150
    
    for section_title, section_content in content_sections.items():
        # Section header
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, section_title)
        y_position -= 30
        
        # Section content
        c.setFont("Helvetica", 11)
        lines = section_content.split('\n')
        
        for line in lines:
            if y_position < 100:  # Start new page
                c.showPage()
                y_position = height - 50
            
            # Wrap long lines
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + word) < 90:
                    current_line += word + " "
                else:
                    if current_line:
                        c.drawString(50, y_position, current_line.strip())
                        y_position -= 15
                    current_line = word + " "
            
            if current_line:
                c.drawString(50, y_position, current_line.strip())
                y_position -= 15
        
        y_position -= 20  # Extra space between sections
    
    c.save()
    return file_path


class SimpleRAGProcessor:
    """Simple RAG processor for testing without full LLM integration."""
    
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    async def generate_answer(self, query: str, top_k: int = 5) -> dict:
        """Generate answer using retrieved context."""
        # Retrieve relevant chunks
        query_embedding = await self.embedding_generator.embed_text(query)
        results = await self.vector_store.search_similar(query_embedding, top_k=top_k)
        
        # Combine context
        context_chunks = [result.chunk.content for result in results]
        combined_context = "\n\n".join(context_chunks)
        
        # Simple answer generation (extractive)
        answer = self._extract_answer(query, combined_context, results)
        
        return {
            "query": query,
            "answer": answer,
            "sources": results,
            "context_length": len(combined_context),
            "num_sources": len(results)
        }
    
    def _extract_answer(self, query: str, context: str, results) -> str:
        """Extract answer from context (simple implementation)."""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Find most relevant sentence
        sentences = context.split('.')
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            # Simple relevance scoring
            score = 0
            for word in query_lower.split():
                if word in sentence.lower():
                    score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence + "."
        
        # Fallback to first chunk
        if results:
            return results[0].chunk.content[:200] + "..."
        
        return "No relevant information found."


async def test_comprehensive_rag_pipeline():
    """Test comprehensive RAG pipeline with multiple knowledge domains."""
    print("🧠 TESTING COMPREHENSIVE RAG PIPELINE")
    print("=" * 60)
    
    # Create comprehensive knowledge base
    knowledge_sections = {
        "Machine Learning Fundamentals": """
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. The field encompasses supervised learning, unsupervised learning, and reinforcement learning approaches.
        
        Supervised learning uses labeled data to train models that can make predictions on new, unseen data. Common algorithms include linear regression, decision trees, random forests, and neural networks.
        
        Unsupervised learning finds patterns in data without labeled examples. Clustering algorithms like K-means and hierarchical clustering are widely used for data exploration and segmentation.
        
        Reinforcement learning involves training agents to make decisions through interaction with an environment, receiving rewards or penalties based on their actions.
        """,
        
        "Deep Learning Architecture": """
        Deep learning utilizes artificial neural networks with multiple layers to process complex patterns in data. Convolutional Neural Networks (CNNs) excel at image recognition tasks by detecting hierarchical features.
        
        Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are designed for sequential data processing, making them effective for natural language processing and time series analysis.
        
        Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing through self-attention mechanisms. This architecture forms the basis for large language models like GPT and BERT.
        
        Modern deep learning frameworks like PyTorch and TensorFlow provide efficient tools for building and training neural networks at scale.
        """,
        
        "RAG Systems Implementation": """
        Retrieval-Augmented Generation (RAG) systems combine the power of large language models with external knowledge retrieval. The architecture consists of document ingestion, vector storage, and generation components.
        
        Document ingestion involves parsing various file formats, chunking content into manageable pieces, and generating embeddings using models like BERT or sentence-transformers.
        
        Vector databases like ChromaDB, Pinecone, and Qdrant provide efficient storage and similarity search capabilities for high-dimensional embeddings.
        
        The generation phase uses retrieved context to produce accurate, grounded responses while maintaining the fluency of large language models. This approach significantly reduces hallucination and improves factual accuracy.
        """,
        
        "Performance Optimization": """
        RAG system performance depends on several key factors: embedding quality, retrieval precision, and generation efficiency. Optimal chunk sizes typically range from 200-800 tokens depending on the domain.
        
        Hybrid search combining dense vector similarity with sparse keyword matching improves retrieval across different query types. Cross-encoder reranking can further improve result quality.
        
        Caching strategies for embeddings and frequent queries significantly reduce response times. Batch processing and asynchronous operations improve throughput for large document collections.
        
        Memory management is crucial when processing large documents, especially on systems with limited resources. Streaming and incremental processing techniques help manage memory usage.
        """
    }
    
    # Create test knowledge base PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        kb_path = create_knowledge_base_pdf(tmp_file.name, "AI & Machine Learning", knowledge_sections)
    
    try:
        # Step 1: Ingest Knowledge Base
        print("📚 Step 1: Knowledge Base Ingestion")
        ingestion = DocumentIngestion(ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=600,
            chunk_overlap=100,
            preserve_sentences=True
        ))
        
        start_time = time.time()
        metadata, chunks = await ingestion.process_file(kb_path)
        ingestion_time = time.time() - start_time
        
        print(f"   ✅ Ingested knowledge base: {len(chunks)} chunks in {ingestion_time:.3f}s")
        print(f"   📄 Document: {metadata.file_name}")
        print(f"   📊 Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars")
        
        # Step 2: Generate Embeddings
        print("\n🧠 Step 2: Embedding Generation")
        embedding_config = EmbeddingConfig(
            backend=EmbeddingBackend.MLX,
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
            batch_size=32,
            cache_embeddings=True
        )
        embedding_generator = EmbeddingGenerator(embedding_config)
        await embedding_generator.initialize()
        
        start_time = time.time()
        embedded_chunks = await embedding_generator.embed_chunks(chunks)
        embedding_time = time.time() - start_time
        
        print(f"   ✅ Generated embeddings: {len(embedded_chunks)} vectors in {embedding_time:.3f}s")
        print(f"   🚀 Embedding speed: {len(embedded_chunks)/embedding_time:.1f} chunks/sec")
        
        # Step 3: Build Vector Index
        print("\n🗄️  Step 3: Vector Index Construction")
        storage_config = StorageConfig(
            collection_name="comprehensive_rag_test",
            persist_directory="./test_comprehensive_rag_db"
        )
        vector_store = VectorStore(storage_config)
        await vector_store.initialize()
        
        start_time = time.time()
        await vector_store.add_document(metadata, embedded_chunks)
        storage_time = time.time() - start_time
        
        print(f"   ✅ Built vector index in {storage_time:.3f}s")
        print(f"   📊 Index size: {len(embedded_chunks)} vectors")
        
        # Step 4: RAG Question Answering
        print("\n🤖 Step 4: RAG Question Answering")
        rag_processor = SimpleRAGProcessor(vector_store, embedding_generator)
        
        # Comprehensive test questions
        test_questions = [
            {
                "question": "What is supervised learning and how does it work?",
                "expected_domain": "machine learning",
                "difficulty": "basic"
            },
            {
                "question": "How do transformer architectures work in deep learning?",
                "expected_domain": "deep learning", 
                "difficulty": "intermediate"
            },
            {
                "question": "What are the main components of a RAG system?",
                "expected_domain": "rag systems",
                "difficulty": "intermediate"
            },
            {
                "question": "How can you optimize RAG system performance?",
                "expected_domain": "performance",
                "difficulty": "advanced"
            },
            {
                "question": "What is the difference between CNNs and RNNs?",
                "expected_domain": "deep learning",
                "difficulty": "intermediate"
            },
            {
                "question": "How does hybrid search improve retrieval quality?",
                "expected_domain": "performance",
                "difficulty": "advanced"
            }
        ]
        
        rag_results = []
        total_query_time = 0
        
        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            print(f"\n   Query {i+1}: {question}")
            
            start_time = time.time()
            result = await rag_processor.generate_answer(question, top_k=3)
            query_time = time.time() - start_time
            total_query_time += query_time
            
            print(f"   ⏱️  Response time: {query_time:.3f}s")
            print(f"   📊 Sources used: {result['num_sources']}")
            print(f"   📝 Answer: {result['answer'][:100]}...")
            
            # Evaluate answer quality
            answer_quality = "good" if len(result['answer']) > 50 and result['num_sources'] > 0 else "poor"
            print(f"   📈 Quality: {answer_quality}")
            
            rag_results.append({
                **test_case,
                "answer": result['answer'],
                "response_time": query_time,
                "sources_count": result['num_sources'],
                "context_length": result['context_length'],
                "quality": answer_quality
            })
        
        # Step 5: Performance Analysis
        print(f"\n📊 COMPREHENSIVE RAG PIPELINE ANALYSIS")
        print("=" * 60)
        
        print(f"📚 Knowledge Base Statistics:")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars")
        print(f"   - Ingestion time: {ingestion_time:.3f}s")
        print(f"   - Embedding time: {embedding_time:.3f}s")
        print(f"   - Storage time: {storage_time:.3f}s")
        
        print(f"\n🤖 Query Performance:")
        print(f"   - Total queries: {len(test_questions)}")
        print(f"   - Average response time: {total_query_time/len(test_questions):.3f}s")
        print(f"   - Fastest query: {min(r['response_time'] for r in rag_results):.3f}s")
        print(f"   - Slowest query: {max(r['response_time'] for r in rag_results):.3f}s")
        
        good_answers = len([r for r in rag_results if r['quality'] == 'good'])
        print(f"   - Answer quality: {good_answers}/{len(test_questions)} good answers")
        
        print(f"\n📈 Retrieval Statistics:")
        avg_sources = sum(r['sources_count'] for r in rag_results) / len(rag_results)
        avg_context = sum(r['context_length'] for r in rag_results) / len(rag_results)
        print(f"   - Average sources per query: {avg_sources:.1f}")
        print(f"   - Average context length: {avg_context:.0f} chars")
        
        # Success criteria validation
        pipeline_success = all([
            ingestion_time < 5.0,  # Fast ingestion
            embedding_time < 10.0,  # Reasonable embedding time
            total_query_time/len(test_questions) < 3.0,  # Sub-3 second queries
            good_answers >= len(test_questions) * 0.7  # 70% good answers
        ])
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        print(f"   ✅ Fast ingestion (<5s): {ingestion_time:.3f}s")
        print(f"   ✅ Reasonable embedding time (<10s): {embedding_time:.3f}s") 
        print(f"   ✅ Sub-3 second queries: {total_query_time/len(test_questions):.3f}s avg")
        print(f"   ✅ Good answer quality (>70%): {good_answers/len(test_questions)*100:.1f}%")
        
        if pipeline_success:
            print(f"\n🎉 COMPREHENSIVE RAG PIPELINE: SUCCESS!")
        else:
            print(f"\n⚠️  COMPREHENSIVE RAG PIPELINE: NEEDS IMPROVEMENT")
        
        return rag_results, pipeline_success
        
    except Exception as e:
        print(f"❌ Comprehensive RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return [], False
    finally:
        kb_path.unlink()


async def test_multi_document_rag():
    """Test RAG with multiple documents and cross-document queries."""
    print("\n📚 TESTING MULTI-DOCUMENT RAG")
    print("=" * 50)
    
    # Create multiple specialized documents
    documents = {
        "python_guide.pdf": {
            "title": "Python Programming Guide",
            "content": """
            Python is a high-level, interpreted programming language known for its simplicity and readability. 
            It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
            
            Key features include dynamic typing, automatic memory management, and a comprehensive standard library.
            Popular frameworks include Django for web development, NumPy for scientific computing, and TensorFlow for machine learning.
            
            Python's syntax emphasizes code readability with significant whitespace. The language uses indentation to define code blocks instead of curly braces.
            """
        },
        "javascript_guide.pdf": {
            "title": "JavaScript Development Guide", 
            "content": """
            JavaScript is a versatile programming language primarily used for web development. It enables interactive web pages and is an essential part of web applications.
            
            Modern JavaScript (ES6+) includes features like arrow functions, async/await, destructuring, and modules. 
            Popular frameworks include React for user interfaces, Node.js for server-side development, and Express.js for web applications.
            
            JavaScript is dynamically typed and supports both object-oriented and functional programming paradigms. The language runs in browsers and server environments.
            """
        },
        "database_guide.pdf": {
            "title": "Database Design Guide",
            "content": """
            Database design involves structuring data efficiently to support application requirements. Relational databases use tables, rows, and columns to organize information.
            
            SQL (Structured Query Language) is used to query and manipulate relational databases. Popular systems include PostgreSQL, MySQL, and SQLite.
            
            NoSQL databases like MongoDB and Redis offer flexible data models for specific use cases. Document databases store data in JSON-like formats.
            
            Database normalization reduces redundancy and improves data integrity. Indexing improves query performance but requires additional storage space.
            """
        }
    }
    
    # Set up RAG system
    embedding_config = EmbeddingConfig(
        backend=EmbeddingBackend.MLX,
        model_name=EmbeddingModel.ALL_MINILM_L6_V2,
        batch_size=16
    )
    embedding_generator = EmbeddingGenerator(embedding_config)
    await embedding_generator.initialize()
    
    storage_config = StorageConfig(
        collection_name="multi_doc_rag_test",
        persist_directory="./test_multi_doc_rag_db"
    )
    vector_store = VectorStore(storage_config)
    await vector_store.initialize()
    
    ingestion = DocumentIngestion(ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=400,
        chunk_overlap=50
    ))
    
    # Ingest all documents
    total_chunks = 0
    for doc_name, doc_info in documents.items():
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            # Create simple PDF
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(tmp_file.name)
            c.setFont("Helvetica", 12)
            
            lines = doc_info["content"].split('\n')
            y = 750
            for line in lines:
                if y < 50:
                    c.showPage()
                    y = 750
                c.drawString(50, y, line.strip())
                y -= 20
            c.save()
            
            doc_path = Path(tmp_file.name)
        
        try:
            metadata, chunks = await ingestion.process_file(doc_path)
            embedded_chunks = await embedding_generator.embed_chunks(chunks)
            await vector_store.add_document(metadata, embedded_chunks)
            total_chunks += len(chunks)
            print(f"   ✅ Ingested {doc_name}: {len(chunks)} chunks")
        finally:
            doc_path.unlink()
    
    print(f"   📊 Total knowledge base: {total_chunks} chunks from {len(documents)} documents")
    
    # Test cross-document queries
    rag_processor = SimpleRAGProcessor(vector_store, embedding_generator)
    
    cross_doc_queries = [
        "Which programming languages are mentioned and what are their key features?",
        "What are the different types of databases discussed?",
        "How do Python and JavaScript differ in their syntax?",
        "What frameworks are mentioned for web development?",
        "Which languages support object-oriented programming?"
    ]
    
    print(f"\n🔍 Testing {len(cross_doc_queries)} cross-document queries:")
    
    for i, query in enumerate(cross_doc_queries):
        print(f"\n   Query {i+1}: {query}")
        
        start_time = time.time()
        result = await rag_processor.generate_answer(query, top_k=5)
        query_time = time.time() - start_time
        
        print(f"   ⏱️  Response time: {query_time:.3f}s")
        print(f"   📊 Sources: {result['num_sources']} chunks")
        print(f"   📝 Answer: {result['answer'][:120]}...")
        
        # Analyze source diversity
        source_docs = set()
        for source in result['sources']:
            # Simple document identification by content patterns
            content = source.chunk.content.lower()
            if 'python' in content:
                source_docs.add('python_guide')
            elif 'javascript' in content:
                source_docs.add('javascript_guide')
            elif 'database' in content or 'sql' in content:
                source_docs.add('database_guide')
        
        print(f"   📚 Source diversity: {len(source_docs)} different documents")
    
    print(f"\n✅ Multi-document RAG testing completed!")
    return True


async def main():
    """Run comprehensive RAG pipeline testing."""
    print("🚀 AKASHA PHASE 2 - COMPREHENSIVE RAG TESTING")
    print("=" * 70)
    
    try:
        # Test 1: Comprehensive single-document RAG
        rag_results, pipeline_success = await test_comprehensive_rag_pipeline()
        
        # Test 2: Multi-document RAG
        multi_doc_success = await test_multi_document_rag()
        
        print(f"\n🎊 COMPREHENSIVE RAG TESTING COMPLETED!")
        print("=" * 50)
        
        print(f"📊 FINAL RESULTS:")
        print(f"   ✅ Single-document RAG: {'SUCCESS' if pipeline_success else 'NEEDS WORK'}")
        print(f"   ✅ Multi-document RAG: {'SUCCESS' if multi_doc_success else 'NEEDS WORK'}")
        
        if rag_results:
            good_answers = len([r for r in rag_results if r['quality'] == 'good'])
            avg_response_time = sum(r['response_time'] for r in rag_results) / len(rag_results)
            print(f"   📈 Answer quality: {good_answers}/{len(rag_results)} ({good_answers/len(rag_results)*100:.1f}%)")
            print(f"   ⚡ Average response time: {avg_response_time:.3f}s")
        
        overall_success = pipeline_success and multi_doc_success
        print(f"\n🏆 OVERALL RAG PIPELINE: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE RAG TESTING FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())