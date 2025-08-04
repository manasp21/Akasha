#!/usr/bin/env python3
"""
Phase 2 PDF Processing Testing.

Tests PDF document processing with the current PyPDF2 implementation.
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


def create_test_pdf(filename: str, content: str, pages: int = 1) -> Path:
    """Create a test PDF file with specified content."""
    file_path = Path(filename)
    
    c = canvas.Canvas(str(file_path), pagesize=letter)
    width, height = letter
    
    # Split content into pages
    words = content.split()
    words_per_page = len(words) // pages
    
    for page in range(pages):
        # Add text content
        text_obj = c.beginText(50, height - 50)
        text_obj.setFont("Helvetica", 12)
        
        start_idx = page * words_per_page
        end_idx = start_idx + words_per_page if page < pages - 1 else len(words)
        page_content = ' '.join(words[start_idx:end_idx])
        
        # Wrap text to fit page
        lines = []
        current_line = ""
        for word in page_content.split():
            if len(current_line + word) < 80:  # Approximate line length
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        for line in lines[:40]:  # Max 40 lines per page
            text_obj.textLine(line)
        
        c.drawText(text_obj)
        c.showPage()
    
    c.save()
    return file_path


async def test_pdf_ingestion():
    """Test PDF document ingestion with current implementation."""
    print("📄 TESTING PDF DOCUMENT INGESTION")
    print("=" * 50)
    
    # Create test PDFs
    test_pdfs = [
        {
            "name": "simple_research.pdf",
            "content": """
            Artificial Intelligence: A Comprehensive Overview
            
            Introduction
            Artificial intelligence (AI) represents one of the most significant technological 
            advancements of the 21st century. This field encompasses machine learning, 
            natural language processing, computer vision, and robotics.
            
            Machine Learning Fundamentals
            Machine learning is a subset of AI that enables systems to automatically learn 
            and improve from experience without being explicitly programmed. Key approaches 
            include supervised learning, unsupervised learning, and reinforcement learning.
            
            Deep Learning Revolution
            Deep learning, utilizing neural networks with multiple layers, has revolutionized 
            AI applications. Convolutional neural networks excel in image recognition, 
            while recurrent neural networks process sequential data effectively.
            
            Applications and Impact
            AI applications span healthcare, finance, transportation, and entertainment. 
            From medical diagnosis to autonomous vehicles, AI systems are transforming 
            industries and improving human capabilities.
            
            Future Directions
            The future of AI includes developments in quantum computing, neuromorphic 
            computing, and artificial general intelligence. Ethical considerations and 
            responsible AI development remain paramount.
            """,
            "pages": 1
        },
        {
            "name": "multi_page_technical.pdf",
            "content": """
            Technical Manual: Advanced RAG Systems
            
            Chapter 1: System Architecture
            Retrieval-Augmented Generation (RAG) systems combine the power of large language 
            models with external knowledge retrieval. The architecture consists of three main 
            components: document ingestion, vector storage, and generation pipeline.
            
            Document ingestion involves parsing various file formats, chunking content into 
            manageable pieces, and extracting meaningful metadata. This process ensures that 
            information is properly structured for efficient retrieval.
            
            Chapter 2: Embedding Systems
            Modern embedding systems utilize transformer-based models to convert text into 
            high-dimensional vector representations. These embeddings capture semantic meaning 
            and enable similarity-based retrieval across large document collections.
            
            Popular embedding models include BERT, RoBERTa, and more recent developments like 
            sentence-transformers. The choice of embedding model significantly impacts retrieval 
            quality and system performance.
            
            Chapter 3: Vector Databases
            Vector databases like ChromaDB, Pinecone, and Qdrant provide efficient storage and 
            retrieval of high-dimensional embeddings. These systems support approximate nearest 
            neighbor search with sub-linear time complexity.
            
            Indexing strategies, distance metrics, and query optimization play crucial roles 
            in vector database performance. Proper configuration ensures fast and accurate 
            retrieval even with millions of documents.
            
            Chapter 4: Retrieval Strategies
            Advanced retrieval strategies include multi-stage retrieval, query expansion, 
            and result reranking. These techniques improve retrieval precision and recall 
            by considering multiple relevance signals.
            
            Hybrid search combines dense vector search with sparse keyword matching, providing 
            robust retrieval across different query types and document characteristics.
            """,
            "pages": 3
        }
    ]
    
    # Set up ingestion pipeline
    ingestion = DocumentIngestion(ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        chunk_overlap=50
    ))
    
    pdf_results = []
    
    for pdf_config in test_pdfs:
        print(f"\n📖 Testing PDF: {pdf_config['name']}")
        print(f"   Pages: {pdf_config['pages']}")
        print(f"   Content length: {len(pdf_config['content'])} chars")
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = create_test_pdf(tmp_file.name, pdf_config['content'], pdf_config['pages'])
        
        try:
            # Test PDF processing
            start_time = time.time()
            metadata, chunks = await ingestion.process_file(pdf_path)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Processed successfully!")
            print(f"   📊 Chunks generated: {len(chunks)}")
            print(f"   ⏱️  Processing time: {processing_time:.3f}s")
            print(f"   📏 Avg chunk length: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars")
            print(f"   🏷️  Document ID: {metadata.document_id}")
            print(f"   📁 File format detected: {metadata.format}")
            print(f"   📄 MIME type: {metadata.mime_type}")
            
            # Show sample chunks
            print(f"   📋 Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk.content[:100].replace('\n', ' ') + "..."
                print(f"      {i+1}. {preview}")
            
            pdf_results.append({
                'name': pdf_config['name'],
                'pages': pdf_config['pages'],
                'chunks': len(chunks),
                'processing_time': processing_time,
                'metadata': metadata
            })
            
        except Exception as e:
            print(f"   ❌ PDF processing failed: {e}")
            pdf_results.append({
                'name': pdf_config['name'],
                'error': str(e)
            })
        finally:
            # Clean up
            pdf_path.unlink()
    
    print(f"\n📊 PDF PROCESSING SUMMARY:")
    print(f"{'PDF Name':<25} {'Pages':<7} {'Chunks':<8} {'Time (s)':<10} {'Status':<10}")
    print("-" * 70)
    
    for result in pdf_results:
        if 'error' in result:
            print(f"{result['name']:<25} {'N/A':<7} {'N/A':<8} {'N/A':<10} {'FAILED':<10}")
        else:
            print(f"{result['name']:<25} {result['pages']:<7} {result['chunks']:<8} {result['processing_time']:<10.3f} {'SUCCESS':<10}")
    
    return pdf_results


async def test_pdf_to_rag_pipeline():
    """Test end-to-end PDF to RAG pipeline."""
    print("\n🔄 TESTING PDF TO RAG PIPELINE")
    print("=" * 50)
    
    # Create a research paper PDF
    research_content = """
    Large Language Models and Retrieval-Augmented Generation
    
    Abstract
    This paper presents a comprehensive analysis of Large Language Models (LLMs) and their 
    integration with Retrieval-Augmented Generation (RAG) systems. We explore the challenges 
    of knowledge cutoffs, hallucination, and the benefits of external knowledge integration.
    
    Introduction
    Large Language Models have revolutionized natural language processing through their ability 
    to understand and generate human-like text. However, these models face limitations including 
    knowledge cutoffs, potential hallucinations, and inability to access real-time information.
    
    Retrieval-Augmented Generation addresses these limitations by combining the generative 
    capabilities of LLMs with external knowledge retrieval. This approach enables models to 
    access up-to-date information and reduce hallucination through grounded generation.
    
    Methodology
    Our RAG system consists of three main components: document ingestion, vector-based retrieval, 
    and contextualized generation. Document ingestion processes various file formats and creates 
    semantic embeddings. Vector retrieval identifies relevant context for user queries.
    
    The generation component uses retrieved context to produce accurate, grounded responses. 
    This approach significantly improves factual accuracy while maintaining the fluency of 
    large language models.
    
    Results
    Experimental results demonstrate substantial improvements in factual accuracy and relevance. 
    RAG systems showed 35% reduction in hallucination rates and 42% improvement in answer quality 
    compared to standalone language models.
    
    Conclusion
    Integration of retrieval mechanisms with large language models represents a significant 
    advancement in AI systems. Future work will focus on improving retrieval precision and 
    developing more sophisticated integration techniques.
    """
    
    # Create PDF and process through full pipeline
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        pdf_path = create_test_pdf(tmp_file.name, research_content, pages=2)
    
    try:
        # Step 1: Document Ingestion
        print("📥 Step 1: Document Ingestion")
        ingestion = DocumentIngestion(ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=400,
            chunk_overlap=50
        ))
        
        metadata, chunks = await ingestion.process_file(pdf_path)
        print(f"   ✅ Ingested {len(chunks)} chunks from PDF")
        
        # Step 2: Embedding Generation
        print("🧠 Step 2: Embedding Generation")
        embedding_config = EmbeddingConfig(
            backend=EmbeddingBackend.MLX,
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
            batch_size=16
        )
        embedding_generator = EmbeddingGenerator(embedding_config)
        await embedding_generator.initialize()
        
        embedded_chunks = await embedding_generator.embed_chunks(chunks)
        print(f"   ✅ Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Step 3: Vector Storage
        print("🗄️  Step 3: Vector Storage")
        storage_config = StorageConfig(
            collection_name="pdf_rag_test",
            persist_directory="./test_pdf_rag_db"
        )
        vector_store = VectorStore(storage_config)
        await vector_store.initialize()
        
        await vector_store.add_document(metadata, embedded_chunks)
        print(f"   ✅ Stored document in vector database")
        
        # Step 4: Test Retrieval
        print("🔍 Step 4: Retrieval Testing")
        test_queries = [
            "What are the main limitations of large language models?",
            "How does RAG improve factual accuracy?",
            "What were the experimental results?",
            "What is the methodology used in this research?"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n   Query {i+1}: {query}")
            
            # Generate query embedding
            query_embedding = await embedding_generator.embed_text(query)
            
            # Search vector store
            results = await vector_store.search_similar(query_embedding, top_k=3)
            
            print(f"   📊 Found {len(results)} relevant chunks:")
            for j, result in enumerate(results):
                preview = result.chunk.content[:80].replace('\n', ' ') + "..."
                print(f"      {j+1}. Score: {result.score:.3f} | {preview}")
        
        print(f"\n✅ END-TO-END PDF RAG PIPELINE SUCCESSFUL!")
        
    except Exception as e:
        print(f"❌ PDF RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pdf_path.unlink()


async def main():
    """Run comprehensive PDF testing."""
    print("🚀 AKASHA PHASE 2 - PDF PROCESSING TESTING")
    print("=" * 70)
    
    try:
        # Test 1: PDF Ingestion
        pdf_results = await test_pdf_ingestion()
        
        # Test 2: End-to-end PDF RAG pipeline
        await test_pdf_to_rag_pipeline()
        
        print("\n🎉 ALL PDF TESTS COMPLETED!")
        
        # Summary
        successful_pdfs = [r for r in pdf_results if 'error' not in r]
        failed_pdfs = [r for r in pdf_results if 'error' in r]
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   ✅ Successful PDFs: {len(successful_pdfs)}")
        print(f"   ❌ Failed PDFs: {len(failed_pdfs)}")
        print(f"   📄 Total chunks generated: {sum(r.get('chunks', 0) for r in successful_pdfs)}")
        print(f"   ⏱️  Average processing time: {sum(r.get('processing_time', 0) for r in successful_pdfs) / max(len(successful_pdfs), 1):.3f}s")
        
        if failed_pdfs:
            print(f"\n❌ Failed PDF processing errors:")
            for result in failed_pdfs:
                print(f"   - {result['name']}: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ PDF TESTING FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())