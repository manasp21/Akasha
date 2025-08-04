"""
Tests for RAG Pipeline system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import time
import asyncio

from src.rag.pipeline import (
    RAGPipeline, RAGPipelineConfig, QueryResult, IngestionResult, QueryMode
)
from src.rag.ingestion import DocumentChunk, DocumentMetadata, ChunkingConfig
from src.rag.embeddings import EmbeddingConfig
from src.rag.storage import StorageConfig
from src.rag.retrieval import RetrievalConfig, RetrievalResult, QueryContext
from src.llm.manager import LLMManager
from src.llm.provider import LLMResponse
from src.llm.templates import TemplateType
from src.core.exceptions import AkashaError


class TestRAGPipelineConfig:
    """Test RAGPipelineConfig model."""
    
    def test_default_config(self):
        """Test default pipeline configuration."""
        config = RAGPipelineConfig()
        
        assert isinstance(config.chunking_config, ChunkingConfig)
        assert isinstance(config.embedding_config, EmbeddingConfig)
        assert isinstance(config.storage_config, StorageConfig)
        assert isinstance(config.retrieval_config, RetrievalConfig)
        assert config.auto_embed_on_ingest is True
        assert config.max_concurrent_ingestions == 3
        assert config.query_timeout == 30.0
        assert config.ingestion_timeout == 300.0
    
    def test_optimized_config_creation(self):
        """Test creating optimized configuration."""
        config = RAGPipelineConfig.create_optimized_config()
        
        assert config.chunking_config.chunk_size == 1000
        assert config.chunking_config.chunk_overlap == 200
        assert config.embedding_config.model_name == "all-MiniLM-L6-v2"
        assert config.storage_config.backend == "chromadb"
        assert config.retrieval_config.strategy == "multi_stage"
        assert config.auto_embed_on_ingest is True
        assert config.cache_embeddings is True


class TestQueryResult:
    """Test QueryResult model."""
    
    def test_query_result_creation(self):
        """Test query result creation."""
        sources = [{"filename": "test.txt", "document_id": "doc1"}]
        
        result = QueryResult(
            query="Test query",
            response="Test response",
            sources=sources,
            processing_time=1.5,
            query_mode=QueryMode.SIMPLE
        )
        
        assert result.query == "Test query"
        assert result.response == "Test response"
        assert result.sources == sources
        assert result.processing_time == 1.5
        assert result.query_mode == QueryMode.SIMPLE


class TestIngestionResult:
    """Test IngestionResult model."""
    
    def test_ingestion_result_success(self):
        """Test successful ingestion result."""
        metadata = DocumentMetadata(
            file_path="test.txt",
            file_name="test.txt",
            file_size=1000,
            file_hash="abc123",
            mime_type="text/plain",
            format="text",
            processed_at=time.time(),
            chunk_count=5,
            processing_time=2.0
        )
        
        result = IngestionResult(
            document_id="doc1",
            file_path="test.txt",
            chunks_created=5,
            processing_time=2.0,
            metadata=metadata,
            success=True
        )
        
        assert result.document_id == "doc1"
        assert result.chunks_created == 5
        assert result.success is True
        assert result.error is None
    
    def test_ingestion_result_failure(self):
        """Test failed ingestion result."""
        metadata = DocumentMetadata(
            file_path="test.txt",
            file_name="test.txt",
            file_size=0,
            file_hash="",
            mime_type="",
            format="text",
            processed_at=time.time(),
            chunk_count=0,
            processing_time=0.0
        )
        
        result = IngestionResult(
            document_id="error",
            file_path="test.txt",
            chunks_created=0,
            processing_time=0.0,
            metadata=metadata,
            success=False,
            error="File not found"
        )
        
        assert result.success is False
        assert result.error == "File not found"
        assert result.chunks_created == 0


class TestRAGPipeline:
    """Test RAGPipeline main class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGPipelineConfig()
        self.llm_manager = Mock(spec=LLMManager)
        self.pipeline = RAGPipeline(self.config, self.llm_manager)
    
    def test_pipeline_initialization(self):
        """Test pipeline component initialization."""
        assert self.pipeline.config == self.config
        assert self.pipeline.llm_manager == self.llm_manager
        assert self.pipeline.initialized is False
        
        # Check component creation
        assert hasattr(self.pipeline, 'document_ingestion')
        assert hasattr(self.pipeline, 'embedding_generator')
        assert hasattr(self.pipeline, 'vector_store')
        assert hasattr(self.pipeline, 'document_retriever')
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test pipeline initialization."""
        # Mock all component initializations
        with patch.object(self.pipeline.embedding_generator, 'initialize') as mock_embed_init, \
             patch.object(self.pipeline.vector_store, 'initialize') as mock_store_init, \
             patch.object(self.pipeline.document_retriever, 'initialize') as mock_retriever_init, \
             patch.object(self.pipeline.llm_manager, 'initialize') as mock_llm_init:
            
            await self.pipeline.initialize()
            
            # Verify all components were initialized
            mock_embed_init.assert_called_once()
            mock_store_init.assert_called_once()
            mock_retriever_init.assert_called_once()
            mock_llm_init.assert_called_once()
            
            assert self.pipeline.initialized is True
    
    @pytest.mark.asyncio
    async def test_ingest_document_success(self):
        """Test successful document ingestion."""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content for ingestion.")
            f.flush()
            file_path = Path(f.name)
        
        try:
            # Mock components
            mock_metadata = DocumentMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=100,
                file_hash="abc123",
                mime_type="text/plain",
                format="text",
                processed_at=time.time(),
                chunk_count=1,
                processing_time=1.0
            )
            
            mock_chunks = [
                DocumentChunk(
                    id="chunk1",
                    content="Test document content for ingestion.",
                    document_id="doc1",
                    chunk_index=0,
                    embedding=[0.1, 0.2, 0.3]
                )
            ]
            
            with patch.object(self.pipeline.document_ingestion, 'process_file', return_value=(mock_metadata, mock_chunks)) as mock_process, \
                 patch.object(self.pipeline.embedding_generator, 'embed_chunks', return_value=mock_chunks) as mock_embed, \
                 patch.object(self.pipeline.vector_store, 'add_document') as mock_add:
                
                self.pipeline.initialized = True
                
                result = await self.pipeline.ingest_document(file_path)
                
                # Verify calls
                mock_process.assert_called_once_with(file_path)
                mock_embed.assert_called_once_with(mock_chunks)
                mock_add.assert_called_once_with(mock_metadata, mock_chunks)
                
                # Verify result
                assert isinstance(result, IngestionResult)
                assert result.success is True
                assert result.chunks_created == 1
                assert result.document_id == "doc1"
                assert result.error is None
                
        finally:
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_ingest_document_failure(self):
        """Test document ingestion failure."""
        file_path = Path("nonexistent_file.txt")
        
        with patch.object(self.pipeline.document_ingestion, 'process_file', side_effect=AkashaError("File not found")):
            self.pipeline.initialized = True
            
            result = await self.pipeline.ingest_document(file_path)
            
            assert isinstance(result, IngestionResult)
            assert result.success is False
            assert result.error == "Failed to ingest document nonexistent_file.txt: File not found"
            assert result.chunks_created == 0
    
    @pytest.mark.asyncio
    async def test_ingest_directory(self):
        """Test directory ingestion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.txt").write_text("First test document.")
            (temp_path / "test2.txt").write_text("Second test document.")
            (temp_path / "ignore.log").write_text("Log file to ignore.")
            
            # Mock successful ingestion results
            mock_result1 = IngestionResult(
                document_id="doc1",
                file_path=str(temp_path / "test1.txt"),
                chunks_created=1,
                processing_time=1.0,
                metadata=Mock(),
                success=True
            )
            
            mock_result2 = IngestionResult(
                document_id="doc2", 
                file_path=str(temp_path / "test2.txt"),
                chunks_created=1,
                processing_time=1.0,
                metadata=Mock(),
                success=True
            )
            
            with patch.object(self.pipeline, 'ingest_document', side_effect=[mock_result1, mock_result2]) as mock_ingest:
                self.pipeline.initialized = True
                
                results = await self.pipeline.ingest_directory(
                    temp_path,
                    recursive=False,
                    file_patterns=["*.txt"]
                )
                
                # Should process 2 .txt files
                assert len(results) == 2
                assert all(result.success for result in results)
                
                # Verify ingest_document was called for each file
                assert mock_ingest.call_count == 2
    
    @pytest.mark.asyncio
    async def test_query_simple(self):
        """Test simple query processing."""
        query = "What is machine learning?"
        
        # Mock retrieval result
        mock_chunks = [
            DocumentChunk(
                id="chunk1",
                content="Machine learning is a subset of AI.",
                document_id="doc1",
                chunk_index=0,
                metadata={"file_name": "ml_guide.txt"}
            )
        ]
        
        mock_retrieval_result = RetrievalResult(
            chunks=mock_chunks,
            scores=[0.8],
            total_score=0.8,
            retrieval_method="vector_similarity",
            processing_time=0.5,
            query_context=QueryContext(
                original_query=query,
                processed_query=query
            )
        )
        
        # Mock LLM response
        mock_llm_response = LLMResponse(
            content="Machine learning is a method of data analysis that automates analytical model building.",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            generation_time=1.0,
            model_name="test-model",
            metadata={"provider_name": "test_provider"}
        )
        
        with patch.object(self.pipeline.document_retriever, 'retrieve', return_value=mock_retrieval_result) as mock_retrieve, \
             patch.object(self.pipeline.llm_manager, 'generate_rag_response', return_value=mock_llm_response) as mock_generate:
            
            self.pipeline.initialized = True
            
            result = await self.pipeline.query(query, mode=QueryMode.SIMPLE)
            
            # Verify calls
            mock_retrieve.assert_called_once_with(
                query=query,
                top_k=self.config.retrieval_config.final_top_k,
                filters=None,
                context={}
            )
            
            mock_generate.assert_called_once_with(
                query=query,
                retrieval_result=mock_retrieval_result,
                template_type=TemplateType.RAG_QA,
                provider_name=None
            )
            
            # Verify result
            assert isinstance(result, QueryResult)
            assert result.query == query
            assert result.response == mock_llm_response.content
            assert result.query_mode == QueryMode.SIMPLE
            assert len(result.sources) == 1
            assert result.sources[0]["filename"] == "ml_guide.txt"
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_query_analytical(self):
        """Test analytical query processing."""
        query = "Analyze the differences between supervised and unsupervised learning."
        
        mock_retrieval_result = Mock(spec=RetrievalResult)
        mock_retrieval_result.chunks = []
        mock_retrieval_result.retrieval_method = "multi_stage"
        
        mock_llm_response = Mock(spec=LLMResponse)
        mock_llm_response.content = "Analysis of learning types..."
        mock_llm_response.metadata = {"provider_name": "test"}
        mock_llm_response.total_tokens = 100
        mock_llm_response.completion_tokens = 50
        
        with patch.object(self.pipeline.document_retriever, 'retrieve', return_value=mock_retrieval_result), \
             patch.object(self.pipeline.llm_manager, 'generate_rag_response', return_value=mock_llm_response) as mock_generate:
            
            self.pipeline.initialized = True
            
            result = await self.pipeline.query(query, mode=QueryMode.ANALYTICAL)
            
            # Should use analytical template type
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args.kwargs["template_type"] == TemplateType.RAG_ANALYSIS
    
    @pytest.mark.asyncio
    async def test_query_timeout(self):
        """Test query timeout handling."""
        query = "Test query"
        
        # Mock slow retrieval
        async def slow_retrieve(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return Mock()
        
        with patch.object(self.pipeline.document_retriever, 'retrieve', side_effect=slow_retrieve):
            self.pipeline.initialized = True
            self.pipeline.config.query_timeout = 0.1  # Very short timeout
            
            with pytest.raises(AkashaError, match="Query timeout"):
                await self.pipeline.query(query)
    
    @pytest.mark.asyncio
    async def test_query_stream(self):
        """Test streaming query processing."""
        query = "What is deep learning?"
        
        # Mock retrieval result
        mock_chunks = [
            DocumentChunk(
                id="chunk1",
                content="Deep learning uses neural networks.",
                document_id="doc1",
                chunk_index=0,
                metadata={"file_name": "dl_guide.txt"}
            )
        ]
        
        mock_retrieval_result = RetrievalResult(
            chunks=mock_chunks,
            scores=[0.9],
            total_score=0.9,
            retrieval_method="hybrid",
            processing_time=0.3,
            query_context=QueryContext(
                original_query=query,
                processed_query=query
            )
        )
        
        # Mock streaming events
        from src.llm.provider import StreamingEvent
        
        async def mock_stream_generator(*args, **kwargs):
            yield StreamingEvent(type="token", content="Deep")
            yield StreamingEvent(type="token", content=" learning") 
            yield StreamingEvent(type="finish", content="", metadata={"total_tokens": 50})
        
        with patch.object(self.pipeline.document_retriever, 'retrieve', return_value=mock_retrieval_result), \
             patch.object(self.pipeline.llm_manager, 'generate_rag_stream', return_value=mock_stream_generator()):
            
            self.pipeline.initialized = True
            
            events = []
            async for event in self.pipeline.query_stream(query):
                events.append(event)
            
            # Should receive status, retrieval_complete, streaming events, and final summary
            assert len(events) >= 4
            
            # Check event types
            event_types = [event.get("type") if isinstance(event, dict) else event.type for event in events]
            assert "status" in event_types
            assert "retrieval_complete" in event_types
    
    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test document deletion."""
        document_id = "doc1"
        
        with patch.object(self.pipeline.vector_store, 'delete_document', return_value=5) as mock_delete:
            self.pipeline.initialized = True
            
            result = await self.pipeline.delete_document(document_id)
            
            mock_delete.assert_called_once_with(document_id)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self):
        """Test deleting non-existent document."""
        document_id = "nonexistent"
        
        with patch.object(self.pipeline.vector_store, 'delete_document', return_value=0):
            self.pipeline.initialized = True
            
            result = await self.pipeline.delete_document(document_id)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_pipeline_stats(self):
        """Test getting pipeline statistics."""
        mock_vector_stats = {"total_chunks": 100, "backend": "chromadb"}
        mock_retrieval_stats = {"strategy": "multi_stage"}
        mock_llm_stats = {"primary_provider": "test_provider"}
        mock_embedding_info = {"model_name": "test_model", "dimensions": 384}
        
        with patch.object(self.pipeline.vector_store, 'get_stats', return_value=mock_vector_stats), \
             patch.object(self.pipeline.document_retriever, 'get_retrieval_stats', return_value=mock_retrieval_stats), \
             patch.object(self.pipeline.llm_manager, 'get_provider_stats', return_value=mock_llm_stats), \
             patch.object(self.pipeline.embedding_generator, 'get_embedding_info', return_value=mock_embedding_info):
            
            self.pipeline.initialized = True
            
            stats = await self.pipeline.get_pipeline_stats()
            
            assert stats["initialized"] is True
            assert stats["vector_store"] == mock_vector_stats
            assert stats["retrieval"] == mock_retrieval_stats
            assert stats["llm_providers"] == mock_llm_stats
            assert stats["embeddings"] == mock_embedding_info
            assert "config" in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test pipeline health check."""
        mock_llm_health = {
            "initialized": True,
            "healthy_providers": ["provider1"],
            "providers": {"provider1": {"status": "healthy"}}
        }
        
        mock_vector_stats = {"backend": "chromadb", "total_chunks": 50}
        mock_embedding_info = {"model_name": "test_model", "dimensions": 384}
        
        with patch.object(self.pipeline.llm_manager, 'health_check', return_value=mock_llm_health), \
             patch.object(self.pipeline.vector_store, 'get_stats', return_value=mock_vector_stats), \
             patch.object(self.pipeline.embedding_generator, 'get_embedding_info', return_value=mock_embedding_info):
            
            self.pipeline.initialized = True
            
            health = await self.pipeline.health_check()
            
            assert health["pipeline_initialized"] is True
            assert health["overall_healthy"] is True
            assert health["llm"] == mock_llm_health
            assert health["vector_store"]["healthy"] is True
            assert health["embeddings"]["healthy"] is True
    
    def test_extract_sources(self):
        """Test source extraction from chunks."""
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Content 1",
                document_id="doc1",
                chunk_index=0,
                metadata={
                    "file_name": "document1.txt",
                    "file_path": "/path/to/document1.txt",
                    "file_size": 1000,
                    "format": "text",
                    "custom_author": "John Doe"
                }
            ),
            DocumentChunk(
                id="chunk2",
                content="Content 2",
                document_id="doc1",  # Same document
                chunk_index=1,
                metadata={
                    "file_name": "document1.txt",
                    "file_path": "/path/to/document1.txt",
                    "file_size": 1000,
                    "format": "text"
                }
            ),
            DocumentChunk(
                id="chunk3",
                content="Content 3",
                document_id="doc2",  # Different document
                chunk_index=0,
                metadata={
                    "file_name": "document2.pdf",
                    "file_path": "/path/to/document2.pdf",
                    "file_size": 2000,
                    "format": "pdf"
                }
            )
        ]
        
        sources = self.pipeline._extract_sources(chunks)
        
        # Should deduplicate by file_name
        assert len(sources) == 2
        
        # Check first source
        source1 = next(s for s in sources if s["filename"] == "document1.txt")
        assert source1["document_id"] == "doc1"
        assert source1["file_path"] == "/path/to/document1.txt"
        assert source1["format"] == "text"
        assert source1["author"] == "John Doe"  # custom_ prefix removed
        
        # Check second source
        source2 = next(s for s in sources if s["filename"] == "document2.pdf")
        assert source2["document_id"] == "doc2"
        assert source2["format"] == "pdf"


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for RAG Pipeline."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = RAGPipelineConfig()
        self.llm_manager = Mock(spec=LLMManager)
        self.pipeline = RAGPipeline(self.config, self.llm_manager)
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_processing(self):
        """Test complete end-to-end document processing."""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """
            Introduction to Machine Learning
            
            Machine learning is a subset of artificial intelligence (AI) that provides 
            systems the ability to automatically learn and improve from experience 
            without being explicitly programmed.
            
            Types of Machine Learning:
            1. Supervised Learning
            2. Unsupervised Learning  
            3. Reinforcement Learning
            
            Applications include image recognition, natural language processing,
            and recommendation systems.
            """
            f.write(test_content)
            f.flush()
            file_path = Path(f.name)
        
        try:
            # Mock all components for integration test
            with patch.object(self.pipeline, 'initialize') as mock_init, \
                 patch.object(self.pipeline.document_ingestion, 'process_file') as mock_process, \
                 patch.object(self.pipeline.embedding_generator, 'embed_chunks') as mock_embed, \
                 patch.object(self.pipeline.vector_store, 'add_document') as mock_add, \
                 patch.object(self.pipeline.document_retriever, 'retrieve') as mock_retrieve, \
                 patch.object(self.pipeline.llm_manager, 'generate_rag_response') as mock_generate:
                
                # Configure mocks
                mock_chunks = [
                    DocumentChunk(
                        id="chunk1",
                        content="Machine learning is a subset of artificial intelligence...",
                        document_id="doc1",
                        chunk_index=0,
                        embedding=[0.1] * 384
                    ),
                    DocumentChunk(
                        id="chunk2",
                        content="Types of Machine Learning: 1. Supervised Learning...",
                        document_id="doc1", 
                        chunk_index=1,
                        embedding=[0.2] * 384
                    )
                ]
                
                mock_metadata = DocumentMetadata(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_size=len(test_content),
                    file_hash="test_hash",
                    mime_type="text/plain",
                    format="text",
                    processed_at=time.time(),
                    chunk_count=2,
                    processing_time=1.0
                )
                
                mock_process.return_value = (mock_metadata, mock_chunks)
                mock_embed.return_value = mock_chunks
                
                mock_retrieval_result = RetrievalResult(
                    chunks=mock_chunks,
                    scores=[0.8, 0.7],
                    total_score=1.5,
                    retrieval_method="multi_stage",
                    processing_time=0.5,
                    query_context=QueryContext(
                        original_query="What is machine learning?",
                        processed_query="What is machine learning?"
                    )
                )
                mock_retrieve.return_value = mock_retrieval_result
                
                mock_llm_response = LLMResponse(
                    content="Machine learning is a subset of AI that enables systems to learn automatically.",
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    generation_time=1.5,
                    model_name="test-model",
                    metadata={"provider_name": "test_provider"}
                )
                mock_generate.return_value = mock_llm_response
                
                # Test complete workflow
                
                # 1. Initialize pipeline
                await self.pipeline.initialize()
                mock_init.assert_called_once()
                
                # 2. Ingest document
                ingestion_result = await self.pipeline.ingest_document(file_path)
                
                assert ingestion_result.success is True
                assert ingestion_result.chunks_created == 2
                mock_process.assert_called_once_with(file_path)
                mock_embed.assert_called_once()
                mock_add.assert_called_once()
                
                # 3. Query the system
                query_result = await self.pipeline.query("What is machine learning?")
                
                assert isinstance(query_result, QueryResult)
                assert query_result.query == "What is machine learning?"
                assert query_result.response == mock_llm_response.content
                assert len(query_result.sources) > 0
                assert query_result.processing_time > 0
                
                mock_retrieve.assert_called_once()
                mock_generate.assert_called_once()
                
        finally:
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent ingestion and querying."""
        # Create multiple test files
        test_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i in range(3):
                file_path = temp_path / f"test_doc_{i}.txt"
                file_path.write_text(f"Test document {i} content about various topics.")
                test_files.append(file_path)
            
            # Mock successful ingestion
            async def mock_ingest_single(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                return IngestionResult(
                    document_id=f"doc_{len(args)}",
                    file_path=str(args[0]),
                    chunks_created=1,
                    processing_time=0.1,
                    metadata=Mock(),
                    success=True
                )
            
            with patch.object(self.pipeline, 'ingest_document', side_effect=mock_ingest_single) as mock_ingest:
                self.pipeline.initialized = True
                
                # Test concurrent ingestion
                start_time = time.time()
                
                ingestion_tasks = [
                    self.pipeline.ingest_document(file_path) 
                    for file_path in test_files
                ]
                
                results = await asyncio.gather(*ingestion_tasks)
                
                end_time = time.time()
                
                # Should complete faster than sequential processing
                assert end_time - start_time < 0.5  # Less than 3 * 0.1 + overhead
                
                # All ingestions should succeed
                assert len(results) == 3
                assert all(result.success for result in results)
                assert mock_ingest.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])