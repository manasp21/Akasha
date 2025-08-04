"""
Tests for RAG vector storage system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json
import time

from src.rag.storage import (
    VectorStore, StorageConfig, VectorStorageBackend, DistanceMetric,
    ChromaDBProvider, SearchResult
)
from src.rag.ingestion import DocumentChunk, DocumentMetadata
from src.core.exceptions import AkashaError


class TestStorageConfig:
    """Test StorageConfig model."""
    
    def test_default_config(self):
        """Test default storage configuration."""
        config = StorageConfig()
        
        assert config.backend == VectorStorageBackend.CHROMADB
        assert config.collection_name == "akasha_documents"
        assert config.distance_metric == DistanceMetric.COSINE
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.embedding_dimensions == 384
        assert config.max_batch_size == 1000
        assert config.enable_hnsw is True


class TestSearchResult:
    """Test SearchResult model."""
    
    def test_search_result_creation(self):
        """Test search result creation."""
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content",
            document_id="doc1",
            chunk_index=0
        )
        
        result = SearchResult(
            chunk=chunk,
            score=0.85,
            distance=0.15,
            metadata={"source": "test"}
        )
        
        assert result.chunk == chunk
        assert result.score == 0.85
        assert result.distance == 0.15
        assert result.metadata["source"] == "test"


class TestChromaDBProvider:
    """Test ChromaDBProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StorageConfig(
            backend=VectorStorageBackend.CHROMADB,
            collection_name="test_collection",
            embedding_dimensions=384
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config.persist_directory = temp_dir
            self.provider = ChromaDBProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_chromadb_not_available(self):
        """Test error when ChromaDB not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'chromadb'")):
            with pytest.raises(AkashaError, match="chromadb not installed"):
                await self.provider._import_chromadb()
    
    @pytest.mark.asyncio
    async def test_initialize_without_persistence(self):
        """Test initialization without persistence directory."""
        config = StorageConfig(persist_directory=None)
        provider = ChromaDBProvider(config)
        
        mock_chromadb = Mock()
        mock_settings = Mock()
        mock_client = Mock()
        mock_collection = Mock()
        
        mock_chromadb.Client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        provider._chromadb = mock_chromadb
        provider._settings = mock_settings
        
        await provider.initialize()
        
        assert provider.initialized is True
        assert provider.client is mock_client
        assert provider.collection is mock_collection
    
    @pytest.mark.asyncio
    async def test_initialize_with_persistence(self):
        """Test initialization with persistence directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(persist_directory=temp_dir)
            provider = ChromaDBProvider(config)
            
            mock_chromadb = Mock()
            mock_settings = Mock()
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_chromadb.Client.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            provider._chromadb = mock_chromadb
            provider._settings = mock_settings
            
            await provider.initialize()
            
            assert provider.initialized is True
            # Should create persistent settings
            mock_settings.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_chunks(self):
        """Test adding chunks to ChromaDB."""
        # Create test chunks with embeddings
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="First test content",
                document_id="doc1",
                chunk_index=0,
                start_offset=0,
                end_offset=20,
                embedding=[0.1, 0.2, 0.3]
            ),
            DocumentChunk(
                id="chunk2",
                content="Second test content", 
                document_id="doc1",
                chunk_index=1,
                start_offset=20,
                end_offset=40,
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        # Mock ChromaDB collection
        mock_collection = Mock()
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        await self.provider.add_chunks(chunks)
        
        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        
        # Verify the data passed to ChromaDB
        call_args = mock_collection.add.call_args
        assert len(call_args.kwargs["ids"]) == 2
        assert len(call_args.kwargs["embeddings"]) == 2
        assert len(call_args.kwargs["documents"]) == 2
        assert len(call_args.kwargs["metadatas"]) == 2
        
        # Check specific values
        assert call_args.kwargs["ids"] == ["chunk1", "chunk2"]
        assert call_args.kwargs["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert call_args.kwargs["documents"] == ["First test content", "Second test content"]
    
    @pytest.mark.asyncio
    async def test_add_chunks_without_embeddings(self):
        """Test error when adding chunks without embeddings."""
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Test content",
                document_id="doc1", 
                chunk_index=0
                # No embedding
            )
        ]
        
        self.provider.initialized = True
        
        with pytest.raises(AkashaError, match="has no embedding"):
            await self.provider.add_chunks(chunks)
    
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Test similarity search."""
        query_embedding = [0.1, 0.2, 0.3]
        
        # Mock ChromaDB query response
        mock_response = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["First content", "Second content"]],
            "metadatas": [[
                {"document_id": "doc1", "chunk_index": 0, "start_offset": 0, "end_offset": 20},
                {"document_id": "doc1", "chunk_index": 1, "start_offset": 20, "end_offset": 40}
            ]],
            "distances": [[0.1, 0.2]]
        }
        
        mock_collection = Mock()
        mock_collection.query.return_value = mock_response
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        results = await self.provider.search_similar(query_embedding, top_k=2)
        
        # Verify query was called correctly
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args.kwargs["query_embeddings"] == [query_embedding]
        assert call_args.kwargs["n_results"] == 2
        
        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.id == "chunk1"
        assert results[0].chunk.content == "First content"
        assert results[0].distance == 0.1
        assert results[0].score == 0.9  # 1 - distance for cosine
    
    @pytest.mark.asyncio
    async def test_search_similar_with_filters(self):
        """Test similarity search with metadata filters."""
        query_embedding = [0.1, 0.2, 0.3]
        filters = {"document_id": "doc1", "custom_author": "test_author"}
        
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        await self.provider.search_similar(query_embedding, top_k=5, filters=filters)
        
        # Verify filters were applied correctly
        call_args = mock_collection.query.call_args
        where_filter = call_args.kwargs["where"]
        
        assert where_filter["document_id"]["$eq"] == "doc1"
        assert where_filter["custom_author"]["$eq"] == "test_author"
    
    @pytest.mark.asyncio
    async def test_get_chunk(self):
        """Test getting specific chunk by ID."""
        chunk_id = "test_chunk"
        
        mock_response = {
            "documents": ["Test content"],
            "metadatas": [{
                "document_id": "doc1",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 12
            }]
        }
        
        mock_collection = Mock()
        mock_collection.get.return_value = mock_response
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        chunk = await self.provider.get_chunk(chunk_id)
        
        # Verify get was called correctly
        mock_collection.get.assert_called_once_with(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        # Verify returned chunk
        assert chunk is not None
        assert chunk.id == chunk_id
        assert chunk.content == "Test content"
        assert chunk.document_id == "doc1"
        assert chunk.chunk_index == 0
    
    @pytest.mark.asyncio
    async def test_get_chunk_not_found(self):
        """Test getting non-existent chunk."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        chunk = await self.provider.get_chunk("nonexistent")
        assert chunk is None
    
    @pytest.mark.asyncio
    async def test_delete_chunks(self):
        """Test deleting chunks by IDs."""
        chunk_ids = ["chunk1", "chunk2"]
        
        mock_collection = Mock()
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        await self.provider.delete_chunks(chunk_ids)
        
        mock_collection.delete.assert_called_once_with(ids=chunk_ids)
    
    @pytest.mark.asyncio
    async def test_delete_document(self):
        """Test deleting all chunks for a document."""
        document_id = "doc1"
        
        # Mock response showing 2 chunks found for the document
        mock_response = {
            "ids": ["chunk1", "chunk2"],
            "metadatas": [
                {"document_id": "doc1"}, 
                {"document_id": "doc1"}
            ]
        }
        
        mock_collection = Mock()
        mock_collection.get.return_value = mock_response
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        deleted_count = await self.provider.delete_document(document_id)
        
        # Verify get was called to find chunks
        mock_collection.get.assert_called_once_with(
            where={"document_id": {"$eq": document_id}},
            include=["metadatas"]
        )
        
        # Verify delete was called with found chunk IDs
        mock_collection.delete.assert_called_once_with(ids=["chunk1", "chunk2"])
        
        assert deleted_count == 2
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        """Test getting collection statistics."""
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        stats = await self.provider.get_collection_stats()
        
        mock_collection.count.assert_called_once()
        
        assert stats["total_chunks"] == 100
        assert stats["collection_name"] == self.config.collection_name
        assert stats["distance_metric"] == self.config.distance_metric
        assert stats["backend"] == "chromadb"
    
    @pytest.mark.asyncio
    async def test_list_documents(self):
        """Test listing all document IDs."""
        mock_response = {
            "metadatas": [
                {"document_id": "doc1"}, 
                {"document_id": "doc2"},
                {"document_id": "doc1"},  # Duplicate
                {"document_id": "doc3"}
            ]
        }
        
        mock_collection = Mock()
        mock_collection.get.return_value = mock_response
        self.provider.collection = mock_collection
        self.provider.initialized = True
        
        document_ids = await self.provider.list_documents()
        
        # Should return unique document IDs
        assert set(document_ids) == {"doc1", "doc2", "doc3"}
        assert len(document_ids) == 3


class TestVectorStore:
    """Test VectorStore main interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StorageConfig(
            backend=VectorStorageBackend.CHROMADB,
            collection_name="test_store"
        )
        self.store = VectorStore(self.config)
    
    def test_provider_initialization(self):
        """Test provider initialization based on backend."""
        assert isinstance(self.store.provider, ChromaDBProvider)
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test store initialization."""
        with patch.object(self.store.provider, 'initialize') as mock_init:
            await self.store.initialize()
            
            mock_init.assert_called_once()
            assert self.store.initialized is True
    
    @pytest.mark.asyncio
    async def test_add_chunks(self):
        """Test adding chunks through store interface."""
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Test content",
                document_id="doc1",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            )
        ]
        
        with patch.object(self.store.provider, 'add_chunks') as mock_add:
            self.store.initialized = True
            await self.store.add_chunks(chunks)
            
            mock_add.assert_called_once_with(chunks)
    
    @pytest.mark.asyncio
    async def test_add_document(self):
        """Test adding complete document with metadata."""
        metadata = DocumentMetadata(
            file_path="test.txt",
            file_name="test.txt", 
            file_size=1000,
            file_hash="abc123",
            mime_type="text/plain",
            format="text",
            processed_at=time.time(),
            chunk_count=2,
            processing_time=1.5
        )
        
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="First content",
                document_id="doc1",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3]
            ),
            DocumentChunk(
                id="chunk2",
                content="Second content",
                document_id="doc1", 
                chunk_index=1,
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        with patch.object(self.store, 'add_chunks') as mock_add:
            self.store.initialized = True
            await self.store.add_document(metadata, chunks)
            
            mock_add.assert_called_once()
            
            # Verify metadata was added to chunks
            added_chunks = mock_add.call_args[0][0]
            for chunk in added_chunks:
                assert chunk.metadata["file_name"] == "test.txt"
                assert chunk.metadata["file_path"] == "test.txt"
                assert chunk.metadata["format"] == "text"
    
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Test similarity search through store interface."""
        query_embedding = [0.1, 0.2, 0.3]
        
        mock_results = [
            SearchResult(
                chunk=DocumentChunk(id="chunk1", content="Test", document_id="doc1", chunk_index=0),
                score=0.8,
                distance=0.2
            )
        ]
        
        with patch.object(self.store.provider, 'search_similar', return_value=mock_results) as mock_search:
            self.store.initialized = True
            results = await self.store.search_similar(query_embedding, top_k=5)
            
            mock_search.assert_called_once_with(query_embedding, 5, None)
            assert results == mock_results
    
    @pytest.mark.asyncio
    async def test_search_by_document(self):
        """Test searching within specific documents."""
        query_embedding = [0.1, 0.2, 0.3]
        document_ids = ["doc1", "doc2"]
        
        with patch.object(self.store, 'search_similar') as mock_search:
            await self.store.search_by_document(query_embedding, document_ids, top_k=10)
            
            mock_search.assert_called_once_with(
                query_embedding, 
                10, 
                {"document_id": document_ids}
            )
    
    @pytest.mark.asyncio
    async def test_search_by_metadata(self):
        """Test searching with metadata filters."""
        query_embedding = [0.1, 0.2, 0.3]
        metadata_filters = {"author": "test_author", "category": "science"}
        
        with patch.object(self.store, 'search_similar') as mock_search:
            await self.store.search_by_metadata(query_embedding, metadata_filters, top_k=10)
            
            # Should add custom_ prefix to metadata filters
            expected_filters = {
                "custom_author": "test_author",
                "custom_category": "science"
            }
            
            mock_search.assert_called_once_with(query_embedding, 10, expected_filters)
    
    @pytest.mark.asyncio
    async def test_get_chunk(self):
        """Test getting specific chunk."""
        chunk_id = "test_chunk"
        
        mock_chunk = DocumentChunk(
            id=chunk_id,
            content="Test content",
            document_id="doc1",
            chunk_index=0
        )
        
        with patch.object(self.store.provider, 'get_chunk', return_value=mock_chunk) as mock_get:
            self.store.initialized = True
            chunk = await self.store.get_chunk(chunk_id)
            
            mock_get.assert_called_once_with(chunk_id)
            assert chunk == mock_chunk
    
    @pytest.mark.asyncio
    async def test_delete_chunks(self):
        """Test deleting chunks."""
        chunk_ids = ["chunk1", "chunk2"]
        
        with patch.object(self.store.provider, 'delete_chunks') as mock_delete:
            self.store.initialized = True
            await self.store.delete_chunks(chunk_ids)
            
            mock_delete.assert_called_once_with(chunk_ids)
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting storage statistics."""
        mock_stats = {
            "total_chunks": 100,
            "backend": "chromadb"
        }
        
        with patch.object(self.store.provider, 'get_collection_stats', return_value=mock_stats) as mock_stats_call:
            self.store.initialized = True
            stats = await self.store.get_stats()
            
            mock_stats_call.assert_called_once()
            assert stats == mock_stats
    
    @pytest.mark.asyncio
    async def test_list_documents(self):
        """Test listing documents."""
        mock_docs = ["doc1", "doc2", "doc3"]
        
        with patch.object(self.store.provider, 'list_documents', return_value=mock_docs) as mock_list:
            self.store.initialized = True
            docs = await self.store.list_documents()
            
            mock_list.assert_called_once()
            assert docs == mock_docs


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for vector storage."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.config = StorageConfig(
                backend=VectorStorageBackend.CHROMADB,
                collection_name="integration_test",
                persist_directory=temp_dir,
                embedding_dimensions=10  # Small for testing
            )
            self.store = VectorStore(self.config)
    
    @pytest.mark.asyncio
    async def test_full_storage_lifecycle(self):
        """Test complete storage lifecycle with mocked ChromaDB."""
        # Mock ChromaDB components
        mock_chromadb = Mock()
        mock_settings = Mock()
        mock_client = Mock()
        mock_collection = Mock()
        
        # Configure mocks
        mock_chromadb.Client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock collection operations
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "ids": [["chunk1"]],
            "documents": [["Test content"]],
            "metadatas": [[{"document_id": "doc1", "chunk_index": 0}]], 
            "distances": [[0.1]]
        }
        
        # Set up provider with mocks
        provider = ChromaDBProvider(self.config)
        provider._chromadb = mock_chromadb
        provider._settings = mock_settings
        
        self.store.provider = provider
        
        # Test initialization
        await self.store.initialize()
        assert self.store.initialized is True
        
        # Test adding chunks
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Test content for storage",
                document_id="doc1",
                chunk_index=0,
                embedding=[0.1] * 10
            ),
            DocumentChunk(
                id="chunk2", 
                content="Second test content",
                document_id="doc1",
                chunk_index=1,
                embedding=[0.2] * 10
            )
        ]
        
        await self.store.add_chunks(chunks)
        mock_collection.add.assert_called_once()
        
        # Test searching
        query_embedding = [0.15] * 10
        results = await self.store.search_similar(query_embedding, top_k=1)
        
        mock_collection.query.assert_called_once()
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        
        # Test getting stats
        stats = await self.store.get_stats()
        assert "total_chunks" in stats
        assert stats["backend"] == "chromadb"
    
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch operations with large datasets."""
        # Create many chunks for batch testing
        chunks = []
        for i in range(50):
            chunk = DocumentChunk(
                id=f"chunk_{i}",
                content=f"Test content number {i}",
                document_id=f"doc_{i // 10}",  # 5 documents with 10 chunks each
                chunk_index=i % 10,
                embedding=[float(i % 10) / 10.0] * 10
            )
            chunks.append(chunk)
        
        # Mock ChromaDB for batch operations
        mock_collection = Mock()
        
        provider = ChromaDBProvider(self.config)
        provider.collection = mock_collection
        provider.initialized = True
        
        self.store.provider = provider
        self.store.initialized = True
        
        # Test batch addition
        await self.store.add_chunks(chunks)
        
        # Should be called multiple times due to batching
        assert mock_collection.add.call_count > 0
        
        # Verify all chunks were processed
        total_chunks_added = 0
        for call in mock_collection.add.call_args_list:
            total_chunks_added += len(call.kwargs["ids"])
        
        assert total_chunks_added == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])