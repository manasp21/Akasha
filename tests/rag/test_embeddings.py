"""
Tests for RAG embedding generation system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time

from src.rag.embeddings import (
    EmbeddingGenerator, EmbeddingConfig, EmbeddingModel, EmbeddingBackend,
    EmbeddingResult, MLXEmbeddingProvider, OpenAIEmbeddingProvider
)
from src.rag.ingestion import DocumentChunk
from src.core.exceptions import AkashaError


class TestEmbeddingConfig:
    """Test EmbeddingConfig model."""
    
    def test_default_config(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        
        assert config.backend == EmbeddingBackend.MLX
        assert config.model_name == EmbeddingModel.ALL_MINILM_L6_V2
        assert config.batch_size == 32
        assert config.max_sequence_length == 512
        assert config.normalize_embeddings is True
        assert config.cache_embeddings is True


class TestEmbeddingResult:
    """Test EmbeddingResult model."""
    
    def test_embedding_result_creation(self):
        """Test embedding result creation."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            dimensions=3,
            processing_time=1.5,
            input_count=2
        )
        
        assert result.embeddings == embeddings
        assert result.model_name == "test-model"
        assert result.dimensions == 3
        assert result.processing_time == 1.5
        assert result.input_count == 2


class TestMLXEmbeddingProvider:
    """Test MLXEmbeddingProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(
            backend=EmbeddingBackend.MLX,
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
            batch_size=2,
            cache_embeddings=False  # Disable caching for tests
        )
        self.provider = MLXEmbeddingProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_mlx_not_available_fallback(self):
        """Test fallback to sentence-transformers when MLX not available."""
        with patch.object(self.provider, '_check_mlx_availability', return_value=False):
            with patch.object(self.provider, '_load_sentence_transformers_model') as mock_load:
                # Ensure model is not loaded initially
                self.provider.model_loaded = False
                mock_load.return_value = None
                
                await self.provider.load_model()
                
                # Should attempt to load sentence transformers model when MLX not available
                mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sentence_transformers_not_available(self):
        """Test error when sentence-transformers not available."""
        with patch.object(self.provider, '_check_mlx_availability', return_value=False):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'sentence_transformers'")):
                with pytest.raises(AkashaError, match="sentence-transformers not installed"):
                    await self.provider._import_sentence_transformers()
    
    def test_get_embedding_dimensions_known_model(self):
        """Test getting embedding dimensions for known models."""
        dimensions = self.provider.get_embedding_dimensions()
        assert dimensions == 384  # Known dimension for ALL_MINILM_L6_V2
    
    @pytest.mark.asyncio
    async def test_embed_texts_empty_input(self):
        """Test embedding empty text list."""
        # Should return empty list without trying to load model
        self.provider.model_loaded = True  # Fake loaded to avoid import errors
        embeddings = await self.provider.embed_texts([])
        assert embeddings == []
    
    @pytest.mark.asyncio
    @patch('asyncio.get_event_loop')
    async def test_embed_texts_with_mock_model(self, mock_loop):
        """Test embedding texts with mocked model."""
        # Mock the model and its encode method
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.provider.model = mock_model
        self.provider.model_loaded = True
        
        # Mock the executor
        mock_executor = Mock()
        mock_executor.run_in_executor = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_loop.return_value = mock_executor
        
        texts = ["First text", "Second text"]
        embeddings = await self.provider.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @pytest.mark.asyncio
    async def test_embed_chunks(self):
        """Test embedding document chunks."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="First chunk content",
                document_id="doc1",
                chunk_index=0
            ),
            DocumentChunk(
                id="chunk2", 
                content="Second chunk content",
                document_id="doc1",
                chunk_index=1
            )
        ]
        
        # Mock the embed_texts method and model loading
        self.provider.model_loaded = True  # Avoid model loading
        with patch.object(self.provider, 'embed_texts', return_value=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
            result_chunks = await self.provider.embed_chunks(chunks)
            
            mock_embed.assert_called_once_with(["First chunk content", "Second chunk content"])
            
            assert len(result_chunks) == 2
            assert result_chunks[0].embedding == [0.1, 0.2]
            assert result_chunks[1].embedding == [0.3, 0.4]


class TestOpenAIEmbeddingProvider:
    """Test OpenAIEmbeddingProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(
            backend=EmbeddingBackend.OPENAI,
            batch_size=10
        )
        self.provider = OpenAIEmbeddingProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_openai_not_available(self):
        """Test error when OpenAI not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            with pytest.raises(AkashaError, match="openai not installed"):
                await self.provider._import_openai()
    
    @pytest.mark.asyncio
    async def test_load_model_no_api_key(self):
        """Test error when no API key provided."""
        with patch.object(self.provider, '_import_openai'):
            with patch('os.getenv', return_value=None):
                with pytest.raises(AkashaError, match="OpenAI API key not found"):
                    await self.provider.load_model()
    
    @pytest.mark.asyncio
    async def test_load_model_with_api_key(self):
        """Test successful model loading with API key."""
        with patch.object(self.provider, '_import_openai'):
            mock_openai = Mock()
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            self.provider._openai = mock_openai
            self.provider.config.api_key = "test-key"
            
            await self.provider.load_model()
            
            assert self.provider.model_loaded is True
            assert self.provider.model is mock_client
            mock_openai.OpenAI.assert_called_once_with(api_key="test-key")
    
    def test_get_embedding_dimensions(self):
        """Test OpenAI embedding dimensions."""
        dimensions = self.provider.get_embedding_dimensions()
        assert dimensions == 1536  # Known OpenAI embedding dimensions
    
    @pytest.mark.asyncio
    async def test_embed_texts_empty(self):
        """Test embedding empty text list."""
        # Should return empty list without trying to load model
        self.provider.model_loaded = True
        embeddings = await self.provider.embed_texts([])
        assert embeddings == []


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator main class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(cache_embeddings=True)
        self.generator = EmbeddingGenerator(self.config)
    
    def test_provider_initialization_mlx(self):
        """Test MLX provider initialization."""
        assert isinstance(self.generator.provider, MLXEmbeddingProvider)
    
    def test_provider_initialization_openai(self):
        """Test OpenAI provider initialization."""
        config = EmbeddingConfig(backend=EmbeddingBackend.OPENAI)
        generator = EmbeddingGenerator(config)
        assert isinstance(generator.provider, OpenAIEmbeddingProvider)
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test generator initialization."""
        with patch.object(self.generator.provider, 'load_model') as mock_load:
            await self.generator.initialize()
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embed_text_single(self):
        """Test embedding single text."""
        with patch.object(self.generator, 'embed_texts', return_value=[[0.1, 0.2, 0.3]]) as mock_embed:
            embedding = await self.generator.embed_text("Test text")
            
            mock_embed.assert_called_once_with(["Test text"])
            assert embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_embed_texts_with_caching(self):
        """Test text embedding with caching enabled."""
        self.generator.config.cache_embeddings = True
        
        with patch.object(self.generator.provider, 'embed_texts', return_value=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
            texts = ["First text", "Second text"]
            
            # First call should hit the provider
            embeddings1 = await self.generator.embed_texts(texts)
            mock_embed.assert_called_once_with(texts)
            assert embeddings1 == [[0.1, 0.2], [0.3, 0.4]]
            
            # Second call should use cache
            mock_embed.reset_mock()
            embeddings2 = await self.generator.embed_texts(texts)
            mock_embed.assert_not_called()  # Should not call provider due to cache
            assert embeddings2 == embeddings1
    
    @pytest.mark.asyncio
    async def test_embed_texts_without_caching(self):
        """Test text embedding with caching disabled."""
        self.generator.config.cache_embeddings = False
        
        with patch.object(self.generator.provider, 'embed_texts', return_value=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
            texts = ["First text", "Second text"]
            
            # Each call should hit the provider
            await self.generator.embed_texts(texts)
            mock_embed.assert_called_once_with(texts)
            
            mock_embed.reset_mock()
            await self.generator.embed_texts(texts)
            mock_embed.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_embed_chunks(self):
        """Test embedding document chunks."""
        chunks = [
            DocumentChunk(id="1", content="Content 1", document_id="doc1", chunk_index=0),
            DocumentChunk(id="2", content="Content 2", document_id="doc1", chunk_index=1)
        ]
        
        with patch.object(self.generator, 'embed_texts', return_value=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
            result_chunks = await self.generator.embed_chunks(chunks)
            
            mock_embed.assert_called_once_with(["Content 1", "Content 2"])
            
            assert len(result_chunks) == 2
            assert result_chunks[0].embedding == [0.1, 0.2]
            assert result_chunks[1].embedding == [0.3, 0.4]
    
    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test query embedding."""
        with patch.object(self.generator, 'embed_text', return_value=[0.1, 0.2, 0.3]) as mock_embed:
            query_embedding = await self.generator.embed_query("Test query")
            
            mock_embed.assert_called_once_with("Test query")
            assert query_embedding == [0.1, 0.2, 0.3]
    
    def test_get_embedding_dimensions(self):
        """Test getting embedding dimensions."""
        with patch.object(self.generator.provider, 'get_embedding_dimensions', return_value=384) as mock_dims:
            dimensions = self.generator.get_embedding_dimensions()
            
            mock_dims.assert_called_once()
            assert dimensions == 384
    
    @pytest.mark.asyncio
    async def test_get_embedding_info(self):
        """Test getting embedding information."""
        with patch.object(self.generator.provider, 'get_embedding_dimensions', return_value=384):
            info = await self.generator.get_embedding_info()
            
            assert info["backend"] == self.config.backend
            assert info["model_name"] == self.config.model_name
            assert info["dimensions"] == 384
            assert info["batch_size"] == self.config.batch_size
            assert info["cache_enabled"] == self.config.cache_embeddings
    
    def test_clear_cache(self):
        """Test clearing embedding cache."""
        # Add some items to cache
        self.generator._embedding_cache[hash("test1")] = [0.1, 0.2]
        self.generator._embedding_cache[hash("test2")] = [0.3, 0.4]
        
        assert len(self.generator._embedding_cache) == 2
        
        self.generator.clear_cache()
        
        assert len(self.generator._embedding_cache) == 0
    
    @pytest.mark.asyncio
    async def test_compute_similarity(self):
        """Test computing cosine similarity."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        embedding3 = [1.0, 0.0, 0.0]
        
        # Orthogonal vectors
        similarity1 = await self.generator.compute_similarity(embedding1, embedding2)
        assert abs(similarity1 - 0.0) < 1e-10
        
        # Identical vectors
        similarity2 = await self.generator.compute_similarity(embedding1, embedding3)
        assert abs(similarity2 - 1.0) < 1e-10
        
        # Empty embeddings
        similarity3 = await self.generator.compute_similarity([], [])
        assert similarity3 == 0.0
    
    @pytest.mark.asyncio
    async def test_find_most_similar(self):
        """Test finding most similar chunks."""
        query_embedding = [1.0, 0.0, 0.0]
        chunk_embeddings = [
            ("chunk1", [1.0, 0.0, 0.0]),  # Identical - similarity 1.0
            ("chunk2", [0.0, 1.0, 0.0]),  # Orthogonal - similarity 0.0
            ("chunk3", [0.5, 0.5, 0.0]),  # Partial overlap
            ("chunk4", [-1.0, 0.0, 0.0])  # Opposite - similarity -1.0
        ]
        
        similar_chunks = await self.generator.find_most_similar(
            query_embedding, 
            chunk_embeddings, 
            top_k=2
        )
        
        assert len(similar_chunks) == 2
        
        # Results should be sorted by similarity (descending)
        assert similar_chunks[0][0] == "chunk1"  # Highest similarity
        assert similar_chunks[0][1] == 1.0
        
        # Second highest should be chunk3 (partial overlap)
        assert similar_chunks[1][0] == "chunk3"
        assert similar_chunks[1][1] > 0.0


@pytest.mark.integration
class TestEmbeddingIntegration:
    """Integration tests for embedding generation."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = EmbeddingConfig(
            backend=EmbeddingBackend.MLX,
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
            batch_size=4,
            cache_embeddings=True
        )
        self.generator = EmbeddingGenerator(self.config)
    
    @pytest.mark.asyncio
    async def test_full_embedding_pipeline(self):
        """Test complete embedding pipeline with realistic data."""
        # Create realistic test chunks
        chunks = [
            DocumentChunk(
                id="chunk1",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                document_id="ml_doc",
                chunk_index=0
            ),
            DocumentChunk(
                id="chunk2", 
                content="Deep learning uses neural networks with multiple layers to process data.",
                document_id="ml_doc",
                chunk_index=1
            ),
            DocumentChunk(
                id="chunk3",
                content="Natural language processing enables computers to understand human language.",
                document_id="nlp_doc", 
                chunk_index=0
            )
        ]
        
        # Mock the sentence transformer model for this test
        with patch.object(self.generator.provider, '_check_mlx_availability', return_value=False):
            mock_model = Mock()
            # Generate realistic-looking embeddings
            mock_embeddings = [
                [0.1] * 384,  # ML content
                [0.2] * 384,  # Deep learning content  
                [0.3] * 384   # NLP content
            ]
            # Mock numpy array with tolist() method
            mock_numpy_result = Mock()
            mock_numpy_result.tolist.return_value = mock_embeddings
            mock_model.encode.return_value = mock_numpy_result
            
            with patch.object(self.generator.provider, '_load_sentence_transformers_model'):
                self.generator.provider.model = mock_model
                self.generator.provider.model_loaded = True
                
                # Test embedding chunks
                embedded_chunks = await self.generator.embed_chunks(chunks)
                
                # Verify all chunks have embeddings
                assert len(embedded_chunks) == 3
                for i, chunk in enumerate(embedded_chunks):
                    assert chunk.embedding is not None
                    assert len(chunk.embedding) == 384
                    assert chunk.embedding == mock_embeddings[i]
                
                # Test query embedding
                query = "What is machine learning?"
                mock_query_result = Mock()
                mock_query_result.tolist.return_value = [[0.15] * 384]  # Similar to ML content
                mock_model.encode.return_value = mock_query_result
                
                query_embedding = await self.generator.embed_query(query)
                assert len(query_embedding) == 384
                
                # Test similarity computation
                chunk_embeddings = [(chunk.id, chunk.embedding) for chunk in embedded_chunks]
                similar_chunks = await self.generator.find_most_similar(
                    query_embedding,
                    chunk_embeddings,
                    top_k=2
                )
                
                assert len(similar_chunks) == 2
                # Results should be ordered by similarity
                assert similar_chunks[0][1] >= similar_chunks[1][1]
    
    @pytest.mark.asyncio
    async def test_embedding_caching_behavior(self):
        """Test embedding caching behavior under load."""
        texts = [
            "This is the first test text for caching.",
            "This is the second test text for caching.", 
            "This is the first test text for caching.",  # Duplicate
            "This is a third unique text for caching."
        ]
        
        with patch.object(self.generator.provider, 'embed_texts') as mock_embed:
            # Configure mock to return unique embeddings for each unique text
            def side_effect(input_texts):
                return [[hash(text) % 100 / 100.0] * 10 for text in input_texts]
            
            mock_embed.side_effect = side_effect
            
            # First call - should embed all texts (including duplicates within the call)
            embeddings1 = await self.generator.embed_texts(texts)
            
            # Should call provider once with all texts
            assert mock_embed.call_count == 1
            called_texts = mock_embed.call_args[0][0]
            assert len(called_texts) == len(texts)  # All texts are sent on first call
            
            # Second call with same texts - should use cache
            mock_embed.reset_mock()
            embeddings2 = await self.generator.embed_texts(texts)
            
            # Should not call provider due to caching
            assert mock_embed.call_count == 0
            assert embeddings1 == embeddings2
            
            # Verify cache statistics
            unique_texts = list(set(texts))
            info = await self.generator.get_embedding_info()
            assert info["cache_size"] == len(unique_texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])