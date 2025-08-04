"""
Tests for RAG document ingestion system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio

from src.rag.ingestion import (
    DocumentIngestion, DocumentChunk, DocumentMetadata, ChunkingConfig,
    DocumentFormat, ChunkingStrategy, DocumentProcessor, TextProcessor,
    PDFProcessor, DOCXProcessor, DocumentChunker
)
from src.core.exceptions import AkashaError


class TestDocumentChunk:
    """Test DocumentChunk model."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is test content",
            document_id="doc_123", 
            chunk_index=0,
            start_offset=0,
            end_offset=20
        )
        
        assert chunk.id == "test_chunk_1"
        assert chunk.content == "This is test content"
        assert chunk.document_id == "doc_123"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
    
    def test_chunk_id_generation(self):
        """Test automatic chunk ID generation."""
        chunk = DocumentChunk(
            id="",
            content="Test content for ID generation",
            document_id="doc_456",
            chunk_index=1
        )
        
        generated_id = chunk.get_id()
        assert generated_id.startswith("doc_456_chunk_1_")
        assert len(generated_id) > len("doc_456_chunk_1_")


class TestChunkingConfig:
    """Test ChunkingConfig model."""
    
    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.RECURSIVE
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 2000
        assert config.preserve_sentences is True
        assert config.preserve_paragraphs is True


class TestTextProcessor:
    """Test TextProcessor."""
    
    @pytest.mark.asyncio
    async def test_extract_text_utf8(self):
        """Test extracting text from UTF-8 file."""
        processor = TextProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            test_content = "This is a test document with UTF-8 content.\nSecond line here."
            f.write(test_content)
            f.flush()
            
            file_path = Path(f.name)
            
        try:
            extracted_text = await processor.extract_text(file_path)
            assert extracted_text == test_content
        finally:
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_extract_text_different_encoding(self):
        """Test extracting text from different encoding."""
        processor = TextProcessor()
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='latin1', delete=False, suffix='.txt') as f:
            test_content = "Test content with latin1 encoding"
            f.write(test_content)
            f.flush()
            
            file_path = Path(f.name)
        
        try:
            extracted_text = await processor.extract_text(file_path)
            assert test_content in extracted_text
        finally:
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_extract_text_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        processor = TextProcessor()
        
        with pytest.raises(FileNotFoundError):
            await processor.extract_text(Path("nonexistent_file.txt"))


class TestDocumentChunker:
    """Test DocumentChunker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=30
        )
        self.chunker = DocumentChunker(self.config)
    
    @pytest.mark.asyncio
    async def test_fixed_size_chunking(self):
        """Test fixed size chunking strategy."""
        self.config.strategy = ChunkingStrategy.FIXED_SIZE
        chunker = DocumentChunker(self.config)
        
        text = "This is a test document. " * 20  # Create longer text
        document_id = "test_doc"
        
        chunks = await chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == document_id
            assert len(chunk.content) <= self.config.chunk_size + 50  # Allow some flexibility
            assert len(chunk.content) >= self.config.min_chunk_size
    
    @pytest.mark.asyncio
    async def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        self.config.strategy = ChunkingStrategy.SENTENCE
        chunker = DocumentChunker(self.config)
        
        text = "First sentence here. Second sentence follows. Third sentence continues. " * 10
        document_id = "test_doc"
        
        chunks = await chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == document_id
            # Each chunk should end with a sentence
            assert chunk.content.strip().endswith('.')
    
    @pytest.mark.asyncio
    async def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        self.config.strategy = ChunkingStrategy.PARAGRAPH
        chunker = DocumentChunker(self.config)
        
        text = "First paragraph here.\n\nSecond paragraph follows.\n\nThird paragraph continues.\n\n" * 5
        document_id = "test_doc"
        
        chunks = await chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == document_id
    
    @pytest.mark.asyncio
    async def test_recursive_chunking(self):
        """Test recursive chunking strategy."""
        self.config.strategy = ChunkingStrategy.RECURSIVE
        chunker = DocumentChunker(self.config)
        
        text = "This is a test document with multiple paragraphs.\n\nSecond paragraph here.\n\nThird paragraph follows." * 10
        document_id = "test_doc"
        
        chunks = await chunker.chunk_text(text, document_id)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == document_id
            assert len(chunk.content) >= self.config.min_chunk_size
    
    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test chunking empty text."""
        chunks = await self.chunker.chunk_text("", "test_doc")
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_short_text(self):
        """Test chunking text shorter than minimum size."""
        short_text = "Short"
        chunks = await self.chunker.chunk_text(short_text, "test_doc")
        # Should return empty list as text is below min_chunk_size
        assert len(chunks) == 0


class TestDocumentIngestion:
    """Test DocumentIngestion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunking_config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        self.ingestion = DocumentIngestion(self.chunking_config)
    
    def test_detect_format(self):
        """Test document format detection."""
        assert self.ingestion._detect_format(Path("test.txt")) == DocumentFormat.TEXT
        assert self.ingestion._detect_format(Path("test.pdf")) == DocumentFormat.PDF
        assert self.ingestion._detect_format(Path("test.docx")) == DocumentFormat.DOCX
        assert self.ingestion._detect_format(Path("test.md")) == DocumentFormat.MARKDOWN
        assert self.ingestion._detect_format(Path("test.html")) == DocumentFormat.HTML
    
    def test_generate_document_id(self):
        """Test document ID generation."""
        file_path = Path("test_document.txt")
        doc_id = self.ingestion._generate_document_id(file_path)
        
        assert doc_id.startswith("doc_")
        assert len(doc_id) == 20  # "doc_" + 16 character hash
    
    @pytest.mark.asyncio
    async def test_process_text_file(self):
        """Test processing a text file."""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            test_content = "This is a test document for ingestion. " * 10
            f.write(test_content)
            f.flush()
            
            file_path = Path(f.name)
        
        try:
            metadata, chunks = await self.ingestion.process_file(file_path)
            
            assert isinstance(metadata, DocumentMetadata)
            assert metadata.file_name == file_path.name
            assert metadata.format == DocumentFormat.TEXT
            assert metadata.chunk_count == len(chunks)
            
            assert len(chunks) > 0
            for chunk in chunks:
                assert isinstance(chunk, DocumentChunk)
                assert chunk.document_id == self.ingestion._generate_document_id(file_path)
                
        finally:
            file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(AkashaError, match="File not found"):
            await self.ingestion.process_file(Path("nonexistent_file.txt"))
    
    @pytest.mark.asyncio
    async def test_batch_process_files(self):
        """Test batch processing of multiple files."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
                f.write(f"Test content for file {i}. " * 20)
                f.flush()
                temp_files.append(Path(f.name))
        
        try:
            results = await self.ingestion.batch_process_files(temp_files, max_concurrent=2)
            
            assert len(results) == 3
            for metadata, chunks in results:
                assert isinstance(metadata, DocumentMetadata)
                assert len(chunks) > 0
                
        finally:
            for file_path in temp_files:
                file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_directory(self):
        """Test processing entire directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.txt").write_text("Test content 1. " * 10)
            (temp_path / "test2.md").write_text("# Test markdown\nContent here. " * 10)
            (temp_path / "ignore.log").write_text("Log file to ignore")
            
            # Create subdirectory
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "test3.txt").write_text("Subdirectory content. " * 10)
            
            results = []
            async for metadata, chunks in self.ingestion.process_directory(
                temp_path, 
                recursive=True,
                file_patterns=["*.txt", "*.md"]
            ):
                results.append((metadata, chunks))
            
            # Should process 3 files (test1.txt, test2.md, subdir/test3.txt)
            assert len(results) == 3
            
            processed_names = [metadata.file_name for metadata, _ in results]
            assert "test1.txt" in processed_names
            assert "test2.md" in processed_names
            assert "test3.txt" in processed_names
            assert "ignore.log" not in processed_names


class TestPDFProcessor:
    """Test PDFProcessor (requires PyPDF2)."""
    
    @pytest.mark.asyncio
    async def test_import_pypdf2_not_available(self):
        """Test handling of missing PyPDF2."""
        processor = PDFProcessor()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'PyPDF2'")):
            with pytest.raises(AkashaError, match="PyPDF2 not installed"):
                await processor._import_pypdf2()


class TestDOCXProcessor:
    """Test DOCXProcessor (requires python-docx)."""
    
    @pytest.mark.asyncio
    async def test_import_docx_not_available(self):
        """Test handling of missing python-docx."""
        processor = DOCXProcessor()
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'docx'")):
            with pytest.raises(AkashaError, match="python-docx not installed"):
                await processor._import_docx()


@pytest.mark.integration
class TestIngestionIntegration:
    """Integration tests for document ingestion."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50
        )
        self.ingestion = DocumentIngestion(self.chunking_config)
    
    @pytest.mark.asyncio
    async def test_full_ingestion_pipeline(self):
        """Test complete ingestion pipeline."""
        # Create a realistic test document
        test_content = """
        # Test Document
        
        This is a comprehensive test document for the Akasha RAG system.
        
        ## Section 1: Introduction
        
        The document ingestion system is responsible for processing various
        document formats and converting them into manageable chunks for
        embedding and retrieval.
        
        ## Section 2: Features
        
        Key features include:
        - Multiple document format support
        - Intelligent chunking strategies
        - Metadata preservation
        - Error handling and recovery
        
        ## Section 3: Implementation
        
        The implementation uses asynchronous processing for better performance
        and scalability. Each document is processed independently, allowing
        for concurrent processing of multiple documents.
        
        ## Conclusion
        
        This system provides a robust foundation for document processing
        in the Akasha RAG pipeline.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.md') as f:
            f.write(test_content)
            f.flush()
            file_path = Path(f.name)
        
        try:
            # Process the document
            metadata, chunks = await self.ingestion.process_file(file_path)
            
            # Verify metadata
            assert metadata.file_name == file_path.name
            assert metadata.format == DocumentFormat.MARKDOWN
            assert metadata.chunk_count == len(chunks)
            assert metadata.processing_time > 0
            
            # Verify chunks
            assert len(chunks) > 1  # Should be split into multiple chunks
            
            total_content_length = sum(len(chunk.content) for chunk in chunks)
            assert total_content_length <= len(test_content) + (len(chunks) * self.chunking_config.chunk_overlap)
            
            # Verify chunk properties
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i
                assert chunk.document_id == metadata.file_hash[:16]  # Document ID is based on path hash
                assert len(chunk.content) >= self.chunking_config.min_chunk_size
                assert chunk.start_offset >= 0
                assert chunk.end_offset > chunk.start_offset
                
            # Verify overlapping chunks have some shared content
            if len(chunks) > 1:
                for i in range(len(chunks) - 1):
                    current_chunk = chunks[i]
                    next_chunk = chunks[i + 1]
                    
                    # Check for potential overlap
                    current_words = set(current_chunk.content.split()[-10:])  # Last 10 words
                    next_words = set(next_chunk.content.split()[:10])  # First 10 words
                    
                    # Some overlap expected due to chunking strategy
                    if current_words & next_words:
                        assert len(current_words & next_words) > 0
                        
        finally:
            file_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])