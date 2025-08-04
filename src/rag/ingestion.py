"""
Document ingestion pipeline for Akasha RAG system.

This module handles loading, parsing, and chunking of various document formats
for embedding and storage in the vector database.
"""

import hashlib
import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
import asyncio
import time

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError


class DocumentFormat(str, Enum):
    """Supported document formats."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    document_id: str
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    mime_type: str
    format: DocumentFormat
    processed_at: float
    chunk_count: int
    processing_time: float
    source: str = "local"
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metadata is None:
            self.custom_metadata = {}


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Index within document")
    start_offset: int = Field(default=0, description="Start character offset")
    end_offset: int = Field(default=0, description="End character offset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Chunk embedding vector")
    
    def get_id(self) -> str:
        """Generate unique chunk ID."""
        if not self.id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"{self.document_id}_chunk_{self.chunk_index}_{content_hash}"
        return self.id


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.RECURSIVE)
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size")
    preserve_sentences: bool = Field(default=True, description="Try to preserve sentence boundaries")
    preserve_paragraphs: bool = Field(default=True, description="Try to preserve paragraph boundaries")


class DocumentProcessor:
    """Base class for document format processors."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text content from document."""
        raise NotImplementedError
    
    async def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract document metadata."""
        return {}


class TextProcessor(DocumentProcessor):
    """Processor for plain text files."""
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin1', 'cp1252', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise AkashaError(f"Could not decode text file: {file_path}")


class PDFProcessor(DocumentProcessor):
    """Processor for PDF files."""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self._pypdf2 = None
    
    async def _import_pypdf2(self):
        """Lazy import PyPDF2."""
        if self._pypdf2 is None:
            try:
                import PyPDF2
                self._pypdf2 = PyPDF2
            except ImportError:
                raise AkashaError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        await self._import_pypdf2()
        
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = self._pypdf2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
        
        return "\n\n".join(text_content)
    
    async def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata."""
        await self._import_pypdf2()
        
        metadata = {}
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = self._pypdf2.PdfReader(file)
                
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata.update({
                        'title': pdf_meta.get('/Title', ''),
                        'author': pdf_meta.get('/Author', ''),
                        'subject': pdf_meta.get('/Subject', ''),
                        'creator': pdf_meta.get('/Creator', ''),
                        'producer': pdf_meta.get('/Producer', ''),
                        'creation_date': str(pdf_meta.get('/CreationDate', '')),
                        'modification_date': str(pdf_meta.get('/ModDate', ''))
                    })
                
                metadata['page_count'] = len(pdf_reader.pages)
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata


class DOCXProcessor(DocumentProcessor):
    """Processor for DOCX files."""
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self._docx = None
    
    async def _import_docx(self):
        """Lazy import python-docx."""
        if self._docx is None:
            try:
                import docx
                self._docx = docx
            except ImportError:
                raise AkashaError("python-docx not installed. Install with: pip install python-docx")
    
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        await self._import_docx()
        
        doc = self._docx.Document(file_path)
        paragraphs = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
        
        return "\n".join(paragraphs)
    
    async def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract DOCX metadata."""
        await self._import_docx()
        
        metadata = {}
        try:
            doc = self._docx.Document(file_path)
            core_props = doc.core_properties
            
            metadata.update({
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'keywords': core_props.keywords or '',
                'category': core_props.category or '',
                'comments': core_props.comments or '',
                'created': str(core_props.created) if core_props.created else '',
                'modified': str(core_props.modified) if core_props.modified else '',
                'last_modified_by': core_props.last_modified_by or '',
                'revision': core_props.revision if core_props.revision else 0,
            })
            
            metadata['paragraph_count'] = len(doc.paragraphs)
        except Exception as e:
            self.logger.warning(f"Failed to extract DOCX metadata: {e}")
        
        return metadata


class DocumentChunker:
    """Handles document chunking with various strategies."""
    
    def __init__(self, config: ChunkingConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
    
    async def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk text based on configured strategy."""
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return await self._chunk_fixed_size(text, document_id)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return await self._chunk_by_sentence(text, document_id)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            return await self._chunk_by_paragraph(text, document_id)
        elif self.config.strategy == ChunkingStrategy.RECURSIVE:
            return await self._chunk_recursive(text, document_id)
        else:
            raise AkashaError(f"Unsupported chunking strategy: {self.config.strategy}")
    
    async def _chunk_fixed_size(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk text into fixed-size pieces with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            
            # Try to break at word boundary if preserving sentences
            if self.config.preserve_sentences and end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 2
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    id="",
                    content=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_offset=start,
                    end_offset=end
                )
                chunk.id = chunk.get_id()
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position considering overlap
            start = max(start + self.config.chunk_size - self.config.chunk_overlap, end)
        
        return chunks
    
    async def _chunk_by_sentence(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk text by sentences, grouping to target size."""
        # Simple sentence splitting - could be enhanced with nltk/spacy
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_offset = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        id="",
                        content=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_offset=start_offset,
                        end_offset=start_offset + len(chunk_text)
                    )
                    chunk.id = chunk.get_id()
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.config.chunk_overlap//100:] if self.config.chunk_overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
                start_offset += len(chunk_text) - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    id="",
                    content=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_offset=start_offset,
                    end_offset=start_offset + len(chunk_text)
                )
                chunk.id = chunk.get_id()
                chunks.append(chunk)
        
        return chunks
    
    async def _chunk_by_paragraph(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Chunk text by paragraphs, grouping to target size."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        start_offset = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > self.config.chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        id="",
                        content=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_offset=start_offset,
                        end_offset=start_offset + len(chunk_text)
                    )
                    chunk.id = chunk.get_id()
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = [paragraph]
                current_size = paragraph_size
                start_offset += len(chunk_text)
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size + 2  # Add 2 for \n\n
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    id="",
                    content=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_offset=start_offset,
                    end_offset=start_offset + len(chunk_text)
                )
                chunk.id = chunk.get_id()
                chunks.append(chunk)
        
        return chunks
    
    async def _chunk_recursive(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Recursive chunking with hierarchical separators."""
        separators = ['\n\n', '\n', '. ', ' ']
        
        def _split_text_recursive(text: str, separators: List[str]) -> List[str]:
            """Recursively split text using hierarchical separators."""
            if not separators or len(text) <= self.config.chunk_size:
                return [text]
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            splits = text.split(separator)
            good_splits = []
            
            for split in splits:
                if len(split) <= self.config.chunk_size:
                    good_splits.append(split)
                else:
                    # Recursively split large chunks
                    sub_splits = _split_text_recursive(split, remaining_separators)
                    good_splits.extend(sub_splits)
            
            return good_splits
        
        # Get initial splits
        splits = _split_text_recursive(text, separators)
        
        # Combine splits to target size with CHARACTER-BASED overlap
        chunks = []
        current_chunk = []
        chunk_index = 0
        text_position = 0  # Track actual position in original text
        
        i = 0
        while i < len(splits):
            current_chunk = []
            current_content = ""
            chunk_start = text_position
            
            # Build chunk by adding splits until we reach target size
            while i < len(splits):
                split = splits[i]
                # Account for separator when joining (except for first split in chunk)
                separator_len = 1 if current_chunk else 0  # '\n' separator
                
                if current_content and len(current_content) + separator_len + len(split) > self.config.chunk_size:
                    break
                
                if current_chunk:  # Add separator for non-first splits
                    current_content += '\n'
                current_content += split
                current_chunk.append(split)
                i += 1
            
            # Create chunk if we have content
            if current_content.strip() and len(current_content.strip()) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    id="",
                    content=current_content.strip(),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(current_content.strip())
                )
                chunk.id = chunk.get_id()
                chunks.append(chunk)
                chunk_index += 1
                
                # Calculate overlap for next chunk (character-based)
                if i < len(splits) and self.config.chunk_overlap > 0:
                    # Find how much content to overlap (from end of current chunk)
                    content_to_overlap = current_content.strip()
                    overlap_chars = min(self.config.chunk_overlap, len(content_to_overlap))
                    
                    if overlap_chars > 0:
                        # Find where the overlap starts in the original content
                        overlap_start = len(content_to_overlap) - overlap_chars
                        overlap_content = content_to_overlap[overlap_start:]
                        
                        # Move back in the splits to include overlapping content
                        # This is a simple approach - we'll go back one split for overlap
                        if len(current_chunk) > 1:
                            i -= 1  # Go back one split for overlap
                            text_position = chunk_start + len(content_to_overlap) - len(splits[i])
                        else:
                            text_position = chunk_start + len(content_to_overlap)
                    else:
                        text_position = chunk_start + len(content_to_overlap)
                else:
                    text_position = chunk_start + len(current_content.strip())
            else:
                # Skip empty/too-small content
                if i < len(splits):
                    text_position += len(splits[i])
                    i += 1
        
        return chunks


class DocumentIngestion:
    """Main document ingestion pipeline."""
    
    def __init__(self, chunking_config: ChunkingConfig = None, logger=None):
        self.logger = logger or get_logger(__name__)
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker = DocumentChunker(self.chunking_config, self.logger)
        
        # Initialize processors
        self.processors = {
            DocumentFormat.TEXT: TextProcessor(self.logger),
            DocumentFormat.PDF: PDFProcessor(self.logger),
            DocumentFormat.DOCX: DOCXProcessor(self.logger),
            DocumentFormat.MARKDOWN: TextProcessor(self.logger),  # Treat as text for now
            DocumentFormat.HTML: TextProcessor(self.logger),      # Could add HTML parser later
        }
    
    def _detect_format(self, file_path: Path) -> DocumentFormat:
        """Detect document format from file extension and MIME type."""
        suffix = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Map extensions to formats
        extension_map = {
            '.txt': DocumentFormat.TEXT,
            '.pdf': DocumentFormat.PDF,
            '.docx': DocumentFormat.DOCX,
            '.doc': DocumentFormat.DOCX,  # Treat .doc as .docx for now
            '.md': DocumentFormat.MARKDOWN,
            '.markdown': DocumentFormat.MARKDOWN,
            '.html': DocumentFormat.HTML,
            '.htm': DocumentFormat.HTML,
            '.json': DocumentFormat.JSON,
            '.csv': DocumentFormat.CSV,
        }
        
        if suffix in extension_map:
            return extension_map[suffix]
        
        # Fallback to MIME type detection
        if mime_type:
            if mime_type.startswith('text/'):
                return DocumentFormat.TEXT
            elif mime_type == 'application/pdf':
                return DocumentFormat.PDF
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return DocumentFormat.DOCX
        
        # Default to text
        return DocumentFormat.TEXT
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return f"doc_{file_hash[:16]}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file content hash for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def process_file(self, file_path: Union[str, Path]) -> tuple[DocumentMetadata, List[DocumentChunk]]:
        """Process a single file and return metadata and chunks."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AkashaError(f"File not found: {file_path}")
        
        start_time = time.time()
        
        async with PerformanceLogger(f"document_ingestion:{file_path.name}", self.logger):
            # Detect format and get processor
            doc_format = self._detect_format(file_path)
            processor = self.processors.get(doc_format)
            
            if not processor:
                raise AkashaError(f"Unsupported document format: {doc_format}")
            
            # Generate document ID and calculate hash
            document_id = self._generate_document_id(file_path)
            file_hash = self._calculate_file_hash(file_path)
            
            # Extract text content
            try:
                text_content = await processor.extract_text(file_path)
            except Exception as e:
                raise AkashaError(f"Failed to extract text from {file_path}: {e}")
            
            if not text_content.strip():
                raise AkashaError(f"No text content extracted from {file_path}")
            
            # Extract additional metadata
            format_metadata = await processor.extract_metadata(file_path)
            
            # Chunk the document
            chunks = await self.chunker.chunk_text(text_content, document_id)
            
            processing_time = time.time() - start_time
            
            # Create document metadata
            file_stat = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            metadata = DocumentMetadata(
                document_id=document_id,
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_stat.st_size,
                file_hash=file_hash,
                mime_type=mime_type or "application/octet-stream",
                format=doc_format,
                processed_at=time.time(),
                chunk_count=len(chunks),
                processing_time=processing_time,
                custom_metadata=format_metadata
            )
            
            self.logger.info(
                "Document processed successfully",
                document_id=document_id,
                file_name=file_path.name,
                format=doc_format.value,
                chunk_count=len(chunks),
                processing_time=processing_time
            )
            
            return metadata, chunks
    
    async def process_directory(self, directory_path: Union[str, Path], 
                               recursive: bool = True,
                               file_patterns: List[str] = None) -> Iterator[tuple[DocumentMetadata, List[DocumentChunk]]]:
        """Process all supported files in a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise AkashaError(f"Directory not found: {directory_path}")
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ["*.txt", "*.pdf", "*.docx", "*.md", "*.html"]
        
        # Collect files
        files_to_process = []
        for pattern in file_patterns:
            if recursive:
                files_to_process.extend(directory_path.rglob(pattern))
            else:
                files_to_process.extend(directory_path.glob(pattern))
        
        self.logger.info(
            "Starting directory processing",
            directory=str(directory_path),
            file_count=len(files_to_process),
            recursive=recursive
        )
        
        # Process files
        for file_path in files_to_process:
            try:
                yield await self.process_file(file_path)
            except Exception as e:
                self.logger.error(
                    "Failed to process file",
                    file_path=str(file_path),
                    error=str(e)
                )
                # Continue with other files
                continue
    
    async def batch_process_files(self, file_paths: List[Union[str, Path]], 
                                  max_concurrent: int = 5) -> List[tuple[DocumentMetadata, List[DocumentChunk]]]:
        """Process multiple files concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_file(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = []
        
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process file in batch: {e}")
                # Continue with other files
                continue
        
        return results