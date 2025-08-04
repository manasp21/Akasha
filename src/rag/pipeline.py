"""
Main RAG Pipeline for Akasha system.

This module provides the complete RAG pipeline that orchestrates document ingestion,
embedding generation, vector storage, retrieval, and LLM response generation.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from ..core.config import AkashaConfig
from .ingestion import DocumentIngestion, DocumentChunk, DocumentMetadata, ChunkingConfig
from .embeddings import EmbeddingGenerator, EmbeddingConfig
from .storage import VectorStore, StorageConfig, SearchResult
from .retrieval import DocumentRetriever, RetrievalConfig, RetrievalResult, QueryContext
from ..llm.manager import LLMManager
from ..llm.config import LLMConfig
from ..llm.provider import LLMResponse, StreamingEvent
from ..llm.templates import TemplateType


class QueryMode(str, Enum):
    """Query processing modes."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"


class RAGPipelineConfig(BaseModel):
    """Configuration for RAG Pipeline."""
    # Component configurations
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage_config: StorageConfig = Field(default_factory=StorageConfig)
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # Pipeline settings
    auto_embed_on_ingest: bool = Field(default=True, description="Automatically generate embeddings during ingestion")
    max_concurrent_ingestions: int = Field(default=3, description="Maximum concurrent document ingestions")
    enable_query_preprocessing: bool = Field(default=True, description="Enable query preprocessing")
    enable_response_postprocessing: bool = Field(default=True, description="Enable response postprocessing")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings for performance")
    
    # Performance settings
    query_timeout: float = Field(default=30.0, description="Query timeout in seconds")
    ingestion_timeout: float = Field(default=300.0, description="Ingestion timeout in seconds")
    
    @classmethod
    def create_optimized_config(cls) -> "RAGPipelineConfig":
        """Create optimized configuration for M4 Pro 48GB."""
        chunking_config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            strategy="recursive"
        )
        
        embedding_config = EmbeddingConfig(
            backend="mlx",
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            cache_embeddings=True
        )
        
        storage_config = StorageConfig(
            backend="chromadb",
            persist_directory="./data/vector_db",
            distance_metric="cosine",
            max_batch_size=1000
        )
        
        retrieval_config = RetrievalConfig(
            strategy="multi_stage",
            initial_top_k=50,
            final_top_k=10,
            reranking_method="bm25",
            min_similarity_threshold=0.3
        )
        
        return cls(
            chunking_config=chunking_config,
            embedding_config=embedding_config,
            storage_config=storage_config,
            retrieval_config=retrieval_config,
            auto_embed_on_ingest=True,
            max_concurrent_ingestions=3,
            cache_embeddings=True
        )


class QueryResult(BaseModel):
    """Result of a RAG query."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    retrieval_result: Optional[RetrievalResult] = Field(default=None, description="Retrieval details")
    llm_response: Optional[LLMResponse] = Field(default=None, description="LLM response details")
    processing_time: float = Field(default=0.0, description="Total processing time")
    query_mode: QueryMode = Field(default=QueryMode.SIMPLE, description="Query processing mode")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    document_id: str = Field(..., description="Document identifier")
    file_path: str = Field(..., description="Path to ingested file")
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Time taken for ingestion")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    success: bool = Field(default=True, description="Whether ingestion succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RAGPipeline:
    """Main RAG Pipeline orchestrating all components."""
    
    def __init__(self, 
                 config: RAGPipelineConfig,
                 llm_manager: LLMManager,
                 logger=None):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = logger or get_logger(__name__)
        
        # Initialize components
        self.document_ingestion = DocumentIngestion(
            chunking_config=config.chunking_config,
            logger=self.logger
        )
        
        self.embedding_generator = EmbeddingGenerator(
            config=config.embedding_config,
            logger=self.logger
        )
        
        self.vector_store = VectorStore(
            config=config.storage_config,
            logger=self.logger
        )
        
        self.document_retriever = DocumentRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            config=config.retrieval_config,
            logger=self.logger
        )
        
        self.initialized = False
        self._ingestion_semaphore = asyncio.Semaphore(config.max_concurrent_ingestions)
    
    async def initialize(self) -> None:
        """Initialize the RAG pipeline."""
        if self.initialized:
            return
        
        async with PerformanceLogger("rag_pipeline_initialization", self.logger):
            # Initialize components in dependency order
            await self.embedding_generator.initialize()
            await self.vector_store.initialize()
            await self.document_retriever.initialize()
            await self.llm_manager.initialize()
            
            self.initialized = True
            
            self.logger.info(
                "RAG Pipeline initialized successfully",
                embedding_model=self.config.embedding_config.model_name,
                vector_backend=self.config.storage_config.backend,
                retrieval_strategy=self.config.retrieval_config.strategy,
                llm_providers=len(self.llm_manager.providers)
            )
    
    async def ingest_document(self, file_path: Union[str, Path]) -> IngestionResult:
        """Ingest a single document into the RAG system."""
        if not self.initialized:
            await self.initialize()
        
        file_path = Path(file_path)
        start_time = time.time()
        
        async with self._ingestion_semaphore:
            try:
                async with PerformanceLogger(f"document_ingestion:{file_path.name}", self.logger):
                    # Process document
                    metadata, chunks = await self.document_ingestion.process_file(file_path)
                    
                    # Generate embeddings if enabled
                    if self.config.auto_embed_on_ingest and chunks:
                        chunks = await self.embedding_generator.embed_chunks(chunks)
                    
                    # Store in vector database
                    if chunks:
                        await self.vector_store.add_document(metadata, chunks)
                    
                    processing_time = time.time() - start_time
                    
                    result = IngestionResult(
                        document_id=chunks[0].document_id if chunks else "unknown",
                        file_path=str(file_path),
                        chunks_created=len(chunks),
                        processing_time=processing_time,
                        metadata=metadata,
                        success=True
                    )
                    
                    self.logger.info(
                        "Document ingested successfully",
                        file_path=str(file_path),
                        chunks_created=len(chunks),
                        processing_time=processing_time
                    )
                    
                    return result
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"Failed to ingest document {file_path}: {e}"
                self.logger.error(error_msg)
                
                return IngestionResult(
                    document_id="error",
                    file_path=str(file_path),
                    chunks_created=0,
                    processing_time=processing_time,
                    metadata=DocumentMetadata(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_size=0,
                        file_hash="",
                        mime_type="",
                        format="text",
                        processed_at=time.time(),
                        chunk_count=0,
                        processing_time=processing_time
                    ),
                    success=False,
                    error=str(e)
                )
    
    async def ingest_directory(self, 
                              directory_path: Union[str, Path],
                              recursive: bool = True,
                              file_patterns: List[str] = None) -> List[IngestionResult]:
        """Ingest all documents in a directory."""
        if not self.initialized:
            await self.initialize()
        
        directory_path = Path(directory_path)
        
        self.logger.info(
            "Starting directory ingestion",
            directory=str(directory_path),
            recursive=recursive
        )
        
        # Get all files to process
        files_to_process = []
        if file_patterns is None:
            file_patterns = ["*.txt", "*.pdf", "*.docx", "*.md", "*.html"]
        
        for pattern in file_patterns:
            if recursive:
                files_to_process.extend(directory_path.rglob(pattern))
            else:
                files_to_process.extend(directory_path.glob(pattern))
        
        # Process files concurrently with semaphore limiting
        ingestion_tasks = [
            self.ingest_document(file_path) 
            for file_path in files_to_process
        ]
        
        results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                self.logger.error(f"Ingestion task failed: {result}")
            else:
                successful_results.append(result)
                if not result.success:
                    failed_count += 1
        
        self.logger.info(
            "Directory ingestion completed",
            total_files=len(files_to_process),
            successful=len([r for r in successful_results if r.success]),
            failed=failed_count
        )
        
        return successful_results
    
    async def query(self, 
                   query: str,
                   mode: QueryMode = QueryMode.SIMPLE,
                   top_k: int = None,
                   filters: Dict[str, Any] = None,
                   llm_provider: Optional[str] = None,
                   template_type: TemplateType = TemplateType.RAG_QA,
                   **kwargs) -> QueryResult:
        """Process a query through the complete RAG pipeline."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            async with asyncio.timeout(self.config.query_timeout):
                async with PerformanceLogger(f"rag_query:{mode.value}", self.logger):
                    # Step 1: Retrieve relevant documents
                    retrieval_result = await self.document_retriever.retrieve(
                        query=query,
                        top_k=top_k or self.config.retrieval_config.final_top_k,
                        filters=filters,
                        context=kwargs.get("context", {})
                    )
                    
                    # Step 2: Generate response using LLM
                    if mode == QueryMode.CONVERSATIONAL:
                        template_type = TemplateType.CONVERSATION
                    elif mode == QueryMode.ANALYTICAL:
                        template_type = TemplateType.RAG_ANALYSIS
                    
                    llm_response = await self.llm_manager.generate_rag_response(
                        query=query,
                        retrieval_result=retrieval_result,
                        template_type=template_type,
                        provider_name=llm_provider,
                        **kwargs
                    )
                    
                    # Step 3: Extract source information
                    sources = self._extract_sources(retrieval_result.chunks)
                    
                    processing_time = time.time() - start_time
                    
                    result = QueryResult(
                        query=query,
                        response=llm_response.content,
                        sources=sources,
                        retrieval_result=retrieval_result,
                        llm_response=llm_response,
                        processing_time=processing_time,
                        query_mode=mode,
                        metadata={
                            "retrieval_method": retrieval_result.retrieval_method,
                            "llm_provider": llm_response.metadata.get("provider_name"),
                            "total_tokens": llm_response.total_tokens,
                            "context_chunks": len(retrieval_result.chunks)
                        }
                    )
                    
                    self.logger.info(
                        "Query processed successfully",
                        query=query[:100],
                        processing_time=processing_time,
                        retrieval_method=retrieval_result.retrieval_method,
                        context_chunks=len(retrieval_result.chunks),
                        response_tokens=llm_response.completion_tokens
                    )
                    
                    return result
                    
        except asyncio.TimeoutError:
            raise AkashaError(f"Query timeout after {self.config.query_timeout} seconds")
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Query processing failed: {e}")
            raise AkashaError(f"Query processing failed: {e}")
    
    async def query_stream(self,
                          query: str,
                          mode: QueryMode = QueryMode.SIMPLE,
                          top_k: int = None,
                          filters: Dict[str, Any] = None,
                          llm_provider: Optional[str] = None,
                          template_type: TemplateType = TemplateType.RAG_QA,
                          **kwargs) -> AsyncIterator[Union[Dict[str, Any], StreamingEvent]]:
        """Process a query with streaming response."""
        if not self.initialized:
            await self.initialize()
        
        try:
            async with asyncio.timeout(self.config.query_timeout):
                start_time = time.time()
                
                # Yield initial status
                yield {
                    "type": "status",
                    "message": "Retrieving relevant documents...",
                    "timestamp": time.time()
                }
                
                # Step 1: Retrieve relevant documents
                retrieval_result = await self.document_retriever.retrieve(
                    query=query,
                    top_k=top_k or self.config.retrieval_config.final_top_k,
                    filters=filters,
                    context=kwargs.get("context", {})
                )
                
                # Yield retrieval results
                sources = self._extract_sources(retrieval_result.chunks)
                yield {
                    "type": "retrieval_complete",
                    "sources": sources,
                    "retrieval_method": retrieval_result.retrieval_method,
                    "context_chunks": len(retrieval_result.chunks),
                    "timestamp": time.time()
                }
                
                # Yield generation start status
                yield {
                    "type": "status", 
                    "message": "Generating response...",
                    "timestamp": time.time()
                }
                
                # Step 2: Stream LLM response
                if mode == QueryMode.CONVERSATIONAL:
                    template_type = TemplateType.CONVERSATION
                elif mode == QueryMode.ANALYTICAL:
                    template_type = TemplateType.RAG_ANALYSIS
                
                async for event in self.llm_manager.generate_rag_stream(
                    query=query,
                    retrieval_result=retrieval_result,
                    template_type=template_type,
                    provider_name=llm_provider,
                    **kwargs
                ):
                    # Add pipeline metadata to streaming events
                    event.metadata.update({
                        "retrieval_method": retrieval_result.retrieval_method,
                        "context_chunks": len(retrieval_result.chunks),
                        "query_mode": mode.value
                    })
                    yield event
                    
                    # Yield final summary on completion
                    if event.type == "finish":
                        processing_time = time.time() - start_time
                        yield {
                            "type": "query_complete",
                            "query": query,
                            "processing_time": processing_time,
                            "total_tokens": event.metadata.get("total_tokens", 0),
                            "timestamp": time.time()
                        }
                        
        except asyncio.TimeoutError:
            yield {
                "type": "error",
                "message": f"Query timeout after {self.config.query_timeout} seconds",
                "timestamp": time.time()
            }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Query processing failed: {e}",
                "timestamp": time.time()
            }
    
    def _extract_sources(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract source information from chunks."""
        sources = []
        seen_sources = set()
        
        for chunk in chunks:
            source_key = chunk.metadata.get('file_name', 'Unknown')
            if source_key in seen_sources:
                continue
            
            seen_sources.add(source_key)
            source = {
                "filename": source_key,
                "document_id": chunk.document_id,
                "file_path": chunk.metadata.get('file_path', ''),
                "file_size": chunk.metadata.get('file_size', 0),
                "format": chunk.metadata.get('format', 'unknown'),
                "processed_at": chunk.metadata.get('processed_at', 0)
            }
            
            # Add custom metadata
            for key, value in chunk.metadata.items():
                if key.startswith('custom_') and key not in source:
                    source[key[7:]] = value  # Remove 'custom_' prefix
            
            sources.append(source)
        
        return sources
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get summary information about a document."""
        if not self.initialized:
            await self.initialize()
        
        # Get document chunks
        # This would require adding a method to vector store to get chunks by document_id
        # For now, return basic info
        stats = await self.vector_store.get_stats()
        
        return {
            "document_id": document_id,
            "total_documents": stats.get("total_chunks", 0),
            "vector_store_backend": self.config.storage_config.backend
        }
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        if not self.initialized:
            await self.initialize()
        
        try:
            deleted_count = await self.vector_store.delete_document(document_id)
            
            self.logger.info(
                "Document deleted",
                document_id=document_id,
                chunks_deleted=deleted_count
            )
            
            return deleted_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        if not self.initialized:
            return {"initialized": False}
        
        # Get stats from all components
        vector_stats = await self.vector_store.get_stats()
        retrieval_stats = await self.document_retriever.get_retrieval_stats()
        llm_stats = self.llm_manager.get_provider_stats()
        embedding_info = await self.embedding_generator.get_embedding_info()
        
        return {
            "initialized": True,
            "vector_store": vector_stats,
            "retrieval": retrieval_stats,
            "llm_providers": llm_stats,
            "embeddings": embedding_info,
            "config": {
                "chunking_strategy": self.config.chunking_config.strategy,
                "embedding_model": self.config.embedding_config.model_name,
                "retrieval_strategy": self.config.retrieval_config.strategy,
                "auto_embed_on_ingest": self.config.auto_embed_on_ingest
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "pipeline_initialized": self.initialized,
            "timestamp": time.time()
        }
        
        if not self.initialized:
            return health
        
        try:
            # Check LLM manager
            llm_health = await self.llm_manager.health_check()
            health["llm"] = llm_health
            
            # Check vector store
            vector_stats = await self.vector_store.get_stats()
            health["vector_store"] = {
                "backend": vector_stats.get("backend"),
                "total_chunks": vector_stats.get("total_chunks", 0),
                "healthy": vector_stats.get("total_chunks", 0) >= 0
            }
            
            # Check embedding generator
            embedding_info = await self.embedding_generator.get_embedding_info()
            health["embeddings"] = {
                "model": embedding_info.get("model_name"),
                "dimensions": embedding_info.get("dimensions"),
                "cache_size": embedding_info.get("cache_size", 0),
                "healthy": True
            }
            
            # Overall health
            health["overall_healthy"] = (
                llm_health.get("initialized", False) and
                len(llm_health.get("healthy_providers", [])) > 0 and
                health["vector_store"]["healthy"] and
                health["embeddings"]["healthy"]
            )
            
        except Exception as e:
            health["error"] = str(e)
            health["overall_healthy"] = False
        
        return health