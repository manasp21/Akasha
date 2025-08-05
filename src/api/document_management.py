"""
Document Management API for Akasha RAG System.

This module implements document upload, CRUD operations, and bulk processing
as specified in Phase 2 requirements.
"""

import asyncio
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.exceptions import AkashaError
from ..core.job_queue import get_job_queue_manager, JobPriority
from ..rag.ingestion import DocumentIngestion, ChunkingConfig, ChunkingStrategy, DocumentFormat
from ..rag.embeddings import EmbeddingGenerator, EmbeddingConfig
from ..rag.storage import VectorStore, StorageConfig
from ..rag.hybrid_search import HybridSearchEngine, HybridSearchConfig, SearchQuery


# Request/Response Models
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    format: str = Field(..., description="Detected document format")
    job_id: Optional[str] = Field(None, description="Processing job ID if async")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str = Field(..., description="Document identifier")
    file_name: str = Field(..., description="File name")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="File hash")
    mime_type: str = Field(..., description="MIME type")
    format: str = Field(..., description="Document format")
    processed_at: float = Field(..., description="Processing timestamp")
    chunk_count: int = Field(..., description="Number of chunks")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentList(BaseModel):
    """List of documents with pagination."""
    documents: List[DocumentInfo] = Field(..., description="Documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class DocumentSearchRequest(BaseModel):
    """Document search request."""
    query: str = Field(..., description="Search query")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    top_k: int = Field(default=10, description="Number of results")
    enable_vector_search: bool = Field(default=True, description="Enable vector search")
    enable_keyword_search: bool = Field(default=True, description="Enable keyword search")


class DocumentSearchResponse(BaseModel):
    """Document search response."""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Search processing time")


class BulkOperationRequest(BaseModel):
    """Bulk operation request."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    operation: str = Field(..., description="Operation type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class BulkOperationResponse(BaseModel):
    """Bulk operation response."""
    operation: str = Field(..., description="Operation type")
    total_documents: int = Field(..., description="Total documents")
    successful: int = Field(..., description="Successful operations")
    failed: int = Field(..., description="Failed operations")
    job_ids: List[str] = Field(default_factory=list, description="Associated job IDs")
    errors: List[str] = Field(default_factory=list, description="Error messages")


# Router
router = APIRouter(prefix="/api/v1/documents", tags=["Document Management"])

# Global instances (would be dependency injected in production)
_document_storage = {}  # Document metadata storage
_upload_directory = Path("./uploads")
_upload_directory.mkdir(exist_ok=True)


# Dependencies
async def get_document_ingestion() -> DocumentIngestion:
    """Get document ingestion instance."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=500,
        chunk_overlap=50
    )
    return DocumentIngestion(config)


async def get_embedding_generator() -> EmbeddingGenerator:
    """Get embedding generator instance."""
    config = EmbeddingConfig()
    generator = EmbeddingGenerator(config)
    await generator.initialize()
    return generator


async def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    config = StorageConfig(collection_name="document_management")
    store = VectorStore(config)
    await store.initialize()
    return store


async def get_hybrid_search() -> HybridSearchEngine:
    """Get hybrid search engine instance."""
    vector_store = await get_vector_store()
    config = HybridSearchConfig()
    search_engine = HybridSearchEngine(vector_store, config)
    await search_engine.initialize()
    return search_engine


# Utility functions
def calculate_file_hash(file_path: Path) -> str:
    """Calculate file hash."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def detect_mime_type(file_path: Path) -> str:
    """Detect MIME type."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


# API Endpoints

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_processing: bool = Query(default=True, description="Process document asynchronously"),
    chunking_strategy: ChunkingStrategy = Query(default=ChunkingStrategy.RECURSIVE, description="Chunking strategy"),
    chunk_size: int = Query(default=500, description="Chunk size"),
    chunk_overlap: int = Query(default=50, description="Chunk overlap")
):
    """Upload and process a document."""
    logger = get_logger("document_upload")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Save uploaded file
        file_path = _upload_directory / f"{document_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Calculate file info
        file_size = len(content)
        file_hash = calculate_file_hash(file_path)
        mime_type = detect_mime_type(file_path)
        
        # Detect format
        if mime_type.startswith("application/pdf"):
            doc_format = DocumentFormat.PDF
        elif mime_type.startswith("text/"):
            doc_format = DocumentFormat.TEXT
        else:
            doc_format = DocumentFormat.OTHER
        
        logger.info(
            "Document uploaded",
            document_id=document_id,
            file_name=file.filename,
            file_size=file_size,
            mime_type=mime_type
        )
        
        if async_processing:
            # Submit processing job
            job_queue = get_job_queue_manager()
            job_id = job_queue.submit_job(
                "akasha.process_document",
                args=[str(file_path)],
                kwargs={
                    "config": {
                        "strategy": chunking_strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap
                    }
                },
                priority=JobPriority.NORMAL,
                queue="documents"
            )
            
            # Store document info
            _document_storage[document_id] = {
                "document_id": document_id,
                "file_name": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_hash": file_hash,
                "mime_type": mime_type,
                "format": doc_format.value,
                "uploaded_at": time.time(),
                "job_id": job_id,
                "status": "processing"
            }
            
            return DocumentUploadResponse(
                document_id=document_id,
                file_name=file.filename,
                file_size=file_size,
                mime_type=mime_type,
                format=doc_format.value,
                job_id=job_id,
                status="processing",
                message="Document uploaded and processing started"
            )
        else:
            # Process synchronously
            ingestion = await get_document_ingestion()
            metadata, chunks = await ingestion.process_file(file_path)
            
            # Generate embeddings
            embedding_generator = await get_embedding_generator()
            embedded_chunks = await embedding_generator.embed_chunks(chunks)
            
            # Store in vector database
            vector_store = await get_vector_store()
            await vector_store.add_document(metadata, embedded_chunks)
            
            # Store document info
            _document_storage[document_id] = {
                "document_id": document_id,
                "file_name": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_hash": file_hash,
                "mime_type": mime_type,
                "format": doc_format.value,
                "processed_at": time.time(),
                "chunk_count": len(chunks),
                "processing_time": metadata.processing_time,
                "status": "completed"
            }
            
            return DocumentUploadResponse(
                document_id=document_id,
                file_name=file.filename,
                file_size=file_size,
                mime_type=mime_type,
                format=doc_format.value,
                job_id=None,
                status="completed",
                message=f"Document processed successfully. {len(chunks)} chunks created."
            )
    
    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get document information by ID."""
    if document_id not in _document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = _document_storage[document_id]
    
    # Check job status if processing
    if doc_info.get("status") == "processing" and doc_info.get("job_id"):
        job_queue = get_job_queue_manager()
        job_result = job_queue.get_job_result(doc_info["job_id"])
        
        if job_result.status.value == "success":
            doc_info["status"] = "completed"
            doc_info["chunk_count"] = job_result.result.get("chunk_count", 0)
            doc_info["processing_time"] = job_result.completed_at - job_result.started_at if job_result.completed_at and job_result.started_at else 0
        elif job_result.status.value == "failure":
            doc_info["status"] = "failed"
            doc_info["error"] = job_result.error
    
    return DocumentInfo(**doc_info)


@router.get("/", response_model=DocumentList)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    size: int = Query(default=10, ge=1, le=100, description="Page size"),
    format_filter: Optional[str] = Query(default=None, description="Filter by format"),
    status_filter: Optional[str] = Query(default=None, description="Filter by status")
):
    """List documents with pagination and filters."""
    documents = list(_document_storage.values())
    
    # Apply filters
    if format_filter:
        documents = [doc for doc in documents if doc.get("format") == format_filter]
    
    if status_filter:
        documents = [doc for doc in documents if doc.get("status") == status_filter]
    
    # Sort by upload time (newest first)
    documents.sort(key=lambda x: x.get("uploaded_at", 0), reverse=True)
    
    # Pagination
    total = len(documents)
    start = (page - 1) * size
    end = start + size
    page_documents = documents[start:end]
    
    return DocumentList(
        documents=[DocumentInfo(**doc) for doc in page_documents],
        total=total,
        page=page,
        size=size,
        has_next=end < total,
        has_prev=page > 1
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    if document_id not in _document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_info = _document_storage[document_id]
    
    try:
        # Delete file
        file_path = Path(doc_info["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Delete from vector store
        vector_store = await get_vector_store()
        await vector_store.delete_document(document_id)
        
        # Remove from storage
        del _document_storage[document_id]
        
        return {"message": "Document deleted successfully", "document_id": document_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    search_engine: HybridSearchEngine = Depends(get_hybrid_search),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator)
):
    """Search documents using hybrid search."""
    start_time = time.time()
    
    try:
        search_query = SearchQuery(
            text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            enable_vector_search=request.enable_vector_search,
            enable_keyword_search=request.enable_keyword_search
        )
        
        results = await search_engine.search(search_query, embedding_generator)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document_id": result.chunk.document_id,
                "chunk_id": result.chunk.id,
                "content": result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content,
                "final_score": result.final_score,
                "vector_score": result.vector_score,
                "keyword_score": result.keyword_score,
                "metadata": result.chunk.metadata
            })
        
        processing_time = time.time() - start_time
        
        return DocumentSearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(results),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/bulk", response_model=BulkOperationResponse)
async def bulk_operations(request: BulkOperationRequest):
    """Perform bulk operations on documents."""
    logger = get_logger("bulk_operations")
    
    try:
        if request.operation == "delete":
            successful = 0
            failed = 0
            errors = []
            
            for document_id in request.document_ids:
                try:
                    if document_id in _document_storage:
                        doc_info = _document_storage[document_id]
                        
                        # Delete file
                        file_path = Path(doc_info["file_path"])
                        if file_path.exists():
                            file_path.unlink()
                        
                        # Delete from vector store
                        vector_store = await get_vector_store()
                        await vector_store.delete_document(document_id)
                        
                        # Remove from storage
                        del _document_storage[document_id]
                        successful += 1
                    else:
                        failed += 1
                        errors.append(f"Document {document_id} not found")
                
                except Exception as e:
                    failed += 1
                    errors.append(f"Failed to delete {document_id}: {str(e)}")
            
            return BulkOperationResponse(
                operation=request.operation,
                total_documents=len(request.document_ids),
                successful=successful,
                failed=failed,
                errors=errors
            )
        
        elif request.operation == "reprocess":
            job_queue = get_job_queue_manager()
            job_ids = []
            successful = 0
            failed = 0
            errors = []
            
            for document_id in request.document_ids:
                try:
                    if document_id in _document_storage:
                        doc_info = _document_storage[document_id]
                        file_path = doc_info["file_path"]
                        
                        # Submit reprocessing job
                        job_id = job_queue.submit_job(
                            "akasha.process_document",
                            args=[file_path],
                            kwargs={"config": request.parameters},
                            priority=JobPriority.NORMAL,
                            queue="documents"
                        )
                        
                        job_ids.append(job_id)
                        doc_info["status"] = "reprocessing"
                        doc_info["job_id"] = job_id
                        successful += 1
                    else:
                        failed += 1
                        errors.append(f"Document {document_id} not found")
                
                except Exception as e:
                    failed += 1
                    errors.append(f"Failed to reprocess {document_id}: {str(e)}")
            
            return BulkOperationResponse(
                operation=request.operation,
                total_documents=len(request.document_ids),
                successful=successful,
                failed=failed,
                job_ids=job_ids,
                errors=errors
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {request.operation}")
    
    except Exception as e:
        logger.error("Bulk operation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Bulk operation failed: {str(e)}")


@router.get("/stats/overview")
async def get_document_stats():
    """Get document statistics overview."""
    documents = list(_document_storage.values())
    
    total_documents = len(documents)
    total_size = sum(doc.get("file_size", 0) for doc in documents)
    
    # Count by format
    format_counts = {}
    for doc in documents:
        doc_format = doc.get("format", "unknown")
        format_counts[doc_format] = format_counts.get(doc_format, 0) + 1
    
    # Count by status
    status_counts = {}
    for doc in documents:
        status = doc.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Recent uploads (last 24 hours)
    recent_cutoff = time.time() - (24 * 60 * 60)
    recent_uploads = len([doc for doc in documents if doc.get("uploaded_at", 0) > recent_cutoff])
    
    return {
        "total_documents": total_documents,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "format_distribution": format_counts,
        "status_distribution": status_counts,
        "recent_uploads_24h": recent_uploads,
        "avg_document_size": round(total_size / total_documents, 2) if total_documents > 0 else 0
    }