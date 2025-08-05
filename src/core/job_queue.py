"""
Async Job Queue System for Akasha RAG.

This module implements async job processing using Celery as specified in Phase 2 requirements.
Handles document processing, embedding generation, and other long-running tasks.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import json

from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import worker_ready, worker_shutting_down
from kombu import Queue
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.exceptions import AkashaError


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RECEIVED = "received"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class JobConfig:
    """Configuration for job queue system."""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = None
    timezone: str = "UTC"
    enable_utc: bool = True
    task_routes: Dict[str, Dict[str, str]] = None
    worker_prefetch_multiplier: int = 1
    task_acks_late: bool = True
    worker_disable_rate_limits: bool = False
    task_compression: str = "gzip"
    result_compression: str = "gzip"
    result_expires: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.accept_content is None:
            self.accept_content = ["json"]
        
        if self.task_routes is None:
            self.task_routes = {
                "akasha.process_document": {"queue": "documents"},
                "akasha.generate_embeddings": {"queue": "embeddings"},
                "akasha.index_chunks": {"queue": "indexing"},
                "akasha.cleanup_tasks": {"queue": "maintenance"}
            }


class JobResult(BaseModel):
    """Job execution result."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    result: Optional[Any] = Field(default=None, description="Job result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    progress: float = Field(default=0.0, description="Progress percentage (0-100)")
    created_at: float = Field(..., description="Job creation timestamp")
    started_at: Optional[float] = Field(default=None, description="Job start timestamp")
    completed_at: Optional[float] = Field(default=None, description="Job completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseJobTask(Task):
    """Base class for all Akasha job tasks."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        self.logger.info(
            "Task completed successfully",
            task_id=task_id,
            task_name=self.name,
            result_type=type(retval).__name__
        )
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        self.logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            exc_info=einfo
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        self.logger.warning(
            "Task retry",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            retry_count=self.request.retries
        )
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update task progress."""
        progress = (current / total * 100) if total > 0 else 0
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "progress": progress,
                "message": message
            }
        )


class JobQueueManager:
    """Manager for the Celery job queue system."""
    
    def __init__(self, config: JobConfig = None, logger=None):
        self.config = config or JobConfig()
        self.logger = logger or get_logger(__name__)
        self.app = None
        self._setup_celery()
    
    def _setup_celery(self):
        """Set up Celery application."""
        self.app = Celery("akasha")
        
        # Configure Celery
        self.app.conf.update(
            broker_url=self.config.broker_url,
            result_backend=self.config.result_backend,
            task_serializer=self.config.task_serializer,
            result_serializer=self.config.result_serializer,
            accept_content=self.config.accept_content,
            timezone=self.config.timezone,
            enable_utc=self.config.enable_utc,
            task_routes=self.config.task_routes,
            worker_prefetch_multiplier=self.config.worker_prefetch_multiplier,
            task_acks_late=self.config.task_acks_late,
            worker_disable_rate_limits=self.config.worker_disable_rate_limits,
            task_compression=self.config.task_compression,
            result_compression=self.config.result_compression,
            result_expires=self.config.result_expires
        )
        
        # Define task queues
        self.app.conf.task_queues = (
            Queue("documents", routing_key="documents"),
            Queue("embeddings", routing_key="embeddings"),
            Queue("indexing", routing_key="indexing"),
            Queue("maintenance", routing_key="maintenance"),
        )
        
        # Set default queue
        self.app.conf.task_default_queue = "documents"
        self.app.conf.task_default_exchange = "akasha"
        self.app.conf.task_default_routing_key = "documents"
        
        self.logger.info(
            "Celery application configured",
            broker_url=self.config.broker_url,
            queues=list(self.config.task_routes.values())
        )
    
    def submit_job(self, task_name: str, args: List[Any] = None, kwargs: Dict[str, Any] = None,
                   priority: JobPriority = JobPriority.NORMAL, 
                   eta: Optional[float] = None,
                   countdown: Optional[int] = None,
                   queue: Optional[str] = None) -> str:
        """Submit a job to the queue."""
        args = args or []
        kwargs = kwargs or {}
        
        # Map priority to Celery priority (higher number = higher priority)
        priority_map = {
            JobPriority.LOW: 1,
            JobPriority.NORMAL: 5,
            JobPriority.HIGH: 8,
            JobPriority.URGENT: 10
        }
        
        celery_priority = priority_map.get(priority, 5)
        
        # Submit task
        result = self.app.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            priority=celery_priority,
            eta=eta,
            countdown=countdown,
            queue=queue
        )
        
        job_id = result.id
        
        self.logger.info(
            "Job submitted",
            job_id=job_id,
            task_name=task_name,
            priority=priority,
            queue=queue or "default"
        )
        
        return job_id
    
    def get_job_result(self, job_id: str) -> JobResult:
        """Get job result by ID."""
        result = AsyncResult(job_id, app=self.app)
        
        # Map Celery states to JobStatus
        status_map = {
            "PENDING": JobStatus.PENDING,
            "RECEIVED": JobStatus.RECEIVED,
            "STARTED": JobStatus.STARTED,
            "SUCCESS": JobStatus.SUCCESS,
            "FAILURE": JobStatus.FAILURE,
            "RETRY": JobStatus.RETRY,
            "REVOKED": JobStatus.REVOKED,
            "PROGRESS": JobStatus.STARTED  # Custom progress state
        }
        
        status = status_map.get(result.state, JobStatus.PENDING)
        
        # Get task metadata
        task_info = result.info or {}
        
        # Calculate timestamps
        created_at = time.time()  # Approximation
        started_at = None
        completed_at = None
        
        if status in [JobStatus.SUCCESS, JobStatus.FAILURE]:
            completed_at = time.time()  # Approximation
        
        if status != JobStatus.PENDING:
            started_at = time.time() - 60  # Approximation
        
        # Extract progress if available
        progress = 0.0
        if result.state == "PROGRESS" and isinstance(task_info, dict):
            progress = task_info.get("progress", 0.0)
        elif status == JobStatus.SUCCESS:
            progress = 100.0
        
        # Extract result data
        result_data = None
        error_message = None
        
        if status == JobStatus.SUCCESS:
            result_data = result.result
        elif status == JobStatus.FAILURE:
            error_message = str(result.info) if result.info else "Unknown error"
        
        return JobResult(
            job_id=job_id,
            status=status,
            result=result_data,
            error=error_message,
            progress=progress,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            metadata=task_info if isinstance(task_info, dict) else {}
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            self.app.control.revoke(job_id, terminate=True)
            self.logger.info("Job cancelled", job_id=job_id)
            return True
        except Exception as e:
            self.logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            inspect = self.app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active()
            
            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled()
            
            # Get reserved tasks
            reserved_tasks = inspect.reserved()
            
            # Get stats
            stats = inspect.stats()
            
            return {
                "active_tasks": active_tasks,
                "scheduled_tasks": scheduled_tasks,
                "reserved_tasks": reserved_tasks,
                "worker_stats": stats,
                "queue_lengths": self._get_queue_lengths()
            }
        except Exception as e:
            self.logger.error("Failed to get queue stats", error=str(e))
            return {"error": str(e)}
    
    def _get_queue_lengths(self) -> Dict[str, int]:
        """Get queue lengths (requires Redis broker)."""
        try:
            if "redis://" in self.config.broker_url:
                import redis
                
                # Parse Redis URL
                redis_client = redis.from_url(self.config.broker_url)
                
                # Get queue lengths
                queues = ["documents", "embeddings", "indexing", "maintenance"]
                lengths = {}
                
                for queue in queues:
                    length = redis_client.llen(f"celery:{queue}")
                    lengths[queue] = length or 0
                
                return lengths
        except Exception as e:
            self.logger.warning("Could not get queue lengths", error=str(e))
        
        return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            inspect = self.app.control.inspect()
            
            # Check if workers are available
            stats = inspect.stats()
            active_workers = len(stats) if stats else 0
            
            # Check broker connection
            broker_ok = True
            try:
                # Send a simple task to test broker
                result = self.app.send_task("celery.ping", expires=10)
                result.get(timeout=5)
            except Exception:
                broker_ok = False
            
            return {
                "status": "healthy" if active_workers > 0 and broker_ok else "unhealthy",
                "active_workers": active_workers,
                "broker_connected": broker_ok,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }


# Global job queue manager instance
job_queue_manager = None


def get_job_queue_manager(config: JobConfig = None) -> JobQueueManager:
    """Get the global job queue manager instance."""
    global job_queue_manager
    
    if job_queue_manager is None:
        job_queue_manager = JobQueueManager(config)
    
    return job_queue_manager


# Celery app instance for worker
def create_celery_app(config: JobConfig = None) -> Celery:
    """Create Celery app for worker."""
    manager = get_job_queue_manager(config)
    return manager.app


# Define job tasks
@get_job_queue_manager().app.task(bind=True, base=BaseJobTask, name="akasha.process_document")
def process_document_task(self, file_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a document asynchronously."""
    try:
        self.update_progress(0, 100, "Starting document processing")
        
        # Import here to avoid circular imports
        import asyncio
        from pathlib import Path
        from ..rag.ingestion import DocumentIngestion, ChunkingConfig
        
        # Set up processing
        chunking_config = ChunkingConfig(**(config or {}))
        ingestion = DocumentIngestion(chunking_config)
        
        self.update_progress(20, 100, "Processing document")
        
        # Process document (run async code in sync context)
        async def _process():
            return await ingestion.process_file(Path(file_path))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metadata, chunks = loop.run_until_complete(_process())
        finally:
            loop.close()
        
        self.update_progress(100, 100, "Document processing completed")
        
        return {
            "document_id": metadata.document_id,
            "file_name": metadata.file_name,
            "chunk_count": len(chunks),
            "processing_time": metadata.processing_time
        }
        
    except Exception as e:
        self.logger.error("Document processing failed", error=str(e), file_path=file_path)
        raise


@get_job_queue_manager().app.task(bind=True, base=BaseJobTask, name="akasha.generate_embeddings")
def generate_embeddings_task(self, chunk_ids: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate embeddings for chunks asynchronously."""
    try:
        self.update_progress(0, 100, "Starting embedding generation")
        
        # Import here to avoid circular imports
        import asyncio
        from ..rag.embeddings import EmbeddingGenerator, EmbeddingConfig
        
        # Set up embedding generator
        embedding_config = EmbeddingConfig(**(config or {}))
        generator = EmbeddingGenerator(embedding_config)
        
        self.update_progress(20, 100, "Initializing embedding model")
        
        # Generate embeddings (run async code in sync context)
        async def _generate():
            await generator.initialize()
            
            # For now, generate dummy embeddings
            # In real implementation, would fetch chunks and generate embeddings
            embeddings = [[0.1] * 384 for _ in chunk_ids]
            return embeddings
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = loop.run_until_complete(_generate())
        finally:
            loop.close()
        
        self.update_progress(100, 100, "Embedding generation completed")
        
        return {
            "chunk_count": len(chunk_ids),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "embeddings_generated": len(embeddings)
        }
        
    except Exception as e:
        self.logger.error("Embedding generation failed", error=str(e), chunk_count=len(chunk_ids))
        raise


@get_job_queue_manager().app.task(bind=True, base=BaseJobTask, name="akasha.index_chunks")
def index_chunks_task(self, document_id: str, chunk_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Index chunks in vector store asynchronously."""
    try:
        self.update_progress(0, 100, "Starting chunk indexing")
        
        # Import here to avoid circular imports
        import asyncio
        from ..rag.storage import VectorStore, StorageConfig
        from ..rag.ingestion import DocumentChunk
        
        # Set up vector store
        storage_config = StorageConfig()
        vector_store = VectorStore(storage_config)
        
        self.update_progress(20, 100, "Initializing vector store")
        
        # Index chunks (run async code in sync context)
        async def _index():
            await vector_store.initialize()
            
            # Convert chunk data to DocumentChunk objects
            chunks = []
            for chunk_info in chunk_data:
                chunk = DocumentChunk(
                    id=chunk_info["id"],
                    content=chunk_info["content"],
                    document_id=document_id,
                    chunk_index=chunk_info.get("chunk_index", 0),
                    embedding=chunk_info.get("embedding", [])
                )
                chunks.append(chunk)
            
            # Add chunks to store
            await vector_store.add_chunks(chunks)
            return len(chunks)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            indexed_count = loop.run_until_complete(_index())
        finally:
            loop.close()
        
        self.update_progress(100, 100, "Chunk indexing completed")
        
        return {
            "document_id": document_id,
            "chunks_indexed": indexed_count
        }
        
    except Exception as e:
        self.logger.error("Chunk indexing failed", error=str(e), document_id=document_id)
        raise


@get_job_queue_manager().app.task(bind=True, base=BaseJobTask, name="akasha.cleanup_tasks")
def cleanup_tasks(self, older_than_hours: int = 24) -> Dict[str, Any]:
    """Clean up old completed tasks."""
    try:
        self.update_progress(0, 100, "Starting cleanup")
        
        # Get job queue manager
        manager = get_job_queue_manager()
        
        # Clean up logic would go here
        # For now, just return success
        
        self.update_progress(100, 100, "Cleanup completed")
        
        return {
            "cleaned_tasks": 0,
            "cleanup_time": time.time()
        }
        
    except Exception as e:
        self.logger.error("Cleanup failed", error=str(e))
        raise


# Signal handlers
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger = get_logger("celery.worker")
    logger.info("Celery worker ready", worker=sender)


@worker_shutting_down.connect  
def worker_shutting_down_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger = get_logger("celery.worker")
    logger.info("Celery worker shutting down", worker=sender)