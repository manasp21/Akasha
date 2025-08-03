# Akasha Component Specifications

## Table of Contents
1. [Ingestion Engine](#1-ingestion-engine)
2. [Embedding Service](#2-embedding-service)
3. [Vector Store](#3-vector-store)
4. [RAG Engine](#4-rag-engine)
5. [LLM Service](#5-llm-service)
6. [Knowledge Graph](#6-knowledge-graph)
7. [Cache Manager](#7-cache-manager)
8. [Plugin Manager](#8-plugin-manager)
9. [Configuration Manager](#9-configuration-manager)
10. [API Gateway](#10-api-gateway)

---

## 1. Ingestion Engine

### 1.1 Overview
The Ingestion Engine is responsible for converting raw documents (primarily PDFs) into structured, searchable content using MinerU 2 and complementary processing tools.

### 1.2 Core Components

#### 1.2.1 MinerU 2 Integration
```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio

@dataclass
class MinerUConfig:
    model_path: str = "mineru-models/"
    batch_size: int = 4
    max_memory_gb: int = 8
    enable_ocr: bool = True
    output_format: str = "markdown"  # markdown, json, hybrid
    extract_images: bool = True
    extract_tables: bool = True
    extract_formulas: bool = True

@dataclass
class DocumentMetadata:
    title: Optional[str] = None
    authors: List[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    processing_timestamp: str = None

@dataclass
class ContentSegment:
    id: str
    type: str  # text, image, table, formula, caption, header, footer
    content: Union[str, bytes]
    metadata: Dict
    page_number: int
    bounding_box: Optional[Dict] = None  # x, y, width, height
    parent_section: Optional[str] = None
    order_index: int = 0

@dataclass
class ProcessedDocument:
    document_id: str
    original_filename: str
    metadata: DocumentMetadata
    segments: List[ContentSegment]
    extracted_images: List[Dict]
    processing_stats: Dict
    error_log: List[str]

class MinerUProcessor:
    def __init__(self, config: MinerUConfig):
        self.config = config
        self._initialize_models()
    
    async def process_pdf(self, pdf_path: Path) -> ProcessedDocument:
        """
        Process a PDF document using MinerU 2
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted content and metadata
            
        Raises:
            ProcessingError: If document cannot be processed
            InvalidFormatError: If file format is not supported
        """
        try:
            # MinerU 2 processing pipeline
            raw_result = await self._extract_with_mineru(pdf_path)
            segments = await self._segment_content(raw_result)
            metadata = await self._extract_metadata(pdf_path, raw_result)
            images = await self._extract_images(raw_result)
            
            return ProcessedDocument(
                document_id=self._generate_id(pdf_path),
                original_filename=pdf_path.name,
                metadata=metadata,
                segments=segments,
                extracted_images=images,
                processing_stats=self._get_stats(),
                error_log=[]
            )
        except Exception as e:
            self._handle_error(e, pdf_path)
            raise
    
    async def _extract_with_mineru(self, pdf_path: Path) -> Dict:
        """Execute MinerU 2 extraction"""
        command = [
            "magic-pdf",
            "-p", str(pdf_path),
            "-o", str(self.config.output_dir),
            "-m", "auto"
        ]
        # Execute and parse results
        pass
    
    async def _segment_content(self, raw_result: Dict) -> List[ContentSegment]:
        """Segment extracted content into logical units"""
        segments = []
        
        # Process text blocks
        for text_block in raw_result.get("text_blocks", []):
            segment = ContentSegment(
                id=self._generate_segment_id(),
                type="text",
                content=text_block["content"],
                metadata=text_block.get("metadata", {}),
                page_number=text_block["page"],
                bounding_box=text_block.get("bbox"),
                order_index=text_block.get("order", 0)
            )
            segments.append(segment)
        
        # Process images
        for image in raw_result.get("images", []):
            segment = ContentSegment(
                id=self._generate_segment_id(),
                type="image",
                content=image["data"],
                metadata={
                    "format": image.get("format"),
                    "caption": image.get("caption"),
                    "alt_text": image.get("alt_text")
                },
                page_number=image["page"],
                bounding_box=image.get("bbox"),
                order_index=image.get("order", 0)
            )
            segments.append(segment)
        
        # Process tables
        for table in raw_result.get("tables", []):
            segment = ContentSegment(
                id=self._generate_segment_id(),
                type="table",
                content=table["content"],  # Structured table data
                metadata={
                    "headers": table.get("headers"),
                    "format": "structured",
                    "caption": table.get("caption")
                },
                page_number=table["page"],
                bounding_box=table.get("bbox"),
                order_index=table.get("order", 0)
            )
            segments.append(segment)
        
        return sorted(segments, key=lambda x: (x.page_number, x.order_index))
```

#### 1.2.2 Content Classification
```python
class ContentClassifier:
    """Classify and tag document segments"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._load_classification_models()
    
    async def classify_segment(self, segment: ContentSegment) -> ContentSegment:
        """
        Classify content type and add semantic tags
        
        Args:
            segment: Content segment to classify
            
        Returns:
            Enhanced segment with classification metadata
        """
        if segment.type == "text":
            return await self._classify_text(segment)
        elif segment.type == "image":
            return await self._classify_image(segment)
        elif segment.type == "table":
            return await self._classify_table(segment)
        else:
            return segment
    
    async def _classify_text(self, segment: ContentSegment) -> ContentSegment:
        """Classify text segments (abstract, body, conclusion, etc.)"""
        classification = await self._text_classifier.predict(segment.content)
        
        segment.metadata.update({
            "section_type": classification.section_type,
            "confidence": classification.confidence,
            "contains_citations": classification.has_citations,
            "contains_formulas": classification.has_formulas,
            "academic_level": classification.academic_level
        })
        
        return segment
    
    async def _classify_image(self, segment: ContentSegment) -> ContentSegment:
        """Classify image type (figure, chart, diagram, etc.)"""
        image_type = await self._image_classifier.predict(segment.content)
        
        segment.metadata.update({
            "image_type": image_type.category,
            "contains_text": image_type.has_text,
            "is_chart": image_type.is_chart,
            "is_diagram": image_type.is_diagram,
            "visual_complexity": image_type.complexity_score
        })
        
        return segment
```

### 1.3 Configuration Schema
```yaml
ingestion:
  mineru:
    model_path: "./models/mineru"
    batch_size: 4
    max_memory_gb: 8
    enable_ocr: true
    ocr_languages: ["en", "fr", "de"]
    output_format: "hybrid"
    
  processing:
    max_file_size_mb: 100
    timeout_seconds: 300
    retry_attempts: 3
    
  classification:
    enable_content_classification: true
    text_classifier_model: "academic-bert"
    image_classifier_model: "vit-academic"
    
  fallback:
    enable_ocr_fallback: true
    ocr_engine: "paddleocr"  # tesseract, paddleocr, easyocr
    
  quality_control:
    min_confidence_threshold: 0.7
    enable_manual_review: false
    suspicious_content_patterns: []
```

### 1.4 Error Handling
```python
class IngestionError(Exception):
    """Base exception for ingestion errors"""
    pass

class ProcessingError(IngestionError):
    """Document processing failed"""
    pass

class InvalidFormatError(IngestionError):
    """Unsupported file format"""
    pass

class ExtractionTimeoutError(IngestionError):
    """Processing timeout exceeded"""
    pass

class ErrorHandler:
    def __init__(self, config: Dict):
        self.config = config
        self.retry_policy = RetryPolicy(config.get("retry", {}))
    
    async def handle_processing_error(self, error: Exception, document_path: Path) -> Optional[ProcessedDocument]:
        """Handle processing errors with fallback strategies"""
        if isinstance(error, ExtractionTimeoutError):
            return await self._handle_timeout(document_path)
        elif isinstance(error, InvalidFormatError):
            return await self._handle_invalid_format(document_path)
        else:
            return await self._handle_generic_error(error, document_path)
```

---

## 2. Embedding Service

### 2.1 Overview
The Embedding Service generates multimodal embeddings using JINA v4, supporting both text and image content in a unified vector space.

### 2.2 Core Components

#### 2.2.1 JINA v4 Integration
```python
from typing import Union, List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

@dataclass
class EmbeddingConfig:
    model_name: str = "jinaai/jina-embeddings-v4"
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 32
    max_length: int = 8192
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    precision: str = "float16"  # float32, float16, bfloat16

@dataclass
class EmbeddingRequest:
    content: Union[str, bytes, List[Union[str, bytes]]]
    content_type: str  # text, image, multimodal
    metadata: Dict = None
    cache_key: Optional[str] = None

@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    dimensions: int
    content_type: str
    processing_time: float
    metadata: Dict
    cache_hit: bool = False

class JINAv4Embedder:
    """JINA v4 multimodal embedding model"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._load_model()
        self._setup_device()
        
    def _load_model(self):
        """Load JINA v4 model and tokenizer"""
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=self._get_torch_dtype()
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        
    async def embed_multimodal(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Generate embeddings for multimodal content
        
        Args:
            request: Embedding request with content and metadata
            
        Returns:
            EmbeddingResult with vectors and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if request.cache_key and self.config.cache_embeddings:
            cached_result = await self._check_cache(request.cache_key)
            if cached_result:
                return cached_result
        
        # Process based on content type
        if request.content_type == "text":
            vectors = await self._embed_text(request.content)
        elif request.content_type == "image":
            vectors = await self._embed_image(request.content)
        elif request.content_type == "multimodal":
            vectors = await self._embed_multimodal_content(request.content)
        else:
            raise ValueError(f"Unsupported content type: {request.content_type}")
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            vectors = self._normalize_vectors(vectors)
        
        processing_time = time.time() - start_time
        
        result = EmbeddingResult(
            vectors=vectors,
            dimensions=vectors.shape[-1],
            content_type=request.content_type,
            processing_time=processing_time,
            metadata=request.metadata or {}
        )
        
        # Cache result
        if request.cache_key and self.config.cache_embeddings:
            await self._cache_result(request.cache_key, result)
        
        return result
    
    async def _embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Embed text content"""
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        return embeddings.cpu().numpy()
    
    async def _embed_image(self, image: Union[bytes, List[bytes]]) -> np.ndarray:
        """Embed image content"""
        if isinstance(image, bytes):
            image = [image]
        
        # Process images
        images = [self._process_image(img) for img in image]
        
        # Generate embeddings using vision encoder
        with torch.no_grad():
            embeddings = self.model.encode_images(images)
        
        return embeddings.cpu().numpy()
    
    async def _embed_multimodal_content(self, content: List[Union[str, bytes]]) -> np.ndarray:
        """Embed mixed text and image content"""
        text_content = [item for item in content if isinstance(item, str)]
        image_content = [item for item in content if isinstance(item, bytes)]
        
        embeddings = []
        
        if text_content:
            text_embeddings = await self._embed_text(text_content)
            embeddings.append(text_embeddings)
        
        if image_content:
            image_embeddings = await self._embed_image(image_content)
            embeddings.append(image_embeddings)
        
        # Combine embeddings (could be concatenation, averaging, or learned fusion)
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return self._fuse_multimodal_embeddings(embeddings)
```

#### 2.2.2 Batch Processing Engine
```python
class BatchProcessor:
    """Efficient batch processing for embeddings"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.queue = asyncio.Queue(maxsize=config.get("queue_size", 1000))
        self.workers = []
        self._start_workers()
    
    async def process_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResult]:
        """Process multiple embedding requests efficiently"""
        
        # Group by content type for optimal batching
        text_requests = [req for req in requests if req.content_type == "text"]
        image_requests = [req for req in requests if req.content_type == "image"]
        multimodal_requests = [req for req in requests if req.content_type == "multimodal"]
        
        results = []
        
        # Process each type in optimized batches
        if text_requests:
            text_results = await self._process_text_batch(text_requests)
            results.extend(text_results)
        
        if image_requests:
            image_results = await self._process_image_batch(image_requests)
            results.extend(image_results)
        
        if multimodal_requests:
            multimodal_results = await self._process_multimodal_batch(multimodal_requests)
            results.extend(multimodal_results)
        
        # Maintain original order
        return self._reorder_results(results, requests)
```

### 2.3 Configuration Schema
```yaml
embedding:
  model:
    name: "jinaai/jina-embeddings-v4"
    device: "auto"
    precision: "float16"
    max_length: 8192
    
  processing:
    batch_size: 32
    max_queue_size: 1000
    worker_count: 4
    normalize_embeddings: true
    
  caching:
    enabled: true
    backend: "redis"  # redis, memory, disk
    ttl_seconds: 3600
    max_cache_size_gb: 10
    
  optimization:
    enable_model_compilation: true
    use_flash_attention: true
    enable_gradient_checkpointing: false
    
  fallback:
    enable_cpu_fallback: true
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 3. Vector Store

### 3.1 Overview
The Vector Store manages vector storage, indexing, and similarity search with support for both ChromaDB and Qdrant backends.

### 3.2 Core Components

#### 3.2.1 Unified Vector Store Interface
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

@dataclass
class VectorStoreConfig:
    backend: str  # chroma, qdrant
    collection_name: str
    dimension: int
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    index_type: str = "hnsw"
    storage_path: str = "./vector_store"

@dataclass
class VectorMetadata:
    document_id: str
    segment_id: str
    content_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    timestamp: str = None
    custom_fields: Dict = None

@dataclass
class VectorDocument:
    id: str
    vector: np.ndarray
    metadata: VectorMetadata
    content: Optional[str] = None  # Original content for reference

@dataclass
class SearchResult:
    document: VectorDocument
    score: float
    rank: int

@dataclass
class SearchQuery:
    vector: np.ndarray
    filters: Optional[Dict] = None
    limit: int = 10
    threshold: float = 0.0
    include_metadata: bool = True
    include_content: bool = False

class VectorStoreInterface(ABC):
    """Abstract interface for vector stores"""
    
    @abstractmethod
    async def create_collection(self, name: str, config: VectorStoreConfig) -> bool:
        """Create a new collection"""
        pass
    
    @abstractmethod
    async def insert_vectors(self, documents: List[VectorDocument]) -> bool:
        """Insert vectors into the collection"""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def update_vector(self, document_id: str, vector: np.ndarray, metadata: VectorMetadata) -> bool:
        """Update an existing vector"""
        pass
    
    @abstractmethod
    async def delete_vectors(self, document_ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        pass

class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._initialize_client()
    
    def _initialize_client(self):
        import chromadb
        from chromadb.config import Settings
        
        self.client = chromadb.PersistentClient(
            path=self.config.storage_path,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
    
    async def insert_vectors(self, documents: List[VectorDocument]) -> bool:
        """Insert vectors into ChromaDB"""
        try:
            ids = [doc.id for doc in documents]
            embeddings = [doc.vector.tolist() for doc in documents]
            metadatas = [self._serialize_metadata(doc.metadata) for doc in documents]
            documents_content = [doc.content or "" for doc in documents]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            
            return True
        except Exception as e:
            self._handle_error(e, "insert_vectors")
            return False
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search vectors in ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query.vector.tolist()],
                n_results=query.limit,
                where=query.filters,
                include=["embeddings", "metadatas", "documents", "distances"]
            )
            
            search_results = []
            for i, (distance, metadata, document) in enumerate(zip(
                results["distances"][0],
                results["metadatas"][0],
                results["documents"][0]
            )):
                score = 1 - distance  # Convert distance to similarity score
                if score >= query.threshold:
                    vector_doc = VectorDocument(
                        id=results["ids"][0][i],
                        vector=np.array(results["embeddings"][0][i]),
                        metadata=self._deserialize_metadata(metadata),
                        content=document if query.include_content else None
                    )
                    
                    search_results.append(SearchResult(
                        document=vector_doc,
                        score=score,
                        rank=i + 1
                    ))
            
            return search_results
            
        except Exception as e:
            self._handle_error(e, "search")
            return []

class QdrantVectorStore(VectorStoreInterface):
    """Qdrant implementation for production use"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._initialize_client()
    
    def _initialize_client(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(path=self.config.storage_path)
        
        # Create collection if it doesn't exist
        try:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension,
                    distance=self._get_qdrant_distance(self.config.distance_metric)
                )
            )
        except Exception:
            pass  # Collection already exists
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search vectors in Qdrant"""
        from qdrant_client.models import Filter
        
        try:
            # Convert filters to Qdrant format
            qdrant_filter = self._convert_filters(query.filters) if query.filters else None
            
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query.vector.tolist(),
                limit=query.limit,
                score_threshold=query.threshold,
                query_filter=qdrant_filter,
                with_payload=query.include_metadata,
                with_vectors=query.include_content
            )
            
            search_results = []
            for i, result in enumerate(results):
                vector_doc = VectorDocument(
                    id=result.id,
                    vector=np.array(result.vector) if result.vector else None,
                    metadata=self._deserialize_metadata(result.payload),
                    content=result.payload.get("content") if query.include_content else None
                )
                
                search_results.append(SearchResult(
                    document=vector_doc,
                    score=result.score,
                    rank=i + 1
                ))
            
            return search_results
            
        except Exception as e:
            self._handle_error(e, "search")
            return []
```

#### 3.2.2 Hybrid Search Engine
```python
class HybridSearchEngine:
    """Combines semantic and keyword search"""
    
    def __init__(self, vector_store: VectorStoreInterface, config: Dict):
        self.vector_store = vector_store
        self.config = config
        self._setup_keyword_search()
    
    def _setup_keyword_search(self):
        """Initialize keyword search index"""
        from whoosh import index
        from whoosh.fields import Schema, TEXT, ID, STORED
        
        schema = Schema(
            id=ID(stored=True),
            content=TEXT(stored=True),
            metadata=STORED()
        )
        
        self.keyword_index = index.create_in(
            self.config["keyword_index_path"],
            schema
        )
    
    async def hybrid_search(self, 
                          semantic_query: np.ndarray,
                          keyword_query: str,
                          filters: Dict = None,
                          limit: int = 10,
                          semantic_weight: float = 0.7) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword results
        
        Args:
            semantic_query: Vector representation of query
            keyword_query: Text query for keyword search
            filters: Metadata filters
            limit: Maximum results to return
            semantic_weight: Weight for semantic vs keyword results (0-1)
            
        Returns:
            Combined and reranked search results
        """
        
        # Perform semantic search
        semantic_results = await self.vector_store.search(SearchQuery(
            vector=semantic_query,
            filters=filters,
            limit=limit * 2,  # Get more for better fusion
            include_metadata=True
        ))
        
        # Perform keyword search
        keyword_results = await self._keyword_search(keyword_query, filters, limit * 2)
        
        # Fuse results using RRF (Reciprocal Rank Fusion)
        fused_results = self._fuse_results(
            semantic_results,
            keyword_results,
            semantic_weight
        )
        
        return fused_results[:limit]
    
    def _fuse_results(self, 
                     semantic_results: List[SearchResult],
                     keyword_results: List[SearchResult],
                     semantic_weight: float) -> List[SearchResult]:
        """Fuse semantic and keyword results using RRF"""
        
        # Create score dictionaries
        semantic_scores = {result.document.id: result.score for result in semantic_results}
        keyword_scores = {result.document.id: result.score for result in keyword_results}
        
        # Get all unique document IDs
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        # Calculate combined scores using RRF
        k = 60  # RRF parameter
        combined_results = []
        
        for doc_id in all_ids:
            semantic_rank = next((i + 1 for i, r in enumerate(semantic_results) 
                                if r.document.id == doc_id), float('inf'))
            keyword_rank = next((i + 1 for i, r in enumerate(keyword_results) 
                               if r.document.id == doc_id), float('inf'))
            
            # RRF score
            rrf_score = (semantic_weight / (k + semantic_rank) + 
                        (1 - semantic_weight) / (k + keyword_rank))
            
            # Find the document from either result set
            document = next((r.document for r in semantic_results if r.document.id == doc_id), 
                          next((r.document for r in keyword_results if r.document.id == doc_id), None))
            
            if document:
                combined_results.append(SearchResult(
                    document=document,
                    score=rrf_score,
                    rank=0  # Will be set after sorting
                ))
        
        # Sort by combined score and set ranks
        combined_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
```

### 3.3 Configuration Schema
```yaml
vector_store:
  backend: "chroma"  # chroma, qdrant
  
  chroma:
    storage_path: "./data/chroma"
    collection_name: "akasha_documents"
    distance_metric: "cosine"
    
  qdrant:
    storage_path: "./data/qdrant"
    collection_name: "akasha_documents"
    distance_metric: "cosine"
    index_type: "hnsw"
    hnsw_config:
      m: 16
      ef_construct: 100
      
  hybrid_search:
    enabled: true
    keyword_index_path: "./data/keyword_index"
    semantic_weight: 0.7
    enable_reranking: true
    reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
  performance:
    batch_insert_size: 1000
    search_timeout_seconds: 30
    connection_pool_size: 10
```

---

## 4. RAG Engine

### 4.1 Overview
The RAG Engine implements state-of-the-art retrieval techniques including GraphRAG, Self-RAG, and advanced context management.

### 4.2 Core Components

#### 4.2.1 Multi-Stage Retriever
```python
@dataclass
class RetrievalConfig:
    stages: List[str] = None  # ["coarse", "fine", "rerank"]
    coarse_limit: int = 100
    fine_limit: int = 20
    final_limit: int = 5
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    context_window_size: int = 4096

@dataclass
class RetrievalContext:
    query: str
    query_vector: np.ndarray
    filters: Dict = None
    conversation_history: List[Dict] = None
    user_preferences: Dict = None

@dataclass
class RetrievalResult:
    contexts: List[Dict]
    sources: List[Dict]
    confidence_score: float
    retrieval_stats: Dict
    graph_paths: List[Dict] = None

class MultiStageRetriever:
    """Advanced multi-stage retrieval pipeline"""
    
    def __init__(self, 
                 vector_store: VectorStoreInterface,
                 config: RetrievalConfig):
        self.vector_store = vector_store
        self.config = config
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize retrieval components"""
        self.query_expander = QueryExpander(self.config.query_expansion)
        self.reranker = CrossEncoderReranker(self.config.reranking)
        self.context_manager = ContextManager(self.config.context)
        
    async def retrieve(self, context: RetrievalContext) -> RetrievalResult:
        """
        Perform multi-stage retrieval
        
        Args:
            context: Retrieval context with query and metadata
            
        Returns:
            RetrievalResult with ranked contexts and sources
        """
        
        # Stage 1: Query expansion and coarse retrieval
        expanded_queries = await self._expand_query(context.query)
        coarse_results = await self._coarse_retrieval(
            expanded_queries, 
            context,
            self.config.coarse_limit
        )
        
        # Stage 2: Fine-grained retrieval with filtering
        fine_results = await self._fine_retrieval(
            coarse_results,
            context,
            self.config.fine_limit
        )
        
        # Stage 3: Reranking with cross-encoder
        if self.config.enable_reranking:
            final_results = await self._rerank_results(
                fine_results,
                context.query,
                self.config.final_limit
            )
        else:
            final_results = fine_results[:self.config.final_limit]
        
        # Stage 4: Context assembly and validation
        assembled_context = await self._assemble_context(final_results, context)
        
        return RetrievalResult(
            contexts=assembled_context["contexts"],
            sources=assembled_context["sources"],
            confidence_score=assembled_context["confidence"],
            retrieval_stats=self._get_retrieval_stats()
        )
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded = await self.query_expander.expand(query)
        return [query] + expanded
    
    async def _coarse_retrieval(self, 
                              queries: List[str],
                              context: RetrievalContext,
                              limit: int) -> List[SearchResult]:
        """Coarse retrieval across expanded queries"""
        all_results = []
        
        for query in queries:
            # Convert query to vector if needed
            if hasattr(self, 'embedding_service'):
                query_vector = await self.embedding_service.embed_text(query)
            else:
                query_vector = context.query_vector
            
            results = await self.vector_store.search(SearchQuery(
                vector=query_vector,
                filters=context.filters,
                limit=limit // len(queries),
                include_metadata=True
            ))
            
            all_results.extend(results)
        
        # Deduplicate and merge scores
        return self._deduplicate_results(all_results)[:limit]
    
    async def _fine_retrieval(self,
                            coarse_results: List[SearchResult],
                            context: RetrievalContext,
                            limit: int) -> List[SearchResult]:
        """Fine-grained retrieval with advanced filtering"""
        
        # Apply semantic filtering
        filtered_results = await self._semantic_filter(coarse_results, context)
        
        # Apply temporal and relevance filters
        if context.conversation_history:
            filtered_results = await self._conversation_filter(
                filtered_results, 
                context.conversation_history
            )
        
        return filtered_results[:limit]
    
    async def _rerank_results(self,
                            results: List[SearchResult],
                            query: str,
                            limit: int) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        reranked = await self.reranker.rerank(query, results)
        return reranked[:limit]

class QueryExpander:
    """Expand queries with related terms and concepts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._load_expansion_models()
    
    async def expand(self, query: str) -> List[str]:
        """
        Expand query with related terms
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded query variations
        """
        expansions = []
        
        # Synonym expansion
        if self.config.get("enable_synonyms", True):
            synonyms = await self._get_synonyms(query)
            expansions.extend(synonyms)
        
        # Concept expansion using knowledge graph
        if self.config.get("enable_concepts", True):
            concepts = await self._get_related_concepts(query)
            expansions.extend(concepts)
        
        # Embedding-based expansion
        if self.config.get("enable_embedding_expansion", True):
            similar_queries = await self._get_similar_queries(query)
            expansions.extend(similar_queries)
        
        return expansions[:self.config.get("max_expansions", 5)]

class CrossEncoderReranker:
    """Rerank results using cross-encoder models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._load_reranker_model()
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder
        
        Args:
            query: Search query
            results: Initial search results
            
        Returns:
            Reranked search results
        """
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [(query, result.document.content or "") for result in results]
        
        # Get cross-encoder scores
        scores = await self._get_cross_encoder_scores(pairs)
        
        # Update result scores and resort
        for result, score in zip(results, scores):
            result.score = score
        
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        return reranked
```

#### 4.2.2 GraphRAG Implementation
```python
@dataclass
class GraphNode:
    id: str
    type: str  # entity, concept, document
    content: str
    metadata: Dict
    embeddings: Optional[np.ndarray] = None

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relationship: str
    weight: float
    metadata: Dict = None

class KnowledgeGraph:
    """Knowledge graph for relationship-based retrieval"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize graph database"""
        import networkx as nx
        self.graph = nx.MultiDiGraph()
        
    async def extract_entities_and_relations(self, document: ProcessedDocument) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Extract entities and relationships from document
        
        Args:
            document: Processed document with segments
            
        Returns:
            Tuple of extracted nodes and edges
        """
        nodes = []
        edges = []
        
        # Extract entities from text segments
        for segment in document.segments:
            if segment.type == "text":
                entities = await self._extract_entities(segment.content)
                relations = await self._extract_relations(segment.content, entities)
                
                # Create nodes for entities
                for entity in entities:
                    node = GraphNode(
                        id=f"{document.document_id}_{entity['id']}",
                        type="entity",
                        content=entity["text"],
                        metadata={
                            "entity_type": entity["type"],
                            "confidence": entity["confidence"],
                            "document_id": document.document_id,
                            "segment_id": segment.id
                        }
                    )
                    nodes.append(node)
                
                # Create edges for relations
                for relation in relations:
                    edge = GraphEdge(
                        source_id=f"{document.document_id}_{relation['source']}",
                        target_id=f"{document.document_id}_{relation['target']}",
                        relationship=relation["type"],
                        weight=relation["confidence"],
                        metadata={
                            "document_id": document.document_id,
                            "segment_id": segment.id
                        }
                    )
                    edges.append(edge)
        
        return nodes, edges
    
    async def graph_based_retrieval(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve information using graph relationships
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            Graph-based retrieval results with relationship paths
        """
        
        # Find query-related entities
        query_entities = await self._extract_entities(query)
        
        # Find paths in graph
        relevant_paths = []
        for entity in query_entities:
            paths = await self._find_relevant_paths(entity, max_depth=3)
            relevant_paths.extend(paths)
        
        # Rank paths by relevance
        ranked_paths = await self._rank_paths(relevant_paths, query)
        
        # Convert paths to retrieval results
        results = await self._paths_to_results(ranked_paths[:limit])
        
        return results

class GraphRAG:
    """GraphRAG implementation for relationship-based retrieval"""
    
    def __init__(self, vector_store: VectorStoreInterface, knowledge_graph: KnowledgeGraph):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
    
    async def graph_enhanced_retrieval(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Perform graph-enhanced retrieval
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            Enhanced search results with graph context
        """
        
        # Traditional vector search
        vector_results = await self.vector_store.search(SearchQuery(
            vector=await self._get_query_vector(query),
            limit=limit * 2
        ))
        
        # Graph-based retrieval
        graph_results = await self.knowledge_graph.graph_based_retrieval(query, limit)
        
        # Combine and enhance with graph context
        enhanced_results = await self._enhance_with_graph_context(vector_results, graph_results)
        
        return enhanced_results[:limit]
```

#### 4.2.3 Self-RAG Implementation
```python
class SelfRAG:
    """Self-reflective RAG with validation and improvement"""
    
    def __init__(self, llm_service, config: Dict):
        self.llm_service = llm_service
        self.config = config
        self._initialize_reflection_prompts()
    
    async def self_reflective_retrieval(self, query: str, retrieval_result: RetrievalResult) -> RetrievalResult:
        """
        Apply self-reflection to improve retrieval results
        
        Args:
            query: Original query
            retrieval_result: Initial retrieval result
            
        Returns:
            Improved retrieval result after self-reflection
        """
        
        # Step 1: Assess retrieval quality
        quality_assessment = await self._assess_retrieval_quality(query, retrieval_result)
        
        # Step 2: If quality is poor, reformulate query and re-retrieve
        if quality_assessment["score"] < self.config.get("quality_threshold", 0.7):
            improved_query = await self._reformulate_query(query, quality_assessment)
            # Re-retrieve with improved query
            # ... (implementation details)
        
        # Step 3: Validate source relevance
        validated_sources = await self._validate_sources(query, retrieval_result.sources)
        
        # Step 4: Improve context assembly
        improved_contexts = await self._improve_context_assembly(
            query, 
            validated_sources,
            quality_assessment
        )
        
        return RetrievalResult(
            contexts=improved_contexts,
            sources=validated_sources,
            confidence_score=quality_assessment["improved_score"],
            retrieval_stats=retrieval_result.retrieval_stats
        )
    
    async def _assess_retrieval_quality(self, query: str, result: RetrievalResult) -> Dict:
        """Assess the quality of retrieval results"""
        
        assessment_prompt = f"""
        Query: {query}
        
        Retrieved contexts: {result.contexts}
        
        Please assess the quality of these retrieval results on a scale of 0-1:
        - Relevance: How well do the contexts address the query?
        - Coverage: Do the contexts provide comprehensive information?
        - Accuracy: Are the contexts factually correct?
        - Coherence: Do the contexts work well together?
        
        Provide a JSON response with scores and suggestions for improvement.
        """
        
        assessment = await self.llm_service.generate(assessment_prompt)
        return self._parse_assessment(assessment)
    
    async def _reformulate_query(self, original_query: str, assessment: Dict) -> str:
        """Reformulate query based on assessment feedback"""
        
        reformulation_prompt = f"""
        Original query: {original_query}
        Assessment feedback: {assessment['feedback']}
        
        Please reformulate the query to address the identified issues and improve retrieval quality.
        Focus on: {assessment['improvement_areas']}
        
        Return only the reformulated query.
        """
        
        reformulated = await self.llm_service.generate(reformulation_prompt)
        return reformulated.strip()
```

### 4.3 Configuration Schema
```yaml
rag:
  retrieval:
    stages: ["coarse", "fine", "rerank"]
    coarse_limit: 100
    fine_limit: 20
    final_limit: 5
    context_window_size: 4096
    
  query_expansion:
    enabled: true
    max_expansions: 5
    enable_synonyms: true
    enable_concepts: true
    enable_embedding_expansion: true
    
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: 32
    
  graph_rag:
    enabled: true
    entity_extraction_model: "spacy/en_core_web_sm"
    relation_extraction_model: "custom-relation-model"
    max_path_depth: 3
    min_relation_confidence: 0.5
    
  self_rag:
    enabled: true
    quality_threshold: 0.7
    max_iterations: 3
    enable_query_reformulation: true
    enable_source_validation: true
```

---

## 5. LLM Service

### 5.1 Overview
The LLM Service provides flexible model inference with support for both llama.cpp and MLX backends, optimized for local deployment.

### 5.2 Core Components

#### 5.2.1 Unified LLM Interface
```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, List
import asyncio

@dataclass
class LLMConfig:
    backend: str  # llama_cpp, mlx
    model_name: str = "gemma-3-27b"
    model_path: str = "./models"
    device: str = "auto"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    conversation_id: Optional[str] = None

@dataclass
class GenerationResponse:
    text: str
    tokens_generated: int
    generation_time: float
    model_name: str
    metadata: Dict = None

class LLMInterface(ABC):
    """Abstract interface for LLM backends"""
    
    @abstractmethod
    async def load_model(self, model_path: str, config: LLMConfig) -> bool:
        """Load a model"""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text response"""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming text response"""
        pass
    
    @abstractmethod
    async def unload_model(self) -> bool:
        """Unload current model"""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        pass

class LlamaCppBackend(LLMInterface):
    """llama.cpp backend for cross-platform inference"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize llama.cpp backend"""
        try:
            from llama_cpp import Llama
            self.llama_cpp = Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed")
    
    async def load_model(self, model_path: str, config: LLMConfig) -> bool:
        """Load model using llama.cpp"""
        try:
            self.model = self.llama_cpp(
                model_path=model_path,
                n_ctx=config.max_tokens * 2,  # Context window
                n_threads=config.get("n_threads", -1),
                n_gpu_layers=config.get("n_gpu_layers", -1),
                verbose=False
            )
            return True
        except Exception as e:
            self._handle_error(e, "load_model")
            return False
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using llama.cpp"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Prepare prompt
        full_prompt = self._prepare_prompt(request)
        
        # Generate
        result = self.model(
            full_prompt,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=request.temperature or self.config.temperature,
            top_p=request.top_p or self.config.top_p,
            top_k=request.top_k or self.config.top_k,
            repeat_penalty=self.config.repetition_penalty,
            stop=["</s>", "\n\nHuman:", "\n\nAssistant:"]
        )
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            text=result["choices"][0]["text"].strip(),
            tokens_generated=result["usage"]["completion_tokens"],
            generation_time=generation_time,
            model_name=self.config.model_name,
            metadata={
                "finish_reason": result["choices"][0]["finish_reason"],
                "total_tokens": result["usage"]["total_tokens"]
            }
        )
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming text using llama.cpp"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        full_prompt = self._prepare_prompt(request)
        
        stream = self.model(
            full_prompt,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=request.temperature or self.config.temperature,
            top_p=request.top_p or self.config.top_p,
            stream=True,
            stop=["</s>", "\n\nHuman:", "\n\nAssistant:"]
        )
        
        for chunk in stream:
            if chunk["choices"][0]["delta"].get("content"):
                yield chunk["choices"][0]["delta"]["content"]

class MLXBackend(LLMInterface):
    """MLX backend for Apple Silicon optimization"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize MLX backend"""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load, generate
            self.mx = mx
            self.mlx_lm = generate
            self.load_func = load
        except ImportError:
            raise ImportError("MLX not installed or not on Apple Silicon")
    
    async def load_model(self, model_path: str, config: LLMConfig) -> bool:
        """Load model using MLX"""
        try:
            self.model, self.tokenizer = self.load_func(model_path)
            return True
        except Exception as e:
            self._handle_error(e, "load_model")
            return False
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using MLX"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Prepare prompt
        full_prompt = self._prepare_prompt(request)
        
        # Generate
        result = self.mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=full_prompt,
            max_tokens=request.max_tokens or self.config.max_tokens,
            temperature=request.temperature or self.config.temperature,
            top_p=request.top_p or self.config.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Extract generated text (remove prompt)
        generated_text = result[len(full_prompt):].strip()
        
        return GenerationResponse(
            text=generated_text,
            tokens_generated=len(self.tokenizer.encode(generated_text)),
            generation_time=generation_time,
            model_name=self.config.model_name,
            metadata={"backend": "mlx"}
        )
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming text using MLX"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        full_prompt = self._prepare_prompt(request)
        
        # MLX streaming implementation
        tokens = self.tokenizer.encode(full_prompt)
        current_tokens = tokens.copy()
        
        for _ in range(request.max_tokens or self.config.max_tokens):
            # Get next token
            logits = self.model(self.mx.array([current_tokens]))
            next_token = self._sample_token(logits, request)
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            current_tokens.append(next_token)
            
            # Decode and yield new token
            new_text = self.tokenizer.decode([next_token])
            yield new_text
            
            # Small delay for streaming effect
            await asyncio.sleep(0.001)

class ModelManager:
    """Manages model loading, switching, and optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loaded_models = {}
        self.active_backend = None
        
    async def load_model(self, model_name: str, backend: str = None) -> bool:
        """Load a specific model"""
        backend = backend or self.config.get("default_backend", "llama_cpp")
        
        if backend == "llama_cpp":
            llm_backend = LlamaCppBackend(self.config)
        elif backend == "mlx":
            llm_backend = MLXBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        model_path = self._get_model_path(model_name)
        success = await llm_backend.load_model(model_path, self.config)
        
        if success:
            self.loaded_models[model_name] = llm_backend
            self.active_backend = llm_backend
            return True
        
        return False
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different loaded model"""
        if model_name in self.loaded_models:
            self.active_backend = self.loaded_models[model_name]
            return True
        else:
            # Load the model if not already loaded
            return await self.load_model(model_name)
    
    async def optimize_model(self, optimization_type: str = "quantize") -> bool:
        """Apply model optimizations"""
        if not self.active_backend:
            return False
        
        if optimization_type == "quantize":
            return await self._quantize_model()
        elif optimization_type == "compile":
            return await self._compile_model()
        else:
            return False
    
    def _get_model_path(self, model_name: str) -> str:
        """Get full path to model file"""
        base_path = self.config.get("model_path", "./models")
        model_mapping = self.config.get("model_mapping", {})
        
        if model_name in model_mapping:
            return f"{base_path}/{model_mapping[model_name]}"
        else:
            return f"{base_path}/{model_name}"

class PromptEngine:
    """Manages prompt templates and conversation formatting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates"""
        self.templates = {
            "default": "{system}\n\nHuman: {user}\n\nAssistant: ",
            "rag": "{system}\n\nContext: {context}\n\nHuman: {user}\n\nAssistant: ",
            "chat": "{conversation_history}\n\nHuman: {user}\n\nAssistant: ",
            "research": """You are a research assistant helping with academic inquiries.

Context from retrieved documents:
{context}

Sources:
{sources}

Question: {user}

Please provide a comprehensive answer based on the retrieved context. Always cite your sources using [Source X] notation."""
        }
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt using the specified template"""
        if template_name not in self.templates:
            template_name = "default"
        
        template = self.templates[template_name]
        return template.format(**kwargs)
    
    def format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation history"""
        formatted = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")
        
        return "\n\n".join(formatted)

### 5.3 Configuration Schema
```yaml
llm:
  backend: "mlx"  # llama_cpp, mlx
  model_name: "gemma-3-27b"
  model_path: "./models"
  
  llama_cpp:
    n_threads: -1
    n_gpu_layers: -1
    n_ctx: 4096
    n_batch: 512
    
  mlx:
    max_cache_size: 1024
    
  generation:
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.9
    top_k: 40
    repetition_penalty: 1.1
    
  optimization:
    enable_quantization: true
    quantization_bits: 4
    enable_compilation: true
    
  models:
    gemma-3-27b: "gemma-3-27b-it.gguf"
    llama-3-8b: "llama-3-8b-instruct.gguf"
    
  prompts:
    template_path: "./prompts"
    default_template: "research"
```

---

## 6. Knowledge Graph

### 6.1 Overview
The Knowledge Graph component extracts and manages entities and relationships from documents to enable graph-based retrieval and reasoning.

### 6.2 Core Components

#### 6.2.1 Entity and Relation Extraction
```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import spacy
import networkx as nx

@dataclass
class Entity:
    id: str
    text: str
    type: str  # PERSON, ORG, CONCEPT, LOCATION, etc.
    confidence: float
    metadata: Dict = None
    embeddings: Optional[np.ndarray] = None

@dataclass
class Relation:
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    metadata: Dict = None

@dataclass
class GraphPath:
    entities: List[Entity]
    relations: List[Relation]
    path_score: float
    path_length: int

class EntityExtractor:
    """Extract entities from text using NLP models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models for entity extraction"""
        # Load spaCy model
        model_name = self.config.get("spacy_model", "en_core_web_sm")
        self.nlp = spacy.load(model_name)
        
        # Add custom entity recognizers if configured
        if self.config.get("custom_entities"):
            self._add_custom_entities()
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted entities
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity = Entity(
                id=self._generate_entity_id(ent.text, ent.label_),
                text=ent.text,
                type=ent.label_,
                confidence=self._calculate_confidence(ent),
                metadata={
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "sentence": ent.sent.text if ent.sent else ""
                }
            )
            entities.append(entity)
        
        # Extract custom entities (concepts, technical terms)
        custom_entities = await self._extract_custom_entities(doc)
        entities.extend(custom_entities)
        
        return self._deduplicate_entities(entities)
    
    async def _extract_custom_entities(self, doc) -> List[Entity]:
        """Extract domain-specific entities"""
        custom_entities = []
        
        # Extract technical terms
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and 
                token.text.lower() not in self.nlp.Defaults.stop_words and
                len(token.text) > 3):
                
                # Check if it's a technical term
                if self._is_technical_term(token.text):
                    entity = Entity(
                        id=self._generate_entity_id(token.text, "CONCEPT"),
                        text=token.text,
                        type="CONCEPT",
                        confidence=0.8,
                        metadata={
                            "pos": token.pos_,
                            "lemma": token.lemma_
                        }
                    )
                    custom_entities.append(entity)
        
        return custom_entities

class RelationExtractor:
    """Extract relationships between entities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize relation extraction models"""
        # Load dependency parser
        self.nlp = spacy.load(self.config.get("spacy_model", "en_core_web_sm"))
        
        # Load custom relation model if available
        if self.config.get("relation_model"):
            self._load_relation_model()
    
    async def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract relationships between entities
        
        Args:
            text: Source text
            entities: List of entities in the text
            
        Returns:
            List of extracted relations
        """
        doc = self.nlp(text)
        relations = []
        
        # Create entity position mapping
        entity_spans = {(ent.metadata["start"], ent.metadata["end"]): ent 
                       for ent in entities if ent.metadata}
        
        # Extract dependency-based relations
        dependency_relations = await self._extract_dependency_relations(doc, entity_spans)
        relations.extend(dependency_relations)
        
        # Extract pattern-based relations
        pattern_relations = await self._extract_pattern_relations(doc, entity_spans)
        relations.extend(pattern_relations)
        
        # Extract semantic relations using embeddings
        semantic_relations = await self._extract_semantic_relations(entities, text)
        relations.extend(semantic_relations)
        
        return self._filter_relations(relations)
    
    async def _extract_dependency_relations(self, doc, entity_spans: Dict) -> List[Relation]:
        """Extract relations using dependency parsing"""
        relations = []
        
        for token in doc:
            # Look for verb relationships
            if token.pos_ == "VERB":
                # Find subject and object
                subject_ents = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                object_ents = [child for child in token.children if child.dep_ in ["dobj", "pobj"]]
                
                for subj in subject_ents:
                    for obj in object_ents:
                        # Check if these tokens correspond to entities
                        subj_entity = self._find_entity_for_token(subj, entity_spans)
                        obj_entity = self._find_entity_for_token(obj, entity_spans)
                        
                        if subj_entity and obj_entity:
                            relation = Relation(
                                id=self._generate_relation_id(subj_entity.id, obj_entity.id, token.lemma_),
                                source_entity=subj_entity.id,
                                target_entity=obj_entity.id,
                                relation_type=token.lemma_,
                                confidence=0.7,
                                metadata={
                                    "extraction_method": "dependency",
                                    "verb": token.text,
                                    "sentence": token.sent.text
                                }
                            )
                            relations.append(relation)
        
        return relations

class KnowledgeGraphManager:
    """Manages the knowledge graph storage and operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._initialize_graph()
        self.entity_extractor = EntityExtractor(config.get("entity_extraction", {}))
        self.relation_extractor = RelationExtractor(config.get("relation_extraction", {}))
    
    def _initialize_graph(self):
        """Initialize graph database"""
        backend = self.config.get("backend", "networkx")
        
        if backend == "networkx":
            self.graph = nx.MultiDiGraph()
        elif backend == "neo4j":
            self._initialize_neo4j()
        else:
            raise ValueError(f"Unsupported graph backend: {backend}")
    
    async def process_document(self, document: ProcessedDocument) -> None:
        """
        Process document and add to knowledge graph
        
        Args:
            document: Processed document with extracted content
        """
        document_entities = []
        document_relations = []
        
        # Process each text segment
        for segment in document.segments:
            if segment.type == "text":
                # Extract entities
                entities = await self.entity_extractor.extract_entities(segment.content)
                
                # Extract relations
                relations = await self.relation_extractor.extract_relations(
                    segment.content, entities
                )
                
                # Add document context to entities and relations
                for entity in entities:
                    entity.metadata.update({
                        "document_id": document.document_id,
                        "segment_id": segment.id,
                        "page_number": segment.page_number
                    })
                
                for relation in relations:
                    relation.metadata.update({
                        "document_id": document.document_id,
                        "segment_id": segment.id,
                        "page_number": segment.page_number
                    })
                
                document_entities.extend(entities)
                document_relations.extend(relations)
        
        # Add to graph
        await self._add_entities_to_graph(document_entities)
        await self._add_relations_to_graph(document_relations)
        
        # Create document node
        await self._create_document_node(document, document_entities)
    
    async def _add_entities_to_graph(self, entities: List[Entity]) -> None:
        """Add entities to the graph"""
        for entity in entities:
            if not self.graph.has_node(entity.id):
                self.graph.add_node(
                    entity.id,
                    text=entity.text,
                    type=entity.type,
                    confidence=entity.confidence,
                    metadata=entity.metadata
                )
            else:
                # Update existing entity (merge metadata)
                existing_data = self.graph.nodes[entity.id]
                existing_data["metadata"].update(entity.metadata)
    
    async def _add_relations_to_graph(self, relations: List[Relation]) -> None:
        """Add relations to the graph"""
        for relation in relations:
            if (self.graph.has_node(relation.source_entity) and 
                self.graph.has_node(relation.target_entity)):
                
                self.graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence,
                    metadata=relation.metadata
                )
    
    async def find_paths(self, 
                        start_entity: str, 
                        end_entity: str, 
                        max_length: int = 3) -> List[GraphPath]:
        """
        Find paths between entities in the graph
        
        Args:
            start_entity: Starting entity ID
            end_entity: Target entity ID
            max_length: Maximum path length
            
        Returns:
            List of paths between entities
        """
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph, 
                start_entity, 
                end_entity, 
                cutoff=max_length
            ))
            
            graph_paths = []
            for path in paths:
                path_entities = [self._get_entity_from_graph(entity_id) for entity_id in path]
                path_relations = []
                
                # Get relations along the path
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                    if edge_data:
                        # Get the first edge (in case of multiple edges)
                        relation_data = list(edge_data.values())[0]
                        relation = Relation(
                            id=f"{path[i]}_{path[i+1]}",
                            source_entity=path[i],
                            target_entity=path[i + 1],
                            relation_type=relation_data.get("relation_type", "related"),
                            confidence=relation_data.get("confidence", 0.5),
                            metadata=relation_data.get("metadata", {})
                        )
                        path_relations.append(relation)
                
                # Calculate path score
                path_score = self._calculate_path_score(path_entities, path_relations)
                
                graph_path = GraphPath(
                    entities=path_entities,
                    relations=path_relations,
                    path_score=path_score,
                    path_length=len(path)
                )
                graph_paths.append(graph_path)
            
            # Sort by score
            graph_paths.sort(key=lambda x: x.path_score, reverse=True)
            return graph_paths
            
        except nx.NetworkXNoPath:
            return []
    
    async def get_entity_neighbors(self, entity_id: str, max_distance: int = 1) -> List[Entity]:
        """Get neighboring entities within specified distance"""
        if not self.graph.has_node(entity_id):
            return []
        
        neighbors = []
        for distance in range(1, max_distance + 1):
            # Get nodes at specific distance
            nodes_at_distance = set()
            
            # Outgoing neighbors
            for node in self.graph.successors(entity_id):
                if distance == 1:
                    nodes_at_distance.add(node)
                else:
                    # BFS for longer distances
                    for path in nx.single_source_shortest_path(
                        self.graph, entity_id, cutoff=distance
                    ).values():
                        if len(path) == distance + 1:
                            nodes_at_distance.add(path[-1])
            
            # Convert to Entity objects
            for node_id in nodes_at_distance:
                entity = self._get_entity_from_graph(node_id)
                if entity:
                    neighbors.append(entity)
        
        return neighbors

### 6.3 Configuration Schema
```yaml
knowledge_graph:
  backend: "networkx"  # networkx, neo4j
  
  entity_extraction:
    spacy_model: "en_core_web_sm"
    custom_entities: true
    technical_terms_threshold: 0.8
    
  relation_extraction:
    spacy_model: "en_core_web_sm"
    enable_dependency_relations: true
    enable_pattern_relations: true
    enable_semantic_relations: true
    min_confidence: 0.5
    
  graph_construction:
    merge_similar_entities: true
    similarity_threshold: 0.9
    max_entity_distance: 3
    
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "password"
    database: "akasha"
```

---

## 7. Cache Manager

### 7.1 Overview
The Cache Manager provides multi-level caching for embeddings, query results, and processed documents to optimize performance.

### 7.2 Core Components

#### 7.2.1 Multi-Level Cache
```python
from typing import Any, Optional, Dict, List
from abc import ABC, abstractmethod
import asyncio
import time
import hashlib
import pickle
import redis
import json

@dataclass
class CacheConfig:
    backend: str = "memory"  # memory, redis, disk
    ttl_seconds: int = 3600
    max_size_mb: int = 1024
    eviction_policy: str = "lru"  # lru, lfu, ttl

@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    size_bytes: int = 0

class CacheInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict:
        """Get cache statistics"""
        pass

class MemoryCache(CacheInterface):
    """In-memory cache with LRU eviction"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self._lock = asyncio.Lock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.current_size_bytes = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > entry.ttl:
                    await self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # Update access
                entry.access_count += 1
                self._update_access_order(key)
                self.hits += 1
                
                return entry.value
            else:
                self.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            ttl = ttl or self.config.ttl_seconds
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to evict
            await self._ensure_space(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.cache:
                await self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_size_bytes += size_bytes
            
            return True
    
    async def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure there's enough space for new entry"""
        max_bytes = self.config.max_size_mb * 1024 * 1024
        
        while (self.current_size_bytes + needed_bytes > max_bytes and 
               self.access_order):
            
            if self.config.eviction_policy == "lru":
                # Remove least recently used
                oldest_key = self.access_order[0]
                await self._remove_entry(oldest_key)
            elif self.config.eviction_policy == "lfu":
                # Remove least frequently used
                lfu_key = min(self.cache.keys(), 
                             key=lambda k: self.cache[k].access_count)
                await self._remove_entry(lfu_key)
            elif self.config.eviction_policy == "ttl":
                # Remove expired entries first
                current_time = time.time()
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if current_time - entry.timestamp > entry.ttl
                ]
                if expired_keys:
                    for key in expired_keys:
                        await self._remove_entry(key)
                else:
                    # Fall back to LRU
                    oldest_key = self.access_order[0]
                    await self._remove_entry(oldest_key)

class RedisCache(CacheInterface):
    """Redis-backed cache for distributed caching"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._connect_redis()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _connect_redis(self):
        """Connect to Redis"""
        redis_config = self.config.redis_config or {}
        self.redis = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            decode_responses=False
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.redis.get(key)
            if value is not None:
                self.hits += 1
                return pickle.loads(value)
            else:
                self.misses += 1
                return None
        except Exception:
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis"""
        try:
            ttl = ttl or self.config.ttl_seconds
            serialized_value = pickle.dumps(value)
            self.redis.setex(key, ttl, serialized_value)
            return True
        except Exception:
            return False

class DiskCache(CacheInterface):
    """Disk-based cache for large objects"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir or "./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Index file for metadata
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _load_index(self) -> Dict:
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception:
            pass
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if key not in self.index:
            self.misses += 1
            return None
        
        entry_info = self.index[key]
        
        # Check TTL
        if time.time() - entry_info["timestamp"] > entry_info["ttl"]:
            await self.delete(key)
            self.misses += 1
            return None
        
        # Load from disk
        file_path = self.cache_dir / f"{key}.pkl"
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access
                entry_info["access_count"] += 1
                entry_info["last_access"] = time.time()
                self._save_index()
                
                self.hits += 1
                return value
            except Exception:
                await self.delete(key)
                self.misses += 1
                return None
        else:
            await self.delete(key)
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache"""
        ttl = ttl or self.config.ttl_seconds
        
        try:
            # Save to disk
            file_path = self.cache_dir / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            self.index[key] = {
                "timestamp": time.time(),
                "ttl": ttl,
                "access_count": 0,
                "last_access": time.time(),
                "size_bytes": file_path.stat().st_size
            }
            self._save_index()
            
            return True
        except Exception:
            return False

class CacheManager:
    """Manages multiple cache layers and strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._initialize_caches()
        
    def _initialize_caches(self):
        """Initialize cache layers"""
        self.caches = {}
        
        # L1 Cache: Memory (fastest)
        if self.config.get("l1_cache", {}).get("enabled", True):
            l1_config = CacheConfig(**self.config.get("l1_cache", {}))
            l1_config.backend = "memory"
            self.caches["l1"] = MemoryCache(l1_config)
        
        # L2 Cache: Redis (shared)
        if self.config.get("l2_cache", {}).get("enabled", False):
            l2_config = CacheConfig(**self.config.get("l2_cache", {}))
            l2_config.backend = "redis"
            self.caches["l2"] = RedisCache(l2_config)
        
        # L3 Cache: Disk (persistent)
        if self.config.get("l3_cache", {}).get("enabled", True):
            l3_config = CacheConfig(**self.config.get("l3_cache", {}))
            l3_config.backend = "disk"
            self.caches["l3"] = DiskCache(l3_config)
    
    async def get(self, key: str, cache_levels: List[str] = None) -> Optional[Any]:
        """
        Get value from cache hierarchy
        
        Args:
            key: Cache key
            cache_levels: Specific cache levels to check (default: all)
            
        Returns:
            Cached value or None
        """
        cache_levels = cache_levels or ["l1", "l2", "l3"]
        
        for level in cache_levels:
            if level in self.caches:
                value = await self.caches[level].get(key)
                if value is not None:
                    # Populate upper levels (cache warming)
                    await self._populate_upper_levels(key, value, level, cache_levels)
                    return value
        
        return None
    
    async def set(self, 
                 key: str, 
                 value: Any, 
                 ttl: Optional[int] = None,
                 cache_levels: List[str] = None) -> bool:
        """
        Set value in cache hierarchy
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_levels: Specific cache levels to set (default: all)
            
        Returns:
            Success status
        """
        cache_levels = cache_levels or ["l1", "l2", "l3"]
        
        success = True
        for level in cache_levels:
            if level in self.caches:
                result = await self.caches[level].set(key, value, ttl)
                success = success and result
        
        return success
    
    async def _populate_upper_levels(self, 
                                   key: str, 
                                   value: Any, 
                                   found_level: str,
                                   cache_levels: List[str]) -> None:
        """Populate upper cache levels with found value"""
        level_order = ["l1", "l2", "l3"]
        found_index = level_order.index(found_level)
        
        # Populate all levels above the found level
        for i in range(found_index):
            level = level_order[i]
            if level in cache_levels and level in self.caches:
                await self.caches[level].set(key, value)

class SpecializedCaches:
    """Specialized caches for different data types"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def _embedding_key(self, content: str, model: str) -> str:
        """Generate cache key for embeddings"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"embedding:{model}:{content_hash}"
    
    def _query_key(self, query: str, filters: Dict, limit: int) -> str:
        """Generate cache key for query results"""
        query_data = {
            "query": query,
            "filters": filters,
            "limit": limit
        }
        query_str = json.dumps(query_data, sort_keys=True)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        return f"query:{query_hash}"
    
    async def get_embedding(self, content: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self._embedding_key(content, model)
        return await self.cache_manager.get(key, ["l1", "l2"])
    
    async def set_embedding(self, content: str, model: str, embedding: np.ndarray) -> bool:
        """Cache embedding"""
        key = self._embedding_key(content, model)
        return await self.cache_manager.set(key, embedding, cache_levels=["l1", "l2"])
    
    async def get_query_result(self, query: str, filters: Dict, limit: int) -> Optional[List]:
        """Get cached query results"""
        key = self._query_key(query, filters, limit)
        return await self.cache_manager.get(key, ["l1"])
    
    async def set_query_result(self, 
                              query: str, 
                              filters: Dict, 
                              limit: int, 
                              results: List) -> bool:
        """Cache query results"""
        key = self._query_key(query, filters, limit)
        # Shorter TTL for query results
        return await self.cache_manager.set(key, results, ttl=300, cache_levels=["l1"])

### 7.3 Configuration Schema
```yaml
cache:
  l1_cache:
    enabled: true
    max_size_mb: 512
    ttl_seconds: 1800
    eviction_policy: "lru"
    
  l2_cache:
    enabled: false
    backend: "redis"
    redis_config:
      host: "localhost"
      port: 6379
      db: 0
    max_size_mb: 2048
    ttl_seconds: 3600
    
  l3_cache:
    enabled: true
    backend: "disk"
    cache_dir: "./cache"
    max_size_mb: 10240
    ttl_seconds: 86400  # 24 hours
    
  specialized:
    embeddings:
      ttl_seconds: 3600
      cache_levels: ["l1", "l2"]
    
    query_results:
      ttl_seconds: 300
      cache_levels: ["l1"]
    
    documents:
      ttl_seconds: 86400
      cache_levels: ["l1", "l3"]
```

---

## 8. Plugin Manager

### 8.1 Overview
The Plugin Manager enables dynamic loading and management of extensions, providing a secure and flexible architecture for community contributions.

### 8.2 Core Components

#### 8.2.1 Plugin Interface and Registry
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
import importlib
import inspect
import sys
from pathlib import Path
import yaml
import hashlib

@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    categories: List[str]
    api_version: str
    entry_point: str

@dataclass
class PluginConfig:
    plugin_name: str
    enabled: bool = True
    config: Dict = None
    priority: int = 0

class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources"""
        pass
    
    @abstractmethod
    async def get_info(self) -> Dict:
        """Get plugin information"""
        pass

class DocumentProcessorPlugin(PluginInterface):
    """Interface for document processing plugins"""
    
    @abstractmethod
    async def process_document(self, document_data: bytes, metadata: Dict) -> ProcessedDocument:
        """Process a document"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported file formats"""
        pass

class EmbeddingModelPlugin(PluginInterface):
    """Interface for embedding model plugins"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate text embeddings"""
        pass
    
    @abstractmethod
    async def embed_image(self, image: bytes) -> np.ndarray:
        """Generate image embeddings"""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        pass

class LLMPlugin(PluginInterface):
    """Interface for LLM plugins"""
    
    @abstractmethod
    async def generate(self, prompt: str, config: Dict) -> str:
        """Generate text response"""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, config: Dict) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        pass

class PluginRegistry:
    """Registry for managing available plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, Type[PluginInterface]] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.categories: Dict[str, List[str]] = {}
    
    def register_plugin(self, plugin_class: Type[PluginInterface]) -> bool:
        """
        Register a plugin class
        
        Args:
            plugin_class: Plugin class to register
            
        Returns:
            Success status
        """
        try:
            # Get metadata
            metadata = plugin_class.metadata
            
            # Validate plugin
            if not self._validate_plugin(plugin_class, metadata):
                return False
            
            # Register
            self.plugins[metadata.name] = plugin_class
            self.plugin_metadata[metadata.name] = metadata
            
            # Update categories
            for category in metadata.categories:
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(metadata.name)
            
            return True
            
        except Exception as e:
            self._handle_error(e, f"register_plugin:{plugin_class.__name__}")
            return False
    
    def get_plugin(self, name: str) -> Optional[Type[PluginInterface]]:
        """Get plugin class by name"""
        return self.plugins.get(name)
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """Get plugin names by category"""
        return self.categories.get(category, [])
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins"""
        return list(self.plugin_metadata.values())
    
    def _validate_plugin(self, plugin_class: Type[PluginInterface], metadata: PluginMetadata) -> bool:
        """Validate plugin class and metadata"""
        # Check required methods
        required_methods = ["initialize", "cleanup", "get_info"]
        for method in required_methods:
            if not hasattr(plugin_class, method):
                return False
        
        # Check metadata completeness
        if not all([metadata.name, metadata.version, metadata.api_version]):
            return False
        
        # Check API version compatibility
        if not self._is_api_compatible(metadata.api_version):
            return False
        
        return True

class PluginLoader:
    """Loads plugins from various sources"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.plugin_paths = config.get("plugin_paths", ["./plugins"])
        self.registry = PluginRegistry()
        
    async def load_plugins(self) -> bool:
        """Load all available plugins"""
        success = True
        
        # Load from configured paths
        for path in self.plugin_paths:
            path_success = await self._load_from_path(Path(path))
            success = success and path_success
        
        # Load built-in plugins
        builtin_success = await self._load_builtin_plugins()
        success = success and builtin_success
        
        return success
    
    async def _load_from_path(self, plugin_path: Path) -> bool:
        """Load plugins from a directory"""
        if not plugin_path.exists():
            return True
        
        success = True
        
        for plugin_dir in plugin_path.iterdir():
            if plugin_dir.is_dir():
                plugin_success = await self._load_plugin_from_dir(plugin_dir)
                success = success and plugin_success
        
        return success
    
    async def _load_plugin_from_dir(self, plugin_dir: Path) -> bool:
        """Load a single plugin from directory"""
        try:
            # Look for plugin.yaml
            config_file = plugin_dir / "plugin.yaml"
            if not config_file.exists():
                return False
            
            # Load plugin configuration
            with open(config_file, 'r') as f:
                plugin_config = yaml.safe_load(f)
            
            # Create metadata
            metadata = PluginMetadata(**plugin_config["metadata"])
            
            # Load plugin module
            sys.path.insert(0, str(plugin_dir))
            try:
                module = importlib.import_module(metadata.entry_point)
                
                # Find plugin class
                plugin_class = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PluginInterface) and 
                        obj != PluginInterface):
                        plugin_class = obj
                        break
                
                if plugin_class:
                    # Set metadata
                    plugin_class.metadata = metadata
                    
                    # Register plugin
                    return self.registry.register_plugin(plugin_class)
                
            finally:
                sys.path.pop(0)
            
            return False
            
        except Exception as e:
            self._handle_error(e, f"load_plugin:{plugin_dir.name}")
            return False

class PluginManager:
    """Main plugin management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loader = PluginLoader(config)
        self.registry = self.loader.registry
        self.active_plugins: Dict[str, PluginInterface] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self._load_plugin_configs()
    
    def _load_plugin_configs(self):
        """Load plugin configurations"""
        configs = self.config.get("plugins", {})
        for name, config_data in configs.items():
            self.plugin_configs[name] = PluginConfig(
                plugin_name=name,
                **config_data
            )
    
    async def initialize(self) -> bool:
        """Initialize plugin system"""
        # Load all available plugins
        await self.loader.load_plugins()
        
        # Initialize enabled plugins
        for name, plugin_config in self.plugin_configs.items():
            if plugin_config.enabled:
                await self.activate_plugin(name, plugin_config.config or {})
        
        return True
    
    async def activate_plugin(self, name: str, config: Dict = None) -> bool:
        """
        Activate a plugin
        
        Args:
            name: Plugin name
            config: Plugin configuration
            
        Returns:
            Success status
        """
        try:
            if name in self.active_plugins:
                return True  # Already active
            
            plugin_class = self.registry.get_plugin(name)
            if not plugin_class:
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Initialize plugin
            success = await plugin_instance.initialize(config or {})
            if success:
                self.active_plugins[name] = plugin_instance
                return True
            
            return False
            
        except Exception as e:
            self._handle_error(e, f"activate_plugin:{name}")
            return False
    
    async def deactivate_plugin(self, name: str) -> bool:
        """
        Deactivate a plugin
        
        Args:
            name: Plugin name
            
        Returns:
            Success status
        """
        try:
            if name not in self.active_plugins:
                return True  # Already inactive
            
            plugin = self.active_plugins[name]
            success = await plugin.cleanup()
            
            if success:
                del self.active_plugins[name]
            
            return success
            
        except Exception as e:
            self._handle_error(e, f"deactivate_plugin:{name}")
            return False
    
    async def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin
        
        Args:
            name: Plugin name
            
        Returns:
            Success status
        """
        # Get current config
        config = {}
        if name in self.active_plugins:
            config = await self.active_plugins[name].get_info()
        
        # Deactivate and reactivate
        await self.deactivate_plugin(name)
        return await self.activate_plugin(name, config)
    
    def get_active_plugins(self, category: str = None) -> List[str]:
        """Get list of active plugins, optionally filtered by category"""
        if category:
            category_plugins = self.registry.get_plugins_by_category(category)
            return [name for name in self.active_plugins.keys() if name in category_plugins]
        else:
            return list(self.active_plugins.keys())
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get active plugin instance"""
        return self.active_plugins.get(name)

class PluginSandbox:
    """Provides sandboxed execution environment for plugins"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.resource_limits = config.get("resource_limits", {})
        self.allowed_modules = config.get("allowed_modules", [])
        self.blocked_modules = config.get("blocked_modules", [])
    
    async def execute_plugin_method(self, 
                                  plugin: PluginInterface, 
                                  method_name: str, 
                                  *args, **kwargs) -> Any:
        """
        Execute plugin method in sandboxed environment
        
        Args:
            plugin: Plugin instance
            method_name: Method to execute
            *args, **kwargs: Method arguments
            
        Returns:
            Method result
        """
        try:
            # Check if method exists
            if not hasattr(plugin, method_name):
                raise AttributeError(f"Plugin has no method '{method_name}'")
            
            method = getattr(plugin, method_name)
            
            # Set resource limits
            await self._set_resource_limits()
            
            # Execute with timeout
            timeout = self.config.get("execution_timeout", 30)
            result = await asyncio.wait_for(
                method(*args, **kwargs),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise PluginExecutionError(f"Plugin method '{method_name}' timed out")
        except Exception as e:
            raise PluginExecutionError(f"Plugin method '{method_name}' failed: {str(e)}")
        finally:
            await self._reset_resource_limits()
    
    async def _set_resource_limits(self):
        """Set resource limits for plugin execution"""
        # Implement resource limiting (memory, CPU, etc.)
        pass
    
    async def _reset_resource_limits(self):
        """Reset resource limits after plugin execution"""
        pass

### 8.3 Configuration Schema
```yaml
plugins:
  plugin_paths:
    - "./plugins"
    - "./community_plugins"
  
  resource_limits:
    max_memory_mb: 512
    max_cpu_percent: 50
    execution_timeout: 30
    
  security:
    enable_sandbox: true
    allowed_modules:
      - "numpy"
      - "pandas"
      - "PIL"
    blocked_modules:
      - "os"
      - "subprocess"
      - "sys"
  
  # Individual plugin configurations
  mineru_enhanced:
    enabled: true
    config:
      ocr_engine: "paddleocr"
      enable_tables: true
  
  custom_embedder:
    enabled: false
    config:
      model_path: "./models/custom_embedder"
      batch_size: 16
```

---

## 9. Configuration Manager

### 9.1 Overview
The Configuration Manager provides centralized, hierarchical configuration management with validation, environment variable support, and hot-reloading capabilities.

### 9.2 Core Components

#### 9.2.1 Configuration Schema and Validation
```python
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os
import re
from pydantic import BaseModel, validator, Field

class SystemConfig(BaseModel):
    """System-level configuration"""
    name: str = "akasha"
    version: str = "1.0.0"
    environment: str = Field(default="development", regex="^(development|staging|production)$")
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    data_directory: str = "./data"
    temp_directory: str = "./temp"

class IngestionConfig(BaseModel):
    """Ingestion engine configuration"""
    backend: str = Field(default="mineru2", regex="^(mineru2|custom)$")
    batch_size: int = Field(default=10, ge=1, le=100)
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    enable_ocr: bool = True
    ocr_languages: List[str] = ["en"]
    output_format: str = Field(default="hybrid", regex="^(markdown|json|hybrid)$")

class EmbeddingConfig(BaseModel):
    """Embedding service configuration"""
    model: str = "jinaai/jina-embeddings-v4"
    device: str = Field(default="auto", regex="^(auto|cpu|cuda|mps)$")
    batch_size: int = Field(default=32, ge=1, le=256)
    max_length: int = Field(default=8192, ge=512, le=32768)
    dimensions: int = Field(default=512, ge=128, le=4096)
    normalize_embeddings: bool = True
    cache_embeddings: bool = True

class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    backend: str = Field(default="chroma", regex="^(chroma|qdrant)$")
    collection_name: str = "akasha_documents"
    distance_metric: str = Field(default="cosine", regex="^(cosine|euclidean|dot)$")
    storage_path: str = "./data/vector_store"

class LLMConfig(BaseModel):
    """LLM service configuration"""
    backend: str = Field(default="mlx", regex="^(llama_cpp|mlx)$")
    model_name: str = "gemma-3-27b"
    model_path: str = "./models"
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)

class RAGConfig(BaseModel):
    """RAG engine configuration"""
    coarse_limit: int = Field(default=100, ge=10, le=1000)
    fine_limit: int = Field(default=20, ge=5, le=100)
    final_limit: int = Field(default=5, ge=1, le=20)
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    enable_graph_rag: bool = True
    enable_self_rag: bool = True

class CacheConfig(BaseModel):
    """Cache configuration"""
    l1_enabled: bool = True
    l1_max_size_mb: int = Field(default=512, ge=64, le=4096)
    l1_ttl_seconds: int = Field(default=1800, ge=300, le=86400)
    l2_enabled: bool = False
    l3_enabled: bool = True
    l3_max_size_mb: int = Field(default=10240, ge=1024, le=102400)

class AkashaConfig(BaseModel):
    """Main Akasha configuration"""
    system: SystemConfig = SystemConfig()
    ingestion: IngestionConfig = IngestionConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    cache: CacheConfig = CacheConfig()
    
    class Config:
        extra = "allow"  # Allow additional fields for plugins

class ConfigurationManager:
    """Manages application configuration with hierarchical loading and validation"""
    
    def __init__(self, config_paths: List[str] = None):
        self.config_paths = config_paths or [
            "/etc/akasha/akasha.yaml",
            "~/.akasha/akasha.yaml", 
            "./akasha.yaml",
            "./config/akasha.yaml"
        ]
        self.config: Optional[AkashaConfig] = None
        self.watchers: List[callable] = []
        self._env_prefix = "AKASHA_"
        
    async def load_config(self) -> AkashaConfig:
        """
        Load configuration from multiple sources with precedence
        
        Precedence order (highest to lowest):
        1. Environment variables
        2. Command line arguments  
        3. Local config files
        4. User config files
        5. System config files
        6. Default values
        
        Returns:
            Validated configuration object
        """
        
        # Start with default configuration
        config_dict = {}
        
        # Load from files (reverse order for precedence)
        for config_path in reversed(self.config_paths):
            file_config = await self._load_from_file(config_path)
            if file_config:
                config_dict = self._merge_configs(config_dict, file_config)
        
        # Override with environment variables
        env_config = self._load_from_env()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Validate and create config object
        try:
            self.config = AkashaConfig(**config_dict)
            return self.config
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    async def _load_from_file(self, file_path: str) -> Optional[Dict]:
        """Load configuration from a YAML file"""
        try:
            path = Path(file_path).expanduser()
            if path.exists():
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config or {}
        except Exception as e:
            # Log warning but don't fail
            pass
        
        return None
    
    def _load_from_env(self) -> Dict:
        """Load configuration from environment variables"""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                # Convert AKASHA_EMBEDDING_MODEL to embedding.model
                config_key = key[len(self._env_prefix):].lower()
                config_path = config_key.split('_')
                
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set nested value
                self._set_nested_value(env_config, config_path, converted_value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        if value.isdigit():
            return int(value)
        
        # Float values
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # String values
        return value
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Set a nested configuration value"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def save_config(self, file_path: str = None) -> bool:
        """Save current configuration to file"""
        if not self.config:
            return False
        
        file_path = file_path or "./akasha.yaml"
        
        try:
            # Convert to dictionary and clean up
            config_dict = self.config.dict()
            
            # Write to file
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            return True
        except Exception as e:
            self._handle_error(e, f"save_config:{file_path}")
            return False
    
    def get_config(self) -> Optional[AkashaConfig]:
        """Get current configuration"""
        return self.config
    
    def get_section(self, section: str) -> Optional[BaseModel]:
        """Get a specific configuration section"""
        if not self.config:
            return None
        
        return getattr(self.config, section, None)
    
    def update_config(self, updates: Dict) -> bool:
        """Update configuration with new values"""
        if not self.config:
            return False
        
        try:
            # Merge updates
            current_dict = self.config.dict()
            updated_dict = self._merge_configs(current_dict, updates)
            
            # Validate updated configuration
            self.config = AkashaConfig(**updated_dict)
            
            # Notify watchers
            self._notify_watchers()
            
            return True
        except Exception as e:
            self._handle_error(e, "update_config")
            return False
    
    def watch_config(self, callback: callable):
        """Register a callback for configuration changes"""
        self.watchers.append(callback)
    
    def _notify_watchers(self):
        """Notify all registered watchers of configuration changes"""
        for watcher in self.watchers:
            try:
                watcher(self.config)
            except Exception as e:
                self._handle_error(e, "config_watcher")

class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass

class EnvironmentManager:
    """Manages environment-specific configurations"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.environments = {
            "development": self._get_development_overrides(),
            "staging": self._get_staging_overrides(),
            "production": self._get_production_overrides()
        }
    
    def _get_development_overrides(self) -> Dict:
        """Development environment overrides"""
        return {
            "system": {
                "log_level": "DEBUG"
            },
            "cache": {
                "l1_max_size_mb": 256,
                "l2_enabled": False
            },
            "embedding": {
                "cache_embeddings": True
            }
        }
    
    def _get_staging_overrides(self) -> Dict:
        """Staging environment overrides"""
        return {
            "system": {
                "log_level": "INFO"
            },
            "cache": {
                "l2_enabled": True
            }
        }
    
    def _get_production_overrides(self) -> Dict:
        """Production environment overrides"""
        return {
            "system": {
                "log_level": "WARNING"
            },
            "cache": {
                "l1_max_size_mb": 1024,
                "l2_enabled": True,
                "l3_max_size_mb": 20480
            },
            "vector_store": {
                "backend": "qdrant"
            }
        }
    
    async def apply_environment(self, environment: str) -> bool:
        """Apply environment-specific configuration overrides"""
        if environment not in self.environments:
            return False
        
        overrides = self.environments[environment]
        return self.config_manager.update_config(overrides)

class ConfigValidator:
    """Validates configuration values and dependencies"""
    
    @staticmethod
    def validate_model_path(model_path: str, model_name: str) -> bool:
        """Validate that model file exists"""
        path = Path(model_path) / f"{model_name}.gguf"
        return path.exists()
    
    @staticmethod
    def validate_storage_path(storage_path: str) -> bool:
        """Validate storage path is writable"""
        try:
            path = Path(storage_path)
            path.mkdir(parents=True, exist_ok=True)
            return os.access(path, os.W_OK)
        except Exception:
            return False
    
    @staticmethod
    def validate_dependencies(config: AkashaConfig) -> List[str]:
        """Validate configuration dependencies and return issues"""
        issues = []
        
        # Check model files exist
        model_path = Path(config.llm.model_path)
        if not model_path.exists():
            issues.append(f"Model directory does not exist: {model_path}")
        
        # Check storage paths are writable
        for path_attr in ["data_directory", "temp_directory"]:
            path_value = getattr(config.system, path_attr)
            if not ConfigValidator.validate_storage_path(path_value):
                issues.append(f"Path not writable: {path_value}")
        
        # Check vector store backend compatibility
        if config.vector_store.backend == "qdrant" and not config.cache.l2_enabled:
            issues.append("Qdrant backend recommended with L2 cache enabled")
        
        return issues

### 9.3 Configuration Schema Example
```yaml
# akasha.yaml
system:
  name: "akasha"
  version: "1.0.0"
  environment: "development"
  log_level: "INFO"
  data_directory: "./data"
  temp_directory: "./temp"

ingestion:
  backend: "mineru2"
  batch_size: 10
  max_file_size_mb: 100
  timeout_seconds: 300
  enable_ocr: true
  ocr_languages: ["en", "fr"]
  output_format: "hybrid"

embedding:
  model: "jinaai/jina-embeddings-v4"
  device: "auto"
  batch_size: 32
  max_length: 8192
  dimensions: 512
  normalize_embeddings: true
  cache_embeddings: true

vector_store:
  backend: "chroma"
  collection_name: "akasha_documents"
  distance_metric: "cosine"
  storage_path: "./data/vector_store"

llm:
  backend: "mlx"
  model_name: "gemma-3-27b"
  model_path: "./models"
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 40

rag:
  coarse_limit: 100
  fine_limit: 20
  final_limit: 5
  enable_query_expansion: true
  enable_reranking: true
  enable_graph_rag: true
  enable_self_rag: true

cache:
  l1_enabled: true
  l1_max_size_mb: 512
  l1_ttl_seconds: 1800
  l2_enabled: false
  l3_enabled: true
  l3_max_size_mb: 10240

# Plugin configurations
plugins:
  mineru_enhanced:
    enabled: true
    config:
      ocr_engine: "paddleocr"
      enable_tables: true
```

---

## 10. API Gateway

### 10.1 Overview
The API Gateway provides a unified REST and WebSocket interface for all system operations, with authentication, rate limiting, and comprehensive documentation.

### 10.2 Core Components

#### 10.2.1 FastAPI Application Structure
```python
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, AsyncGenerator
import asyncio
import json
import time
from pathlib import Path

# Request/Response Models
class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    filename: str
    content_type: str = "application/pdf"
    metadata: Optional[Dict] = None

class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    status: str
    processing_time: float
    message: str

class SearchRequest(BaseModel):
    """Search request"""
    query: str
    filters: Optional[Dict] = None
    limit: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True
    include_content: bool = False
    search_type: str = Field(default="hybrid", regex="^(semantic|keyword|hybrid)$")

class SearchResponse(BaseModel):
    """Search response"""
    results: List[Dict]
    total_results: int
    search_time: float
    query_id: str

class ChatRequest(BaseModel):
    """Chat request"""
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    conversation_id: str
    sources: List[Dict]
    generation_time: float
    tokens_generated: int

class SystemStatus(BaseModel):
    """System status response"""
    status: str
    version: str
    uptime: float
    components: Dict[str, str]
    performance: Dict[str, Any]

class AkashaAPI:
    """Main API application"""
    
    def __init__(self, 
                 ingestion_engine,
                 embedding_service,
                 vector_store,
                 rag_engine,
                 llm_service,
                 plugin_manager,
                 config):
        
        self.ingestion_engine = ingestion_engine
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.rag_engine = rag_engine
        self.llm_service = llm_service
        self.plugin_manager = plugin_manager
        self.config = config
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Akasha API",
            description="Advanced Multimodal RAG System API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Track active connections
        self.active_connections: List[WebSocket] = []
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("allowed_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for logging and metrics
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log request
            self._log_request(request, response, process_time)
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Document management
        @self.app.post("/api/v1/documents", response_model=DocumentUploadResponse)
        async def upload_document(file: UploadFile = File(...)):
            """Upload and process a document"""
            return await self._handle_upload(file)
        
        @self.app.get("/api/v1/documents")
        async def list_documents(limit: int = 100, offset: int = 0):
            """List processed documents"""
            return await self._handle_list_documents(limit, offset)
        
        @self.app.get("/api/v1/documents/{document_id}")
        async def get_document(document_id: str):
            """Get document details"""
            return await self._handle_get_document(document_id)
        
        @self.app.delete("/api/v1/documents/{document_id}")
        async def delete_document(document_id: str):
            """Delete a document"""
            return await self._handle_delete_document(document_id)
        
        # Search endpoints
        @self.app.post("/api/v1/search", response_model=SearchResponse)
        async def search(request: SearchRequest):
            """Search documents"""
            return await self._handle_search(request)
        
        @self.app.get("/api/v1/search/similar/{document_id}")
        async def find_similar(document_id: str, limit: int = 10):
            """Find similar documents"""
            return await self._handle_find_similar(document_id, limit)
        
        # Chat endpoints
        @self.app.post("/api/v1/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Chat with documents"""
            if request.stream:
                return StreamingResponse(
                    self._handle_chat_stream(request),
                    media_type="text/plain"
                )
            else:
                return await self._handle_chat(request)
        
        @self.app.get("/api/v1/conversations/{conversation_id}")
        async def get_conversation(conversation_id: str):
            """Get conversation history"""
            return await self._handle_get_conversation(conversation_id)
        
        # Model management
        @self.app.get("/api/v1/models")
        async def list_models():
            """List available models"""
            return await self._handle_list_models()
        
        @self.app.post("/api/v1/models/{model_name}/load")
        async def load_model(model_name: str):
            """Load a specific model"""
            return await self._handle_load_model(model_name)
        
        @self.app.get("/api/v1/models/current")
        async def get_current_model():
            """Get current model info"""
            return await self._handle_get_current_model()
        
        # Plugin management
        @self.app.get("/api/v1/plugins")
        async def list_plugins():
            """List available plugins"""
            return await self._handle_list_plugins()
        
        @self.app.post("/api/v1/plugins/{plugin_name}/activate")
        async def activate_plugin(plugin_name: str, config: Dict = None):
            """Activate a plugin"""
            return await self._handle_activate_plugin(plugin_name, config)
        
        @self.app.post("/api/v1/plugins/{plugin_name}/deactivate")
        async def deactivate_plugin(plugin_name: str):
            """Deactivate a plugin"""
            return await self._handle_deactivate_plugin(plugin_name)
        
        # System endpoints
        @self.app.get("/api/v1/status", response_model=SystemStatus)
        async def get_status():
            """Get system status"""
            return await self._handle_get_status()
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get system metrics"""
            return await self._handle_get_metrics()
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time interactions"""
            await self._handle_websocket(websocket)
    
    # Route handlers
    async def _handle_upload(self, file: UploadFile) -> DocumentUploadResponse:
        """Handle document upload"""
        try:
            start_time = time.time()
            
            # Read file content
            content = await file.read()
            
            # Process document
            result = await self.ingestion_engine.process_document(
                content, 
                {"filename": file.filename, "content_type": file.content_type}
            )
            
            processing_time = time.time() - start_time
            
            return DocumentUploadResponse(
                document_id=result.document_id,
                status="processed",
                processing_time=processing_time,
                message=f"Successfully processed {file.filename}"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_search(self, request: SearchRequest) -> SearchResponse:
        """Handle search request"""
        try:
            start_time = time.time()
            
            # Perform search based on type
            if request.search_type == "semantic":
                results = await self._semantic_search(request)
            elif request.search_type == "keyword":
                results = await self._keyword_search(request)
            else:  # hybrid
                results = await self._hybrid_search(request)
            
            search_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                total_results=len(results),
                search_time=search_time,
                query_id=self._generate_query_id()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat request"""
        try:
            start_time = time.time()
            
            # Retrieve relevant context
            context_result = await self.rag_engine.retrieve(
                query=request.message,
                conversation_id=request.conversation_id
            )
            
            # Generate response
            generation_result = await self.llm_service.generate(
                prompt=request.message,
                context=context_result.contexts,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            generation_time = time.time() - start_time
            
            return ChatResponse(
                response=generation_result.text,
                conversation_id=request.conversation_id or self._generate_conversation_id(),
                sources=context_result.sources,
                generation_time=generation_time,
                tokens_generated=generation_result.tokens_generated
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat request"""
        try:
            # Retrieve context
            context_result = await self.rag_engine.retrieve(
                query=request.message,
                conversation_id=request.conversation_id
            )
            
            # Stream generation
            async for chunk in self.llm_service.generate_stream(
                prompt=request.message,
                context=context_result.contexts,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send final metadata
            yield f"data: {json.dumps({'sources': context_result.sources, 'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message["type"] == "chat":
                    await self._handle_websocket_chat(websocket, message)
                elif message["type"] == "search":
                    await self._handle_websocket_search(websocket, message)
                elif message["type"] == "status":
                    await self._handle_websocket_status(websocket)
                
        except Exception as e:
            self.active_connections.remove(websocket)
    
    async def _handle_websocket_chat(self, websocket: WebSocket, message: Dict):
        """Handle WebSocket chat message"""
        try:
            # Send typing indicator
            await websocket.send_text(json.dumps({
                "type": "typing",
                "status": "started"
            }))
            
            # Process chat request
            chat_request = ChatRequest(**message["data"])
            
            # Stream response
            async for chunk in self._handle_chat_stream(chat_request):
                await websocket.send_text(chunk)
            
            # Send completion
            await websocket.send_text(json.dumps({
                "type": "typing",
                "status": "completed"
            }))
            
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))

# Authentication and authorization
class AuthManager:
    """Manages API authentication and authorization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.security = HTTPBearer()
        self.api_keys = config.get("api_keys", {})
        self.rate_limits = config.get("rate_limits", {})
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify API token"""
        token = credentials.credentials
        
        if token not in self.api_keys:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return self.api_keys[token]
    
    async def check_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Check rate limits for user and endpoint"""
        # Implement rate limiting logic
        return True

# API Documentation
def create_app(akasha_system) -> FastAPI:
    """Create and configure the FastAPI application"""
    
    api = AkashaAPI(
        ingestion_engine=akasha_system.ingestion_engine,
        embedding_service=akasha_system.embedding_service,
        vector_store=akasha_system.vector_store,
        rag_engine=akasha_system.rag_engine,
        llm_service=akasha_system.llm_service,
        plugin_manager=akasha_system.plugin_manager,
        config=akasha_system.config
    )
    
    return api.app

### 10.3 Configuration Schema
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
  cors:
    allowed_origins: ["*"]
    allowed_methods: ["*"]
    allowed_headers: ["*"]
    
  authentication:
    enabled: true
    api_keys:
      "your-api-key": "user1"
      "admin-key": "admin"
    
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_size: 10
    
  file_upload:
    max_file_size_mb: 100
    allowed_types: ["pdf", "txt", "docx"]
    upload_path: "./uploads"
    
  websocket:
    max_connections: 100
    heartbeat_interval: 30
    
  documentation:
    enable_docs: true
    enable_redoc: true
    contact:
      name: "Akasha Team"
      email: "support@akasha.ai"
```

This completes the comprehensive Component Specifications document covering all major components of the Akasha system. Each component is detailed with interfaces, implementations, configuration schemas, and integration patterns.