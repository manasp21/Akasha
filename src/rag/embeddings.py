"""
Embedding generation for Akasha RAG system using MLX backend.

This module provides embedding generation optimized for Apple Silicon using MLX,
with support for various embedding models and efficient batch processing.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .ingestion import DocumentChunk


class EmbeddingBackend(str, Enum):
    """Supported embedding backends."""
    MLX = "mlx"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    # MLX optimized models
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    BGE_SMALL_EN = "bge-small-en"
    BGE_BASE_EN = "bge-base-en"
    BGE_LARGE_EN = "bge-large-en"
    E5_SMALL_V2 = "e5-small-v2"
    E5_BASE_V2 = "e5-base-v2"
    E5_LARGE_V2 = "e5-large-v2"
    # JINA models
    JINA_V2_BASE = "jinaai/jina-embeddings-v2-base-en"
    JINA_V2_SMALL = "jinaai/jina-embeddings-v2-small-en"
    JINA_V3 = "jinaai/jina-embeddings-v3"
    JINA_V4 = "jinaai/jina-embeddings-v4"
    # Multimodal models
    CLIP_VIT_B32 = "openai/clip-vit-base-patch32"
    CLIP_VIT_L14 = "openai/clip-vit-large-patch14"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    backend: EmbeddingBackend = EmbeddingBackend.MLX
    model_name: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6_V2
    model_path: Optional[str] = None
    device: str = "auto"
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_dir: Optional[str] = None
    api_key: Optional[str] = None
    multimodal: bool = False  # Enable multimodal processing


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model_name: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Embedding dimensions")
    processing_time: float = Field(..., description="Time taken to generate embeddings")
    input_count: int = Field(..., description="Number of inputs processed")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: EmbeddingConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.model = None
        self.model_loaded = False
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the embedding model."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    async def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of images."""
        # Default implementation - override in multimodal providers
        raise NotImplementedError("Image embedding not supported by this provider")
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the model."""
        pass
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks."""
        if not self.model_loaded:
            await self.load_model()
        
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks


class MLXEmbeddingProvider(EmbeddingProvider):
    """MLX-based embedding provider optimized for Apple Silicon."""
    
    def __init__(self, config: EmbeddingConfig, logger=None):
        super().__init__(config, logger)
        self._mlx_available = None
        self._sentence_transformers = None
    
    async def _check_mlx_availability(self) -> bool:
        """Check if MLX is available."""
        if self._mlx_available is None:
            try:
                import mlx.core as mx
                import mlx.nn as nn
                self._mlx_available = True
                self.logger.info("MLX backend available")
            except ImportError:
                self._mlx_available = False
                self.logger.warning("MLX not available, falling back to sentence-transformers")
        return self._mlx_available
    
    async def _import_sentence_transformers(self):
        """Lazy import sentence-transformers."""
        if self._sentence_transformers is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformers = SentenceTransformer
            except ImportError:
                raise AkashaError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
    
    async def load_model(self) -> None:
        """Load the embedding model."""
        if self.model_loaded:
            return
        
        async with PerformanceLogger(f"load_embedding_model:{self.config.model_name}", self.logger):
            # Check if MLX is available, otherwise fall back to sentence-transformers
            mlx_available = await self._check_mlx_availability()
            
            if mlx_available:
                await self._load_mlx_model()
            else:
                await self._load_sentence_transformers_model()
            
            self.model_loaded = True
            self.logger.info(
                "Embedding model loaded successfully",
                model_name=self.config.model_name,
                backend="mlx" if mlx_available else "sentence_transformers",
                dimensions=self.get_embedding_dimensions()
            )
    
    async def _load_mlx_model(self) -> None:
        """Load model using MLX (when available)."""
        # For now, we'll use sentence-transformers as MLX embedding models
        # are still being developed. This can be updated when MLX embedding
        # models become available.
        await self._load_sentence_transformers_model()
    
    async def _load_sentence_transformers_model(self) -> None:
        """Load model using sentence-transformers."""
        await self._import_sentence_transformers()
        
        model_name = self.config.model_name.value
        if self.config.model_path:
            model_name = self.config.model_path
        
        # Load model in a thread to avoid blocking
        def _load_model():
            return self._sentence_transformers(
                model_name,
                device="mps" if self.config.device == "auto" else self.config.device,
                cache_folder=self.config.cache_dir
            )
        
        self.model = await asyncio.get_event_loop().run_in_executor(None, _load_model)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.model_loaded:
            await self.load_model()
        
        if not texts:
            return []
        
        async with PerformanceLogger(f"embed_texts:batch_size_{len(texts)}", self.logger):
            # Process in batches to manage memory
            all_embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings in a thread to avoid blocking
                def _encode_batch():
                    embeddings = self.model.encode(
                        batch_texts,
                        normalize_embeddings=self.config.normalize_embeddings,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    return embeddings.tolist()
                
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, _encode_batch
                )
                all_embeddings.extend(batch_embeddings)
            
            self.logger.debug(
                "Generated embeddings",
                text_count=len(texts),
                embedding_dimensions=len(all_embeddings[0]) if all_embeddings else 0
            )
            
            return all_embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions."""
        if not self.model_loaded or not self.model:
            # Return known dimensions for common models
            dimensions_map = {
                EmbeddingModel.ALL_MINILM_L6_V2: 384,
                EmbeddingModel.ALL_MPNET_BASE_V2: 768,
                EmbeddingModel.BGE_SMALL_EN: 384,
                EmbeddingModel.BGE_BASE_EN: 768,
                EmbeddingModel.BGE_LARGE_EN: 1024,
                EmbeddingModel.E5_SMALL_V2: 384,
                EmbeddingModel.E5_BASE_V2: 768,
                EmbeddingModel.E5_LARGE_V2: 1024,
                EmbeddingModel.JINA_V2_BASE: 768,
                EmbeddingModel.JINA_V2_SMALL: 512,
                EmbeddingModel.JINA_V3: 1024,
                EmbeddingModel.JINA_V4: 1024,
                EmbeddingModel.CLIP_VIT_B32: 512,
                EmbeddingModel.CLIP_VIT_L14: 768,
            }
            return dimensions_map.get(self.config.model_name, 384)
        
        # Get actual dimensions from loaded model
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        elif hasattr(self.model, '_modules') and 'word_embedding_dimension' in self.model._modules:
            return self.model._modules['word_embedding_dimension']
        else:
            # Fallback: encode a dummy text to get dimensions
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return dummy_embedding.shape[1]


class JINAEmbeddingProvider(EmbeddingProvider):
    """JINA embedding provider with fallback for different versions."""
    
    def __init__(self, config: EmbeddingConfig, logger=None):
        super().__init__(config, logger)
        self._sentence_transformers = None
        self.jina_version = None
        self.requires_trust_remote_code = True
    
    async def _import_sentence_transformers(self):
        """Lazy import sentence-transformers."""
        if self._sentence_transformers is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformers = SentenceTransformer
            except ImportError:
                raise AkashaError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
    
    def _detect_jina_version(self, model_name: str) -> str:
        """Detect JINA model version from model name."""
        if "jina-v4" in model_name or "jina-embeddings-v4" in model_name:
            return "v4"
        elif "jina-v3" in model_name or "jina-embeddings-v3" in model_name:
            return "v3"
        elif "jina-v2" in model_name or "jina-embeddings-v2" in model_name:
            return "v2"
        else:
            return "unknown"
    
    def _get_fallback_models(self, target_version: str) -> List[str]:
        """Get fallback models if target version fails."""
        if target_version == "v4":
            return [
                "jinaai/jina-embeddings-v3",
                "jinaai/jina-embeddings-v2-base-en",
                "all-MiniLM-L6-v2"  # Ultimate fallback
            ]
        elif target_version == "v3":
            return [
                "jinaai/jina-embeddings-v2-base-en",
                "all-MiniLM-L6-v2"
            ]
        else:
            return ["all-MiniLM-L6-v2"]
    
    async def load_model(self) -> None:
        """Load JINA embedding model with fallbacks."""
        if self.model_loaded:
            return
        
        await self._import_sentence_transformers()
        
        model_name = self.config.model_name.value
        if self.config.model_path:
            model_name = self.config.model_path
        
        self.jina_version = self._detect_jina_version(model_name)
        
        # Try to load the requested model first
        success = await self._try_load_model(model_name)
        
        if not success:
            # Try fallback models
            fallback_models = self._get_fallback_models(self.jina_version)
            for fallback_model in fallback_models:
                self.logger.warning(f"Trying fallback model: {fallback_model}")
                success = await self._try_load_model(fallback_model)
                if success:
                    model_name = fallback_model
                    self.jina_version = self._detect_jina_version(fallback_model)
                    break
        
        if not success:
            raise AkashaError(f"Failed to load any JINA embedding model")
        
        self.model_loaded = True
        self.logger.info(
            "JINA embedding model loaded successfully",
            model_name=model_name,
            version=self.jina_version,
            dimensions=self.get_embedding_dimensions()
        )
    
    async def _try_load_model(self, model_name: str) -> bool:
        """Try to load a specific model."""
        try:
            # Load model in a thread to avoid blocking
            def _load_model():
                return self._sentence_transformers(
                    model_name,
                    device="mps" if self.config.device == "auto" else self.config.device,
                    cache_folder=self.config.cache_dir,
                    trust_remote_code=self.requires_trust_remote_code
                )
            
            self.model = await asyncio.get_event_loop().run_in_executor(None, _load_model)
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load model {model_name}: {e}")
            
            # Check for specific dependency errors and log helpful messages
            error_str = str(e).lower()
            if "peft" in error_str:
                self.logger.warning("JINA v4 requires 'peft' package: pip install peft")
            elif "einops" in error_str:
                self.logger.warning("JINA v3 requires 'einops' package: pip install einops")
            elif "trust_remote_code" in error_str:
                self.logger.warning("Model requires trust_remote_code=True")
            
            return False
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using JINA model."""
        if not self.model_loaded:
            await self.load_model()
        
        async with PerformanceLogger(f"jina_embed_texts:count_{len(texts)}", self.logger):
            def _encode_texts():
                # JINA models work well with sentence-transformers interface
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            
            embeddings = await asyncio.get_event_loop().run_in_executor(None, _encode_texts)
            
            self.logger.debug(
                "Generated JINA embeddings",
                text_count=len(texts),
                embedding_count=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0,
                model_version=self.jina_version
            )
            
            return embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions for JINA models."""
        if not self.model:
            # Fallback dimensions based on known JINA models
            dimensions_map = {
                EmbeddingModel.JINA_V2_BASE: 768,
                EmbeddingModel.JINA_V2_SMALL: 512,
                EmbeddingModel.JINA_V3: 1024,
                EmbeddingModel.JINA_V4: 1024,  # May vary
            }
            return dimensions_map.get(self.config.model_name, 768)
        
        # Get dimensions from loaded model
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        else:
            # Fallback: encode a dummy text to get dimensions
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return dummy_embedding.shape[1]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded JINA model."""
        if not self.model_loaded:
            await self.load_model()
        
        info = {
            "provider": "JINA",
            "version": self.jina_version,
            "model_name": str(self.config.model_name.value),
            "dimensions": self.get_embedding_dimensions(),
            "supports_multimodal": self.jina_version in ["v3", "v4"],
            "max_sequence_length": getattr(self.model, 'max_seq_length', None),
        }
        
        # Add model-specific capabilities
        if hasattr(self.model, '_modules'):
            info["has_vision_model"] = 'vision_model' in str(self.model._modules)
        
        return info


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider (for comparison/fallback)."""
    
    def __init__(self, config: EmbeddingConfig, logger=None):
        super().__init__(config, logger)
        self._openai = None
        self.api_key = None
    
    async def _import_openai(self):
        """Lazy import OpenAI."""
        if self._openai is None:
            try:
                import openai
                self._openai = openai
            except ImportError:
                raise AkashaError("openai not installed. Install with: pip install openai")
    
    async def load_model(self) -> None:
        """Load OpenAI client."""
        if self.model_loaded:
            return
        
        await self._import_openai()
        
        # Get API key from config or environment
        import os
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AkashaError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.")
        
        self.model = self._openai.OpenAI(api_key=self.api_key)
        self.model_loaded = True
        
        self.logger.info("OpenAI embedding provider initialized")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not self.model_loaded:
            await self.load_model()
        
        if not texts:
            return []
        
        async with PerformanceLogger(f"openai_embed_texts:batch_size_{len(texts)}", self.logger):
            # OpenAI has rate limits, so process in smaller batches
            batch_size = min(self.config.batch_size, 100)  # OpenAI recommends max 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                def _get_embeddings():
                    response = self.model.embeddings.create(
                        input=batch_texts,
                        model="text-embedding-ada-002"  # OpenAI's latest embedding model
                    )
                    return [embedding.embedding for embedding in response.data]
                
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, _get_embeddings
                )
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get OpenAI embedding dimensions."""
        return 1536  # text-embedding-ada-002 dimensions


class MultimodalEmbeddingProvider(EmbeddingProvider):
    """Multimodal embedding provider using Hugging Face Transformers with CLIP."""
    
    def __init__(self, config: EmbeddingConfig, logger=None):
        super().__init__(config, logger)
        self._transformers = None
        self._torch = None
        self._pil = None
        self.processor = None
        
    async def _import_dependencies(self):
        """Lazy import dependencies."""
        if self._transformers is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                from PIL import Image
                self._transformers = (CLIPProcessor, CLIPModel)
                self._torch = torch
                self._pil = Image
            except ImportError as e:
                raise AkashaError(f"Required dependencies not installed: {e}")
    
    async def load_model(self) -> None:
        """Load the multimodal CLIP model."""
        if self.model_loaded:
            return
        
        await self._import_dependencies()
        CLIPProcessor, CLIPModel = self._transformers
        
        async with PerformanceLogger(f"load_multimodal_model:{self.config.model_name}", self.logger):
            model_name = self.config.model_name.value
            if self.config.model_path:
                model_name = self.config.model_path
            
            def _load_model():
                processor = CLIPProcessor.from_pretrained(model_name)
                model = CLIPModel.from_pretrained(model_name)
                
                # Move to appropriate device
                device = "mps" if self.config.device == "auto" and self._torch.backends.mps.is_available() else "cpu"
                model = model.to(device)
                
                return processor, model, device
            
            self.processor, self.model, self.device = await asyncio.get_event_loop().run_in_executor(
                None, _load_model
            )
            
            self.model_loaded = True
            self.logger.info(
                "Multimodal CLIP model loaded successfully",
                model_name=model_name,
                device=self.device,
                dimensions=self.get_embedding_dimensions()
            )
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using CLIP."""
        if not self.model_loaded:
            await self.load_model()
        
        if not texts:
            return []
        
        async with PerformanceLogger(f"clip_embed_texts:batch_size_{len(texts)}", self.logger):
            def _encode_texts():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with self._torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                    
                if self.config.normalize_embeddings:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                return text_features.cpu().numpy().tolist()
            
            embeddings = await asyncio.get_event_loop().run_in_executor(None, _encode_texts)
            
            self.logger.debug(
                "Generated text embeddings with CLIP",
                text_count=len(texts),
                embedding_dimensions=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
    
    async def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Generate embeddings for images using CLIP."""
        if not self.model_loaded:
            await self.load_model()
        
        if not image_paths:
            return []
        
        async with PerformanceLogger(f"clip_embed_images:batch_size_{len(image_paths)}", self.logger):
            def _encode_images():
                # Load images
                images = []
                for image_path in image_paths:
                    try:
                        image = self._pil.open(image_path).convert('RGB')
                        images.append(image)
                    except Exception as e:
                        self.logger.warning(f"Failed to load image {image_path}: {e}")
                        # Add a blank image as placeholder
                        images.append(self._pil.new('RGB', (224, 224), color='white'))
                
                if not images:
                    return []
                
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with self._torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    
                if self.config.normalize_embeddings:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().tolist()
            
            embeddings = await asyncio.get_event_loop().run_in_executor(None, _encode_images)
            
            self.logger.debug(
                "Generated image embeddings with CLIP",
                image_count=len(image_paths),
                embedding_dimensions=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
    
    def get_embedding_dimensions(self) -> int:
        """Get CLIP embedding dimensions."""
        if self.config.model_name == EmbeddingModel.CLIP_VIT_B32:
            return 512  # CLIP ViT-B/32 dimensions
        elif self.config.model_name == EmbeddingModel.CLIP_VIT_L14:
            return 768  # CLIP ViT-L/14 dimensions
        else:
            return 512  # Default CLIP dimensions


class EmbeddingGenerator:
    """Main embedding generator with support for multiple backends."""
    
    def __init__(self, config: EmbeddingConfig = None, logger=None):
        self.config = config or EmbeddingConfig()
        self.logger = logger or get_logger(__name__)
        
        # Initialize provider based on backend, model, and multimodal setting
        if self.config.multimodal or self.config.model_name in [EmbeddingModel.CLIP_VIT_B32, EmbeddingModel.CLIP_VIT_L14]:
            self.provider = MultimodalEmbeddingProvider(self.config, self.logger)
        elif self.config.model_name in [EmbeddingModel.JINA_V2_BASE, EmbeddingModel.JINA_V2_SMALL, 
                                       EmbeddingModel.JINA_V3, EmbeddingModel.JINA_V4]:
            # Use JINA provider for JINA models
            self.provider = JINAEmbeddingProvider(self.config, self.logger)
        elif self.config.backend == EmbeddingBackend.MLX:
            self.provider = MLXEmbeddingProvider(self.config, self.logger)
        elif self.config.backend == EmbeddingBackend.OPENAI:
            self.provider = OpenAIEmbeddingProvider(self.config, self.logger)
        else:
            # Default to MLX
            self.provider = MLXEmbeddingProvider(self.config, self.logger)
        
        self._embedding_cache = {}
    
    async def initialize(self) -> None:
        """Initialize the embedding generator."""
        await self.provider.load_model()
        
        self.logger.info(
            "Embedding generator initialized",
            backend=self.config.backend,
            model=self.config.model_name,
            dimensions=self.provider.get_embedding_dimensions()
        )
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        start_time = time.time()
        
        # Check cache if enabled
        if self.config.cache_embeddings:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                text_hash = hash(text)
                if text_hash in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text_hash]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = await self.provider.embed_texts(uncached_texts)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[hash(text)] = embedding
            else:
                new_embeddings = []
            
            # Combine cached and new embeddings in correct order
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for orig_idx, embedding in cached_embeddings:
                all_embeddings[orig_idx] = embedding
            
            # Place new embeddings
            for uncached_idx, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[uncached_idx] = embedding
            
            embeddings = all_embeddings
        else:
            embeddings = await self.provider.embed_texts(texts)
        
        processing_time = time.time() - start_time
        
        self.logger.info(
            "Generated embeddings",
            text_count=len(texts),
            processing_time=processing_time,
            cached_count=len(texts) - len(uncached_texts) if self.config.cache_embeddings else 0
        )
        
        return embeddings
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for document chunks."""
        if not chunks:
            return []
        
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        # For retrieval, we might want to add a query prefix for some models
        # For now, just embed the query as-is
        return await self.embed_text(query)
    
    async def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Generate embeddings for images (multimodal only)."""
        if hasattr(self.provider, 'embed_images'):
            return await self.provider.embed_images(image_paths)
        else:
            raise AkashaError("Image embedding not supported by current provider. Use multimodal configuration.")
    
    async def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for a single image."""
        embeddings = await self.embed_images([image_path])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.provider.get_embedding_dimensions()
    
    async def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "backend": self.config.backend,
            "model_name": self.config.model_name,
            "dimensions": self.provider.get_embedding_dimensions(),
            "batch_size": self.config.batch_size,
            "max_sequence_length": self.config.max_sequence_length,
            "cache_enabled": self.config.cache_embeddings,
            "cache_size": len(self._embedding_cache) if self.config.cache_embeddings else 0
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays for efficient computation
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)
    
    async def find_most_similar(self, query_embedding: List[float], 
                               chunk_embeddings: List[Tuple[str, List[float]]],
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar chunks to a query embedding."""
        if not query_embedding or not chunk_embeddings:
            return []
        
        similarities = []
        
        for chunk_id, chunk_embedding in chunk_embeddings:
            similarity = await self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]