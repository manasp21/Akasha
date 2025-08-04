"""
Large Language Model integration for Akasha RAG system.

This module provides LLM capabilities optimized for Apple Silicon using MLX,
with support for local model inference and response generation.
"""

from .provider import LLMProvider, MLXProvider
from .config import LLMConfig, ModelConfig, GenerationConfig
from .manager import LLMManager
from .templates import PromptTemplate, RAGTemplate

__all__ = [
    "LLMProvider",
    "MLXProvider", 
    "LLMConfig",
    "ModelConfig",
    "GenerationConfig",
    "LLMManager",
    "PromptTemplate",
    "RAGTemplate"
]