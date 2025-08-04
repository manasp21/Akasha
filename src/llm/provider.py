"""
LLM provider implementations for Akasha RAG system.

This module provides various LLM backends including MLX for Apple Silicon optimization,
with support for local and cloud-based language models.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable
import threading
from queue import Queue

from pydantic import BaseModel, Field

from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import AkashaError
from .config import LLMConfig, LLMBackend, GenerationConfig


class StreamingEvent(BaseModel):
    """Event in streaming response."""
    type: str = Field(..., description="Event type (token, finish, error)")
    content: str = Field(default="", description="Event content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    content: str = Field(..., description="Generated content")
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, description="Number of completion tokens") 
    total_tokens: int = Field(default=0, description="Total tokens used")
    generation_time: float = Field(default=0.0, description="Time taken for generation")
    model_name: str = Field(..., description="Model used for generation")
    finish_reason: str = Field(default="stop", description="Reason for completion")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig, logger=None):
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self._generation_lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[StreamingEvent]:
        """Generate text with streaming response."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion from messages."""
        # Convert messages to single prompt
        prompt = self._messages_to_prompt(messages)
        return await self.generate(prompt, **kwargs)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to single prompt."""
        # Simple implementation - can be enhanced based on model format
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "backend": self.config.backend,
            "model_name": self.config.model.name,
            "initialized": self.initialized,
            "context_length": self.config.model.context_length,
            "quantization": self.config.model.quantization
        }


class MLXProvider(LLMProvider):
    """MLX-based LLM provider optimized for Apple Silicon."""
    
    def __init__(self, config: LLMConfig, logger=None):
        super().__init__(config, logger)
        self._mlx_lm = None
        self._mlx = None
        self._mx = None
        self._kv_cache = None
        self._model_loaded = False
    
    async def _import_mlx(self):
        """Lazy import MLX modules."""
        if self._mlx_lm is None:
            try:
                import mlx.core as mx
                import mlx_lm
                from mlx_lm import load, generate
                
                self._mx = mx
                self._mlx_lm = mlx_lm
                self._load = load
                self._generate = generate
                
                self.logger.info("MLX modules imported successfully")
            except ImportError as e:
                raise AkashaError(f"MLX not available: {e}. Install with: pip install mlx-lm")
    
    async def initialize(self) -> None:
        """Initialize MLX model and tokenizer."""
        if self.initialized:
            return
        
        await self._import_mlx()
        
        async with PerformanceLogger(f"mlx_model_loading:{self.config.model.name}", self.logger):
            # Load model in thread to avoid blocking
            def _load_model():
                try:
                    # Determine model path or repo
                    model_path = self.config.model.path
                    if not model_path and self.config.model.hf_repo:
                        model_path = self.config.model.hf_repo
                    elif not model_path:
                        # Default to model name if no path specified
                        model_path = self.config.model.name
                    
                    self.logger.info(f"Loading MLX model from: {model_path}")
                    
                    # Load model and tokenizer
                    model, tokenizer = self._load(model_path)
                    
                    # Enable quantization if specified
                    if self.config.model.quantization.value != "none":
                        self.logger.info(f"Applying {self.config.model.quantization} quantization")
                        # MLX will handle quantization based on model format
                    
                    return model, tokenizer
                    
                except Exception as e:
                    raise AkashaError(f"Failed to load MLX model: {e}")
            
            self.model, self.tokenizer = await asyncio.get_event_loop().run_in_executor(
                None, _load_model
            )
            
            # Initialize KV cache if enabled
            if self.config.enable_kv_cache:
                self._initialize_kv_cache()
            
            self._model_loaded = True
            self.initialized = True
            
            self.logger.info(
                "MLX model initialized successfully",
                model_name=self.config.model.name,
                quantization=self.config.model.quantization,
                kv_cache_enabled=self.config.enable_kv_cache
            )
    
    def _initialize_kv_cache(self):
        """Initialize KV cache for efficient generation."""
        try:
            # MLX handles KV cache internally, but we can configure it
            cache_size_mb = self.config.mlx_cache_size
            self.logger.info(f"KV cache configured: {cache_size_mb}MB")
        except Exception as e:
            self.logger.warning(f"KV cache initialization failed: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using MLX."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Merge generation config with kwargs
        gen_config = self._merge_generation_config(kwargs)
        
        async with self._generation_lock:
            async with PerformanceLogger(f"mlx_generation:tokens_{gen_config['max_tokens']}", self.logger):
                # Generate in thread to avoid blocking
                def _generate():
                    try:
                        # Tokenize prompt
                        prompt_tokens = self.tokenizer.encode(prompt)
                        
                        # Check context length
                        if len(prompt_tokens) > self.config.model.context_length:
                            raise AkashaError(
                                f"Prompt too long: {len(prompt_tokens)} tokens > {self.config.model.context_length}"
                            )
                        
                        # Generate response
                        response = self._generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=gen_config["max_tokens"],
                            temp=gen_config["temperature"],
                            top_p=gen_config["top_p"],
                            repetition_penalty=gen_config["repetition_penalty"],
                            verbose=False
                        )
                        
                        # Extract generated text (MLX returns full text including prompt)
                        generated_text = response
                        if generated_text.startswith(prompt):
                            generated_text = generated_text[len(prompt):].strip()
                        
                        # Calculate token counts
                        completion_tokens = len(self.tokenizer.encode(generated_text))
                        total_tokens = len(prompt_tokens) + completion_tokens
                        
                        return {
                            "content": generated_text,
                            "prompt_tokens": len(prompt_tokens),
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                        
                    except Exception as e:
                        raise AkashaError(f"MLX generation failed: {e}")
                
                result = await asyncio.get_event_loop().run_in_executor(None, _generate)
                
                generation_time = time.time() - start_time
                
                response = LLMResponse(
                    content=result["content"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"], 
                    total_tokens=result["total_tokens"],
                    generation_time=generation_time,
                    model_name=self.config.model.name,
                    finish_reason="stop",
                    metadata={
                        "backend": "mlx",
                        "quantization": self.config.model.quantization.value,
                        "generation_config": gen_config
                    }
                )
                
                self.logger.debug(
                    "MLX generation completed",
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    generation_time=generation_time
                )
                
                return response
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[StreamingEvent]:
        """Generate streaming response using MLX."""
        if not self.initialized:
            await self.initialize()
        
        gen_config = self._merge_generation_config(kwargs)
        
        async with self._generation_lock:
            # For now, MLX doesn't support native streaming, so we'll simulate it
            # by generating in chunks or tokens
            try:
                response = await self.generate(prompt, **kwargs)
                
                # Simulate streaming by yielding words
                words = response.content.split()
                current_content = ""
                
                for word in words:
                    current_content += word + " "
                    yield StreamingEvent(
                        type="token",
                        content=word + " ",
                        metadata={"total_content": current_content.strip()}
                    )
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)
                
                # Final event
                yield StreamingEvent(
                    type="finish",
                    content="",
                    metadata={
                        "finish_reason": "stop",
                        "total_tokens": response.total_tokens,
                        "generation_time": response.generation_time
                    }
                )
                
            except Exception as e:
                yield StreamingEvent(
                    type="error",
                    content=str(e),
                    metadata={"error_type": type(e).__name__}
                )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tokenizer."""
        if not self.initialized or not self.tokenizer:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # Fallback to character estimation
            return len(text) // 4
    
    def _merge_generation_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default generation config with provided kwargs."""
        config = {
            "max_tokens": self.config.generation.max_tokens,
            "temperature": self.config.generation.temperature,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k,
            "repetition_penalty": self.config.generation.repetition_penalty,
            "stop_sequences": self.config.generation.stop_sequences,
            "stream": self.config.generation.stream,
            "seed": self.config.generation.seed
        }
        
        # Override with provided kwargs
        config.update(kwargs)
        return config
    
    async def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            if self._mx:
                # Get MLX memory info
                memory_info = self._mx.metal.get_active_memory() / (1024 ** 3)  # Convert to GB
                return {
                    "mlx_memory_gb": memory_info,
                    "kv_cache_mb": self.config.mlx_cache_size,
                    "model_loaded": self._model_loaded
                }
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
        
        return {
            "mlx_memory_gb": 0.0,
            "kv_cache_mb": self.config.mlx_cache_size,
            "model_loaded": self._model_loaded
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig, logger=None):
        super().__init__(config, logger)
        self._openai = None
        self.client = None
    
    async def _import_openai(self):
        """Lazy import OpenAI."""
        if self._openai is None:
            try:
                import openai
                self._openai = openai
            except ImportError:
                raise AkashaError("openai not installed. Install with: pip install openai")
    
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self.initialized:
            return
        
        await self._import_openai()
        
        # Get API key
        api_key = self.config.api_key
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise AkashaError("OpenAI API key not found. Set api_key in config or OPENAI_API_KEY environment variable.")
        
        # Create client
        client_kwargs = {"api_key": api_key}
        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base
        
        self.client = self._openai.OpenAI(**client_kwargs)
        self.initialized = True
        
        self.logger.info("OpenAI provider initialized")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using OpenAI API."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        gen_config = self._merge_generation_config(kwargs)
        
        async with self._generation_lock:
            def _generate():
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.model.name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=gen_config["max_tokens"],
                        temperature=gen_config["temperature"],
                        top_p=gen_config["top_p"],
                        frequency_penalty=gen_config.get("frequency_penalty", 0.0),
                        presence_penalty=gen_config.get("presence_penalty", 0.0),
                        stop=gen_config["stop_sequences"] if gen_config["stop_sequences"] else None
                    )
                    return response
                except Exception as e:
                    raise AkashaError(f"OpenAI API error: {e}")
            
            response = await asyncio.get_event_loop().run_in_executor(None, _generate)
            
            generation_time = time.time() - start_time
            choice = response.choices[0]
            
            return LLMResponse(
                content=choice.message.content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                generation_time=generation_time,
                model_name=self.config.model.name,
                finish_reason=choice.finish_reason,
                metadata={"backend": "openai"}
            )
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[StreamingEvent]:
        """Generate streaming response using OpenAI API."""
        if not self.initialized:
            await self.initialize()
        
        gen_config = self._merge_generation_config(kwargs)
        
        async with self._generation_lock:
            def _create_stream():
                try:
                    return self.client.chat.completions.create(
                        model=self.config.model.name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=gen_config["max_tokens"],
                        temperature=gen_config["temperature"],
                        top_p=gen_config["top_p"],
                        stream=True,
                        stop=gen_config["stop_sequences"] if gen_config["stop_sequences"] else None
                    )
                except Exception as e:
                    raise AkashaError(f"OpenAI API error: {e}")
            
            stream = await asyncio.get_event_loop().run_in_executor(None, _create_stream)
            
            try:
                for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.delta.content:
                            yield StreamingEvent(
                                type="token",
                                content=choice.delta.content
                            )
                        
                        if choice.finish_reason:
                            yield StreamingEvent(
                                type="finish",
                                content="",
                                metadata={"finish_reason": choice.finish_reason}
                            )
                            break
                            
            except Exception as e:
                yield StreamingEvent(
                    type="error",
                    content=str(e),
                    metadata={"error_type": type(e).__name__}
                )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for OpenAI models."""
        # Rough estimation for OpenAI models
        return len(text) // 4


def create_llm_provider(config: LLMConfig, logger=None) -> LLMProvider:
    """Factory function to create appropriate LLM provider."""
    if config.backend == LLMBackend.MLX:
        return MLXProvider(config, logger)
    elif config.backend == LLMBackend.OPENAI:
        return OpenAIProvider(config, logger)
    else:
        raise AkashaError(f"Unsupported LLM backend: {config.backend}")