"""
Configuration classes for LLM integration in Akasha.

This module defines configuration classes for various LLM backends
and generation parameters optimized for Apple Silicon.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class LLMBackend(str, Enum):
    """Supported LLM backends."""
    MLX = "mlx"
    LLAMACPP = "llamacpp"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelFormat(str, Enum):
    """Supported model formats."""
    MLX = "mlx"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"


class QuantizationType(str, Enum):
    """Quantization types for model optimization."""
    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="Model name/identifier")
    path: Optional[str] = Field(default=None, description="Local path to model files")
    hf_repo: Optional[str] = Field(default=None, description="HuggingFace repository")
    format: ModelFormat = Field(default=ModelFormat.MLX, description="Model format")
    quantization: QuantizationType = Field(default=QuantizationType.INT4, description="Quantization type")
    context_length: int = Field(default=4096, description="Maximum context length")
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size")
    hidden_size: Optional[int] = Field(default=None, description="Hidden dimension size")
    num_layers: Optional[int] = Field(default=None, description="Number of transformer layers")
    num_attention_heads: Optional[int] = Field(default=None, description="Number of attention heads")
    rope_theta: float = Field(default=10000.0, description="RoPE theta parameter")
    rope_scaling: Optional[Dict[str, Any]] = Field(default=None, description="RoPE scaling configuration")
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Custom model configuration")


class GenerationConfig(BaseModel):
    """Configuration for text generation."""
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(default=True, description="Enable streaming response")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class LLMConfig(BaseModel):
    """Main LLM configuration."""
    backend: LLMBackend = Field(default=LLMBackend.MLX, description="LLM backend to use")
    model: ModelConfig = Field(..., description="Model configuration")
    generation: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation parameters")
    
    # Performance settings
    batch_size: int = Field(default=1, description="Batch size for inference")
    max_concurrent_requests: int = Field(default=4, description="Maximum concurrent requests")
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit in GB")
    device: str = Field(default="auto", description="Device to use (auto, cpu, mps)")
    
    # MLX-specific settings
    mlx_cache_size: int = Field(default=1024, description="MLX KV cache size in MB")
    mlx_use_quantized_kv_cache: bool = Field(default=True, description="Use quantized KV cache")
    mlx_prefill_step_size: int = Field(default=512, description="Prefill step size")
    mlx_memory_pressure_threshold: float = Field(default=0.8, description="Memory pressure threshold")
    
    # Caching and optimization
    enable_kv_cache: bool = Field(default=True, description="Enable key-value caching")
    cache_warmup: bool = Field(default=True, description="Warm up cache on initialization")
    lazy_loading: bool = Field(default=True, description="Enable lazy model loading")
    
    # API settings (for external providers)
    api_key: Optional[str] = Field(default=None, description="API key for external services")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
    @classmethod
    def create_gemma_3_config(cls, model_path: Optional[str] = None) -> "LLMConfig":
        """Create optimized configuration for Gemma 3 27B on M4 Pro."""
        model = ModelConfig(
            name="gemma-3-27b-it",
            path=model_path,
            hf_repo="google/gemma-2-27b-it" if not model_path else None,
            format=ModelFormat.MLX,
            quantization=QuantizationType.INT4,
            context_length=8192,
            vocab_size=256000,
            hidden_size=4608,
            num_layers=46,
            num_attention_heads=32,
            rope_theta=10000.0
        )
        
        generation = GenerationConfig(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            stop_sequences=["<end_of_turn>", "<eos>"],
            stream=True
        )
        
        return cls(
            backend=LLMBackend.MLX,
            model=model,
            generation=generation,
            batch_size=1,
            max_concurrent_requests=2,
            memory_limit_gb=32.0,  # Conservative for 48GB system
            device="auto",
            mlx_cache_size=2048,
            mlx_use_quantized_kv_cache=True,
            mlx_prefill_step_size=512,
            mlx_memory_pressure_threshold=0.75,
            enable_kv_cache=True,
            cache_warmup=True,
            lazy_loading=True
        )
    
    @classmethod 
    def create_llama_config(cls, model_path: Optional[str] = None, size: str = "7b") -> "LLMConfig":
        """Create configuration for Llama models."""
        size_configs = {
            "7b": {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32},
            "13b": {"hidden_size": 5120, "num_layers": 40, "num_attention_heads": 40},
            "70b": {"hidden_size": 8192, "num_layers": 80, "num_attention_heads": 64}
        }
        
        config = size_configs.get(size, size_configs["7b"])
        
        model = ModelConfig(
            name=f"llama-2-{size}-chat",
            path=model_path,
            format=ModelFormat.MLX,
            quantization=QuantizationType.INT4,
            context_length=4096,
            **config
        )
        
        generation = GenerationConfig(
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["</s>", "[INST]", "[/INST]"],
            stream=True
        )
        
        return cls(
            backend=LLMBackend.MLX,
            model=model,
            generation=generation,
            memory_limit_gb=16.0 if size == "7b" else 32.0,
            mlx_cache_size=1024 if size == "7b" else 2048
        )
    
    @classmethod
    def create_openai_config(cls, model_name: str = "gpt-4", api_key: Optional[str] = None) -> "LLMConfig":
        """Create configuration for OpenAI models."""
        model = ModelConfig(
            name=model_name,
            format=ModelFormat.PYTORCH,  # Not used for API models
            context_length=8192 if "gpt-4" in model_name else 4096
        )
        
        generation = GenerationConfig(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )
        
        return cls(
            backend=LLMBackend.OPENAI,
            model=model,
            generation=generation,
            api_key=api_key,
            timeout=120
        )
    
    def validate_memory_requirements(self, available_memory_gb: float) -> bool:
        """Validate if the configuration fits within available memory."""
        # Rough memory estimation for different quantization types
        param_count_estimates = {
            "7b": 7,
            "13b": 13, 
            "27b": 27,
            "70b": 70
        }
        
        # Extract parameter count from model name
        param_count = 7  # Default
        for size, count in param_count_estimates.items():
            if size in self.model.name.lower():
                param_count = count
                break
        
        # Memory requirements based on quantization
        memory_multipliers = {
            QuantizationType.INT4: 0.5,
            QuantizationType.INT8: 1.0,
            QuantizationType.FP16: 2.0,
            QuantizationType.BF16: 2.0,
            QuantizationType.NONE: 4.0
        }
        
        multiplier = memory_multipliers.get(self.model.quantization, 2.0)
        estimated_memory = param_count * multiplier
        
        # Add overhead for KV cache and system
        overhead = 3.0 + (self.mlx_cache_size / 1024.0)  # Convert MB to GB
        total_estimated = estimated_memory + overhead
        
        return total_estimated <= available_memory_gb
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """Get detailed memory usage estimate."""
        param_count_estimates = {
            "7b": 7, "13b": 13, "27b": 27, "70b": 70
        }
        
        param_count = 7
        for size, count in param_count_estimates.items():
            if size in self.model.name.lower():
                param_count = count
                break
        
        memory_multipliers = {
            QuantizationType.INT4: 0.5,
            QuantizationType.INT8: 1.0, 
            QuantizationType.FP16: 2.0,
            QuantizationType.BF16: 2.0,
            QuantizationType.NONE: 4.0
        }
        
        multiplier = memory_multipliers.get(self.model.quantization, 2.0)
        model_memory = param_count * multiplier
        kv_cache_memory = self.mlx_cache_size / 1024.0
        system_overhead = 3.0
        
        total_memory = model_memory + kv_cache_memory + system_overhead
        
        return {
            "model_memory_gb": model_memory,
            "kv_cache_memory_gb": kv_cache_memory,
            "system_overhead_gb": system_overhead,
            "total_estimated_gb": total_memory,
            "parameter_count_b": param_count,
            "quantization": self.model.quantization.value
        }