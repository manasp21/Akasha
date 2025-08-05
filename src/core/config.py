"""
Configuration management for Akasha system.

This module provides comprehensive configuration management using Pydantic for
validation and YAML for human-readable configuration files. It supports
environment-specific configurations and runtime overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class SystemConfig(BaseModel):
    """System-level configuration."""
    
    name: str = Field(default="akasha", description="System name")
    version: str = Field(default="0.1.0", description="System version")
    environment: str = Field(default="development", description="Environment (development, production, testing)")
    debug: bool = Field(default=True, description="Enable debug mode")
    max_memory_gb: int = Field(default=40, description="Maximum memory usage in GB (for M4 Pro 48GB)")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["development", "production", "testing"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v


class APIConfig(BaseModel):
    """API server configuration."""
    
    host: str = Field(default="127.0.0.1", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    max_request_size: int = Field(default=100 * 1024 * 1024, description="Max request size in bytes (100MB)")
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class LLMConfig(BaseModel):
    """LLM service configuration optimized for Apple Silicon M4 Pro."""
    
    backend: str = Field(default="mlx", description="LLM backend (mlx, llama_cpp)")
    model_name: str = Field(default="gemma-3-27b", description="Model name")
    model_path: str = Field(default="./models", description="Path to model files")
    quantization_bits: int = Field(default=4, description="Quantization bits (4, 8, 16)")
    max_tokens: int = Field(default=2048, description="Maximum tokens per generation")
    temperature: float = Field(default=0.7, description="Generation temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    memory_limit_gb: int = Field(default=16, description="Memory limit for LLM in GB")
    
    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        valid_backends = ["mlx", "llama_cpp"]
        if v not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}")
        return v
    
    @field_validator("quantization_bits")
    @classmethod
    def validate_quantization(cls, v):
        valid_bits = [4, 8, 16]
        if v not in valid_bits:
            raise ValueError(f"Quantization bits must be one of {valid_bits}")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""
    
    model: str = Field(default="jina-v4", description="Embedding model")
    backend: str = Field(default="local", description="Embedding backend")
    dimensions: int = Field(default=512, description="Embedding dimensions")
    batch_size: int = Field(default=16, description="Batch size for processing")
    memory_limit_gb: int = Field(default=4, description="Memory limit for embeddings in GB")
    cache_embeddings: bool = Field(default=True, description="Enable embedding caching")
    
    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v


class VectorStoreConfig(BaseModel):
    """Vector storage configuration."""
    
    backend: str = Field(default="chroma", description="Vector store backend (chroma, qdrant)")
    collection_name: str = Field(default="documents", description="Collection name")
    memory_limit_gb: int = Field(default=8, description="Memory limit for vector store in GB")
    cache_size_gb: int = Field(default=4, description="Cache size in GB")
    persist_directory: str = Field(default="./data/vectors", description="Persistence directory")
    
    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        valid_backends = ["chroma", "qdrant"]
        if v not in valid_backends:
            raise ValueError(f"Backend must be one of {valid_backends}")
        return v


class IngestionConfig(BaseModel):
    """Document ingestion configuration."""
    
    backend: str = Field(default="mineru2", description="Ingestion backend")
    batch_size: int = Field(default=4, description="Processing batch size")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    enable_ocr: bool = Field(default=True, description="Enable OCR processing")
    extract_images: bool = Field(default=True, description="Extract images from documents")
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    output_format: str = Field(default="markdown", description="Output format (markdown, json)")
    temp_directory: str = Field(default="./data/temp", description="Temporary processing directory")


class CacheConfig(BaseModel):
    """Cache configuration."""
    
    memory_limit_gb: int = Field(default=6, description="Memory limit for cache in GB")
    enable_disk_cache: bool = Field(default=True, description="Enable disk-based cache")
    cache_directory: str = Field(default="./data/cache", description="Cache directory")
    ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_entries: int = Field(default=10000, description="Maximum cache entries")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="structured", description="Log format (structured, plain)")
    output: str = Field(default="console", description="Log output (console, file, both)")
    file_path: str = Field(default="./logs/akasha.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    jwt_secret_key: str = Field(default="dev-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration in minutes")
    api_key_length: int = Field(default=32, description="API key length")
    bcrypt_rounds: int = Field(default=12, description="Bcrypt rounds")


class PluginConfig(BaseModel):
    """Plugin system configuration."""
    
    plugin_directory: str = Field(default="./plugins", description="Plugin directory")
    enable_hot_reload: bool = Field(default=True, description="Enable plugin hot reloading")
    max_plugins: int = Field(default=50, description="Maximum number of plugins")
    sandbox_enabled: bool = Field(default=True, description="Enable plugin sandboxing")
    timeout_seconds: int = Field(default=30, description="Plugin execution timeout")


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    performance_logging: bool = Field(default=True, description="Enable performance logging")


class AkashaConfig(BaseSettings):
    """
    Main configuration class for Akasha system.
    
    This class aggregates all configuration sections and provides
    methods for loading from YAML files and environment variables.
    """
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    model_config = ConfigDict(
        env_prefix="AKASHA_",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "AkashaConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            AkashaConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
        
        if config_data is None:
            config_data = {}
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def validate_memory_limits(self) -> None:
        """
        Validate that total memory limits don't exceed system maximum.
        
        Raises:
            ValueError: If memory limits exceed system maximum
        """
        total_memory = (
            self.llm.memory_limit_gb +
            self.embedding.memory_limit_gb +
            self.vector_store.memory_limit_gb +
            self.cache.memory_limit_gb
        )
        
        if total_memory > self.system.max_memory_gb:
            raise ValueError(
                f"Total memory allocation ({total_memory}GB) exceeds "
                f"system maximum ({self.system.max_memory_gb}GB)"
            )
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.vector_store.persist_directory,
            self.ingestion.temp_directory,
            self.cache.cache_directory,
            self.plugins.plugin_directory,
            Path(self.logging.file_path).parent,
            Path(self.llm.model_path),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration overrides.
        
        Returns:
            Dictionary of environment-specific settings
        """
        env_configs = {
            "development": {
                "system": {"debug": True},
                "api": {"reload": True, "workers": 1},
                "logging": {"level": "DEBUG"},
                "security": {"jwt_expire_minutes": 60},
            },
            "production": {
                "system": {"debug": False},
                "api": {"reload": False, "workers": 4},
                "logging": {"level": "INFO", "output": "file"},
                "security": {"jwt_expire_minutes": 15},
            },
            "testing": {
                "system": {"debug": True},
                "api": {"reload": False, "workers": 1},
                "logging": {"level": "WARNING"},
                "cache": {"enable_disk_cache": False},
            }
        }
        
        return env_configs.get(self.system.environment, {})


def load_config(config_path: Optional[Union[str, Path]] = None) -> AkashaConfig:
    """
    Load Akasha configuration from file or environment.
    
    Args:
        config_path: Optional path to configuration file.
                    If not provided, looks for AKASHA_CONFIG env var
                    or defaults to ./config/akasha.yaml
    
    Returns:
        AkashaConfig instance
    """
    if config_path is None:
        # Try environment variable first
        config_path = os.getenv("AKASHA_CONFIG")
        
        if config_path is None:
            # Default configuration paths
            default_paths = [
                Path("./config/akasha.yaml"),
                Path("./akasha.yaml"),
                Path("./config/config.yaml"),
            ]
            
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
    
    if config_path is None:
        # No config file found, use defaults with environment overrides
        config = AkashaConfig()
    else:
        # Load from file
        config = AkashaConfig.from_yaml(config_path)
    
    # Apply environment-specific overrides
    env_overrides = config.get_environment_specific_config()
    if env_overrides:
        for section, values in env_overrides.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    # Validate memory limits
    config.validate_memory_limits()
    
    # Create necessary directories
    config.create_directories()
    
    return config


# Global configuration instance
config: Optional[AkashaConfig] = None

# Export main config class for easier imports
Config = AkashaConfig


def get_config() -> AkashaConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AkashaConfig instance
    """
    global config
    if config is None:
        config = load_config()
    return config


def reload_config(config_path: Optional[Union[str, Path]] = None) -> AkashaConfig:
    """
    Reload the global configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        New AkashaConfig instance
    """
    global config
    config = load_config(config_path)
    return config