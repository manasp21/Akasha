"""
Tests for the Akasha configuration system.

This module tests configuration loading, validation, and environment
variable handling.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.config import (
    AkashaConfig,
    SystemConfig,
    APIConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    load_config,
    get_config,
    reload_config
)
from src.core.exceptions import ConfigurationError


class TestSystemConfig:
    """Test system configuration validation."""
    
    def test_default_values(self):
        """Test default system configuration values."""
        config = SystemConfig()
        
        assert config.name == "akasha"
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.debug is True
        assert config.max_memory_gb == 40
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "production", "testing"]:
            config = SystemConfig(environment=env)
            assert config.environment == env
        
        # Invalid environment
        with pytest.raises(ValueError, match="Environment must be one of"):
            SystemConfig(environment="invalid")


class TestAPIConfig:
    """Test API configuration validation."""
    
    def test_default_values(self):
        """Test default API configuration values."""
        config = APIConfig()
        
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1
        assert config.reload is True
        assert config.cors_origins == ["*"]
    
    def test_port_validation(self):
        """Test port number validation."""
        # Valid ports
        for port in [80, 8000, 65535]:
            config = APIConfig(port=port)
            assert config.port == port
        
        # Invalid ports
        for port in [0, 65536, -1]:
            with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
                APIConfig(port=port)


class TestLLMConfig:
    """Test LLM configuration validation."""
    
    def test_default_values(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        
        assert config.backend == "mlx"
        assert config.model_name == "gemma-3-27b"
        assert config.quantization_bits == 4
        assert config.max_tokens == 2048
        assert config.memory_limit_gb == 16
    
    def test_backend_validation(self):
        """Test backend validation."""
        # Valid backends
        for backend in ["mlx", "llama_cpp"]:
            config = LLMConfig(backend=backend)
            assert config.backend == backend
        
        # Invalid backend
        with pytest.raises(ValueError, match="Backend must be one of"):
            LLMConfig(backend="invalid")
    
    def test_quantization_validation(self):
        """Test quantization bits validation."""
        # Valid quantization bits
        for bits in [4, 8, 16]:
            config = LLMConfig(quantization_bits=bits)
            assert config.quantization_bits == bits
        
        # Invalid quantization bits
        with pytest.raises(ValueError, match="Quantization bits must be one of"):
            LLMConfig(quantization_bits=32)


class TestEmbeddingConfig:
    """Test embedding configuration validation."""
    
    def test_default_values(self):
        """Test default embedding configuration values."""
        config = EmbeddingConfig()
        
        assert config.model == "jina-v4"
        assert config.backend == "local"
        assert config.dimensions == 512
        assert config.batch_size == 16
        assert config.memory_limit_gb == 4
    
    def test_dimensions_validation(self):
        """Test dimensions validation."""
        # Valid dimensions
        config = EmbeddingConfig(dimensions=768)
        assert config.dimensions == 768
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            EmbeddingConfig(dimensions=0)


class TestVectorStoreConfig:
    """Test vector store configuration validation."""
    
    def test_default_values(self):
        """Test default vector store configuration values."""
        config = VectorStoreConfig()
        
        assert config.backend == "chroma"
        assert config.collection_name == "documents"
        assert config.memory_limit_gb == 8
        assert config.persist_directory == "./data/vectors"
    
    def test_backend_validation(self):
        """Test backend validation."""
        # Valid backends
        for backend in ["chroma", "qdrant"]:
            config = VectorStoreConfig(backend=backend)
            assert config.backend == backend
        
        # Invalid backend
        with pytest.raises(ValueError, match="Backend must be one of"):
            VectorStoreConfig(backend="invalid")


class TestAkashaConfig:
    """Test main Akasha configuration."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = AkashaConfig()
        
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.vector_store, VectorStoreConfig)
    
    def test_memory_validation_success(self):
        """Test successful memory validation."""
        config = AkashaConfig()
        # Should not raise an exception with default values
        config.validate_memory_limits()
    
    def test_memory_validation_failure(self):
        """Test memory validation failure."""
        config = AkashaConfig()
        config.system.max_memory_gb = 10  # Set very low limit
        
        with pytest.raises(ValueError, match="Total memory allocation.*exceeds system maximum"):
            config.validate_memory_limits()
    
    def test_from_yaml(self, temp_dir: Path):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "test_config.yaml"
        
        config_data = {
            "system": {
                "name": "test-akasha",
                "environment": "testing",
                "max_memory_gb": 20
            },
            "api": {
                "port": 9000
            },
            "llm": {
                "backend": "llama_cpp",
                "model_name": "test-model"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = AkashaConfig.from_yaml(config_file)
        
        assert config.system.name == "test-akasha"
        assert config.system.environment == "testing"
        assert config.system.max_memory_gb == 20
        assert config.api.port == 9000
        assert config.llm.backend == "llama_cpp"
        assert config.llm.model_name == "test-model"
    
    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            AkashaConfig.from_yaml("non_existent_file.yaml")
    
    def test_from_yaml_invalid_yaml(self, temp_dir: Path):
        """Test loading from invalid YAML file."""
        config_file = temp_dir / "invalid.yaml"
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            AkashaConfig.from_yaml(config_file)
    
    def test_to_yaml(self, temp_dir: Path):
        """Test saving configuration to YAML file."""
        config = AkashaConfig()
        config.system.name = "test-save"
        config.api.port = 9001
        
        output_file = temp_dir / "output_config.yaml"
        config.to_yaml(output_file)
        
        assert output_file.exists()
        
        # Load and verify
        loaded_config = AkashaConfig.from_yaml(output_file)
        assert loaded_config.system.name == "test-save"
        assert loaded_config.api.port == 9001
    
    def test_create_directories(self, temp_dir: Path):
        """Test directory creation."""
        config = AkashaConfig()
        config.vector_store.persist_directory = str(temp_dir / "vectors")
        config.ingestion.temp_directory = str(temp_dir / "temp")
        config.cache.cache_directory = str(temp_dir / "cache")
        config.plugins.plugin_directory = str(temp_dir / "plugins")
        config.logging.file_path = str(temp_dir / "logs" / "test.log")
        config.llm.model_path = str(temp_dir / "models")
        
        config.create_directories()
        
        assert (temp_dir / "vectors").exists()
        assert (temp_dir / "temp").exists()
        assert (temp_dir / "cache").exists()
        assert (temp_dir / "plugins").exists()
        assert (temp_dir / "logs").exists()
        assert (temp_dir / "models").exists()
    
    def test_get_environment_specific_config(self):
        """Test environment-specific configuration overrides."""
        config = AkashaConfig()
        
        # Test development environment
        config.system.environment = "development"
        env_config = config.get_environment_specific_config()
        assert env_config["system"]["debug"] is True
        assert env_config["logging"]["level"] == "DEBUG"
        
        # Test production environment
        config.system.environment = "production"
        env_config = config.get_environment_specific_config()
        assert env_config["system"]["debug"] is False
        assert env_config["logging"]["level"] == "INFO"
        
        # Test testing environment
        config.system.environment = "testing"
        env_config = config.get_environment_specific_config()
        assert env_config["cache"]["enable_disk_cache"] is False


class TestConfigurationLoading:
    """Test configuration loading functions."""
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, AkashaConfig)
    
    def test_load_config_from_file(self, temp_dir: Path):
        """Test loading configuration from specific file."""
        config_file = temp_dir / "custom_config.yaml"
        
        config_data = {
            "system": {"name": "custom-akasha"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        assert config.system.name == "custom-akasha"
    
    def test_load_config_environment_variable(self, temp_dir: Path, monkeypatch):
        """Test loading configuration from environment variable."""
        config_file = temp_dir / "env_config.yaml"
        
        config_data = {
            "system": {"name": "env-akasha"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        monkeypatch.setenv("AKASHA_CONFIG", str(config_file))
        
        config = load_config()
        assert config.system.name == "env-akasha"
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_reload_config(self, temp_dir: Path):
        """Test configuration reloading."""
        config_file = temp_dir / "reload_config.yaml"
        
        config_data = {
            "system": {"name": "reload-test"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = reload_config(config_file)
        assert config.system.name == "reload-test"