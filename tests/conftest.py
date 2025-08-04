"""
Pytest configuration and shared fixtures for Akasha tests.

This module provides common test fixtures and configuration for
the entire test suite.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.core.config import AkashaConfig
from src.api.main import create_app


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> AkashaConfig:
    """Create a test configuration."""
    config_data = {
        "system": {
            "name": "akasha-test",
            "version": "0.1.0",
            "environment": "testing",
            "debug": True,
            "max_memory_gb": 8
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8001,
            "workers": 1,
            "reload": False
        },
        "llm": {
            "backend": "mlx",
            "model_name": "test-model",
            "model_path": str(temp_dir / "models"),
            "quantization_bits": 4,
            "memory_limit_gb": 2
        },
        "embedding": {
            "model": "test-embedding",
            "batch_size": 4,
            "memory_limit_gb": 1
        },
        "vector_store": {
            "backend": "chroma",
            "collection_name": "test_documents",
            "memory_limit_gb": 1,
            "persist_directory": str(temp_dir / "vectors")
        },
        "ingestion": {
            "temp_directory": str(temp_dir / "temp")
        },
        "cache": {
            "memory_limit_gb": 1,
            "cache_directory": str(temp_dir / "cache"),
            "enable_disk_cache": False
        },
        "logging": {
            "level": "WARNING",
            "output": "console",
            "file_path": str(temp_dir / "test.log")
        },
        "plugins": {
            "plugin_directory": str(temp_dir / "plugins")
        }
    }
    
    return AkashaConfig(**config_data)


@pytest.fixture
def client(test_config: AkashaConfig) -> TestClient:
    """Create a test client for the FastAPI application."""
    # Set test config as global config for the duration of the test
    import src.core.config
    original_config = src.core.config.config
    src.core.config.config = test_config
    
    try:
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Restore original config
        src.core.config.config = original_config


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create sample PDF bytes for testing."""
    # Simple PDF header (for testing purposes only)
    return b"%PDF-1.4\n%Test PDF content\nendobj\n%%EOF"


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    This is a sample text document for testing purposes.
    It contains multiple paragraphs and some basic formatting.
    
    The document discusses machine learning and natural language processing.
    These are important topics in artificial intelligence research.
    """


@pytest.fixture
def sample_document_metadata() -> dict:
    """Sample document metadata for testing."""
    return {
        "title": "Test Document",
        "authors": ["Test Author"],
        "publication_date": "2025-01-01",
        "tags": ["test", "sample"],
        "category": "testing"
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Set test environment variables
    monkeypatch.setenv("AKASHA_SYSTEM__ENVIRONMENT", "testing")
    monkeypatch.setenv("AKASHA_SYSTEM__DEBUG", "true")
    monkeypatch.setenv("AKASHA_LOGGING__LEVEL", "WARNING")
    
    # Ensure we don't accidentally use production resources
    monkeypatch.setenv("AKASHA_VECTOR_STORE__BACKEND", "chroma")
    monkeypatch.setenv("AKASHA_CACHE__ENABLE_DISK_CACHE", "false")


@pytest.fixture
def mock_plugin_metadata():
    """Create mock plugin metadata for testing."""
    from src.plugins.base import PluginMetadata, PluginType, PluginCapability
    
    capability = PluginCapability(
        name="test_capability",
        description="Test capability for unit tests",
        input_types=["text"],
        output_types=["text"],
        parameters={}
    )
    
    return PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        description="Test plugin for unit tests",
        author="Test Author",
        plugin_type=PluginType.CUSTOM,
        capabilities=[capability],
        akasha_version="0.1.0",
        timeout_seconds=10
    )