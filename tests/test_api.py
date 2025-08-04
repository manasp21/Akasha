"""
Tests for the Akasha API endpoints.

This module tests the FastAPI application endpoints including
health checks, system status, and error handling.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.core.config import AkashaConfig
from src.core.exceptions import AkashaError


class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert "metadata" in data
        
        assert data["data"]["name"] == "akasha-test"
        assert data["data"]["version"] == "0.1.0"
        assert data["data"]["environment"] == "testing"
        assert "Welcome to Akasha" in data["data"]["message"]


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["version"] == "0.1.0"
        assert data["data"]["environment"] == "testing"
        assert "checks" in data["data"]
        assert data["data"]["checks"]["api"] == "pass"
        assert data["data"]["checks"]["config"] == "pass"
        assert data["data"]["checks"]["logging"] == "pass"
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_system_status_with_psutil(self, mock_disk, mock_cpu, mock_memory, client: TestClient):
        """Test system status endpoint with psutil available."""
        # Mock psutil responses
        mock_memory.return_value = MagicMock(
            percent=45.2,
            available=20 * 1024 * 1024 * 1024,  # 20GB
            total=48 * 1024 * 1024 * 1024        # 48GB
        )
        mock_cpu.return_value = 25.5
        mock_disk.return_value = MagicMock(
            used=100 * 1024 * 1024 * 1024,      # 100GB
            total=500 * 1024 * 1024 * 1024,     # 500GB
            free=400 * 1024 * 1024 * 1024       # 400GB
        )
        
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert data["data"]["version"] == "0.1.0"
        
        # Check resource information
        resources = data["data"]["resources"]
        assert resources["cpu_usage_percent"] == 25.5
        assert resources["memory_usage_percent"] == 45.2
        assert resources["memory_available_gb"] == 20.0
        assert resources["memory_total_gb"] == 48.0
        assert "disk_usage_percent" in resources
        assert "disk_free_gb" in resources
        
        # Check components
        components = data["data"]["components"]
        assert components["api_server"] == "healthy"
        assert components["configuration"] == "healthy"
        assert components["logging"] == "healthy"
        
        # Check configuration
        config = data["data"]["configuration"]
        assert config["max_memory_gb"] == 8
        assert config["debug_mode"] is True
        assert config["llm_backend"] == "mlx"
        assert config["llm_model"] == "test-model"
        assert config["vector_store"] == "chroma"
    
    def test_system_status_without_psutil(self, client: TestClient):
        """Test system status endpoint without psutil."""
        with patch.dict('sys.modules', {'psutil': None}):
            response = client.get("/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["data"]["status"] == "healthy"
            assert "Resource monitoring unavailable" in data["data"]["message"]


class TestConfigurationEndpoint:
    """Test configuration endpoint."""
    
    def test_get_configuration(self, client: TestClient):
        """Test getting system configuration."""
        response = client.get("/config")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        config = data["data"]
        
        # Check system configuration
        assert config["system"]["name"] == "akasha-test"
        assert config["system"]["version"] == "0.1.0"
        assert config["system"]["environment"] == "testing"
        assert config["system"]["debug"] is True
        assert config["system"]["max_memory_gb"] == 8
        
        # Check API configuration
        assert config["api"]["host"] == "127.0.0.1"
        assert config["api"]["port"] == 8001
        
        # Check LLM configuration
        assert config["llm"]["backend"] == "mlx"
        assert config["llm"]["model_name"] == "test-model"
        assert config["llm"]["quantization_bits"] == 4
        assert config["llm"]["memory_limit_gb"] == 2
        
        # Check features
        features = config["features"]
        assert features["multimodal_search"] is True
        assert features["streaming_chat"] is True
        assert features["plugin_support"] is True
        assert features["graph_rag"] is False
        
        # Ensure sensitive data is not exposed
        assert "jwt_secret_key" not in json.dumps(config)
        assert "secret" not in json.dumps(config).lower()


class TestErrorHandling:
    """Test API error handling."""
    
    def test_akasha_error_handling(self, client: TestClient):
        """Test custom Akasha error handling."""
        # Test that we have proper error handling structure by patching get_config
        # to raise an AkashaError
        
        def mock_config():
            raise AkashaError(
                message="Test error",
                error_code="TEST_ERROR",
                details="This is a test error",
                context={"test": "value"}
            )
        
        with patch('src.api.main.get_config', side_effect=mock_config):
            response = client.get("/config")
            
            assert response.status_code == 400
            data = response.json()
            
            assert data["success"] is False
            assert "error" in data
            assert data["error"]["error_code"] == "TEST_ERROR"
            assert data["error"]["message"] == "Test error"
            assert "metadata" in data
    
    def test_validation_error_handling(self, client: TestClient):
        """Test request validation error handling."""
        # Make a request with invalid data (if we had POST endpoints)
        # For now, this is more of a placeholder since our current endpoints
        # don't accept complex input data
        pass
    
    def test_404_error(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["code"] == "HTTP_404"


class TestMiddleware:
    """Test API middleware functionality."""
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are present."""
        response = client.get("/")
        
        # Check that the response completed successfully
        assert response.status_code == 200
        
        # In a real test, we'd check for CORS headers, but TestClient
        # doesn't simulate browser CORS behavior exactly
    
    def test_correlation_id_header(self, client: TestClient):
        """Test that correlation ID is added to response headers."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        
        # Correlation ID should be a valid UUID-like string
        correlation_id = response.headers["X-Correlation-ID"]
        assert len(correlation_id) > 0
        assert "-" in correlation_id  # Basic UUID format check
    
    def test_request_logging(self, client: TestClient):
        """Test that requests are logged properly."""
        # This would require checking log output, which is complex to test
        # For now, we just ensure the request completes successfully
        response = client.get("/health")
        assert response.status_code == 200


class TestResponseFormat:
    """Test API response format consistency."""
    
    def test_success_response_format(self, client: TestClient):
        """Test that successful responses follow the standard format."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check standard response structure
        assert "success" in data
        assert "data" in data
        assert "metadata" in data
        
        assert data["success"] is True
        assert isinstance(data["data"], dict)
        assert isinstance(data["metadata"], dict)
        
        # Check metadata structure
        metadata = data["metadata"]
        assert "timestamp" in metadata
        assert isinstance(metadata["timestamp"], (int, float))
    
    def test_error_response_format(self, client: TestClient):
        """Test that error responses follow the standard format."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        
        # Check standard error response structure
        assert "success" in data
        assert "error" in data
        assert "metadata" in data
        
        assert data["success"] is False
        assert isinstance(data["error"], dict)
        assert isinstance(data["metadata"], dict)
        
        # Check error structure
        error = data["error"]
        assert "code" in error
        assert "message" in error
        
        # Check metadata
        metadata = data["metadata"]
        assert "timestamp" in metadata
        assert "path" in metadata