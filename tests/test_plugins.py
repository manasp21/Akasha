"""
Tests for the Akasha plugin system.

This module tests plugin interfaces, manager, and registry functionality.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.plugins.base import (
    PluginInterface,
    PluginMetadata,
    PluginType,
    PluginStatus,
    PluginCapability,
    DocumentProcessorInterface,
    EmbeddingModelInterface,
    get_plugin_interface
)
from src.plugins.manager import PluginManager
from src.plugins.registry import PluginRegistry, PluginRegistryEntry
from src.core.exceptions import (
    PluginError,
    PluginNotFoundError,
    PluginLoadError,
    PluginExecutionError,
    PluginTimeoutError
)


class TestPluginInterface:
    """Test base plugin interface."""
    
    def test_plugin_metadata_creation(self, mock_plugin_metadata):
        """Test plugin metadata creation."""
        metadata = mock_plugin_metadata
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.CUSTOM
        assert len(metadata.capabilities) == 1
        assert metadata.capabilities[0].name == "test_capability"
    
    def test_get_plugin_interface(self):
        """Test getting plugin interface by type."""
        # Test known interfaces
        assert get_plugin_interface(PluginType.DOCUMENT_PROCESSOR) == DocumentProcessorInterface
        assert get_plugin_interface(PluginType.EMBEDDING_MODEL) == EmbeddingModelInterface
        assert get_plugin_interface(PluginType.CUSTOM) == PluginInterface
        
        # Test unknown interface
        with pytest.raises(PluginError):
            get_plugin_interface("unknown_type")


class MockPlugin(PluginInterface):
    """Mock plugin for testing."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.initialized = False
        self.processed_data = []
        self.cleanup_called = False
    
    async def initialize(self, config):
        """Initialize the mock plugin."""
        await asyncio.sleep(0.01)  # Simulate some work
        self.config = config
        self.initialized = True
        self.status = PluginStatus.ACTIVE
    
    async def process(self, input_data, **kwargs):
        """Process input data."""
        if not self.initialized:
            raise PluginError("Plugin not initialized")
        
        await asyncio.sleep(0.01)  # Simulate processing
        processed = f"processed_{input_data}"
        self.processed_data.append(processed)
        return processed
    
    async def cleanup(self):
        """Clean up the plugin."""
        self.cleanup_called = True
        self.status = PluginStatus.INACTIVE


class SlowPlugin(PluginInterface):
    """Plugin that takes a long time to process (for timeout testing)."""
    
    async def initialize(self, config):
        """Initialize the plugin."""
        self.config = config
        self.status = PluginStatus.ACTIVE
    
    async def process(self, input_data, **kwargs):
        """Process with intentional delay."""
        await asyncio.sleep(1.0)  # Longer than typical timeout
        return f"slow_{input_data}"
    
    async def cleanup(self):
        """Clean up the plugin."""
        self.status = PluginStatus.INACTIVE


class FailingPlugin(PluginInterface):
    """Plugin that always fails (for error testing)."""
    
    async def initialize(self, config):
        """Initialize the plugin."""
        raise PluginError("Initialization failed")
    
    async def process(self, input_data, **kwargs):
        """Process that always fails."""
        raise PluginError("Processing failed")
    
    async def cleanup(self):
        """Clean up the plugin."""
        pass


class TestPluginManager:
    """Test plugin manager functionality."""
    
    @pytest.fixture
    def plugin_manager(self, temp_dir):
        """Create a plugin manager for testing."""
        manager = PluginManager()
        manager.config.plugin_directory = str(temp_dir / "plugins")
        return manager
    
    @pytest.fixture
    def mock_plugin(self, mock_plugin_metadata):
        """Create a mock plugin instance."""
        return MockPlugin(mock_plugin_metadata)
    
    def test_plugin_manager_initialization(self, plugin_manager):
        """Test plugin manager initialization."""
        assert isinstance(plugin_manager._plugins, dict)
        assert isinstance(plugin_manager._active_plugins, set)
        assert len(plugin_manager._plugins) == 0
        assert len(plugin_manager._active_plugins) == 0
    
    @pytest.mark.asyncio
    async def test_plugin_activation_success(self, plugin_manager, mock_plugin):
        """Test successful plugin activation."""
        plugin_name = "test_plugin"
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._plugin_metadata[plugin_name] = mock_plugin.metadata
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 0,
            "error_count": 0
        }
        
        config = {"test_param": "test_value"}
        result = await plugin_manager.activate_plugin(plugin_name, config)
        
        assert result is True
        assert mock_plugin.status == PluginStatus.ACTIVE
        assert mock_plugin.initialized is True
        assert mock_plugin.config == config
        assert plugin_name in plugin_manager._active_plugins
    
    @pytest.mark.asyncio
    async def test_plugin_activation_not_found(self, plugin_manager):
        """Test plugin activation with non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            await plugin_manager.activate_plugin("nonexistent_plugin")
    
    @pytest.mark.asyncio
    async def test_plugin_activation_failure(self, plugin_manager, mock_plugin_metadata):
        """Test plugin activation failure."""
        plugin_name = "failing_plugin"
        failing_plugin = FailingPlugin(mock_plugin_metadata)
        
        plugin_manager._plugins[plugin_name] = failing_plugin
        plugin_manager._plugin_metadata[plugin_name] = mock_plugin_metadata
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 0,
            "error_count": 0
        }
        
        with pytest.raises(PluginExecutionError):
            await plugin_manager.activate_plugin(plugin_name)
        
        assert failing_plugin.status == PluginStatus.ERROR
        assert plugin_name not in plugin_manager._active_plugins
    
    @pytest.mark.asyncio
    async def test_plugin_deactivation(self, plugin_manager, mock_plugin):
        """Test plugin deactivation."""
        plugin_name = "test_plugin"
        mock_plugin.status = PluginStatus.ACTIVE
        
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._active_plugins.add(plugin_name)
        
        result = await plugin_manager.deactivate_plugin(plugin_name)
        
        assert result is True
        assert mock_plugin.cleanup_called is True
        assert mock_plugin.status == PluginStatus.INACTIVE
        assert plugin_name not in plugin_manager._active_plugins
    
    @pytest.mark.asyncio
    async def test_plugin_execution_success(self, plugin_manager, mock_plugin):
        """Test successful plugin execution."""
        plugin_name = "test_plugin"
        mock_plugin.status = PluginStatus.ACTIVE
        mock_plugin.initialized = True
        
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 0,
            "total_execution_time": 0.0,
            "error_count": 0
        }
        
        input_data = "test_input"
        result = await plugin_manager.execute_plugin(plugin_name, input_data)
        
        assert result == "processed_test_input"
        assert input_data in [data.replace("processed_", "") for data in mock_plugin.processed_data]
        
        # Check statistics were updated
        stats = plugin_manager._plugin_stats[plugin_name]
        assert stats["execution_count"] == 1
        assert stats["total_execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_plugin_execution_timeout(self, plugin_manager, mock_plugin_metadata):
        """Test plugin execution timeout."""
        plugin_name = "slow_plugin"
        
        # Create metadata with very short timeout
        metadata = PluginMetadata(
            name=plugin_name,
            version="1.0.0",
            description="Slow plugin",
            author="Test",
            plugin_type=PluginType.CUSTOM,
            akasha_version="0.1.0",
            timeout_seconds=1  # Very short timeout (1 second)
        )
        
        slow_plugin = SlowPlugin(metadata)
        slow_plugin.status = PluginStatus.ACTIVE
        
        plugin_manager._plugins[plugin_name] = slow_plugin
        plugin_manager._plugin_metadata[plugin_name] = metadata
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 0,
            "error_count": 0
        }
        
        with pytest.raises(PluginTimeoutError):
            await plugin_manager.execute_plugin(plugin_name, "test_input")
    
    @pytest.mark.asyncio
    async def test_plugin_info(self, plugin_manager, mock_plugin):
        """Test getting plugin information."""
        plugin_name = "test_plugin"
        
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._plugin_metadata[plugin_name] = mock_plugin.metadata
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 5,
            "total_execution_time": 1.0,
            "error_count": 1,
            "load_time": 1234567890
        }
        plugin_manager._execution_times[plugin_name] = [0.1, 0.2, 0.3, 0.2, 0.2]
        
        info = await plugin_manager.get_plugin_info(plugin_name)
        
        assert info is not None
        assert info["name"] == "test_plugin"
        assert info["version"] == "1.0.0"
        assert info["type"] == "custom"
        assert info["status"] == mock_plugin.status.value
        assert info["statistics"]["execution_count"] == 5
        assert info["statistics"]["error_count"] == 1
        assert info["statistics"]["average_execution_time"] == 0.2
        assert info["statistics"]["success_rate"] == 0.8  # 4/5 success rate
    
    @pytest.mark.asyncio
    async def test_list_plugins(self, plugin_manager, mock_plugin):
        """Test listing plugins."""
        plugin_name = "test_plugin"
        
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._plugin_metadata[plugin_name] = mock_plugin.metadata
        plugin_manager._plugin_stats[plugin_name] = {
            "execution_count": 0,
            "error_count": 0
        }
        
        plugins = await plugin_manager.list_plugins()
        
        assert len(plugins) == 1
        assert plugins[0]["name"] == "test_plugin"
        
        # Test filtering by type
        custom_plugins = await plugin_manager.list_plugins(PluginType.CUSTOM)
        assert len(custom_plugins) == 1
        
        doc_plugins = await plugin_manager.list_plugins(PluginType.DOCUMENT_PROCESSOR)
        assert len(doc_plugins) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, plugin_manager, mock_plugin):
        """Test plugin health check."""
        plugin_name = "test_plugin"
        mock_plugin.status = PluginStatus.ACTIVE
        
        plugin_manager._plugins[plugin_name] = mock_plugin
        plugin_manager._active_plugins.add(plugin_name)
        
        health = await plugin_manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["total_plugins"] == 1
        assert health["active_plugins"] == 1
        assert health["failed_plugins"] == 0
        assert plugin_name in health["plugin_health"]
        assert health["plugin_health"][plugin_name]["healthy"] is True


class TestPluginRegistry:
    """Test plugin registry functionality."""
    
    @pytest.fixture
    def plugin_registry(self, temp_dir):
        """Create a plugin registry for testing."""
        registry = PluginRegistry()
        registry.config.plugin_directory = str(temp_dir / "plugins")
        registry._registry_file = temp_dir / "plugins" / "registry.json"
        
        # Clear any existing entries to ensure test isolation
        registry._entries.clear()
        registry._dependencies.clear()
        registry._dependents.clear()
        
        return registry
    
    def test_registry_initialization(self, plugin_registry):
        """Test registry initialization."""
        assert isinstance(plugin_registry._entries, dict)
        assert len(plugin_registry._entries) == 0
    
    def test_register_plugin(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test plugin registration."""
        plugin_name = "test_plugin"
        plugin_file = temp_dir / "test_plugin.py"
        
        # Create a dummy plugin file
        plugin_file.write_text("# Test plugin")
        
        result = plugin_registry.register_plugin(
            plugin_name,
            mock_plugin_metadata,
            str(plugin_file)
        )
        
        assert result is True
        assert plugin_name in plugin_registry._entries
        
        entry = plugin_registry._entries[plugin_name]
        assert entry.metadata.name == "test_plugin"
        assert entry.file_path == str(plugin_file)
        assert entry.enabled is True
    
    def test_register_plugin_duplicate(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test registering duplicate plugin."""
        plugin_name = "test_plugin"
        plugin_file = temp_dir / "test_plugin.py"
        plugin_file.write_text("# Test plugin")
        
        # Register first time
        plugin_registry.register_plugin(plugin_name, mock_plugin_metadata, str(plugin_file))
        
        # Try to register again without force
        with pytest.raises(PluginError, match="already exists"):
            plugin_registry.register_plugin(plugin_name, mock_plugin_metadata, str(plugin_file))
        
        # Register with force should work
        result = plugin_registry.register_plugin(
            plugin_name, mock_plugin_metadata, str(plugin_file), force_update=True
        )
        assert result is True
    
    def test_unregister_plugin(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test plugin unregistration."""
        plugin_name = "test_plugin"
        plugin_file = temp_dir / "test_plugin.py"
        plugin_file.write_text("# Test plugin")
        
        # Register plugin first
        plugin_registry.register_plugin(plugin_name, mock_plugin_metadata, str(plugin_file))
        assert plugin_name in plugin_registry._entries
        
        # Unregister plugin
        result = plugin_registry.unregister_plugin(plugin_name)
        assert result is True
        assert plugin_name not in plugin_registry._entries
    
    def test_unregister_nonexistent_plugin(self, plugin_registry):
        """Test unregistering non-existent plugin."""
        with pytest.raises(PluginNotFoundError):
            plugin_registry.unregister_plugin("nonexistent_plugin")
    
    def test_enable_disable_plugin(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test enabling and disabling plugins."""
        plugin_name = "test_plugin"
        plugin_file = temp_dir / "test_plugin.py"
        plugin_file.write_text("# Test plugin")
        
        # Register plugin
        plugin_registry.register_plugin(plugin_name, mock_plugin_metadata, str(plugin_file))
        
        # Plugin should be enabled by default
        entry = plugin_registry._entries[plugin_name]
        assert entry.enabled is True
        
        # Disable plugin
        result = plugin_registry.disable_plugin(plugin_name)
        assert result is True
        assert entry.enabled is False
        
        # Enable plugin
        result = plugin_registry.enable_plugin(plugin_name)
        assert result is True
        assert entry.enabled is True
    
    def test_list_plugins(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test listing plugins."""
        plugin_file = temp_dir / "test_plugin.py"
        plugin_file.write_text("# Test plugin")
        
        # Register plugin
        plugin_registry.register_plugin("test_plugin", mock_plugin_metadata, str(plugin_file))
        
        # List all plugins
        all_plugins = plugin_registry.list_plugins()
        assert "test_plugin" in all_plugins
        
        # List enabled plugins only
        enabled_plugins = plugin_registry.list_plugins(enabled_only=True)
        assert "test_plugin" in enabled_plugins
        
        # Disable plugin and list enabled only
        plugin_registry.disable_plugin("test_plugin")
        enabled_plugins = plugin_registry.list_plugins(enabled_only=True)
        assert "test_plugin" not in enabled_plugins
    
    def test_get_registry_stats(self, plugin_registry, mock_plugin_metadata, temp_dir):
        """Test getting registry statistics."""
        plugin_file = temp_dir / "test_plugin.py"
        plugin_file.write_text("# Test plugin")
        
        # Register plugin
        plugin_registry.register_plugin("test_plugin", mock_plugin_metadata, str(plugin_file))
        
        stats = plugin_registry.get_registry_stats()
        
        assert stats["total_plugins"] == 1
        assert stats["enabled_plugins"] == 1
        assert stats["disabled_plugins"] == 0
        assert "custom" in stats["plugins_by_type"]
        assert stats["plugins_by_type"]["custom"] == 1