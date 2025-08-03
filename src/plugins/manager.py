"""
Plugin manager for the Akasha plugin system.

This module provides comprehensive plugin management including discovery,
loading, validation, execution, and lifecycle management with security
and resource monitoring.
"""

import asyncio
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set
import weakref

from ..core.config import get_config
from ..core.logging import get_logger, PerformanceLogger
from ..core.exceptions import (
    PluginError, 
    PluginNotFoundError, 
    PluginLoadError, 
    PluginExecutionError,
    PluginTimeoutError
)
from .base import (
    PluginInterface, 
    PluginMetadata, 
    PluginType, 
    PluginStatus,
    get_plugin_interface
)


class PluginManager:
    """
    Manages the lifecycle of plugins in the Akasha system.
    
    Provides plugin discovery, loading, validation, execution,
    and resource management with security controls.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config().plugins
        
        # Plugin storage
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
        self._plugin_modules: Dict[str, Any] = {}
        
        # Plugin tracking
        self._active_plugins: Set[str] = set()
        self._failed_plugins: Set[str] = set()
        self._plugin_stats: Dict[str, Dict[str, Any]] = {}
        
        # Resource monitoring
        self._execution_times: Dict[str, List[float]] = {}
        self._memory_usage: Dict[str, List[float]] = {}
        
        # Security tracking
        self._plugin_permissions: Dict[str, Set[str]] = {}
        
        self.logger.info("Plugin manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the plugin manager and discover plugins."""
        self.logger.info("Initializing plugin manager")
        
        # Create plugin directory if it doesn't exist
        plugin_dir = Path(self.config.plugin_directory)
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover and load plugins
        await self.discover_plugins()
        
        self.logger.info(
            "Plugin manager initialized",
            total_plugins=len(self._plugins),
            active_plugins=len(self._active_plugins),
            failed_plugins=len(self._failed_plugins)
        )
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover plugins in the plugin directory.
        
        Returns:
            List of discovered plugin names
        """
        plugin_dir = Path(self.config.plugin_directory)
        discovered_plugins = []
        
        if not plugin_dir.exists():
            self.logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return discovered_plugins
        
        self.logger.info(f"Discovering plugins in {plugin_dir}")
        
        # Look for Python files and packages
        for item in plugin_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                plugin_name = item.stem
                await self._load_plugin_from_file(item, plugin_name)
                discovered_plugins.append(plugin_name)
            
            elif item.is_dir() and not item.name.startswith('_'):
                plugin_file = item / '__init__.py'
                if plugin_file.exists():
                    plugin_name = item.name
                    await self._load_plugin_from_file(plugin_file, plugin_name)
                    discovered_plugins.append(plugin_name)
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins", plugins=discovered_plugins)
        return discovered_plugins
    
    async def _load_plugin_from_file(self, plugin_file: Path, plugin_name: str) -> bool:
        """
        Load a plugin from a Python file.
        
        Args:
            plugin_file: Path to plugin file
            plugin_name: Name of the plugin
            
        Returns:
            True if plugin loaded successfully
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            if spec is None or spec.loader is None:
                raise PluginLoadError(plugin_name, "Could not create module spec")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin class and metadata
            plugin_class = None
            metadata = None
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Look for plugin metadata
                if isinstance(attr, PluginMetadata):
                    metadata = attr
                
                # Look for plugin class
                elif (inspect.isclass(attr) and 
                      issubclass(attr, PluginInterface) and 
                      attr != PluginInterface):
                    plugin_class = attr
            
            if plugin_class is None:
                raise PluginLoadError(plugin_name, "No plugin class found")
            
            if metadata is None:
                raise PluginLoadError(plugin_name, "No plugin metadata found")
            
            # Validate plugin
            await self._validate_plugin(plugin_class, metadata)
            
            # Create plugin instance
            plugin_instance = plugin_class(metadata)
            plugin_instance.logger = get_logger(f"plugin.{plugin_name}")
            
            # Store plugin
            self._plugins[plugin_name] = plugin_instance
            self._plugin_metadata[plugin_name] = metadata
            self._plugin_modules[plugin_name] = module
            
            # Initialize plugin statistics
            self._plugin_stats[plugin_name] = {
                "load_time": time.time(),
                "execution_count": 0,
                "total_execution_time": 0.0,
                "last_execution": None,
                "error_count": 0,
                "last_error": None
            }
            
            self.logger.info(
                "Plugin loaded successfully",
                plugin_name=plugin_name,
                plugin_type=metadata.plugin_type.value,
                version=metadata.version
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to load plugin",
                plugin_name=plugin_name,
                error=str(e),
                exc_info=True
            )
            self._failed_plugins.add(plugin_name)
            return False
    
    async def _validate_plugin(self, plugin_class: Type[PluginInterface], metadata: PluginMetadata) -> None:
        """
        Validate a plugin class and metadata.
        
        Args:
            plugin_class: Plugin class to validate
            metadata: Plugin metadata to validate
            
        Raises:
            PluginLoadError: If validation fails
        """
        # Check if plugin type interface is implemented correctly
        expected_interface = get_plugin_interface(metadata.plugin_type)
        
        if not issubclass(plugin_class, expected_interface):
            raise PluginLoadError(
                metadata.name,
                f"Plugin does not implement required interface: {expected_interface.__name__}"
            )
        
        # Validate metadata
        if not metadata.name:
            raise PluginLoadError(metadata.name, "Plugin name is required")
        
        if not metadata.version:
            raise PluginLoadError(metadata.name, "Plugin version is required")
        
        # Check resource limits
        if metadata.memory_limit_mb and metadata.memory_limit_mb > 1024:  # 1GB limit
            raise PluginLoadError(
                metadata.name,
                f"Memory limit too high: {metadata.memory_limit_mb}MB"
            )
        
        # Validate dependencies (basic check)
        for dep in metadata.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                raise PluginLoadError(
                    metadata.name,
                    f"Required dependency not available: {dep}"
                )
    
    async def activate_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Activate a plugin with optional configuration.
        
        Args:
            plugin_name: Name of plugin to activate
            config: Optional configuration dictionary
            
        Returns:
            True if plugin activated successfully
            
        Raises:
            PluginNotFoundError: If plugin not found
            PluginExecutionError: If activation fails
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(plugin_name)
        
        plugin = self._plugins[plugin_name]
        
        if plugin.status == PluginStatus.ACTIVE:
            self.logger.info(f"Plugin {plugin_name} is already active")
            return True
        
        try:
            plugin.status = PluginStatus.LOADING
            
            # Use default config if none provided
            if config is None:
                config = plugin.metadata.default_config.copy()
            
            # Validate configuration
            await plugin.validate_config(config)
            
            # Initialize plugin with timeout
            with PerformanceLogger(f"plugin_activation:{plugin_name}", self.logger):
                await asyncio.wait_for(
                    plugin.initialize(config),
                    timeout=plugin.metadata.timeout_seconds
                )
            
            plugin.config = config
            plugin.status = PluginStatus.ACTIVE
            self._active_plugins.add(plugin_name)
            
            self.logger.info(
                "Plugin activated successfully",
                plugin_name=plugin_name,
                plugin_type=plugin.metadata.plugin_type.value
            )
            
            return True
            
        except asyncio.TimeoutError:
            plugin.status = PluginStatus.ERROR
            error_msg = f"Plugin activation timed out after {plugin.metadata.timeout_seconds}s"
            self.logger.error("Plugin activation timeout", plugin_name=plugin_name)
            raise PluginTimeoutError(plugin_name, plugin.metadata.timeout_seconds)
            
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            self._plugin_stats[plugin_name]["error_count"] += 1
            self._plugin_stats[plugin_name]["last_error"] = str(e)
            
            self.logger.error(
                "Plugin activation failed",
                plugin_name=plugin_name,
                error=str(e),
                exc_info=True
            )
            raise PluginExecutionError(plugin_name, "activate", str(e))
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate a plugin.
        
        Args:
            plugin_name: Name of plugin to deactivate
            
        Returns:
            True if plugin deactivated successfully
        """
        if plugin_name not in self._plugins:
            self.logger.warning(f"Plugin {plugin_name} not found for deactivation")
            return False
        
        plugin = self._plugins[plugin_name]
        
        if plugin.status != PluginStatus.ACTIVE:
            self.logger.info(f"Plugin {plugin_name} is not active")
            return True
        
        try:
            await plugin.cleanup()
            plugin.status = PluginStatus.INACTIVE
            self._active_plugins.discard(plugin_name)
            
            self.logger.info("Plugin deactivated", plugin_name=plugin_name)
            return True
            
        except Exception as e:
            self.logger.error(
                "Plugin deactivation failed",
                plugin_name=plugin_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def execute_plugin(
        self, 
        plugin_name: str, 
        input_data: Any, 
        **kwargs
    ) -> Any:
        """
        Execute a plugin's main processing method.
        
        Args:
            plugin_name: Name of plugin to execute
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            Plugin execution result
            
        Raises:
            PluginNotFoundError: If plugin not found
            PluginExecutionError: If execution fails
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(plugin_name)
        
        plugin = self._plugins[plugin_name]
        
        if plugin.status != PluginStatus.ACTIVE:
            raise PluginExecutionError(
                plugin_name, 
                "execute", 
                f"Plugin is not active (status: {plugin.status.value})"
            )
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                plugin.process(input_data, **kwargs),
                timeout=plugin.metadata.timeout_seconds
            )
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_plugin_stats(plugin_name, execution_time, success=True)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self._update_plugin_stats(plugin_name, execution_time, success=False)
            
            self.logger.error(
                "Plugin execution timeout",
                plugin_name=plugin_name,
                timeout=plugin.metadata.timeout_seconds
            )
            raise PluginTimeoutError(plugin_name, plugin.metadata.timeout_seconds)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_plugin_stats(plugin_name, execution_time, success=False)
            
            self.logger.error(
                "Plugin execution failed",
                plugin_name=plugin_name,
                error=str(e),
                execution_time=execution_time,
                exc_info=True
            )
            raise PluginExecutionError(plugin_name, "execute", str(e))
    
    def _update_plugin_stats(self, plugin_name: str, execution_time: float, success: bool) -> None:
        """Update plugin execution statistics."""
        stats = self._plugin_stats.get(plugin_name, {})
        
        stats["execution_count"] = stats.get("execution_count", 0) + 1
        stats["total_execution_time"] = stats.get("total_execution_time", 0.0) + execution_time
        stats["last_execution"] = time.time()
        
        if not success:
            stats["error_count"] = stats.get("error_count", 0) + 1
        
        # Track execution times for performance monitoring
        if plugin_name not in self._execution_times:
            self._execution_times[plugin_name] = []
        
        self._execution_times[plugin_name].append(execution_time)
        
        # Keep only last 100 execution times
        if len(self._execution_times[plugin_name]) > 100:
            self._execution_times[plugin_name] = self._execution_times[plugin_name][-100:]
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin information dictionary or None if not found
        """
        if plugin_name not in self._plugins:
            return None
        
        plugin = self._plugins[plugin_name]
        metadata = self._plugin_metadata[plugin_name]
        stats = self._plugin_stats.get(plugin_name, {})
        
        # Calculate performance metrics
        execution_times = self._execution_times.get(plugin_name, [])
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "type": metadata.plugin_type.value,
            "status": plugin.status.value,
            "capabilities": [cap.dict() for cap in metadata.capabilities],
            "config": plugin.config,
            "statistics": {
                **stats,
                "average_execution_time": round(avg_execution_time, 3),
                "success_rate": self._calculate_success_rate(plugin_name)
            },
            "metadata": {
                "akasha_version": metadata.akasha_version,
                "python_version": metadata.python_version,
                "dependencies": metadata.dependencies,
                "memory_limit_mb": metadata.memory_limit_mb,
                "timeout_seconds": metadata.timeout_seconds,
                "sandbox_enabled": metadata.sandbox_enabled
            }
        }
    
    def _calculate_success_rate(self, plugin_name: str) -> float:
        """Calculate plugin success rate."""
        stats = self._plugin_stats.get(plugin_name, {})
        total_executions = stats.get("execution_count", 0)
        error_count = stats.get("error_count", 0)
        
        if total_executions == 0:
            return 1.0
        
        return round((total_executions - error_count) / total_executions, 3)
    
    async def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """
        List all plugins with optional filtering by type.
        
        Args:
            plugin_type: Optional plugin type filter
            
        Returns:
            List of plugin information dictionaries
        """
        plugins = []
        
        for plugin_name in self._plugins:
            plugin_info = await self.get_plugin_info(plugin_name)
            if plugin_info:
                if plugin_type is None or plugin_info["type"] == plugin_type.value:
                    plugins.append(plugin_info)
        
        return sorted(plugins, key=lambda x: x["name"])
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """
        Get list of plugin names by type.
        
        Args:
            plugin_type: Plugin type to filter by
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for plugin_name, metadata in self._plugin_metadata.items():
            if metadata.plugin_type == plugin_type:
                plugins.append(plugin_name)
        
        return sorted(plugins)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all active plugins.
        
        Returns:
            Health check results
        """
        health_status = {
            "status": "healthy",
            "total_plugins": len(self._plugins),
            "active_plugins": len(self._active_plugins),
            "failed_plugins": len(self._failed_plugins),
            "plugin_health": {}
        }
        
        for plugin_name in self._active_plugins:
            try:
                plugin = self._plugins[plugin_name]
                plugin_health = await plugin.health_check()
                health_status["plugin_health"][plugin_name] = plugin_health
                
                if not plugin_health.get("healthy", False):
                    health_status["status"] = "degraded"
                    
            except Exception as e:
                health_status["plugin_health"][plugin_name] = {
                    "status": "error",
                    "error": str(e),
                    "healthy": False
                }
                health_status["status"] = "degraded"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up all plugins and resources."""
        self.logger.info("Cleaning up plugin manager")
        
        # Deactivate all active plugins
        for plugin_name in list(self._active_plugins):
            await self.deactivate_plugin(plugin_name)
        
        # Clear all data structures
        self._plugins.clear()
        self._plugin_metadata.clear()
        self._plugin_modules.clear()
        self._active_plugins.clear()
        self._failed_plugins.clear()
        self._plugin_stats.clear()
        self._execution_times.clear()
        self._memory_usage.clear()
        
        self.logger.info("Plugin manager cleanup completed")


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.
    
    Returns:
        PluginManager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager