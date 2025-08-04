"""
Plugin registry for the Akasha plugin system.

This module provides a centralized registry for managing plugin metadata,
versions, dependencies, and providing plugin discovery and installation
capabilities.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import PluginError, PluginNotFoundError
from .base import PluginMetadata, PluginType, PluginCapability


class PluginRegistryEntry:
    """Represents an entry in the plugin registry."""
    
    def __init__(
        self,
        metadata: PluginMetadata,
        file_path: str,
        file_hash: str,
        install_date: datetime,
        last_updated: datetime,
        enabled: bool = True
    ):
        self.metadata = metadata
        self.file_path = file_path
        self.file_hash = file_hash
        self.install_date = install_date
        self.last_updated = last_updated
        self.enabled = enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "metadata": self.metadata.model_dump(),
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "install_date": self.install_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginRegistryEntry":
        """Create entry from dictionary."""
        # Convert capability dictionaries back to PluginCapability objects
        capabilities = []
        for cap_data in data["metadata"].get("capabilities", []):
            capability = PluginCapability(
                name=cap_data["name"],
                description=cap_data["description"],
                input_types=cap_data["input_types"],
                output_types=cap_data["output_types"],
                parameters=cap_data["parameters"]
            )
            capabilities.append(capability)
        
        # Create metadata object
        metadata_dict = data["metadata"].copy()
        metadata_dict["capabilities"] = capabilities
        metadata_dict["plugin_type"] = PluginType(metadata_dict["plugin_type"])
        
        metadata = PluginMetadata(**metadata_dict)
        
        return cls(
            metadata=metadata,
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            install_date=datetime.fromisoformat(data["install_date"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            enabled=data.get("enabled", True)
        )


class PluginRegistry:
    """
    Central registry for managing plugin metadata and installation.
    
    Provides functionality for plugin registration, discovery, dependency
    resolution, and version management.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config().plugins
        
        # Registry storage
        self._entries: Dict[str, PluginRegistryEntry] = {}
        self._registry_file = Path(self.config.plugin_directory) / "registry.json"
        
        # Plugin relationships
        self._dependencies: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load plugin registry from file."""
        if not self._registry_file.exists():
            self.logger.info("No existing plugin registry found, creating new one")
            return
        
        try:
            with open(self._registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            for plugin_name, entry_data in registry_data.get("plugins", {}).items():
                try:
                    entry = PluginRegistryEntry.from_dict(entry_data)
                    self._entries[plugin_name] = entry
                    self._build_dependency_graph(plugin_name, entry.metadata)
                except Exception as e:
                    self.logger.error(
                        "Failed to load registry entry",
                        plugin_name=plugin_name,
                        error=str(e)
                    )
            
            self.logger.info(
                "Plugin registry loaded",
                total_plugins=len(self._entries),
                enabled_plugins=len([e for e in self._entries.values() if e.enabled])
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load plugin registry",
                registry_file=str(self._registry_file),
                error=str(e)
            )
    
    def _save_registry(self) -> None:
        """Save plugin registry to file."""
        try:
            # Ensure registry directory exists
            self._registry_file.parent.mkdir(parents=True, exist_ok=True)
            
            registry_data = {
                "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "plugins": {
                    name: entry.to_dict() 
                    for name, entry in self._entries.items()
                }
            }
            
            with open(self._registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Plugin registry saved")
            
        except Exception as e:
            self.logger.error(
                "Failed to save plugin registry",
                registry_file=str(self._registry_file),
                error=str(e)
            )
    
    def _build_dependency_graph(self, plugin_name: str, metadata: PluginMetadata) -> None:
        """Build dependency relationships for a plugin."""
        # Clear existing dependencies for this plugin
        if plugin_name in self._dependencies:
            for dep in self._dependencies[plugin_name]:
                self._dependents.get(dep, set()).discard(plugin_name)
        
        # Build new dependencies
        dependencies = set()
        
        # Add explicit dependencies from metadata
        for dep in metadata.dependencies:
            dependencies.add(dep)
        
        self._dependencies[plugin_name] = dependencies
        
        # Update dependents
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(plugin_name)
    
    def register_plugin(
        self,
        plugin_name: str,
        metadata: PluginMetadata,
        file_path: str,
        force_update: bool = False
    ) -> bool:
        """
        Register a plugin in the registry.
        
        Args:
            plugin_name: Name of the plugin
            metadata: Plugin metadata
            file_path: Path to plugin file
            force_update: Force update if plugin already exists
            
        Returns:
            True if registration successful
            
        Raises:
            PluginError: If registration fails
        """
        # Check if plugin already exists
        if plugin_name in self._entries and not force_update:
            existing_version = self._entries[plugin_name].metadata.version
            new_version = metadata.version
            
            raise PluginError(
                f"Plugin {plugin_name} already exists with version {existing_version}. "
                f"Use force_update=True to register version {new_version}"
            )
        
        # Validate plugin file
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise PluginError(f"Plugin file not found: {file_path}")
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path_obj)
        
        # Validate dependencies
        self._validate_dependencies(metadata)
        
        # Create registry entry
        now = datetime.now(timezone.utc)
        entry = PluginRegistryEntry(
            metadata=metadata,
            file_path=file_path,
            file_hash=file_hash,
            install_date=now,
            last_updated=now,
            enabled=True
        )
        
        # Register the plugin
        self._entries[plugin_name] = entry
        self._build_dependency_graph(plugin_name, metadata)
        
        # Save registry
        self._save_registry()
        
        self.logger.info(
            "Plugin registered successfully",
            plugin_name=plugin_name,
            version=metadata.version,
            plugin_type=metadata.plugin_type.value
        )
        
        return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_dependencies(self, metadata: PluginMetadata) -> None:
        """
        Validate plugin dependencies.
        
        Args:
            metadata: Plugin metadata to validate
            
        Raises:
            PluginError: If dependencies are invalid
        """
        for dep in metadata.dependencies:
            # Check if dependency is a Python module
            try:
                import importlib
                importlib.import_module(dep)
            except ImportError:
                # Check if it's another plugin
                if dep not in self._entries:
                    raise PluginError(
                        f"Dependency not found: {dep}. "
                        "Ensure all required packages are installed or dependent plugins are registered."
                    )
    
    def unregister_plugin(self, plugin_name: str, force: bool = False) -> bool:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_name: Name of plugin to unregister
            force: Force unregistration even if other plugins depend on it
            
        Returns:
            True if unregistration successful
            
        Raises:
            PluginNotFoundError: If plugin not found
            PluginError: If plugin has dependents and force=False
        """
        if plugin_name not in self._entries:
            raise PluginNotFoundError(plugin_name)
        
        # Check for dependents
        dependents = self._dependents.get(plugin_name, set())
        if dependents and not force:
            raise PluginError(
                f"Cannot unregister plugin {plugin_name}. "
                f"Other plugins depend on it: {', '.join(dependents)}. "
                "Use force=True to unregister anyway."
            )
        
        # Remove from registry
        del self._entries[plugin_name]
        
        # Clean up dependency graph
        if plugin_name in self._dependencies:
            for dep in self._dependencies[plugin_name]:
                self._dependents.get(dep, set()).discard(plugin_name)
            del self._dependencies[plugin_name]
        
        if plugin_name in self._dependents:
            del self._dependents[plugin_name]
        
        # Save registry
        self._save_registry()
        
        self.logger.info("Plugin unregistered", plugin_name=plugin_name)
        return True
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginRegistryEntry]:
        """
        Get plugin registry entry.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin registry entry or None if not found
        """
        return self._entries.get(plugin_name)
    
    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        enabled_only: bool = False
    ) -> List[str]:
        """
        List registered plugins.
        
        Args:
            plugin_type: Optional filter by plugin type
            enabled_only: Only return enabled plugins
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for plugin_name, entry in self._entries.items():
            # Filter by enabled status
            if enabled_only and not entry.enabled:
                continue
            
            # Filter by plugin type
            if plugin_type is not None and entry.metadata.plugin_type != plugin_type:
                continue
            
            plugins.append(plugin_name)
        
        return sorted(plugins)
    
    def get_plugins_by_capability(self, capability_name: str) -> List[str]:
        """
        Get plugins that provide a specific capability.
        
        Args:
            capability_name: Name of capability to search for
            
        Returns:
            List of plugin names that provide the capability
        """
        plugins = []
        
        for plugin_name, entry in self._entries.items():
            if not entry.enabled:
                continue
            
            for capability in entry.metadata.capabilities:
                if capability.name == capability_name:
                    plugins.append(plugin_name)
                    break
        
        return sorted(plugins)
    
    def get_dependency_order(self, plugin_names: List[str]) -> List[str]:
        """
        Get plugins in dependency order for loading/activation.
        
        Args:
            plugin_names: List of plugin names to order
            
        Returns:
            List of plugin names in dependency order
            
        Raises:
            PluginError: If circular dependencies detected
        """
        # Topological sort using Kahn's algorithm
        in_degree = {}
        graph = {}
        
        # Initialize
        for plugin in plugin_names:
            in_degree[plugin] = 0
            graph[plugin] = []
        
        # Build graph and calculate in-degrees
        for plugin in plugin_names:
            if plugin in self._dependencies:
                for dep in self._dependencies[plugin]:
                    if dep in plugin_names:  # Only consider dependencies in the given list
                        graph[dep].append(plugin)
                        in_degree[plugin] += 1
        
        # Topological sort
        queue = [plugin for plugin in plugin_names if in_degree[plugin] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(plugin_names):
            remaining = [p for p in plugin_names if p not in result]
            raise PluginError(f"Circular dependencies detected among plugins: {remaining}")
        
        return result
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin in the registry.
        
        Args:
            plugin_name: Name of plugin to enable
            
        Returns:
            True if successful
            
        Raises:
            PluginNotFoundError: If plugin not found
        """
        if plugin_name not in self._entries:
            raise PluginNotFoundError(plugin_name)
        
        entry = self._entries[plugin_name]
        if entry.enabled:
            return True
        
        entry.enabled = True
        entry.last_updated = datetime.now(timezone.utc)
        
        self._save_registry()
        
        self.logger.info("Plugin enabled", plugin_name=plugin_name)
        return True
    
    def disable_plugin(self, plugin_name: str, force: bool = False) -> bool:
        """
        Disable a plugin in the registry.
        
        Args:
            plugin_name: Name of plugin to disable
            force: Force disable even if other plugins depend on it
            
        Returns:
            True if successful
            
        Raises:
            PluginNotFoundError: If plugin not found
            PluginError: If plugin has dependents and force=False
        """
        if plugin_name not in self._entries:
            raise PluginNotFoundError(plugin_name)
        
        # Check for enabled dependents
        enabled_dependents = []
        for dependent in self._dependents.get(plugin_name, set()):
            if dependent in self._entries and self._entries[dependent].enabled:
                enabled_dependents.append(dependent)
        
        if enabled_dependents and not force:
            raise PluginError(
                f"Cannot disable plugin {plugin_name}. "
                f"Other enabled plugins depend on it: {', '.join(enabled_dependents)}. "
                "Use force=True to disable anyway."
            )
        
        entry = self._entries[plugin_name]
        entry.enabled = False
        entry.last_updated = datetime.now(timezone.utc)
        
        self._save_registry()
        
        self.logger.info("Plugin disabled", plugin_name=plugin_name)
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total_plugins = len(self._entries)
        enabled_plugins = len([e for e in self._entries.values() if e.enabled])
        
        # Count by type
        type_counts = {}
        for entry in self._entries.values():
            plugin_type = entry.metadata.plugin_type.value
            type_counts[plugin_type] = type_counts.get(plugin_type, 0) + 1
        
        # Calculate dependency statistics
        total_dependencies = sum(len(deps) for deps in self._dependencies.values())
        
        return {
            "total_plugins": total_plugins,
            "enabled_plugins": enabled_plugins,
            "disabled_plugins": total_plugins - enabled_plugins,
            "plugins_by_type": type_counts,
            "total_dependencies": total_dependencies,
            "registry_file": str(self._registry_file),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }


# Global registry instance
_plugin_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        PluginRegistry instance
    """
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry