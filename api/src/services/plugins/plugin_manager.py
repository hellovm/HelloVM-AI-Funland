"""
Plugin system for HelloVM AI Funland
Provides hot-loading, lifecycle management, and plugin communication
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class PluginType(Enum):
    """Types of plugins supported"""
    MODEL = "model"
    HARDWARE = "hardware"
    UI = "ui"
    INTEGRATION = "integration"
    TOOL = "tool"
    EXTENSION = "extension"


class PluginState(Enum):
    """Plugin lifecycle states"""
    INSTALLED = "installed"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    entry_points: Dict[str, str] = field(default_factory=dict)
    compatibility: Dict[str, str] = field(default_factory=dict)


@dataclass
class PluginInfo:
    """Complete plugin information"""
    metadata: PluginMetadata
    path: str
    state: PluginState
    instance: Optional[Any] = None
    error: Optional[str] = None
    loaded_at: Optional[float] = None
    started_at: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)


class PluginHook:
    """Plugin hook for event handling"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.callbacks: List[Callable] = []
        
    def register(self, callback: Callable):
        """Register a callback for this hook"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.debug(f"Registered callback for hook '{self.name}'")
            
    def unregister(self, callback: Callable):
        """Unregister a callback for this hook"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Unregistered callback for hook '{self.name}'")
            
    async def trigger(self, *args, **kwargs):
        """Trigger all callbacks for this hook"""
        results = []
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook callback failed for '{self.name}'", error=str(e))
                results.append(None)
        return results


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.config: Dict[str, Any] = {}
        self.plugin_manager = None
        self.hooks: Dict[str, PluginHook] = {}
        self._initialized = False
        
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
        
    def initialize(self, config: Dict[str, Any] = None, plugin_manager=None) -> bool:
        """Initialize the plugin"""
        try:
            self.metadata = self.get_metadata()
            self.config = config or {}
            self.plugin_manager = plugin_manager
            
            # Setup default hooks
            self._setup_hooks()
            
            # Call plugin-specific initialization
            success = self.on_initialize()
            
            self._initialized = success
            logger.info(f"Plugin '{self.metadata.name}' initialized successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin '{self.metadata.name if self.metadata else 'unknown'}'", error=str(e))
            return False
            
    def start(self) -> bool:
        """Start the plugin"""
        if not self._initialized:
            logger.error(f"Plugin '{self.metadata.name}' not initialized")
            return False
            
        try:
            success = self.on_start()
            logger.info(f"Plugin '{self.metadata.name}' started successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to start plugin '{self.metadata.name}'", error=str(e))
            return False
            
    def stop(self) -> bool:
        """Stop the plugin"""
        if not self._initialized:
            return True
            
        try:
            success = self.on_stop()
            logger.info(f"Plugin '{self.metadata.name}' stopped successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop plugin '{self.metadata.name}'", error=str(e))
            return False
            
    def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        try:
            success = self.on_cleanup()
            self._initialized = False
            logger.info(f"Plugin '{self.metadata.name}' cleaned up successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cleanup plugin '{self.metadata.name}'", error=str(e))
            return False
            
    def get_config_schema(self) -> Optional[Dict[str, Any]]:
        """Get plugin configuration schema"""
        return self.metadata.config_schema if self.metadata else None
        
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            self.config.update(config)
            return self.on_config_update(config)
            
        except Exception as e:
            logger.error(f"Failed to update config for plugin '{self.metadata.name}'", error=str(e))
            return False
            
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = PluginHook(hook_name)
            
        self.hooks[hook_name].register(callback)
        
    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """Trigger a plugin hook"""
        if hook_name in self.hooks:
            return self.hooks[hook_name].trigger(*args, **kwargs)
        return []
        
    def _setup_hooks(self):
        """Setup default plugin hooks"""
        # Override in subclasses to define custom hooks
        pass
        
    # Plugin lifecycle callbacks (override in subclasses)
    
    def on_initialize(self) -> bool:
        """Called during plugin initialization"""
        return True
        
    def on_start(self) -> bool:
        """Called when plugin starts"""
        return True
        
    def on_stop(self) -> bool:
        """Called when plugin stops"""
        return True
        
    def on_cleanup(self) -> bool:
        """Called during plugin cleanup"""
        return True
        
    def on_config_update(self, config: Dict[str, Any]) -> bool:
        """Called when configuration is updated"""
        return True
        
    def on_error(self, error: Exception):
        """Called when an error occurs"""
        logger.error(f"Plugin '{self.metadata.name}' error", error=str(error))


class PluginManager:
    """Manages plugin lifecycle, loading, and communication"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, PluginInfo] = {}
        self.hooks: Dict[str, PluginHook] = {}
        self._loading_lock = asyncio.Lock()
        self._watcher_task = None
        self._stop_event = None
        
        # Ensure plugins directory exists
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Setup default hooks
        self._setup_default_hooks()
        
    def _setup_default_hooks(self):
        """Setup default system hooks"""
        default_hooks = [
            ("plugin_loaded", "Triggered when a plugin is loaded"),
            ("plugin_started", "Triggered when a plugin starts"),
            ("plugin_stopped", "Triggered when a plugin stops"),
            ("plugin_unloaded", "Triggered when a plugin is unloaded"),
            ("plugin_error", "Triggered when a plugin encounters an error"),
            ("model_loaded", "Triggered when a model is loaded"),
            ("inference_complete", "Triggered when inference completes"),
            ("hardware_changed", "Triggered when hardware status changes"),
        ]
        
        for hook_name, description in default_hooks:
            self.hooks[hook_name] = PluginHook(hook_name, description)
            
    async def start(self):
        """Start the plugin manager"""
        logger.info("Starting plugin manager")
        self._stop_event = asyncio.Event()
        
        # Load existing plugins
        await self.discover_and_load_plugins()
        
        # Start plugin directory watcher (for hot-loading)
        self._watcher_task = asyncio.create_task(self._watch_plugins_directory())
        
        logger.info(f"Plugin manager started with {len(self.plugins)} plugins")
        
    async def stop(self):
        """Stop the plugin manager"""
        logger.info("Stopping plugin manager")
        
        if self._stop_event:
            self._stop_event.set()
            
        # Stop all running plugins
        await self.stop_all_plugins()
        
        # Cancel watcher task
        if self._watcher_task:
            self._watcher_task.cancel()
            
        logger.info("Plugin manager stopped")
        
    async def discover_and_load_plugins(self):
        """Discover and load all plugins in the plugins directory"""
        logger.info("Discovering plugins")
        
        plugin_files = []
        
        # Scan for plugin files
        for item in self.plugins_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                plugin_files.append(item)
            elif item.is_dir() and (item / 'plugin.py').exists():
                plugin_files.append(item / 'plugin.py')
                
        logger.info(f"Found {len(plugin_files)} potential plugins")
        
        # Load each plugin
        for plugin_file in plugin_files:
            try:
                await self.load_plugin(str(plugin_file))
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}", error=str(e))
                
    async def load_plugin(self, plugin_path: str) -> bool:
        """Load a plugin from a file path"""
        async with self._loading_lock:
            try:
                logger.info(f"Loading plugin from {plugin_path}")
                
                # Load the plugin module
                spec = importlib.util.spec_from_file_location("plugin", plugin_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"Failed to load plugin spec from {plugin_path}")
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin class
                plugin_class = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BasePlugin) and obj != BasePlugin:
                        plugin_class = obj
                        break
                        
                if plugin_class is None:
                    raise RuntimeError(f"No plugin class found in {plugin_path}")
                    
                # Create plugin instance
                plugin_instance = plugin_class()
                
                # Get metadata
                metadata = plugin_instance.get_metadata()
                
                # Check for conflicts
                if metadata.name in self.plugins:
                    logger.warning(f"Plugin '{metadata.name}' already loaded, unloading first")
                    await self.unload_plugin(metadata.name)
                    
                # Create plugin info
                plugin_info = PluginInfo(
                    metadata=metadata,
                    path=plugin_path,
                    state=PluginState.LOADED,
                    instance=plugin_instance,
                    loaded_at=time.time()
                )
                
                # Store plugin
                self.plugins[metadata.name] = plugin_info
                
                # Trigger hook
                await self.trigger_hook("plugin_loaded", plugin_info)
                
                logger.info(f"Plugin '{metadata.name}' loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_path}", error=str(e))
                return False
                
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        async with self._loading_lock:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin '{plugin_name}' not found")
                return False
                
            plugin_info = self.plugins[plugin_name]
            
            try:
                logger.info(f"Unloading plugin '{plugin_name}'")
                
                # Stop plugin if running
                if plugin_info.state == PluginState.RUNNING:
                    await self.stop_plugin(plugin_name)
                    
                # Cleanup plugin
                if plugin_info.instance:
                    plugin_info.instance.cleanup()
                    
                # Update state
                plugin_info.state = PluginState.UNLOADED
                plugin_info.instance = None
                
                # Remove from registry
                del self.plugins[plugin_name]
                
                # Trigger hook
                await self.trigger_hook("plugin_unloaded", plugin_name)
                
                logger.info(f"Plugin '{plugin_name}' unloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin '{plugin_name}'", error=str(e))
                plugin_info.state = PluginState.ERROR
                plugin_info.error = str(e)
                return False
                
    async def start_plugin(self, plugin_name: str) -> bool:
        """Start a plugin"""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return False
            
        plugin_info = self.plugins[plugin_name]
        
        try:
            logger.info(f"Starting plugin '{plugin_name}'")
            
            # Initialize plugin if not already done
            if plugin_info.state == PluginState.LOADED:
                plugin_info.state = PluginState.INITIALIZING
                success = plugin_info.instance.initialize(plugin_info.config, self)
                if not success:
                    plugin_info.state = PluginState.ERROR
                    return False
                    
            # Start plugin
            plugin_info.state = PluginState.STARTING
            success = plugin_info.instance.start()
            
            if success:
                plugin_info.state = PluginState.RUNNING
                plugin_info.started_at = time.time()
                
                # Trigger hook
                await self.trigger_hook("plugin_started", plugin_info)
                
                logger.info(f"Plugin '{plugin_name}' started successfully")
                return True
            else:
                plugin_info.state = PluginState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to start plugin '{plugin_name}'", error=str(e))
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            return False
            
    async def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a plugin"""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return False
            
        plugin_info = self.plugins[plugin_name]
        
        try:
            logger.info(f"Stopping plugin '{plugin_name}'")
            
            plugin_info.state = PluginState.STOPPING
            success = plugin_info.instance.stop()
            
            if success:
                plugin_info.state = PluginState.STOPPED
                
                # Trigger hook
                await self.trigger_hook("plugin_stopped", plugin_info)
                
                logger.info(f"Plugin '{plugin_name}' stopped successfully")
                return True
            else:
                plugin_info.state = PluginState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop plugin '{plugin_name}'", error=str(e))
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            return False
            
    async def start_all_plugins(self):
        """Start all loaded plugins"""
        for plugin_name in self.plugins:
            plugin_info = self.plugins[plugin_name]
            if plugin_info.state == PluginState.LOADED:
                await self.start_plugin(plugin_name)
                
    async def stop_all_plugins(self):
        """Stop all running plugins"""
        for plugin_name in list(self.plugins.keys()):
            plugin_info = self.plugins[plugin_name]
            if plugin_info.state == PluginState.RUNNING:
                await self.stop_plugin(plugin_name)
                
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.plugins.get(plugin_name)
        
    def get_all_plugins(self) -> List[PluginInfo]:
        """Get all plugins"""
        return list(self.plugins.values())
        
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type"""
        return [p for p in self.plugins.values() if p.metadata.plugin_type == plugin_type]
        
    async def trigger_hook(self, hook_name: str, *args, **kwargs):
        """Trigger a system hook"""
        if hook_name in self.hooks:
            return await self.hooks[hook_name].trigger(*args, **kwargs)
        return []
        
    def register_hook_callback(self, hook_name: str, callback: Callable):
        """Register a callback for a system hook"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = PluginHook(hook_name)
            
        self.hooks[hook_name].register(callback)
        
    async def _watch_plugins_directory(self):
        """Watch plugins directory for changes (hot-loading)"""
        logger.info("Starting plugin directory watcher")
        
        # This is a simplified watcher - in production, use file system watchers like watchdog
        last_scan = time.time()
        
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = time.time()
                
                # Scan for new or modified plugins
                for item in self.plugins_dir.iterdir():
                    if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                        file_time = item.stat().st_mtime
                        
                        # Check if file is newer than last scan
                        if file_time > last_scan:
                            plugin_name = item.stem
                            
                            # Check if plugin is already loaded and needs reload
                            if plugin_name in self.plugins:
                                logger.info(f"Plugin '{plugin_name}' modified, reloading")
                                await self.unload_plugin(plugin_name)
                                
                            # Load the plugin
                            await self.load_plugin(str(item))
                            
                last_scan = current_time
                
            except Exception as e:
                logger.error("Plugin directory watcher error", error=str(e))
                
        logger.info("Plugin directory watcher stopped")