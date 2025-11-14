"""
Example Hardware Plugin - Demonstrates hardware monitoring and control
"""

from typing import Dict, Any, List
from ..plugin_manager import BasePlugin, PluginMetadata, PluginType


class ExampleHardwarePlugin(BasePlugin):
    """Example plugin for hardware monitoring and control"""
    
    def __init__(self):
        super().__init__()
        self.monitoring_active = False
        self.metrics_history = []
        
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="example_hardware_plugin",
            version="1.0.0",
            description="Example plugin for hardware monitoring and control",
            author="HelloVM Team",
            plugin_type=PluginType.HARDWARE,
            dependencies=["psutil"],
            required_permissions=["system_monitoring", "hardware_access"],
            config_schema={
                "monitoring_interval": {"type": "number", "default": 5.0},
                "enable_temperature": {"type": "boolean", "default": True},
                "enable_power": {"type": "boolean", "default": True},
                "alerts_enabled": {"type": "boolean", "default": True},
                "cpu_threshold": {"type": "number", "default": 80.0},
                "memory_threshold": {"type": "number", "default": 85.0}
            },
            entry_points={
                "get_metrics": "get_metrics",
                "start_monitoring": "start_monitoring",
                "stop_monitoring": "stop_monitoring",
                "get_history": "get_history"
            }
        )
        
    def on_initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            import psutil
            
            self.monitoring_active = False
            self.metrics_history = []
            self.monitoring_task = None
            
            logger.info("ExampleHardwarePlugin initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            return False
            
    def on_start(self) -> bool:
        """Start the plugin"""
        try:
            # Register for system events
            self.register_system_hooks()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ExampleHardwarePlugin: {e}")
            return False
            
    def register_system_hooks(self):
        """Register hooks for system events"""
        # Register for hardware change events
        if self.plugin_manager:
            self.plugin_manager.register_hook_callback(
                "hardware_changed",
                self.on_hardware_changed
            )
            
    def on_hardware_changed(self, hardware_info: Dict[str, Any]):
        """Handle hardware change events"""
        logger.info(f"Hardware change detected: {hardware_info}")
        
        # Check for critical conditions
        if self.config.get("alerts_enabled", True):
            self.check_critical_conditions(hardware_info)
            
    def check_critical_conditions(self, hardware_info: Dict[str, Any]):
        """Check for critical hardware conditions"""
        cpu_threshold = self.config.get("cpu_threshold", 80.0)
        memory_threshold = self.config.get("memory_threshold", 85.0)
        
        # Check CPU usage
        if hardware_info.get("cpu_percent", 0) > cpu_threshold:
            self.trigger_alert("high_cpu_usage", hardware_info)
            
        # Check memory usage
        if hardware_info.get("memory_percent", 0) > memory_threshold:
            self.trigger_alert("high_memory_usage", hardware_info)
            
    def trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert"""
        alert = {
            "type": alert_type,
            "timestamp": time.time(),
            "data": data,
            "severity": "warning"
        }
        
        logger.warning(f"Hardware alert: {alert_type}", alert=alert)
        
        # Trigger plugin hook for alerts
        self.trigger_hook("hardware_alert", alert)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current hardware metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Temperature (if available)
            temperature_info = {}
            if self.config.get("enable_temperature", True):
                try:
                    temperatures = psutil.sensors_temperatures()
                    if temperatures:
                        for name, entries in temperatures.items():
                            temperature_info[name] = [
                                {"label": entry.label, "current": entry.current}
                                for entry in entries
                            ]
                except:
                    pass
                    
            metrics = {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else None,
                    "frequency_max": cpu_freq.max if cpu_freq else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "temperature": temperature_info
            }
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
                
            return {
                "success": True,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get hardware metrics: {e}")
            return {"success": False, "error": str(e)}
            
    def start_monitoring(self) -> Dict[str, Any]:
        """Start continuous monitoring"""
        if self.monitoring_active:
            return {"success": False, "error": "Monitoring already active"}
            
        try:
            import asyncio
            
            self.monitoring_active = True
            interval = self.config.get("monitoring_interval", 5.0)
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
            
            logger.info(f"Hardware monitoring started with interval {interval}s")
            return {"success": True, "interval": interval}
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return {"success": False, "error": str(e)}
            
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop continuous monitoring"""
        if not self.monitoring_active:
            return {"success": False, "error": "Monitoring not active"}
            
        try:
            self.monitoring_active = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                self.monitoring_task = None
                
            logger.info("Hardware monitoring stopped")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return {"success": False, "error": str(e)}
            
    async def _monitoring_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                # Get metrics
                result = self.get_metrics()
                
                if result["success"]:
                    # Check for critical conditions
                    if self.config.get("alerts_enabled", True):
                        self.check_critical_conditions(result["metrics"])
                        
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
                
    def get_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get metrics history"""
        return {
            "success": True,
            "history": self.metrics_history[-limit:],
            "count": len(self.metrics_history)
        }