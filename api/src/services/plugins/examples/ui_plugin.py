"""
Example UI Plugin - Demonstrates UI component and interface extensions
"""

from typing import Dict, Any, List
from ..plugin_manager import BasePlugin, PluginMetadata, PluginType


class ExampleUIPlugin(BasePlugin):
    """Example plugin for UI extensions and components"""
    
    def __init__(self):
        super().__init__()
        self.ui_components = {}
        self.themes = {}
        
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="example_ui_plugin",
            version="1.0.0",
            description="Example plugin for UI extensions and components",
            author="HelloVM Team",
            plugin_type=PluginType.UI,
            dependencies=[],
            required_permissions=["ui_access", "theme_management"],
            config_schema={
                "theme": {"type": "string", "default": "light"},
                "enable_animations": {"type": "boolean", "default": True},
                "custom_css": {"type": "string", "default": ""},
                "component_positions": {"type": "object", "default": {}}
            },
            entry_points={
                "register_component": "register_component",
                "get_components": "get_components",
                "apply_theme": "apply_theme",
                "get_themes": "get_themes"
            }
        )
        
    def on_initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            # Initialize UI components
            self._initialize_default_components()
            self._initialize_themes()
            
            logger.info("ExampleUIPlugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ExampleUIPlugin: {e}")
            return False
            
    def _initialize_default_components(self):
        """Initialize default UI components"""
        self.ui_components = {
            "status_indicator": {
                "type": "component",
                "name": "Hardware Status Indicator",
                "description": "Shows hardware acceleration status",
                "template": "status_indicator.html",
                "props": {
                    "show_gpu": True,
                    "show_npu": True,
                    "show_cpu": True,
                    "refresh_interval": 5000
                }
            },
            "model_selector": {
                "type": "component",
                "name": "Model Selector",
                "description": "Enhanced model selection interface",
                "template": "model_selector.html",
                "props": {
                    "show_download_status": True,
                    "show_performance_info": True,
                    "filter_options": ["size", "type", "hardware"]
                }
            },
            "performance_monitor": {
                "type": "component",
                "name": "Performance Monitor",
                "description": "Real-time performance monitoring",
                "template": "performance_monitor.html",
                "props": {
                    "update_interval": 1000,
                    "show_charts": True,
                    "show_metrics": ["cpu", "memory", "gpu", "inference_time"]
                }
            }
        }
        
    def _initialize_themes(self):
        """Initialize available themes"""
        self.themes = {
            "light": {
                "name": "Light Theme",
                "colors": {
                    "primary": "#007bff",
                    "secondary": "#6c757d",
                    "success": "#28a745",
                    "danger": "#dc3545",
                    "warning": "#ffc107",
                    "info": "#17a2b8",
                    "background": "#ffffff",
                    "surface": "#f8f9fa",
                    "text": "#212529"
                },
                "typography": {
                    "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto",
                    "font_size": "14px",
                    "line_height": "1.5"
                }
            },
            "dark": {
                "name": "Dark Theme",
                "colors": {
                    "primary": "#0d6efd",
                    "secondary": "#6c757d",
                    "success": "#198754",
                    "danger": "#dc3545",
                    "warning": "#ffc107",
                    "info": "#0dcaf0",
                    "background": "#212529",
                    "surface": "#343a40",
                    "text": "#ffffff"
                },
                "typography": {
                    "font_family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto",
                    "font_size": "14px",
                    "line_height": "1.5"
                }
            },
            "high_contrast": {
                "name": "High Contrast Theme",
                "colors": {
                    "primary": "#0000ff",
                    "secondary": "#800080",
                    "success": "#008000",
                    "danger": "#ff0000",
                    "warning": "#ff8c00",
                    "info": "#00bfff",
                    "background": "#000000",
                    "surface": "#1a1a1a",
                    "text": "#ffffff"
                },
                "typography": {
                    "font_family": "Arial, sans-serif",
                    "font_size": "16px",
                    "line_height": "1.6",
                    "font_weight": "bold"
                }
            }
        }
        
    def register_component(self, component_id: str, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new UI component"""
        try:
            # Validate component configuration
            if "type" not in component_config or "template" not in component_config:
                return {"success": False, "error": "Component must have 'type' and 'template' fields"}
                
            # Register component
            self.ui_components[component_id] = component_config
            
            logger.info(f"UI component registered: {component_id}")
            return {
                "success": True,
                "component_id": component_id,
                "message": f"Component '{component_id}' registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return {"success": False, "error": str(e)}
            
    def get_components(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """Get registered UI components"""
        try:
            if component_type:
                components = {
                    k: v for k, v in self.ui_components.items()
                    if v.get("type") == component_type
                }
            else:
                components = self.ui_components.copy()
                
            return {
                "success": True,
                "components": components,
                "count": len(components)
            }
            
        except Exception as e:
            logger.error(f"Failed to get components: {e}")
            return {"success": False, "error": str(e)}
            
    def apply_theme(self, theme_name: str, custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a theme with optional custom overrides"""
        try:
            if theme_name not in self.themes:
                return {"success": False, "error": f"Theme '{theme_name}' not found"}
                
            theme_config = self.themes[theme_name].copy()
            
            # Apply custom overrides
            if custom_overrides:
                self._deep_merge(theme_config, custom_overrides)
                
            # Update configuration
            self.config["theme"] = theme_name
            self.config["custom_css"] = self._generate_css_from_theme(theme_config)
            
            logger.info(f"Theme applied: {theme_name}")
            return {
                "success": True,
                "theme": theme_name,
                "theme_config": theme_config,
                "message": f"Theme '{theme_name}' applied successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to apply theme {theme_name}: {e}")
            return {"success": False, "error": str(e)}
            
    def get_themes(self) -> Dict[str, Any]:
        """Get available themes"""
        try:
            return {
                "success": True,
                "themes": self.themes,
                "current_theme": self.config.get("theme", "light"),
                "count": len(self.themes)
            }
            
        except Exception as e:
            logger.error(f"Failed to get themes: {e}")
            return {"success": False, "error": str(e)}
            
    def _deep_merge(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
        
    def _generate_css_from_theme(self, theme_config: Dict[str, Any]) -> str:
        """Generate CSS from theme configuration"""
        colors = theme_config.get("colors", {})
        typography = theme_config.get("typography", {})
        
        css_parts = []
        
        # Generate color CSS variables
        css_parts.append(":root {")
        for color_name, color_value in colors.items():
            css_parts.append(f"  --color-{color_name}: {color_value};")
        css_parts.append("}")
        
        # Generate typography CSS
        if typography:
            css_parts.append("body {")
            if "font_family" in typography:
                css_parts.append(f"  font-family: {typography['font_family']};")
            if "font_size" in typography:
                css_parts.append(f"  font-size: {typography['font_size']};")
            if "line_height" in typography:
                css_parts.append(f"  line-height: {typography['line_height']};")
            if "font_weight" in typography:
                css_parts.append(f"  font-weight: {typography['font_weight']};")
            css_parts.append("}")
            
        return "\n".join(css_parts)