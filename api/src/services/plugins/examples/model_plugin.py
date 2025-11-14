"""
Example Model Plugin - Demonstrates model loading and inference capabilities
"""

from typing import Dict, Any, Optional
from ..plugin_manager import BasePlugin, PluginMetadata, PluginType


class ExampleModelPlugin(BasePlugin):
    """Example plugin for model loading and inference"""
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="example_model_plugin",
            version="1.0.0",
            description="Example plugin for model loading and inference",
            author="HelloVM Team",
            plugin_type=PluginType.MODEL,
            dependencies=["torch", "transformers"],
            required_permissions=["model_access", "inference"],
            config_schema={
                "model_name": {"type": "string", "default": "microsoft/DialoGPT-medium"},
                "device": {"type": "string", "default": "auto"},
                "max_length": {"type": "integer", "default": 1000}
            },
            entry_points={
                "load_model": "load_model",
                "generate_text": "generate_text",
                "get_model_info": "get_model_info"
            }
        )
        
    def on_initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.model = None
            self.tokenizer = None
            self.device = self.config.get("device", "auto")
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            logger.info(f"ExampleModelPlugin initialized with device: {self.device}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            return False
            
    def on_start(self) -> bool:
        """Start the plugin"""
        try:
            # Load model in start() to avoid blocking initialization
            model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
            self.load_model(model_name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ExampleModelPlugin: {e}")
            return False
            
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=self.device
            )
            
            self.model.eval()
            
            return {
                "success": True,
                "model_name": model_name,
                "device": self.device,
                "model_size": sum(p.numel() for p in self.model.parameters())
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return {"success": False, "error": str(e)}
            
    def generate_text(self, prompt: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """Generate text using the loaded model"""
        try:
            if not self.model or not self.tokenizer:
                return {"success": False, "error": "Model not loaded"}
                
            max_len = max_length or self.config.get("max_length", 100)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_len,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "prompt": prompt,
                "generated_text": generated_text,
                "model_name": self.config.get("model_name")
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {"success": False, "error": str(e)}
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"loaded": False}
            
        return {
            "loaded": True,
            "model_name": self.config.get("model_name"),
            "device": self.device,
            "model_size": sum(p.numel() for p in self.model.parameters()),
            "config": self.config
        }