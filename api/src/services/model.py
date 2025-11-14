"""
Model management service
Handles model loading, unloading, and Modelscope API integration
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

from ..core.config import settings
from ..core.logging import get_logger
from .modelscope import ModelscopeService, ModelscopeModel

logger = get_logger(__name__)


class ModelInfo:
    """Model information"""
    def __init__(self, model_id: str, name: str, description: str, **kwargs):
        self.id = model_id
        self.name = name
        self.description = description
        self.size = kwargs.get("size", 0)
        self.format = kwargs.get("format", "gguf")
        self.quantization = kwargs.get("quantization")
        self.context_length = kwargs.get("context_length", 2048)
        self.tags = kwargs.get("tags", [])
        self.download_url = kwargs.get("download_url", "")
        self.sha256 = kwargs.get("sha256")
        self.downloaded = kwargs.get("downloaded", False)
        self.path = kwargs.get("path")
        self.loaded = kwargs.get("loaded", False)
        self.load_time = kwargs.get("load_time")
        self.memory_usage = kwargs.get("memory_usage", 0)


class ModelService:
    """Model management service"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.modelscope_service = ModelscopeService()
        self._load_models_from_config()
        
    def _load_models_from_config(self):
        """Load initial models from configuration"""
        # This would load from database or configuration file
        # For now, we'll use mock data and sync with Modelscope
        self._initialize_mock_models()
        
    def _initialize_mock_models(self):
        """Initialize with mock models for demonstration"""
        mock_models = [
            {
                "id": "qwen2.5-7b-instruct-q4",
                "name": "Qwen2.5-7B-Instruct-Q4",
                "description": "Qwen 2.5 7B Instruct model with 4-bit quantization",
                "size": 4.2,
                "format": "gguf",
                "quantization": "Q4_K_M",
                "context_length": 32768,
                "tags": ["qwen", "instruct", "chat", "multilingual"],
                "download_url": "https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
                "downloaded": False
            },
            {
                "id": "llama3.1-8b-instruct-q6",
                "name": "Llama 3.1 8B Instruct Q6",
                "description": "Llama 3.1 8B Instruct model with 6-bit quantization",
                "size": 6.8,
                "format": "gguf",
                "quantization": "Q6_K",
                "context_length": 128000,
                "tags": ["llama", "instruct", "chat", "english"],
                "download_url": "https://www.modelscope.cn/models/meta-llama/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct-q6_k.gguf",
                "downloaded": False
            },
            {
                "id": "deepseek-coder-7b-q5",
                "name": "DeepSeek Coder 7B Q5",
                "description": "DeepSeek Coder 7B model optimized for coding tasks",
                "size": 5.1,
                "format": "gguf",
                "quantization": "Q5_K_M",
                "context_length": 16384,
                "tags": ["deepseek", "coder", "programming", "code"],
                "download_url": "https://www.modelscope.cn/models/deepseek-ai/DeepSeek-Coder-7B-GGUF/resolve/main/deepseek-coder-7b-q5_k_m.gguf",
                "downloaded": True,
                "path": "/models/deepseek-coder-7b-q5_k_m.gguf"
            }
        ]
        
        for model_data in mock_models:
            model = ModelInfo(**model_data)
            self.models[model.id] = model
            
        logger.info(f"Loaded {len(self.models)} models from configuration")
        
    async def sync_with_modelscope(self, query: str = "llm gguf") -> List[Dict[str, Any]]:
        """Sync models with Modelscope API"""
        try:
            async with self.modelscope_service as service:
                # Search for models on Modelscope
                modelscope_models = await service.search_models(query, limit=50)
                
                # Convert to our model format
                new_models = []
                for ms_model in modelscope_models:
                    if ms_model.id not in self.models:
                        # Create new model from Modelscope data
                        model_info = ModelInfo(
                            model_id=ms_model.id,
                            name=ms_model.name,
                            description=ms_model.description,
                            size=ms_model.size,
                            format=ms_model.format.lower(),
                            quantization=ms_model.quantization,
                            context_length=ms_model.context_length,
                            tags=ms_model.tags,
                            download_url=ms_model.download_url,
                            downloaded=ms_model.downloaded,
                            path=ms_model.local_path
                        )
                        self.models[ms_model.id] = model_info
                        new_models.append(self._model_to_dict(model_info))
                        
                logger.info(f"Synced {len(new_models)} new models from Modelscope")
                return new_models
                
        except Exception as e:
            logger.error("Failed to sync with Modelscope", error=str(e))
            return []
        
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all available models"""
        return [self._model_to_dict(model) for model in self.models.values()]
        
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get specific model details"""
        model = self.models.get(model_id)
        return self._model_to_dict(model) if model else None
        
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get currently loaded models"""
        return [self._model_to_dict(model) for model in self.loaded_models.values()]
        
    async def load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
            
        model = self.models[model_id]
        
        if not model.downloaded:
            logger.error(f"Model {model_id} not downloaded")
            return False
            
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return True
            
        try:
            logger.info(f"Loading model {model_id}")
            
            # Simulate model loading
            await asyncio.sleep(2)  # Simulate loading time
            
            # Update model status
            model.loaded = True
            model.load_time = datetime.now()
            model.memory_usage = model.size * 1.2  # Estimate memory usage
            
            # Add to loaded models
            self.loaded_models[model_id] = model
            
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}", error=str(e))
            return False
            
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id not in self.loaded_models:
            logger.info(f"Model {model_id} not loaded")
            return True
            
        try:
            logger.info(f"Unloading model {model_id}")
            
            # Simulate model unloading
            await asyncio.sleep(1)  # Simulate unloading time
            
            # Update model status
            model = self.models[model_id]
            model.loaded = False
            model.load_time = None
            model.memory_usage = 0
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}", error=str(e))
            return False
            
    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for models"""
        query = query.lower()
        results = []
        
        for model in self.models.values():
            # Search in name, description, and tags
            if (query in model.name.lower() or 
                query in model.description.lower() or 
                any(query in tag.lower() for tag in model.tags)):
                results.append(self._model_to_dict(model))
                
        return results[:limit]
        
    def filter_by_format(self, format: str) -> List[Dict[str, Any]]:
        """Filter models by format"""
        return [self._model_to_dict(model) for model in self.models.values() 
                if model.format == format]
        
    def filter_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Filter models by tag"""
        return [self._model_to_dict(model) for model in self.models.values() 
                if tag in model.tags]
        
    async def download_model(self, model_id: str, quantization: str = "Q4_K_M", 
                           threads: int = 4, resume: bool = True) -> str:
        """Download a model"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            raise ValueError(f"Model {model_id} not found")
            
        model = self.models[model_id]
        
        if model.downloaded:
            logger.info(f"Model {model_id} already downloaded")
            return "already_downloaded"
            
        try:
            # Use Modelscope service to download
            async with self.modelscope_service as service:
                task_id = await service.download_model(
                    model_id, quantization, threads, resume
                )
                logger.info(f"Model download started", model_id=model_id, task_id=task_id)
                return task_id
                
        except Exception as e:
            logger.error(f"Failed to start model download", model_id=model_id, error=str(e))
            raise
            
    def get_download_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get download task status"""
        try:
            task = self.modelscope_service.get_download_task(task_id)
            if task:
                return {
                    "id": task.id,
                    "model_id": task.model_id,
                    "status": task.status,
                    "progress": task.progress,
                    "speed": task.speed,
                    "eta": task.eta,
                    "error": task.error,
                    "threads": task.threads,
                    "resume_supported": task.resume_supported,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get download status", task_id=task_id, error=str(e))
            return None
            
    def get_all_download_tasks(self) -> List[Dict[str, Any]]:
        """Get all download tasks"""
        try:
            tasks = self.modelscope_service.get_all_download_tasks()
            return [
                {
                    "id": task.id,
                    "model_id": task.model_id,
                    "status": task.status,
                    "progress": task.progress,
                    "speed": task.speed,
                    "eta": task.eta,
                    "error": task.error,
                    "threads": task.threads,
                    "resume_supported": task.resume_supported,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None
                }
                for task in tasks
            ]
        except Exception as e:
            logger.error(f"Failed to get download tasks", error=str(e))
            return []
            
    async def pause_download(self, task_id: str) -> bool:
        """Pause a download task"""
        try:
            return await self.modelscope_service.pause_download(task_id)
        except Exception as e:
            logger.error(f"Failed to pause download", task_id=task_id, error=str(e))
            return False
            
    async def resume_download(self, task_id: str) -> bool:
        """Resume a download task"""
        try:
            return await self.modelscope_service.resume_download(task_id)
        except Exception as e:
            logger.error(f"Failed to resume download", task_id=task_id, error=str(e))
            return False
            
    async def cancel_download(self, task_id: str) -> bool:
        """Cancel a download task"""
        try:
            return await self.modelscope_service.cancel_download(task_id)
        except Exception as e:
            logger.error(f"Failed to cancel download", task_id=task_id, error=str(e))
            return False
        
    def _model_to_dict(self, model: ModelInfo) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "size": model.size,
            "format": model.format,
            "quantization": model.quantization,
            "context_length": model.context_length,
            "tags": model.tags,
            "download_url": model.download_url,
            "sha256": model.sha256,
            "downloaded": model.downloaded,
            "path": model.path,
            "loaded": model.loaded,
            "load_time": model.load_time.isoformat() if model.load_time else None,
            "memory_usage": model.memory_usage
        }