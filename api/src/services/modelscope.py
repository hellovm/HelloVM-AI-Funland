"""
Modelscope API integration service
Handles model discovery, metadata retrieval, and download management
"""

import asyncio
import aiohttp
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import structlog
from pathlib import Path

from ..core.config import settings
from ..core.exceptions import ModelNotFoundError, ModelDownloadError, ModelscopeAPIError

logger = structlog.get_logger(__name__)


@dataclass
class ModelscopeModel:
    """Modelscope model information"""
    id: str
    name: str
    description: str
    model_id: str  # Modelscope model ID (e.g., "qwen/Qwen2.5-7B-Instruct")
    size: float  # Size in GB
    format: str  # GGUF, GGML, etc.
    quantization: str  # Q4_K_M, Q6_K, etc.
    context_length: int
    tags: List[str]
    download_url: str
    downloaded: bool = False
    local_path: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ModelDownloadTask:
    """Model download task information"""
    id: str
    model_id: str
    status: str  # pending, downloading, paused, completed, error
    progress: float  # 0-100
    speed: float  # MB/s
    eta: Optional[int] = None  # seconds
    error: Optional[str] = None
    threads: int = 4
    resume_supported: bool = True
    created_at: datetime = None
    updated_at: datetime = None


class ModelscopeService:
    """Service for interacting with Modelscope API"""
    
    BASE_URL = "https://www.modelscope.cn/api/v1"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.download_tasks: Dict[str, ModelDownloadTask] = {}
        self.models_cache: Dict[str, ModelscopeModel] = {}
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'HelloVM-AI-Funland/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def search_models(self, query: str, limit: int = 20) -> List[ModelscopeModel]:
        """Search for models on Modelscope"""
        try:
            url = f"{self.BASE_URL}/models"
            params = {
                'search': query,
                'limit': limit,
                'task': 'text-generation',
                'framework': 'pytorch'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise ModelscopeAPIError(f"Search failed with status {response.status}")
                    
                data = await response.json()
                models = []
                
                for item in data.get('data', []):
                    model = self._parse_modelscope_model(item)
                    if model:
                        models.append(model)
                        
                logger.info("Model search completed", query=query, count=len(models))
                return models
                
        except Exception as e:
            logger.error("Model search failed", query=query, error=str(e))
            raise ModelscopeAPIError(f"Failed to search models: {str(e)}")
            
    async def get_model_info(self, model_id: str) -> ModelscopeModel:
        """Get detailed model information"""
        try:
            # Check cache first
            if model_id in self.models_cache:
                return self.models_cache[model_id]
                
            url = f"{self.BASE_URL}/models/{model_id}"
            
            async with self.session.get(url) as response:
                if response.status == 404:
                    raise ModelNotFoundError(f"Model {model_id} not found")
                elif response.status != 200:
                    raise ModelscopeAPIError(f"Get model failed with status {response.status}")
                    
                data = await response.json()
                model = self._parse_modelscope_model(data)
                
                if model:
                    self.models_cache[model_id] = model
                    return model
                else:
                    raise ModelscopeAPIError(f"Failed to parse model data for {model_id}")
                    
        except Exception as e:
            logger.error("Get model info failed", model_id=model_id, error=str(e))
            raise
            
    async def get_model_files(self, model_id: str) -> List[Dict[str, Any]]:
        """Get available files for a model"""
        try:
            url = f"{self.BASE_URL}/models/{model_id}/files"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise ModelscopeAPIError(f"Get model files failed with status {response.status}")
                    
                data = await response.json()
                return data.get('data', [])
                
        except Exception as e:
            logger.error("Get model files failed", model_id=model_id, error=str(e))
            raise
            
    async def download_model(self, model_id: str, quantization: str = "Q4_K_M", 
                           threads: int = 4, resume: bool = True) -> str:
        """Download a model with specified quantization"""
        try:
            # Get model info
            model = await self.get_model_info(model_id)
            
            # Find appropriate GGUF file
            files = await self.get_model_files(model_id)
            gguf_files = [f for f in files if f.get('name', '').endswith('.gguf')]
            
            if not gguf_files:
                raise ModelDownloadError(f"No GGUF files found for model {model_id}")
                
            # Find file with matching quantization
            target_file = None
            for file in gguf_files:
                if quantization in file.get('name', ''):
                    target_file = file
                    break
                    
            if not target_file:
                # Fallback to first GGUF file
                target_file = gguf_files[0]
                logger.warning("Target quantization not found, using first available GGUF file", 
                             model_id=model_id, quantization=quantization)
                
            # Create download task
            task_id = f"download_{model_id}_{datetime.now().timestamp()}"
            task = ModelDownloadTask(
                id=task_id,
                model_id=model_id,
                status='pending',
                progress=0.0,
                speed=0.0,
                threads=threads,
                resume_supported=resume,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.download_tasks[task_id] = task
            
            # Start download in background
            asyncio.create_task(self._download_file(
                task_id, target_file['download_url'], target_file['name'], model_id
            ))
            
            logger.info("Model download started", 
                       model_id=model_id, 
                       task_id=task_id,
                       file=target_file['name'])
            
            return task_id
            
        except Exception as e:
            logger.error("Model download initiation failed", model_id=model_id, error=str(e))
            raise
            
    async def _download_file(self, task_id: str, download_url: str, filename: str, model_id: str):
        """Download file with progress tracking"""
        task = self.download_tasks.get(task_id)
        if not task:
            return
            
        try:
            task.status = 'downloading'
            task.updated_at = datetime.now()
            
            # Local file path
            local_path = self.models_dir / filename
            
            # Check if file exists and get size for resume
            existing_size = 0
            if local_path.exists():
                existing_size = local_path.stat().st_size
                
            # Prepare headers for resume
            headers = {}
            if existing_size > 0 and task.resume_supported:
                headers['Range'] = f'bytes={existing_size}-'
                
            # Start download
            async with self.session.get(download_url, headers=headers) as response:
                if response.status not in [200, 206]:  # 206 for partial content (resume)
                    raise ModelDownloadError(f"Download failed with status {response.status}")
                    
                total_size = int(response.headers.get('content-length', 0))
                if existing_size > 0 and response.status == 206:
                    total_size += existing_size
                    
                downloaded_size = existing_size
                start_time = datetime.now()
                
                # Open file for writing (append mode for resume)
                mode = 'ab' if existing_size > 0 else 'wb'
                with open(local_path, mode) as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if task.status != 'downloading':
                            break
                            
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Update progress
                        if total_size > 0:
                            task.progress = (downloaded_size / total_size) * 100
                            
                        # Calculate speed
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > 0:
                            task.speed = (downloaded_size / (1024 * 1024)) / elapsed  # MB/s
                            
                        # Calculate ETA
                        if task.speed > 0 and total_size > 0:
                            remaining_mb = (total_size - downloaded_size) / (1024 * 1024)
                            task.eta = int(remaining_mb / task.speed)
                            
                        task.updated_at = datetime.now()
                        
                # Mark as completed
                if task.status == 'downloading':
                    task.status = 'completed'
                    task.progress = 100.0
                    task.updated_at = datetime.now()
                    
                    # Update model cache
                    if model_id in self.models_cache:
                        self.models_cache[model_id].downloaded = True
                        self.models_cache[model_id].local_path = str(local_path)
                        
                    logger.info("Model download completed", 
                               task_id=task_id, 
                               model_id=model_id,
                               path=str(local_path))
                    
        except Exception as e:
            task.status = 'error'
            task.error = str(e)
            task.updated_at = datetime.now()
            logger.error("Model download failed", task_id=task_id, error=str(e))
            
    def get_download_task(self, task_id: str) -> Optional[ModelDownloadTask]:
        """Get download task status"""
        return self.download_tasks.get(task_id)
        
    def get_all_download_tasks(self) -> List[ModelDownloadTask]:
        """Get all download tasks"""
        return list(self.download_tasks.values())
        
    async def pause_download(self, task_id: str) -> bool:
        """Pause a download task"""
        task = self.download_tasks.get(task_id)
        if task and task.status == 'downloading':
            task.status = 'paused'
            task.updated_at = datetime.now()
            logger.info("Download paused", task_id=task_id)
            return True
        return False
        
    async def resume_download(self, task_id: str) -> bool:
        """Resume a paused download task"""
        task = self.download_tasks.get(task_id)
        if task and task.status == 'paused':
            # Restart download
            task.status = 'downloading'
            task.updated_at = datetime.now()
            logger.info("Download resumed", task_id=task_id)
            return True
        return False
        
    async def cancel_download(self, task_id: str) -> bool:
        """Cancel a download task"""
        task = self.download_tasks.get(task_id)
        if task and task.status in ['downloading', 'paused']:
            task.status = 'cancelled'
            task.updated_at = datetime.now()
            logger.info("Download cancelled", task_id=task_id)
            return True
        return False
        
    def _parse_modelscope_model(self, data: Dict[str, Any]) -> Optional[ModelscopeModel]:
        """Parse Modelscope API response into ModelscopeModel"""
        try:
            # Extract basic information
            model_id = data.get('id', '')
            name = data.get('name', model_id)
            description = data.get('description', '')
            
            # Extract model card information
            card = data.get('card', {})
            task = card.get('task', 'text-generation')
            
            # Only process text-generation models
            if task not in ['text-generation', 'chat', 'conversational']:
                return None
                
            # Extract tags
            tags = card.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
                
            # Estimate size (this would be more accurate with actual file info)
            size = self._estimate_model_size(name)
            
            # Determine format and quantization from name
            format, quantization = self._extract_format_quantization(name)
            
            # Context length estimation
            context_length = self._estimate_context_length(name, tags)
            
            # Generate download URL (this would be constructed from actual file info)
            download_url = f"https://www.modelscope.cn/models/{model_id}/resolve/main/{name}.gguf"
            
            return ModelscopeModel(
                id=model_id.replace('/', '-'),
                name=name,
                description=description,
                model_id=model_id,
                size=size,
                format=format,
                quantization=quantization,
                context_length=context_length,
                tags=tags,
                download_url=download_url,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error("Failed to parse model data", data=data, error=str(e))
            return None
            
    def _estimate_model_size(self, name: str) -> float:
        """Estimate model size based on name patterns"""
        # Extract parameter size from name
        import re
        
        # Look for patterns like "7B", "13B", "70B"
        param_match = re.search(r'(\d+)B', name)
        if param_match:
            param_size = int(param_match.group(1))
            # Rough estimation: 1B params ≈ 0.5-2GB depending on quantization
            base_size = param_size * 0.8  # Conservative estimate
            
            # Adjust for quantization
            if 'Q4' in name:
                return base_size * 0.5
            elif 'Q5' in name:
                return base_size * 0.6
            elif 'Q6' in name:
                return base_size * 0.75
            elif 'Q8' in name:
                return base_size * 1.0
            else:
                return base_size * 0.6  # Default to Q5 equivalent
                
        return 4.0  # Default size if can't parse
        
    def _extract_format_quantization(self, name: str) -> tuple[str, str]:
        """Extract format and quantization from model name"""
        name_upper = name.upper()
        
        # Format detection
        if 'GGUF' in name_upper:
            format = 'GGUF'
        elif 'GGML' in name_upper:
            format = 'GGML'
        else:
            format = 'GGUF'  # Default
            
        # Quantization detection
        quant_patterns = ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'Q4_0', 'Q5_0']
        for pattern in quant_patterns:
            if pattern in name_upper:
                return format, pattern
                
        # Default quantization
        return format, 'Q4_K_M'
        
    def _estimate_context_length(self, name: str, tags: List[str]) -> int:
        """Estimate context length based on model characteristics"""
        # Look for context length indicators
        import re
        
        # Check for explicit context length in name
        context_match = re.search(r'(\d+)K', name)
        if context_match:
            return int(context_match.group(1)) * 1000
            
        # Check tags for context length
        for tag in tags:
            if '32k' in tag.lower():
                return 32000
            elif '16k' in tag.lower():
                return 16000
            elif '8k' in tag.lower():
                return 8000
            elif '128k' in tag.lower():
                return 128000
                
        # Default based on model size
        if '70B' in name or '65B' in name:
            return 32000
        elif '30B' in name or '13B' in name:
            return 16000
        elif '7B' in name or '6B' in name:
            return 8000
        else:
            return 4096  # Default