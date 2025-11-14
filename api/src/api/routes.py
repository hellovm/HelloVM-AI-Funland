"""
API routes for HelloVM AI Funland Backend
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import structlog

from ..services.hardware import HardwareService, HardwareDevice
from ..services.model import ModelService
from ..services.chat import ChatService
from ..core.config import settings

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/v1", tags=["api"])

# Service instances (will be injected)
hardware_service: Optional[HardwareService] = None
model_service: Optional[ModelService] = None
chat_service: Optional[ChatService] = None


def set_services(hw_service: HardwareService, m_service: ModelService, c_service: ChatService):
    """Set service instances"""
    global hardware_service, model_service, chat_service
    hardware_service = hw_service
    model_service = m_service
    chat_service = c_service


# Hardware endpoints
@router.get("/hardware/status", response_model=Dict[str, Any])
async def get_hardware_status():
    """Get current hardware status and available devices"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        status = hardware_service.get_status()
        return {
            "success": True,
            "data": {
                "devices": [device.__dict__ for device in status.devices],
                "primary_device": status.primary_device.__dict__ if status.primary_device else None,
                "acceleration_mode": status.acceleration_mode,
                "last_scan": status.last_scan
            }
        }
    except Exception as e:
        logger.error("Failed to get hardware status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get hardware status")


@router.post("/hardware/scan")
async def scan_hardware():
    """Trigger hardware detection scan"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        status = await hardware_service.scan_hardware()
        return {
            "success": True,
            "data": {
                "devices_found": len(status.devices),
                "primary_device": status.primary_device.name if status.primary_device else None,
                "scan_time": status.last_scan
            }
        }
    except Exception as e:
        logger.error("Hardware scan failed", error=str(e))
        raise HTTPException(status_code=500, detail="Hardware scan failed")


@router.post("/hardware/select/{device_id}")
async def select_hardware_device(device_id: str):
    """Select primary hardware device"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        success = hardware_service.select_primary_device(device_id)
        if success:
            return {
                "success": True,
                "data": {"device_id": device_id, "status": "selected"}
            }
        else:
            raise HTTPException(status_code=404, detail="Device not found")
    except Exception as e:
        logger.error("Failed to select hardware device", device_id=device_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to select hardware device")


@router.get("/hardware/metrics/{device_id}")
async def get_hardware_metrics(device_id: str):
    """Get real-time metrics for a specific hardware device"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        metrics = await hardware_service.get_device_metrics(device_id)
        if metrics is None:
            raise HTTPException(status_code=404, detail="Device not found")
            
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        logger.error("Failed to get hardware metrics", device_id=device_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get hardware metrics")


# Hardware acceleration endpoints
@router.get("/hardware/acceleration/performance")
async def get_acceleration_performance():
    """Get performance summary for all acceleration devices"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        performance = hardware_service.get_acceleration_performance_summary()
        return {
            "success": True,
            "data": performance
        }
    except Exception as e:
        logger.error("Failed to get acceleration performance", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get acceleration performance")


@router.post("/hardware/acceleration/benchmark")
async def benchmark_acceleration_devices(model_path: str, iterations: int = 50):
    """Benchmark all available acceleration devices with a model"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        results = hardware_service.benchmark_acceleration_devices(model_path, iterations)
        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        logger.error("Benchmark failed", model_path=model_path, error=str(e))
        raise HTTPException(status_code=500, detail="Benchmark failed")


@router.post("/hardware/acceleration/load-model")
async def load_model_with_acceleration(model_path: str, device_preference: Optional[str] = None):
    """Load a model with hardware acceleration"""
    if not hardware_service:
        raise HTTPException(status_code=503, detail="Hardware service not available")
        
    try:
        result = await hardware_service.load_model_with_acceleration(model_path, device_preference)
        return {
            "success": True,
            "data": {
                "device": result["device"],
                "accelerator": result["accelerator"].device_name,
                "model_loaded": True
            }
        }
    except Exception as e:
        logger.error("Failed to load model with acceleration", model_path=model_path, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load model with acceleration")


# Model endpoints
@router.get("/models")
async def get_models():
    """Get available models"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        models = model_service.get_all_models()
        return {
            "success": True,
            "data": models
        }
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get models")


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model details"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        model = model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
            
        return {
            "success": True,
            "data": model
        }
    except Exception as e:
        logger.error("Failed to get model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get model")


@router.post("/models/{model_id}/load")
async def load_model(model_id: str, background_tasks: BackgroundTasks):
    """Load a model into memory"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        # Check if model is available
        model = model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
            
        if not model.get("downloaded", False):
            raise HTTPException(status_code=400, detail="Model not downloaded")
            
        # Start loading in background
        background_tasks.add_task(model_service.load_model, model_id)
        
        return {
            "success": True,
            "data": {"message": "Model loading started", "model_id": model_id}
        }
    except Exception as e:
        logger.error("Failed to load model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load model")


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        success = await model_service.unload_model(model_id)
        if success:
            return {
                "success": True,
                "data": {"message": "Model unloaded", "model_id": model_id}
            }
        else:
            raise HTTPException(status_code=404, detail="Model not loaded")
    except Exception as e:
        logger.error("Failed to unload model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to unload model")


@router.get("/models/search")
async def search_models(query: str, limit: int = 10):
    """Search for models"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        results = model_service.search_models(query, limit)
        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        logger.error("Model search failed", query=query, error=str(e))
        raise HTTPException(status_code=500, detail="Model search failed")


@router.post("/models/sync")
async def sync_models(query: str = "llm gguf"):
    """Sync models with Modelscope API"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        new_models = await model_service.sync_with_modelscope(query)
        return {
            "success": True,
            "data": {
                "new_models": new_models,
                "total_models": len(model_service.get_all_models())
            }
        }
    except Exception as e:
        logger.error("Model sync failed", error=str(e))
        raise HTTPException(status_code=500, detail="Model sync failed")


@router.post("/models/{model_id}/download")
async def download_model(model_id: str, quantization: str = "Q4_K_M", threads: int = 4):
    """Download a model"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        # Check if model exists
        model = model_service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
            
        # Start download
        task_id = await model_service.download_model(model_id, quantization, threads)
        
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "model_id": model_id,
                "quantization": quantization,
                "threads": threads
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Model download failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Model download failed")


@router.get("/models/downloads")
async def get_download_tasks():
    """Get all download tasks"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        tasks = model_service.get_all_download_tasks()
        return {
            "success": True,
            "data": tasks
        }
    except Exception as e:
        logger.error("Failed to get download tasks", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get download tasks")


@router.get("/models/downloads/{task_id}")
async def get_download_task(task_id: str):
    """Get specific download task status"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        task = model_service.get_download_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Download task not found")
            
        return {
            "success": True,
            "data": task
        }
    except Exception as e:
        logger.error("Failed to get download task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get download task")


@router.post("/models/downloads/{task_id}/pause")
async def pause_download(task_id: str):
    """Pause a download task"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        success = await model_service.pause_download(task_id)
        return {
            "success": success,
            "data": {"task_id": task_id, "status": "paused" if success else "failed"}
        }
    except Exception as e:
        logger.error("Failed to pause download", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to pause download")


@router.post("/models/downloads/{task_id}/resume")
async def resume_download(task_id: str):
    """Resume a download task"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        success = await model_service.resume_download(task_id)
        return {
            "success": success,
            "data": {"task_id": task_id, "status": "resumed" if success else "failed"}
        }
    except Exception as e:
        logger.error("Failed to resume download", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resume download")


@router.post("/models/downloads/{task_id}/cancel")
async def cancel_download(task_id: str):
    """Cancel a download task"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
        
    try:
        success = await model_service.cancel_download(task_id)
        return {
            "success": success,
            "data": {"task_id": task_id, "status": "cancelled" if success else "failed"}
        }
    except Exception as e:
        logger.error("Failed to cancel download", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel download")


# Chat endpoints
@router.post("/chat/sessions")
async def create_chat_session(title: Optional[str] = None):
    """Create a new chat session"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
        
    try:
        session_id = await chat_service.create_session(title)
        return {
            "success": True,
            "data": {"session_id": session_id, "title": title}
        }
    except Exception as e:
        logger.error("Failed to create chat session", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get chat session details"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
        
    try:
        session = chat_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {
            "success": True,
            "data": session
        }
    except Exception as e:
        logger.error("Failed to get chat session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get chat session")


@router.post("/chat/sessions/{session_id}/messages")
async def send_chat_message(session_id: str, message: str, model_id: Optional[str] = None):
    """Send a message to a chat session"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
        
    try:
        # This would integrate with the actual chat processing
        response = await chat_service.send_message(session_id, message, model_id)
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "message": message,
                "response": response
            }
        }
    except Exception as e:
        logger.error("Failed to send chat message", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to send chat message")


@router.get("/chat/sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, limit: int = 50, offset: int = 0):
    """Get messages from a chat session"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
        
    try:
        messages = chat_service.get_messages(session_id, limit, offset)
        return {
            "success": True,
            "data": messages
        }
    except Exception as e:
        logger.error("Failed to get chat messages", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get chat messages")


# Download endpoints
@router.get("/downloads")
async def get_downloads():
    """Get download tasks"""
    # This would integrate with download service
    return {
        "success": True,
        "data": []  # Placeholder
    }


@router.post("/downloads")
async def create_download(model_id: str, url: str):
    """Create a new download task"""
    # This would integrate with download service
    return {
        "success": True,
        "data": {"task_id": "placeholder", "model_id": model_id, "status": "pending"}
    }


# System endpoints
@router.get("/system/info")
async def get_system_info():
    """Get system information"""
    import platform
    import sys
    
    return {
        "success": True,
        "data": {
            "platform": platform.system(),
            "architecture": platform.architecture(),
            "python_version": sys.version,
            "app_version": settings.APP_VERSION,
            "debug": settings.DEBUG
        }
    }


@router.get("/system/health")
async def health_check():
    """System health check"""
    services_status = {
        "hardware": hardware_service is not None,
        "models": model_service is not None,
        "chat": chat_service is not None
    }
    
    all_healthy = all(services_status.values())
    
    return {
        "success": all_healthy,
        "data": {
            "status": "healthy" if all_healthy else "degraded",
            "services": services_status,
            "timestamp": "2024-01-01T00:00:00Z"  # Placeholder
        }
    }