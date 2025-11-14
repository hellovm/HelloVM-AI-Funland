"""
Base accelerator interface for hardware acceleration backends
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import structlog

logger = structlog.get_logger(__name__)


class BaseAccelerator(ABC):
    """Base class for all hardware accelerators"""
    
    def __init__(self, device_type: str, device_name: str):
        self.device_type = device_type
        self.device_name = device_name
        self.model = None
        self.config = {}
        self._is_initialized = False
        self._performance_metrics = {
            "total_inferences": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
            "last_inference_time": 0.0,
            "errors": 0
        }
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the accelerator is available on this system"""
        pass
        
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this accelerator"""
        pass
        
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a model for this accelerator"""
        pass
        
    @abstractmethod
    def infer(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference on the loaded model"""
        pass
        
    def initialize(self, **config) -> bool:
        """Initialize the accelerator with configuration"""
        try:
            if not self.is_available():
                logger.warning(f"{self.device_type} accelerator not available")
                return False
                
            self.config.update(config)
            self._is_initialized = True
            logger.info(f"{self.device_type} accelerator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.device_type} accelerator", error=str(e))
            return False
            
    def cleanup(self):
        """Cleanup accelerator resources"""
        try:
            if self.model:
                self._cleanup_model(self.model)
                self.model = None
            self._is_initialized = False
            logger.info(f"{self.device_type} accelerator cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up {self.device_type} accelerator", error=str(e))
            
    def _cleanup_model(self, model: Any):
        """Cleanup model-specific resources"""
        # Override in subclasses if needed
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get accelerator information"""
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "is_available": self.is_available(),
            "is_initialized": self._is_initialized,
            "config": self.config,
            "performance_metrics": self._performance_metrics
        }
        
    def _update_performance_metrics(self, inference_time: float, success: bool = True):
        """Update performance metrics after inference"""
        if success:
            self._performance_metrics["total_inferences"] += 1
            self._performance_metrics["total_inference_time"] += inference_time
            self._performance_metrics["last_inference_time"] = inference_time
            
            # Update average
            total = self._performance_metrics["total_inferences"]
            if total > 0:
                self._performance_metrics["average_inference_time"] = (
                    self._performance_metrics["total_inference_time"] / total
                )
        else:
            self._performance_metrics["errors"] += 1
            
    def _time_inference(self, func, *args, **kwargs):
        """Time an inference operation and update metrics"""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            inference_time = time.time() - start_time
            self._update_performance_metrics(inference_time, success=True)
            return result
        except Exception as e:
            inference_time = time.time() - start_time
            self._update_performance_metrics(inference_time, success=False)
            raise e
            
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()