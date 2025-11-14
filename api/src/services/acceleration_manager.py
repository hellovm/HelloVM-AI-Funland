"""
Hardware Acceleration Manager for coordinating different acceleration backends
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import structlog

from .accelerators import CPUAccelerator, OpenVINOAccelerator, CUDAAccelerator
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AccelerationResult:
    """Result from hardware acceleration"""
    output: Any
    device: str
    acceleration_type: str
    inference_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AccelerationConfig:
    """Configuration for hardware acceleration"""
    preferred_devices: List[str]  # Priority order: ["cuda", "openvino", "cpu"]
    fallback_enabled: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 2
    benchmark_mode: bool = False


class HardwareAccelerationManager:
    """Manages multiple hardware acceleration backends"""
    
    def __init__(self):
        self.accelerators = {}
        self.acceleration_configs = {}
        self.performance_history = []
        self._initialized = False
        
    async def initialize(self, config: Optional[AccelerationConfig] = None) -> bool:
        """Initialize hardware acceleration manager"""
        try:
            logger.info("Initializing hardware acceleration manager")
            
            # Default configuration
            if config is None:
                config = AccelerationConfig(
                    preferred_devices=["cuda", "openvino", "cpu"],
                    fallback_enabled=True,
                    timeout_seconds=30.0,
                    retry_count=2
                )
                
            self.acceleration_configs["default"] = config
            
            # Initialize available accelerators
            await self._initialize_accelerators()
            
            self._initialized = True
            logger.info("Hardware acceleration manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hardware acceleration manager: {e}")
            return False
            
    async def _initialize_accelerators(self):
        """Initialize available accelerators"""
        # CPU Accelerator
        try:
            cpu_accel = CPUAccelerator()
            if cpu_accel.initialize():
                self.accelerators["cpu"] = cpu_accel
                logger.info("CPU accelerator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CPU accelerator: {e}")
            
        # OpenVINO Accelerator
        try:
            openvino_accel = OpenVINOAccelerator()
            if openvino_accel.initialize():
                self.accelerators["openvino"] = openvino_accel
                logger.info("OpenVINO accelerator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenVINO accelerator: {e}")
            
        # CUDA Accelerator
        try:
            cuda_accel = CUDAAccelerator()
            if cuda_accel.initialize():
                self.accelerators["cuda"] = cuda_accel
                logger.info("CUDA accelerator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CUDA accelerator: {e}")
            
        logger.info(f"Initialized {len(self.accelerators)} accelerators: {list(self.accelerators.keys())}")
        
    async def load_model(self, model_path: str, device_preference: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load a model on the best available device"""
        if not self._initialized:
            raise RuntimeError("Hardware acceleration manager not initialized")
            
        config = self.acceleration_configs.get("default")
        if not config:
            raise RuntimeError("No acceleration configuration found")
            
        # Determine device preference
        if device_preference and device_preference in self.accelerators:
            preferred_devices = [device_preference] + [d for d in config.preferred_devices if d != device_preference]
        else:
            preferred_devices = config.preferred_devices
            
        # Try to load model on preferred devices
        for device in preferred_devices:
            if device not in self.accelerators:
                logger.warning(f"Device {device} not available")
                continue
                
            try:
                accelerator = self.accelerators[device]
                model = accelerator.load_model(model_path, **kwargs)
                
                result = {
                    "model": model,
                    "device": device,
                    "accelerator": accelerator,
                    "success": True
                }
                
                logger.info(f"Model loaded successfully on {device}: {model_path}")
                return result
                
            except Exception as e:
                logger.warning(f"Failed to load model on {device}: {e}")
                if not config.fallback_enabled:
                    raise RuntimeError(f"Failed to load model on {device}: {e}")
                    
        # No device available
        raise RuntimeError(f"Failed to load model on any available device: {preferred_devices}")
        
    async def infer(self, model_info: Dict[str, Any], input_data: Any, **kwargs) -> AccelerationResult:
        """Run inference with hardware acceleration"""
        if not self._initialized:
            raise RuntimeError("Hardware acceleration manager not initialized")
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            accelerator = model_info["accelerator"]
            model = model_info["model"]
            device = model_info["device"]
            
            # Run inference
            result = accelerator.infer(model, input_data, **kwargs)
            
            inference_time = asyncio.get_event_loop().time() - start_time
            
            acceleration_result = AccelerationResult(
                output=result.get("output", result.get("response")),
                device=device,
                acceleration_type=result.get("acceleration", device),
                inference_time=inference_time,
                success=True,
                metadata={
                    "original_result": result,
                    "device_info": accelerator.get_info(),
                    "performance_metrics": accelerator.get_performance_metrics()
                }
            )
            
            # Record performance history
            self._record_performance(acceleration_result)
            
            logger.info(f"Inference completed on {device} in {inference_time:.3f}s")
            return acceleration_result
            
        except Exception as e:
            inference_time = asyncio.get_event_loop().time() - start_time
            
            acceleration_result = AccelerationResult(
                output=None,
                device=model_info.get("device", "unknown"),
                acceleration_type="failed",
                inference_time=inference_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Inference failed on {model_info.get('device', 'unknown')}: {e}")
            return acceleration_result
            
    def get_available_devices(self) -> List[str]:
        """Get list of available acceleration devices"""
        return list(self.accelerators.keys())
        
    def get_device_info(self, device: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific device"""
        if device not in self.accelerators:
            return None
            
        return self.accelerators[device].get_info()
        
    def get_performance_summary(self, device: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for devices"""
        if device:
            if device not in self.accelerators:
                return {"error": f"Device {device} not available"}
                
            accelerator = self.accelerators[device]
            return {
                "device": device,
                "info": accelerator.get_info(),
                "performance_metrics": accelerator.get_performance_metrics()
            }
        else:
            # Summary for all devices
            summary = {}
            for device_name, accelerator in self.accelerators.items():
                summary[device_name] = {
                    "info": accelerator.get_info(),
                    "performance_metrics": accelerator.get_performance_metrics()
                }
            return summary
            
    def benchmark_devices(self, model_path: str, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark all available devices with a model"""
        results = {}
        
        for device_name, accelerator in self.accelerators.items():
            try:
                logger.info(f"Benchmarking {device_name} with {iterations} iterations")
                
                if hasattr(accelerator, 'benchmark_model'):
                    # Use device-specific benchmark if available
                    benchmark_result = accelerator.benchmark_model(model_path, iterations=iterations)
                    results[device_name] = benchmark_result
                else:
                    # Generic benchmark
                    model_info = asyncio.run(self.load_model(model_path, device_name))
                    
                    # Create dummy input
                    import numpy as np
                    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    
                    times = []
                    for _ in range(iterations):
                        start_time = time.time()
                        result = asyncio.run(self.infer(model_info, dummy_input))
                        end_time = time.time()
                        
                        if result.success:
                            times.append(end_time - start_time)
                            
                    if times:
                        import numpy as np
                        results[device_name] = {
                            "device": device_name,
                            "iterations": len(times),
                            "avg_time_ms": np.mean(times) * 1000,
                            "min_time_ms": np.min(times) * 1000,
                            "max_time_ms": np.max(times) * 1000,
                            "std_time_ms": np.std(times) * 1000,
                            "throughput_fps": 1.0 / np.mean(times)
                        }
                    else:
                        results[device_name] = {"error": "No successful inferences"}
                        
            except Exception as e:
                logger.error(f"Benchmark failed for {device_name}: {e}")
                results[device_name] = {"error": str(e)}
                
        return results
        
    def _record_performance(self, result: AccelerationResult):
        """Record performance metrics"""
        self.performance_history.append({
            "timestamp": datetime.now(),
            "device": result.device,
            "inference_time": result.inference_time,
            "success": result.success,
            "metadata": result.metadata
        })
        
        # Keep only recent history (last 1000 entries)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance history"""
        return self.performance_history[-limit:]
        
    def cleanup(self):
        """Cleanup all accelerators"""
        logger.info("Cleaning up hardware acceleration manager")
        
        for device_name, accelerator in self.accelerators.items():
            try:
                accelerator.cleanup()
                logger.info(f"Cleaned up {device_name} accelerator")
            except Exception as e:
                logger.error(f"Failed to cleanup {device_name} accelerator: {e}")
                
        self.accelerators.clear()
        self._initialized = False
        logger.info("Hardware acceleration manager cleanup completed")