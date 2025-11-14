"""
OpenVINO Accelerator implementation for Intel hardware acceleration
"""

import os
import time
import structlog
from typing import Any, Dict, List, Optional
from .base import BaseAccelerator

logger = structlog.get_logger(__name__)


class OpenVINOAccelerator(BaseAccelerator):
    """OpenVINO-based accelerator for Intel hardware (CPU, GPU, NPU)"""
    
    def __init__(self):
        super().__init__("openvino", "Intel OpenVINO")
        self.core = None
        self.available_devices = []
        self.compiled_models = {}
        
    def is_available(self) -> bool:
        """Check if OpenVINO is available"""
        try:
            import openvino as ov
            
            # Try to create a Core object
            core = ov.Core()
            available_devices = core.available_devices
            
            # Check for Intel devices
            intel_devices = [d for d in available_devices if any(x in d.upper() for x in ['CPU', 'GPU', 'NPU'])]
            
            return len(intel_devices) > 0
            
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"OpenVINO availability check failed: {e}")
            return False
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get OpenVINO performance metrics"""
        try:
            if not self.core:
                return {"error": "OpenVINO not initialized"}
                
            metrics = {
                "available_devices": self.available_devices,
                "compiled_models_count": len(self.compiled_models),
                "openvino_version": self.core.get_version() if self.core else "unknown",
                "device_metrics": {}
            }
            
            # Get device-specific metrics
            for device in self.available_devices:
                try:
                    device_metrics = self._get_device_metrics(device)
                    metrics["device_metrics"][device] = device_metrics
                except Exception as e:
                    metrics["device_metrics"][device] = {"error": str(e)}
                    
            metrics["inference_metrics"] = self._performance_metrics.copy()
            return metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "inference_metrics": self._performance_metrics.copy()
            }
            
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a model for OpenVINO inference"""
        return self._time_inference(self._load_model_internal, model_path, **kwargs)
        
    def infer(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference with OpenVINO"""
        return self._time_inference(self._run_inference, model, input_data, **kwargs)
        
    def initialize(self, **config) -> bool:
        """Initialize OpenVINO with configuration"""
        try:
            if not self.is_available():
                logger.warning("OpenVINO not available")
                return False
                
            import openvino as ov
            
            self.core = ov.Core()
            self.available_devices = self.core.available_devices
            
            # Filter for Intel devices
            intel_devices = [d for d in self.available_devices if any(x in d.upper() for x in ['CPU', 'GPU', 'NPU'])]
            self.available_devices = intel_devices
            
            self.config.update(config)
            self._is_initialized = True
            
            logger.info(f"OpenVINO initialized with devices: {self.available_devices}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO: {e}")
            return False
            
    def _load_model_internal(self, model_path: str, **kwargs) -> Any:
        """Internal model loading logic"""
        if not self.core:
            raise RuntimeError("OpenVINO not initialized")
            
        # Determine target device
        target_device = kwargs.get('device', 'CPU')
        if target_device not in self.available_devices:
            # Fallback to CPU if requested device not available
            target_device = 'CPU' if 'CPU' in self.available_devices else self.available_devices[0]
            
        # Check if model is already compiled
        model_key = f"{model_path}_{target_device}"
        if model_key in self.compiled_models:
            logger.info(f"Using cached compiled model: {model_key}")
            return self.compiled_models[model_key]
            
        try:
            # Handle different model formats
            if model_path.endswith('.xml') and os.path.exists(model_path.replace('.xml', '.bin')):
                # OpenVINO IR format
                model = self.core.read_model(model_path)
            elif model_path.endswith('.onnx'):
                # ONNX format
                model = self.core.read_model(model_path)
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch format - would need conversion
                raise RuntimeError("PyTorch models need to be converted to OpenVINO format first")
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
                
            # Compile model for target device
            compiled_model = self.core.compile_model(model, target_device)
            
            # Cache the compiled model
            self.compiled_models[model_key] = compiled_model
            
            logger.info(f"Model loaded on {target_device}: {model_path}")
            return compiled_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO model {model_path}: {e}")
            
    def _run_inference(self, compiled_model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run OpenVINO inference"""
        try:
            # Get input and output layers
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            
            # Prepare input data
            import numpy as np
            
            if isinstance(input_data, dict):
                # Handle dictionary input (multiple inputs)
                inputs = {}
                for i, (key, value) in enumerate(input_data.items()):
                    if i < len(compiled_model.inputs):
                        input_name = compiled_model.input(i).get_any_name()
                        inputs[input_name] = np.array(value, dtype=np.float32)
            elif isinstance(input_data, (list, tuple)):
                # Handle list/tuple input
                if len(input_data) == 1:
                    inputs = {input_layer.get_any_name(): np.array(input_data[0], dtype=np.float32)}
                else:
                    inputs = {}
                    for i, data in enumerate(input_data):
                        if i < len(compiled_model.inputs):
                            input_name = compiled_model.input(i).get_any_name()
                            inputs[input_name] = np.array(data, dtype=np.float32)
            else:
                # Single input
                inputs = {input_layer.get_any_name(): np.array(input_data, dtype=np.float32)}
            
            # Run inference
            result = compiled_model(inputs)
            
            # Extract output
            if len(result) == 1:
                output = result[output_layer]
            else:
                output = {out.get_any_name(): result[out] for out in compiled_model.outputs}
            
            # Get device info
            device_name = compiled_model.get_property("EXECUTION_DEVICES")[0] if hasattr(compiled_model, 'get_property') else "Unknown"
            
            return {
                'output': output,
                'device': device_name,
                'acceleration': 'openvino',
                'model_format': 'openvino_ir',
                'input_shape': str(input_layer.get_partial_shape()),
                'output_shape': str(output_layer.get_partial_shape())
            }
            
        except Exception as e:
            raise RuntimeError(f"OpenVINO inference failed: {e}")
            
    def _get_device_metrics(self, device: str) -> Dict[str, Any]:
        """Get device-specific metrics"""
        try:
            if not self.core:
                return {"error": "OpenVINO not initialized"}
                
            metrics = {
                "name": device,
                "full_name": self.core.get_property(device, "FULL_DEVICE_NAME") if hasattr(self.core, 'get_property') else device,
                "type": self._get_device_type(device),
                "is_available": True
            }
            
            # Get device-specific properties
            if device.upper() == "GPU":
                try:
                    # Intel GPU metrics
                    metrics["driver_version"] = self.core.get_property(device, "DRIVER_VERSION")
                    metrics["device_id"] = self.core.get_property(device, "DEVICE_ID")
                except:
                    pass
                    
            elif device.upper() == "CPU":
                try:
                    # CPU metrics
                    metrics["supported_precision"] = self.core.get_property(device, "OPTIMIZATION_CAPABILITIES")
                    metrics["cores"] = self.core.get_property(device, "NUMBER_OF_CORES")
                except:
                    pass
                    
            elif "NPU" in device.upper():
                try:
                    # NPU metrics
                    metrics["driver_version"] = self.core.get_property(device, "DRIVER_VERSION")
                except:
                    pass
                    
            return metrics
            
        except Exception as e:
            return {"error": str(e), "name": device}
            
    def _get_device_type(self, device: str) -> str:
        """Get device type from device name"""
        device_upper = device.upper()
        if "GPU" in device_upper:
            return "intel_gpu"
        elif "NPU" in device_upper:
            return "intel_npu"
        elif "CPU" in device_upper:
            return "cpu"
        else:
            return "unknown"
            
    def get_optimal_device(self, model_size_mb: int, precision: str = "FP16") -> str:
        """Get optimal device for model based on size and precision"""
        if not self.available_devices:
            return "CPU"  # Fallback
            
        # Priority: GPU > NPU > CPU
        device_priority = ["GPU", "NPU", "CPU"]
        
        for device_type in device_priority:
            available_device = next((d for d in self.available_devices if device_type in d.upper()), None)
            if available_device:
                return available_device
                
        return self.available_devices[0]  # Fallback to first available
        
    def optimize_model(self, model_path: str, device: str = None, **kwargs) -> str:
        """Optimize model for specific device"""
        if not self.core:
            raise RuntimeError("OpenVINO not initialized")
            
        target_device = device or self.get_optimal_device(0)
        
        try:
            # Load model
            model = self.core.read_model(model_path)
            
            # Apply optimizations
            from openvino import pass_manager
            
            # Create optimization passes
            manager = pass_manager.Manager()
            
            # Add optimization passes based on device
            if "GPU" in target_device.upper():
                # GPU-specific optimizations
                manager.register_pass("ConvertFP32ToFP16")  # Use FP16 for GPU
            elif "NPU" in target_device.upper():
                # NPU-specific optimizations
                manager.register_pass("ConvertFP32ToFP16")
                manager.register_pass("ConvertMatMulToFullyConnected")
            else:
                # CPU optimizations
                manager.register_pass("ConvertFP32ToFP16")
                
            # Run optimizations
            manager.run_passes(model)
            
            # Save optimized model
            optimized_path = model_path.replace('.xml', f'_optimized_{target_device}.xml')
            self.core.save_model(model, optimized_path)
            
            logger.info(f"Model optimized for {target_device}: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            raise RuntimeError(f"Model optimization failed: {e}")
            
    def benchmark_model(self, model_path: str, device: str = None, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance on specific device"""
        if not self.core:
            raise RuntimeError("OpenVINO not initialized")
            
        target_device = device or self.get_optimal_device(0)
        
        try:
            # Load and compile model
            compiled_model = self._load_model_internal(model_path, device=target_device)
            
            # Create dummy input
            import numpy as np
            input_shape = compiled_model.input(0).get_partial_shape()
            dummy_input = np.random.randn(*input_shape.get_shape()).astype(np.float32)
            
            # Warm up
            for _ in range(10):
                compiled_model({compiled_model.input(0).get_any_name(): dummy_input})
                
            # Benchmark
            import time
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                compiled_model({compiled_model.input(0).get_any_name(): dummy_input})
                end_time = time.time()
                times.append(end_time - start_time)
                
            return {
                "device": target_device,
                "iterations": iterations,
                "avg_time_ms": np.mean(times) * 1000,
                "min_time_ms": np.min(times) * 1000,
                "max_time_ms": np.max(times) * 1000,
                "std_time_ms": np.std(times) * 1000,
                "throughput_fps": 1.0 / np.mean(times)
            }
            
        except Exception as e:
            raise RuntimeError(f"Benchmark failed: {e}")
            
    def cleanup(self):
        """Cleanup OpenVINO resources"""
        try:
            # Clear compiled models
            self.compiled_models.clear()
            
            # Reset core
            self.core = None
            self.available_devices = []
            
            super().cleanup()
            
        except Exception as e:
            logger.error(f"OpenVINO cleanup failed: {e}")
            
    def _cleanup_model(self, model: Any):
        """Cleanup model resources"""
        # OpenVINO models are automatically cleaned up
        pass