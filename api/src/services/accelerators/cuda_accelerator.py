"""
CUDA Accelerator implementation for NVIDIA GPU acceleration
"""

import torch
import time
import structlog
from typing import Any, Dict, List, Optional
from .base import BaseAccelerator

# Optional imports with fallback
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = structlog.get_logger(__name__)


class CUDAAccelerator(BaseAccelerator):
    """CUDA-based accelerator for NVIDIA GPU acceleration"""
    
    def __init__(self):
        super().__init__("cuda", "NVIDIA CUDA")
        self.device = None
        self.device_id = 0
        self.gpu_handle = None
        self._gpu_info = None
        
    def is_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            return torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"CUDA availability check failed: {e}")
            return False
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get CUDA/GPU performance metrics"""
        try:
            if not self.is_available():
                return {"error": "CUDA not available"}
                
            if not self.gpu_handle:
                self._initialize_nvidia_ml()
                
            metrics = {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_name": torch.cuda.get_device_name(self.device_id),
                "device_properties": self._get_device_properties(),
                "memory_info": self._get_memory_info(),
                "utilization": self._get_gpu_utilization(),
                "temperature": self._get_gpu_temperature(),
                "power": self._get_gpu_power(),
                "inference_metrics": self._performance_metrics.copy()
            }
            
            return metrics
            
        except Exception as e:
            return {
                "error": str(e),
                "inference_metrics": self._performance_metrics.copy()
            }
            
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a model for CUDA inference"""
        return self._time_inference(self._load_model_internal, model_path, **kwargs)
        
    def infer(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference with CUDA"""
        return self._time_inference(self._run_inference, model, input_data, **kwargs)
        
    def initialize(self, **config) -> bool:
        """Initialize CUDA with configuration"""
        try:
            if not self.is_available():
                logger.warning("CUDA not available")
                return False
                
            # Set device
            self.device_id = config.get('device_id', 0)
            if self.device_id >= torch.cuda.device_count():
                logger.warning(f"Device {self.device_id} not available, using device 0")
                self.device_id = 0
                
            self.device = torch.device(f'cuda:{self.device_id}')
            torch.cuda.set_device(self.device)
            
            # Initialize NVIDIA ML for monitoring
            self._initialize_nvidia_ml()
            
            # Set memory fraction if specified
            if 'memory_fraction' in config:
                torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
                
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                
            # Set cuDNN settings
            torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
            torch.backends.cudnn.deterministic = config.get('cudnn_deterministic', False)
            torch.backends.cudnn.enabled = config.get('cudnn_enabled', True)
            
            self.config.update(config)
            self._is_initialized = True
            
            logger.info(f"CUDA initialized on device {self.device_id}: {torch.cuda.get_device_name(self.device_id)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return False
            
    def _initialize_nvidia_ml(self):
        """Initialize NVIDIA ML library for monitoring"""
        if not HAS_PYNVML:
            logger.warning("pynvml not available for NVIDIA GPU monitoring")
            return
            
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            logger.debug(f"NVIDIA ML initialized for device {self.device_id}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize NVIDIA ML: {e}")
            
    def _load_model_internal(self, model_path: str, **kwargs) -> Any:
        """Internal model loading logic"""
        if not self.device:
            raise RuntimeError("CUDA not initialized")
            
        try:
            # Determine model format and load accordingly
            if model_path.endswith('.gguf') or model_path.endswith('.ggml'):
                return self._load_gguf_model(model_path, **kwargs)
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                return self._load_pytorch_model(model_path, **kwargs)
            elif model_path.endswith('.onnx'):
                return self._load_onnx_model(model_path, **kwargs)
            else:
                # Try to load as PyTorch model by default
                return self._load_pytorch_model(model_path, **kwargs)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path} on CUDA: {e}")
            
    def _load_gguf_model(self, model_path: str, **kwargs) -> Any:
        """Load GGUF model with CUDA support"""
        try:
            from llama_cpp import Llama
            
            # Configure for CUDA
            model_kwargs = {
                'model_path': model_path,
                'n_gpu_layers': kwargs.get('n_gpu_layers', -1),  # Use all layers on GPU by default
                'n_batch': kwargs.get('n_batch', 512),
                'n_ctx': kwargs.get('n_ctx', 2048),
                'use_mmap': kwargs.get('use_mmap', True),
                'use_mlock': kwargs.get('use_mlock', False),
                'verbose': kwargs.get('verbose', False)
            }
            
            model = Llama(**model_kwargs)
            
            self.model = model
            logger.info(f"GGUF model loaded on CUDA: {model_path}")
            return model
            
        except ImportError:
            raise RuntimeError("llama-cpp-python with CUDA support is required")
            
    def _load_pytorch_model(self, model_path: str, **kwargs) -> Any:
        """Load PyTorch model to CUDA"""
        try:
            # Load model and move to GPU
            model = torch.load(model_path, map_location=self.device)
            
            # Handle different model types
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
                model.eval()  # Set to evaluation mode
            elif isinstance(model, dict) and 'model_state_dict' in model:
                # Load state dict - this would need the model architecture
                raise RuntimeError("Model architecture needed to load state dict")
                
            self.model = model
            logger.info(f"PyTorch model loaded on CUDA device {self.device_id}: {model_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model on CUDA: {e}")
            
    def _load_onnx_model(self, model_path: str, **kwargs) -> Any:
        """Load ONNX model with CUDA execution provider"""
        try:
            import onnxruntime as ort
            
            # Configure CUDA execution provider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.device_id,
                    'gpu_mem_limit': kwargs.get('gpu_mem_limit', 2 * 1024 * 1024 * 1024),  # 2GB default
                    'arena_extend_strategy': kwargs.get('arena_extend_strategy', 'kNextPowerOfTwo'),
                    'cudnn_conv_algo_search': kwargs.get('cudnn_conv_algo_search', 'EXHAUSTIVE'),
                    'do_copy_in_default_stream': kwargs.get('do_copy_in_default_stream', True),
                    'cudnn_conv_use_max_workspace': kwargs.get('cudnn_conv_use_max_workspace', True),
                }),
                'CPUExecutionProvider'  # Fallback
            ]
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            model = ort.InferenceSession(model_path, session_options, providers=providers)
            
            # Verify CUDA provider is being used
            available_providers = model.get_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                logger.warning("CUDA execution provider not available, using CPU fallback")
                
            self.model = model
            logger.info(f"ONNX model loaded on CUDA: {model_path}")
            return model
            
        except ImportError:
            raise RuntimeError("onnxruntime-gpu is required for CUDA ONNX inference")
            
    def _run_inference(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference on CUDA"""
        try:
            if hasattr(model, 'create_chat_completion'):
                # GGUF chat completion
                response = model.create_chat_completion(
                    messages=input_data,
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.95),
                    top_k=kwargs.get('top_k', 40),
                    max_tokens=kwargs.get('max_tokens', 512),
                    stream=kwargs.get('stream', False),
                    stop=kwargs.get('stop', [])
                )
                
                return {
                    'response': response,
                    'device': f'cuda:{self.device_id}',
                    'acceleration': 'cuda',
                    'model_format': 'gguf'
                }
                
            elif hasattr(model, 'run'):
                # ONNX model
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name
                
                # Convert input to numpy array (ONNX Runtime handles GPU transfer)
                import numpy as np
                if isinstance(input_data, dict):
                    input_feed = {k: np.array(v, dtype=np.float32) for k, v in input_data.items()}
                else:
                    input_feed = {input_name: np.array(input_data, dtype=np.float32)}
                
                outputs = model.run([output_name], input_feed)
                
                return {
                    'output': outputs[0],
                    'device': f'cuda:{self.device_id}',
                    'acceleration': 'cuda',
                    'model_format': 'onnx'
                }
                
            else:
                # PyTorch model
                with torch.cuda.device(self.device):
                    with torch.no_grad():
                        if isinstance(input_data, torch.Tensor):
                            input_tensor = input_data.to(self.device)
                        else:
                            input_tensor = torch.tensor(input_data, device=self.device)
                            
                        # Handle batch dimension
                        if input_tensor.dim() == 1:
                            input_tensor = input_tensor.unsqueeze(0)
                            
                        output = model(input_tensor)
                        
                        # Move output back to CPU for compatibility
                        if isinstance(output, torch.Tensor):
                            output_cpu = output.cpu()
                        else:
                            output_cpu = output
                
                return {
                    'output': output_cpu.numpy() if hasattr(output_cpu, 'numpy') else output_cpu,
                    'device': f'cuda:{self.device_id}',
                    'acceleration': 'cuda',
                    'model_format': 'pytorch'
                }
                
        except Exception as e:
            raise RuntimeError(f"CUDA inference failed: {e}")
            
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get CUDA device properties"""
        try:
            props = torch.cuda.get_device_properties(self.device_id)
            
            return {
                "name": props.name,
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
                "warp_size": props.warp_size,
                "max_threads_per_block": props.max_threads_per_block,
                "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
                "max_blocks_per_multiprocessor": props.max_blocks_per_multiprocessor,
                "compute_capability": f"{props.major}.{props.minor}"
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        try:
            if self.gpu_handle and HAS_PYNVML:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                return {
                    "total": memory_info.total,
                    "free": memory_info.free,
                    "used": memory_info.used,
                    "total_gb": memory_info.total / (1024**3),
                    "free_gb": memory_info.free / (1024**3),
                    "used_gb": memory_info.used / (1024**3),
                    "utilization_percent": (memory_info.used / memory_info.total) * 100
                }
            else:
                # Fallback to PyTorch
                total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(self.device_id)
                cached_memory = torch.cuda.memory_reserved(self.device_id)
                
                return {
                    "total": total_memory,
                    "used": allocated_memory,
                    "cached": cached_memory,
                    "total_gb": total_memory / (1024**3),
                    "used_gb": allocated_memory / (1024**3),
                    "cached_gb": cached_memory / (1024**3),
                    "utilization_percent": (allocated_memory / total_memory) * 100
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            if self.gpu_handle and HAS_PYNVML:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                return utilization.gpu
            return None
        except:
            return None
            
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature"""
        try:
            if self.gpu_handle and HAS_PYNVML:
                temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                return temperature
            return None
        except:
            return None
            
    def _get_gpu_power(self) -> Optional[float]:
        """Get GPU power usage"""
        try:
            if self.gpu_handle and HAS_PYNVML:
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # Convert to watts
                return power
            return None
        except:
            return None
            
    def optimize_for_inference(self, **kwargs) -> Dict[str, Any]:
        """Optimize CUDA for inference"""
        optimizations = {}
        
        try:
            # Enable TensorFloat-32 for better performance on Ampere and newer GPUs
            if kwargs.get('enable_tf32', True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations['tf32_enabled'] = True
                
            # Set memory allocation strategy
            if kwargs.get('memory_efficient', True):
                torch.cuda.empty_cache()  # Clear cache first
                optimizations['memory_cleared'] = True
                
            # Enable cuDNN auto-tuner
            if kwargs.get('cudnn_benchmark', True):
                torch.backends.cudnn.benchmark = True
                optimizations['cudnn_benchmark'] = True
                
            # Set memory fraction
            if 'memory_fraction' in kwargs:
                torch.cuda.set_per_process_memory_fraction(kwargs['memory_fraction'])
                optimizations['memory_fraction'] = kwargs['memory_fraction']
                
            logger.info(f"CUDA optimizations applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            logger.error(f"Failed to apply CUDA optimizations: {e}")
            return {"error": str(e)}
            
    def get_optimal_settings(self, model_size_mb: int) -> Dict[str, Any]:
        """Get optimal CUDA settings based on model size"""
        try:
            memory_info = self._get_memory_info()
            total_memory_gb = memory_info.get('total_gb', 8)  # Default to 8GB
            
            settings = {
                "device_id": self.device_id,
                "memory_fraction": 0.8,  # Use 80% of GPU memory
                "cudnn_benchmark": True,
                "enable_tf32": True,
                "memory_efficient": True
            }
            
            # Adjust based on model size
            if model_size_mb < 1000:  # Small models (< 1GB)
                settings["memory_fraction"] = 0.5  # Use less memory for small models
                settings["n_gpu_layers"] = -1  # Use all layers on GPU
                
            elif model_size_mb > 5000:  # Large models (> 5GB)
                settings["memory_fraction"] = 0.9  # Use more memory for large models
                settings["memory_efficient"] = True
                
                # Check if we have enough memory
                if model_size_mb / 1024 > total_memory_gb * 0.8:
                    settings["n_gpu_layers"] = 20  # Limit layers on GPU
                else:
                    settings["n_gpu_layers"] = -1  # Use all layers
                    
            # Adjust based on available memory
            if total_memory_gb < 8:
                settings["memory_fraction"] = min(settings["memory_fraction"], 0.7)
                settings["n_gpu_layers"] = 10  # Conservative for low-memory GPUs
                
            return settings
            
        except Exception as e:
            logger.error(f"Failed to get optimal CUDA settings: {e}")
            return self.config
            
    def benchmark_model(self, model_path: str, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model performance on CUDA"""
        try:
            # Load model
            model = self._load_model_internal(model_path)
            
            # Create dummy input
            import numpy as np
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)  # Example input
            
            # Warm up
            for _ in range(10):
                self._run_inference(model, dummy_input)
                
            # Synchronize to ensure all GPU operations are complete
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            
            for _ in range(iterations):
                torch.cuda.synchronize()  # Wait for previous operations
                start_time = time.time()
                
                self._run_inference(model, dummy_input)
                
                torch.cuda.synchronize()  # Wait for current operation
                end_time = time.time()
                times.append(end_time - start_time)
                
            return {
                "device": f"cuda:{self.device_id}",
                "device_name": torch.cuda.get_device_name(self.device_id),
                "iterations": iterations,
                "avg_time_ms": np.mean(times) * 1000,
                "min_time_ms": np.min(times) * 1000,
                "max_time_ms": np.max(times) * 1000,
                "std_time_ms": np.std(times) * 1000,
                "throughput_fps": 1.0 / np.mean(times),
                "memory_peak_gb": torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
            }
            
        except Exception as e:
            raise RuntimeError(f"CUDA benchmark failed: {e}")
            
    def cleanup(self):
        """Cleanup CUDA resources"""
        try:
            # Clear memory cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Shutdown NVIDIA ML
            if self.gpu_handle and HAS_PYNVML:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                    
            self.gpu_handle = None
            self.device = None
            
            super().cleanup()
            
        except Exception as e:
            logger.error(f"CUDA cleanup failed: {e}")
            
    def _cleanup_model(self, model: Any):
        """Cleanup model resources"""
        try:
            # Move model back to CPU before cleanup
            if hasattr(model, 'cpu'):
                model.cpu()
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Error during CUDA model cleanup: {e}")