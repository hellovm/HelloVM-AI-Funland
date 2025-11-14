"""
CPU Accelerator implementation
"""

import torch
import time
import psutil
from typing import Any, Dict, List, Optional
from .base import BaseAccelerator


class CPUAccelerator(BaseAccelerator):
    """CPU-based accelerator for model inference"""
    
    def __init__(self):
        super().__init__("cpu", "CPU")
        self.device = torch.device('cpu')
        self._cpu_info = None
        self._supported_instructions = []
        
    def is_available(self) -> bool:
        """CPU is always available"""
        return True
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get CPU performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "cpu_count": psutil.cpu_count(),
                "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                "supported_instructions": self._get_supported_instructions(),
                "temperature_celsius": self._get_cpu_temperature(),
                "inference_metrics": self._performance_metrics.copy()
            }
        except Exception as e:
            return {
                "error": str(e),
                "inference_metrics": self._performance_metrics.copy()
            }
            
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load a model for CPU inference"""
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
            raise RuntimeError(f"Failed to load model {model_path} on CPU: {e}")
            
    def infer(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference on CPU"""
        return self._time_inference(self._run_inference, model, input_data, **kwargs)
        
    def _load_gguf_model(self, model_path: str, **kwargs) -> Any:
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            model = Llama(
                model_path=model_path,
                n_threads=kwargs.get('n_threads', min(4, psutil.cpu_count())),
                n_batch=kwargs.get('n_batch', 512),
                n_ctx=kwargs.get('n_ctx', 2048),
                use_mmap=kwargs.get('use_mmap', True),
                use_mlock=kwargs.get('use_mlock', False),
                verbose=kwargs.get('verbose', False)
            )
            
            self.model = model
            logger.info(f"GGUF model loaded on CPU: {model_path}")
            return model
            
        except ImportError:
            raise RuntimeError("llama-cpp-python is required for GGUF model loading")
            
    def _load_pytorch_model(self, model_path: str, **kwargs) -> Any:
        """Load PyTorch model"""
        try:
            # Load model to CPU
            model = torch.load(model_path, map_location=self.device)
            model.eval()  # Set to evaluation mode
            
            self.model = model
            logger.info(f"PyTorch model loaded on CPU: {model_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
            
    def _load_onnx_model(self, model_path: str, **kwargs) -> Any:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            
            # Configure ONNX Runtime for CPU
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set number of threads
            session_options.intra_op_num_threads = kwargs.get('intra_op_num_threads', min(4, psutil.cpu_count()))
            session_options.inter_op_num_threads = kwargs.get('inter_op_num_threads', 1)
            
            model = ort.InferenceSession(model_path, session_options, providers=providers)
            
            self.model = model
            logger.info(f"ONNX model loaded on CPU: {model_path}")
            return model
            
        except ImportError:
            raise RuntimeError("onnxruntime is required for ONNX model loading")
            
    def _run_inference(self, model: Any, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run the actual inference"""
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
                    'device': 'cpu',
                    'acceleration': 'native',
                    'model_format': 'gguf'
                }
                
            elif hasattr(model, 'run'):
                # ONNX model
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name
                
                # Convert input to numpy array
                import numpy as np
                if isinstance(input_data, dict):
                    # Handle dictionary input
                    input_feed = {k: np.array(v, dtype=np.float32) for k, v in input_data.items()}
                else:
                    input_feed = {input_name: np.array(input_data, dtype=np.float32)}
                
                outputs = model.run([output_name], input_feed)
                
                return {
                    'output': outputs[0],
                    'device': 'cpu',
                    'acceleration': 'onnx',
                    'model_format': 'onnx'
                }
                
            else:
                # PyTorch model
                with torch.no_grad():
                    if isinstance(input_data, torch.Tensor):
                        input_tensor = input_data.to(self.device)
                    else:
                        input_tensor = torch.tensor(input_data, device=self.device)
                        
                    output = model(input_tensor)
                
                return {
                    'output': output.cpu().numpy() if isinstance(output, torch.Tensor) else output,
                    'device': 'cpu',
                    'acceleration': 'pytorch',
                    'model_format': 'pytorch'
                }
                
        except Exception as e:
            raise RuntimeError(f"CPU inference failed: {e}")
            
    def _get_supported_instructions(self) -> List[str]:
        """Get supported CPU instruction sets"""
        if self._supported_instructions:
            return self._supported_instructions
            
        instructions = []
        
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            
            # Check for various instruction sets
            instruction_sets = {
                'AVX': ['avx'],
                'AVX2': ['avx2'],
                'AVX512': ['avx512f'],
                'SSE': ['sse', 'sse2', 'sse3'],
                'SSE4': ['sse4_1', 'sse4_2'],
                'FMA': ['fma'],
                'BMI': ['bmi1', 'bmi2']
            }
            
            for instruction, required_flags in instruction_sets.items():
                if all(flag in flags for flag in required_flags):
                    instructions.append(instruction)
                    
        except ImportError:
            # Fallback detection
            import platform
            import subprocess
            
            try:
                # Try to detect using system commands
                if platform.system() == "Windows":
                    result = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True)
                    cpu_name = result.stdout.strip()
                    
                    # Basic detection based on CPU name
                    if "Core" in cpu_name:
                        instructions.extend(['SSE', 'SSE2', 'SSE3', 'SSE4'])
                        if "i5" in cpu_name or "i7" in cpu_name or "i9" in cpu_name:
                            instructions.extend(['AVX', 'AVX2'])
                            
                elif platform.system() == "Linux":
                    result = subprocess.run(['grep', 'flags', '/proc/cpuinfo'], capture_output=True, text=True)
                    flags = result.stdout.lower()
                    
                    if 'avx' in flags:
                        instructions.append('AVX')
                    if 'avx2' in flags:
                        instructions.append('AVX2')
                    if 'sse' in flags:
                        instructions.append('SSE')
                        
            except:
                pass
                
        self._supported_instructions = instructions
        return instructions
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            temperatures = psutil.sensors_temperatures()
            
            # Try different temperature sensors
            for sensor_name in ['coretemp', 'cpu_thermal', 'k10temp', 'zenpower']:
                if sensor_name in temperatures:
                    # Get the first temperature reading
                    temp = temperatures[sensor_name][0]
                    return temp.current
                    
            return None
            
        except:
            return None
            
    def optimize_for_inference(self, **kwargs) -> Dict[str, Any]:
        """Optimize CPU for inference"""
        optimizations = {}
        
        try:
            # Set number of threads for PyTorch
            if 'torch_threads' in kwargs:
                torch.set_num_threads(kwargs['torch_threads'])
                optimizations['torch_threads'] = kwargs['torch_threads']
                
            # Set number of interop threads
            if 'torch_interop_threads' in kwargs:
                torch.set_num_interop_threads(kwargs['torch_interop_threads'])
                optimizations['torch_interop_threads'] = kwargs['torch_interop_threads']
                
            # Enable/disable optimizations
            if 'torch_jit' in kwargs and kwargs['torch_jit']:
                torch.jit.set_fuser('fuser2')
                optimizations['torch_jit'] = True
                
            logger.info(f"CPU optimizations applied: {optimizations}")
            return optimizations
            
        except Exception as e:
            logger.error(f"Failed to apply CPU optimizations: {e}")
            return {"error": str(e)}
            
    def _cleanup_model(self, model: Any):
        """Cleanup model resources"""
        try:
            if hasattr(model, 'close'):
                model.close()
            elif hasattr(model, 'free'):
                model.free()
                
            # Clear any cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
            
    def get_optimal_settings(self, model_size_mb: int) -> Dict[str, Any]:
        """Get optimal settings based on model size and CPU capabilities"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        settings = {
            "n_threads": min(cpu_count, 8),  # Cap at 8 threads for most models
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": False,
            "torch_threads": min(cpu_count, 8),
            "torch_interop_threads": 1
        }
        
        # Adjust based on model size
        if model_size_mb < 1000:  # Small models (< 1GB)
            settings["n_batch"] = 1024
            settings["n_threads"] = min(cpu_count, 4)
        elif model_size_mb > 5000:  # Large models (> 5GB)
            settings["n_batch"] = 256
            settings["use_mmap"] = True
            settings["use_mlock"] = memory_gb > 16  # Lock memory if we have enough
            
        # Adjust based on available memory
        if memory_gb < 8:
            settings["n_batch"] = min(settings["n_batch"], 256)
            settings["use_mlock"] = False
            
        return settings