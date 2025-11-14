"""
Hardware detection and acceleration service
Supports CPU, Intel GPU, Intel NPU, and NVIDIA GPU
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog

# Hardware-specific imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import pyadl
    HAS_PYADL = True
except ImportError:
    HAS_PYADL = False

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

try:
    from intel_npu_acceleration_library import NPUCompiler
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

from ..core.config import settings
from ..core.logging import get_logger
from .acceleration_manager import HardwareAccelerationManager, AccelerationConfig

logger = get_logger(__name__)


@dataclass
class HardwareDevice:
    """Hardware device information"""
    id: str
    name: str
    type: str  # 'cpu', 'intel_gpu', 'intel_npu', 'nvidia_gpu', 'amd_gpu'
    memory_total: Optional[float] = None  # GB
    memory_used: Optional[float] = None   # GB
    utilization: Optional[float] = None   # percentage
    temperature: Optional[float] = None   # celsius
    supported: bool = False
    selected: bool = False
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None
    last_updated: Optional[datetime] = None


@dataclass
class HardwareStatus:
    """Overall hardware status"""
    devices: List[HardwareDevice]
    primary_device: Optional[HardwareDevice]
    acceleration_mode: str  # 'single', 'multi', 'hybrid'
    scan_interval: int
    last_scan: Optional[datetime]


class HardwareService:
    """Hardware detection and monitoring service"""
    
    def __init__(self):
        self.devices: Dict[str, HardwareDevice] = {}
        self.primary_device: Optional[HardwareDevice] = None
        self.acceleration_mode: str = "single"
        self.scan_interval: int = settings.HARDWARE_SCAN_INTERVAL
        self.last_scan: Optional[datetime] = None
        self._scanning: bool = False
        self._stop_event: Optional[asyncio.Event] = None
        self.acceleration_manager: Optional[HardwareAccelerationManager] = None
        
    async def start_detection(self):
        """Start hardware detection service"""
        logger.info("Starting hardware detection service")
        self._stop_event = asyncio.Event()
        
        # Initialize hardware acceleration manager
        self.acceleration_manager = HardwareAccelerationManager()
        await self.acceleration_manager.initialize()
        
        # Initial scan
        await self.scan_hardware()
        
        # Start periodic scanning
        asyncio.create_task(self._periodic_scan())
        
    async def stop_detection(self):
        """Stop hardware detection service"""
        logger.info("Stopping hardware detection service")
        if self._stop_event:
            self._stop_event.set()
            
        # Cleanup acceleration manager
        if self.acceleration_manager:
            self.acceleration_manager.cleanup()
            
    async def _periodic_scan(self):
        """Periodic hardware scanning"""
        while not self._stop_event.is_set():
            try:
                await self.scan_hardware()
                await asyncio.sleep(self.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error during periodic hardware scan", error=str(e))
                await asyncio.sleep(self.scan_interval)  # Wait before retrying
                
    async def scan_hardware(self) -> HardwareStatus:
        """Scan for available hardware devices"""
        if self._scanning:
            logger.warning("Hardware scan already in progress")
            return self.get_status()
            
        self._scanning = True
        logger.info("Starting hardware detection scan")
        
        try:
            devices = []
            
            # Detect CPU
            cpu_device = await self._detect_cpu()
            if cpu_device:
                devices.append(cpu_device)
                
            # Detect Intel GPU
            intel_gpu_devices = await self._detect_intel_gpu()
            devices.extend(intel_gpu_devices)
            
            # Detect Intel NPU
            npu_devices = await self._detect_intel_npu()
            devices.extend(npu_devices)
            
            # Detect NVIDIA GPU
            nvidia_devices = await self._detect_nvidia_gpu()
            devices.extend(nvidia_devices)
            
            # Update device registry
            self.devices = {device.id: device for device in devices}
            
            # Set primary device if not set
            if not self.primary_device and devices:
                self.primary_device = devices[0]  # Default to first available device
                
            self.last_scan = datetime.now()
            
            logger.info(
                "Hardware scan completed",
                devices_found=len(devices),
                primary_device=self.primary_device.name if self.primary_device else None
            )
            
            return self.get_status()
            
        finally:
            self._scanning = False
            
    async def _detect_cpu(self) -> Optional[HardwareDevice]:
        """Detect CPU hardware"""
        try:
            if not HAS_PSUTIL:
                logger.warning("psutil not available for CPU detection")
                return None
                
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            device = HardwareDevice(
                id="cpu_0",
                name=f"CPU ({cpu_count} cores)",
                type="cpu",
                memory_total=memory.total / (1024**3),  # Convert to GB
                memory_used=memory.used / (1024**3),
                utilization=psutil.cpu_percent(interval=1),
                supported=True,
                selected=True,
                last_updated=datetime.now()
            )
            
            logger.debug("CPU detected", device=asdict(device))
            return device
            
        except Exception as e:
            logger.error("CPU detection failed", error=str(e))
            return None
            
    async def _detect_intel_gpu(self) -> List[HardwareDevice]:
        """Detect Intel GPU hardware"""
        devices = []
        
        try:
            if HAS_OPENVINO:
                # Use OpenVINO to detect Intel GPU
                core = ov.Core()
                available_devices = core.available_devices
                
                for i, device in enumerate(available_devices):
                    if "GPU" in device.upper():
                        try:
                            gpu_info = core.get_property(device, "FULL_DEVICE_NAME")
                            device_obj = HardwareDevice(
                                id=f"intel_gpu_{i}",
                                name=f"Intel GPU: {gpu_info}",
                                type="intel_gpu",
                                supported=True,
                                selected=False,
                                driver_version=core.get_version(),
                                last_updated=datetime.now()
                            )
                            devices.append(device_obj)
                            logger.debug("Intel GPU detected", device=asdict(device_obj))
                        except Exception as e:
                            logger.warning(f"Failed to get Intel GPU {device} info", error=str(e))
                            
            elif HAS_PYADL:
                # Fallback to ADL (AMD Display Library) - works for some Intel GPUs
                try:
                    adapters = pyadl.ADLManager.getInstance().getDevices()
                    for i, adapter in enumerate(adapters):
                        device_obj = HardwareDevice(
                            id=f"intel_gpu_{i}",
                            name=f"Intel GPU: {adapter.adapterName}",
                            type="intel_gpu",
                            supported=True,
                            selected=False,
                            last_updated=datetime.now()
                        )
                        devices.append(device_obj)
                        logger.debug("Intel GPU detected via ADL", device=asdict(device_obj))
                except Exception as e:
                    logger.warning("ADL detection failed", error=str(e))
                    
        except Exception as e:
            logger.error("Intel GPU detection failed", error=str(e))
            
        return devices
        
    async def _detect_intel_npu(self) -> List[HardwareDevice]:
        """Detect Intel NPU hardware"""
        devices = []
        
        try:
            if HAS_NPU:
                # Try to detect Intel NPU
                try:
                    compiler = NPUCompiler()
                    device_info = compiler.get_device_info()
                    
                    device_obj = HardwareDevice(
                        id="intel_npu_0",
                        name=f"Intel NPU: {device_info.get('name', 'Unknown')}",
                        type="intel_npu",
                        supported=True,
                        selected=False,
                        compute_capability=device_info.get("compute_capability"),
                        last_updated=datetime.now()
                    )
                    devices.append(device_obj)
                    logger.debug("Intel NPU detected", device=asdict(device_obj))
                    
                except Exception as e:
                    logger.debug("Intel NPU not detected", error=str(e))
            else:
                logger.debug("Intel NPU library not available")
                
        except Exception as e:
            logger.error("Intel NPU detection failed", error=str(e))
            
        return devices
        
    async def _detect_nvidia_gpu(self) -> List[HardwareDevice]:
        """Detect NVIDIA GPU hardware"""
        devices = []
        
        try:
            if HAS_PYNVML:
                # Initialize NVML
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                            
                            # Get compute capability
                            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                            compute_capability = f"{major}.{minor}"
                            
                            device_obj = HardwareDevice(
                                id=f"nvidia_gpu_{i}",
                                name=f"NVIDIA GPU: {name}",
                                type="nvidia_gpu",
                                memory_total=memory_info.total / (1024**3),  # Convert to GB
                                memory_used=memory_info.used / (1024**3),
                                utilization=utilization.gpu,
                                temperature=temperature,
                                supported=True,
                                selected=False,
                                driver_version=driver_version,
                                compute_capability=compute_capability,
                                last_updated=datetime.now()
                            )
                            devices.append(device_obj)
                            logger.debug("NVIDIA GPU detected", device=asdict(device_obj))
                            
                        except Exception as e:
                            logger.warning(f"Failed to get NVIDIA GPU {i} info", error=str(e))
                            
                    pynvml.nvmlShutdown()
                    
                except pynvml.NVMLError as e:
                    logger.debug("NVIDIA GPU not available", error=str(e))
            else:
                logger.debug("NVIDIA ML library not available")
                
        except Exception as e:
            logger.error("NVIDIA GPU detection failed", error=str(e))
            
        return devices
        
    def get_status(self) -> HardwareStatus:
        """Get current hardware status"""
        return HardwareStatus(
            devices=list(self.devices.values()),
            primary_device=self.primary_device,
            acceleration_mode=self.acceleration_mode,
            scan_interval=self.scan_interval,
            last_scan=self.last_scan
        )
        
    def select_primary_device(self, device_id: str) -> bool:
        """Select primary hardware device"""
        if device_id in self.devices:
            self.primary_device = self.devices[device_id]
            logger.info("Primary device selected", device_id=device_id, device_name=self.primary_device.name)
            return True
        return False
        
    def set_acceleration_mode(self, mode: str) -> bool:
        """Set hardware acceleration mode"""
        if mode in ["single", "multi", "hybrid"]:
            self.acceleration_mode = mode
            logger.info("Acceleration mode set", mode=mode)
            return True
        return False
        
    async def load_model_with_acceleration(self, model_path: str, device_preference: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load a model with hardware acceleration"""
        if not self.acceleration_manager:
            raise RuntimeError("Hardware acceleration manager not initialized")
            
        return await self.acceleration_manager.load_model(model_path, device_preference, **kwargs)
        
    async def run_inference_with_acceleration(self, model_info: Dict[str, Any], input_data: Any, **kwargs) -> Any:
        """Run inference with hardware acceleration"""
        if not self.acceleration_manager:
            raise RuntimeError("Hardware acceleration manager not initialized")
            
        return await self.acceleration_manager.infer(model_info, input_data, **kwargs)
        
    def get_acceleration_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all acceleration devices"""
        if not self.acceleration_manager:
            return {"error": "Hardware acceleration manager not initialized"}
            
        return self.acceleration_manager.get_performance_summary()
        
    def benchmark_acceleration_devices(self, model_path: str, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark all available acceleration devices"""
        if not self.acceleration_manager:
            return {"error": "Hardware acceleration manager not initialized"}
            
        return self.acceleration_manager.benchmark_devices(model_path, iterations)
        
    async def get_device_metrics(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time metrics for a specific device"""
        if device_id not in self.devices:
            return None
            
        device = self.devices[device_id]
        
        try:
            if device.type == "cpu":
                return await self._get_cpu_metrics(device)
            elif device.type == "intel_gpu":
                return await self._get_intel_gpu_metrics(device)
            elif device.type == "intel_npu":
                return await self._get_intel_npu_metrics(device)
            elif device.type == "nvidia_gpu":
                return await self._get_nvidia_gpu_metrics(device)
        except Exception as e:
            logger.error(f"Failed to get metrics for {device_id}", error=str(e))
            
        return None
        
    async def _get_cpu_metrics(self, device: HardwareDevice) -> Dict[str, Any]:
        """Get CPU metrics"""
        if not HAS_PSUTIL:
            return {}
            
        return {
            "utilization": psutil.cpu_percent(interval=0.1),
            "memory_used": psutil.virtual_memory().used / (1024**3),
            "memory_total": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _get_intel_gpu_metrics(self, device: HardwareDevice) -> Dict[str, Any]:
        """Get Intel GPU metrics"""
        # This would require Intel-specific libraries
        # For now, return basic info
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    async def _get_intel_npu_metrics(self, device: HardwareDevice) -> Dict[str, Any]:
        """Get Intel NPU metrics"""
        # This would require Intel NPU-specific libraries
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }
        
    async def _get_nvidia_gpu_metrics(self, device: HardwareDevice) -> Dict[str, Any]:
        """Get NVIDIA GPU metrics"""
        if not HAS_PYNVML:
            return {}
            
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device.id.split("_")[-1]))
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            
            pynvml.nvmlShutdown()
            
            return {
                "utilization": utilization.gpu,
                "memory_used": memory_info.used / (1024**3),
                "memory_total": memory_info.total / (1024**3),
                "temperature": temperature,
                "power": power,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get NVIDIA GPU metrics", error=str(e))
            return {}