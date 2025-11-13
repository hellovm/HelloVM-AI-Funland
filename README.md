# LLM Interaction Platform

A comprehensive LLM interaction platform with hardware acceleration support, multilingual interface, and advanced model management capabilities.

## Features

- 🚀 **Multi-Hardware Acceleration**: Support for CPU, iGPU, NPU, and GPU acceleration
- 🌍 **Multilingual Interface**: Chinese and English language support
- 🤖 **Model Management**: Download, quantize, and manage LLM models
- 📊 **System Monitoring**: Real-time hardware utilization and performance metrics
- 🔌 **Plugin Architecture**: Extensible plugin system for future enhancements
- ⚡ **High Performance**: Optimized inference with hardware-specific backends

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web UI  │    │  Electron App   │    │  Python Backend │
│  (Chinese/EN)   │◄──►│   (Desktop)     │◄──►│  (FastAPI)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Hardware      │    │   Model         │
                       │   Detection     │    │   Management    │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+ (Embedded environment provided)
- Node.js 18+ (for development)
- 8GB+ RAM recommended
- Supported hardware: Intel CPU/GPU, NVIDIA GPU, Intel NPU

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-interaction-platform
   ```

2. **Set up Python environment**
   ```bash
   # Use the provided embedded Python
   python/python.exe -m pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   python/python.exe main.py
   ```

### Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
python main.py --dev
```

## Hardware Support

### CPU Acceleration
- Intel AVX-512 optimization
- Multi-threading support
- Memory-mapped model loading

### GPU Acceleration
- NVIDIA CUDA support
- Intel Arc GPU support
- OpenVINO optimization

### NPU Acceleration
- Intel Ultra NPU support
- Dedicated AI inference engine
- Ultra-low power consumption

## Model Management

### Supported Formats
- GGUF (llama.cpp)
- ONNX (OpenVINO)
- PyTorch (transformers)
- SafeTensors

### Download Sources
- ModelScope (Chinese models)
- Hugging Face Hub
- Local model files

### Quantization Options
- 4-bit integer (INT4)
- 8-bit integer (INT8)
- 16-bit floating point (FP16)
- Dynamic quantization

## API Documentation

### Hardware Detection
```http
GET /api/hardware/detect
```

### Model Management
```http
GET    /api/models
POST   /api/models/download
DELETE /api/models/{id}
```

### Inference
```http
POST /api/inference/chat
```

## Configuration

### Environment Variables
```bash
# Hardware configuration
HARDWARE_PREFERENCE=auto  # auto, cpu, gpu, npu
MAX_MEMORY_USAGE=0.8     # 80% of available memory

# Model configuration
DEFAULT_MODEL=qwen-7b-chat
MODEL_CACHE_DIR=./models

# Server configuration
API_HOST=localhost
API_PORT=8000
WEB_UI_PORT=3000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide