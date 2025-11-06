# hellovm-localAI-NPU
This is a local AI platform supports Intel Ultra series NPU, with loads of AI models, and fuctions of chat, text2image, text2video and so on
# 本地AI问答平台系统

一个功能完善的本地AI问答平台，支持NPU和GPU硬件加速，提供友好的用户界面和强大的模型管理功能。

## 功能特点

- 🚀 硬件加速：支持Ultra系列CPU自带NPU和GPU的协同加速
- 🎨 响应式UI：直观友好的用户界面，支持多种设备
- 🤖 模型管理：多模型集成管理，支持动态切换
- 💬 智能问答：基于上下文的智能对话功能
- ⚡ 模型量化：支持模型NPU量化，提升推理性能
- 📦 绿色部署：一键打包，无需额外环境配置

## 系统要求

- Python 3.13+
- OpenVINO 2025.3+
- Intel NPU Acceleration Library
- 支持NPU的Intel CPU (Ultra系列)

## 快速开始

1. 克隆项目
2. 安装依赖：`pip install -r requirements.txt`
3. 运行应用：`python main.py`

## 项目结构

```
ai_qa_platform/
├── app/                    # 应用主目录
│   ├── core/              # 核心功能模块
│   ├── hardware/          # 硬件加速模块
│   ├── models/            # 模型管理模块
│   ├── ui/                # 用户界面模块
│   └── utils/             # 工具函数
├── assets/                # 静态资源
├── config/                # 配置文件
├── docs/                  # 文档
├── models/                # 模型文件存储
├── requirements.txt       # 依赖列表
└── main.py               # 应用入口
```

## 技术文档

详细的技术文档请参考 [docs/](docs/) 目录。

## 许可证

MIT License
