import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config import settings
from src.core.database import init_db
from src.api.routes import router
from src.core.hardware_detector import HardwareDetector
from src.core.model_manager import ModelManager
from src.core.inference_engine import InferenceEngine
from src.services.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_platform.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global services
hardware_detector: Optional[HardwareDetector] = None
model_manager: Optional[ModelManager] = None
inference_engine: Optional[InferenceEngine] = None
websocket_manager: Optional[WebSocketManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global hardware_detector, model_manager, inference_engine, websocket_manager
    
    logger.info("Starting LLM Interaction Platform...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize hardware detector
        hardware_detector = HardwareDetector()
        await hardware_detector.initialize()
        logger.info("Hardware detector initialized")
        
        # Initialize model manager
        model_manager = ModelManager()
        await model_manager.initialize()
        logger.info("Model manager initialized")
        
        # Initialize inference engine
        inference_engine = InferenceEngine(hardware_detector)
        await inference_engine.initialize()
        logger.info("Inference engine initialized")
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager()
        logger.info("WebSocket manager initialized")
        
        logger.info("LLM Interaction Platform started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down LLM Interaction Platform...")
        if inference_engine:
            await inference_engine.cleanup()
        if model_manager:
            await model_manager.cleanup()
        if hardware_detector:
            await hardware_detector.cleanup()
        logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="LLM Interaction Platform",
    description="A comprehensive LLM interaction platform with hardware acceleration support",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM交互平台</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 600px;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                font-size: 2.5em;
            }
            .status {
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                font-weight: 500;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .feature {
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .feature h3 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .feature p {
                margin: 0;
                color: #666;
                font-size: 0.9em;
            }
            .api-links {
                margin-top: 30px;
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .api-links a {
                padding: 10px 20px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            .api-links a:hover {
                background: #5a6fd8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 LLM交互平台</h1>
            <div class="status success">
                ✅ 系统运行正常 - 欢迎使用LLM交互平台！
            </div>
            <p>一个支持硬件加速的多语言大语言模型交互系统</p>
            
            <div class="features">
                <div class="feature">
                    <h3>🚀 多硬件加速</h3>
                    <p>支持CPU、iGPU、NPU、GPU等多种硬件加速方案</p>
                </div>
                <div class="feature">
                    <h3>🌍 多语言界面</h3>
                    <p>中英文双语切换，友好的用户界面</p>
                </div>
                <div class="feature">
                    <h3>📊 模型管理</h3>
                    <p>模型下载、量化、加载、卸载全生命周期管理</p>
                </div>
                <div class="feature">
                    <h3>📈 性能监控</h3>
                    <p>实时监控系统资源和模型运行状态</p>
                </div>
            </div>
            
            <div class="api-links">
                <a href="/docs">📚 API文档</a>
                <a href="/api/hardware/detect">🔧 硬件检测</a>
                <a href="/api/models">📦 模型列表</a>
                <a href="/monitor">📊 监控面板</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "version": "1.0.0",
        "services": {
            "hardware_detector": hardware_detector is not None,
            "model_manager": model_manager is not None,
            "inference_engine": inference_engine is not None,
            "websocket_manager": websocket_manager is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )