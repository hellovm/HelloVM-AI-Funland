"""
Custom exceptions for the HelloVM AI Funland backend
"""


class HelloVMException(Exception):
    """Base exception for HelloVM AI Funland"""
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)


class HardwareDetectionError(HelloVMException):
    """Hardware detection failed"""
    pass


class ModelNotFoundError(HelloVMException):
    """Model not found"""
    pass


class ModelDownloadError(HelloVMException):
    """Model download failed"""
    pass


class ModelscopeAPIError(HelloVMException):
    """Modelscope API error"""
    pass


class ChatServiceError(HelloVMException):
    """Chat service error"""
    pass


class WebSocketError(HelloVMException):
    """WebSocket communication error"""
    pass


class PluginError(HelloVMException):
    """Plugin system error"""
    pass


class ConfigurationError(HelloVMException):
    """Configuration error"""
    pass


class ValidationError(HelloVMException):
    """Input validation error"""
    pass