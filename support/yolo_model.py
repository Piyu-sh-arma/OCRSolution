from ultralytics import YOLO

import config
class YoloObjDetectionModel:
    """Singleton class for PaddleOCR text detection."""
    
    _instance = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._model = YOLO(config.YOLO_MODEL_PATH)
            self.__class__._initialized = True
    
    def get_model(self):
        return self._model  