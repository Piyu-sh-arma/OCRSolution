# support/model_manager.py (NEW FILE)
from support.paddleocr_model import PaddleOCRModel
from support.yolo_model import YoloObjDetectionModel


class ModelManager:
    """Centralized model management - load once, use everywhere"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_yolo_model(self):
        if 'yolo' not in self._models:
            print(" >>> Loading YOLO model...")
            self._models['yolo'] = YoloObjDetectionModel().get_model()
        return self._models['yolo']
    
    def get_ocr_model(self):
        if 'ocr' not in self._models:
            print(" >>> Loading OCR model...")
            self._models['ocr'] = PaddleOCRModel()
        return self._models['ocr']
    
    def warmup_models(self, dummy_frame):
        """Run inference once to warm up models"""
        print(" >>> Warming up models...")
        yolo = self.get_yolo_model()
        ocr = self.get_ocr_model()
        
        # Dummy inference to load weights
        if yolo is not None:
            yolo.predict(
                    source=dummy_frame,
                    conf=0.8,
                    verbose=False,
                    stream=False,
                    device="intel:cpu"
                )
        if ocr is not None:
            ocr.read_text(dummy_frame)
        print("Models warmed up!")