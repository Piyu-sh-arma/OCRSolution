from paddleocr import PaddleOCR  

class PaddleOCRModel:
    """Singleton class for PaddleOCR text detection."""
    
    _instance = None
    _ocr = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device="cpu"):
        if not self._initialized:
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
                device=device,
            )
            self.__class__._initialized = True
    
    def read_text(self, frame, verbose=False):
        """
        Read text from image using PaddleOCR.
        
        Args:
            frame: Input frame
            verbose: Whether to print OCR results
            
        Returns:
            List of detected text strings
        """
        if self._ocr is None:
            raise RuntimeError("PaddleOCR model not initialized. Call __init__ first.")
        result = self._ocr.predict(input=frame)
        
        if result:
            detected_texts = result[0]['rec_texts']
            if verbose:
                print("OCR Results:", detected_texts)
            return detected_texts
        else:
            if verbose:
                print("No text detected.")
            return []