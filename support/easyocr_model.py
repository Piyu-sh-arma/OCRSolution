# import cv2
# import easyocr

# class EasyOCRModel:
#     """Singleton class for EasyOCR text detection."""
    
#     _instance = None
#     _reader = None
#     _initialized = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def __init__(self, lang=['en'], gpu=True):
#         if not self._initialized:
#             self._reader = easyocr.Reader(lang, gpu=gpu)
#             self.__class__._initialized = True
    
#     def read_text(self, input_img, expected_str=None, text_threshold=0.8):
#         """
#         Read text from image using EasyOCR.
        
#         Args:
#             input_img: Input image (BGR or grayscale)
#             expected_str: Allowlist of expected characters (optional)
#             text_threshold: Confidence threshold for text detection
            
#         Returns:
#             List of detected text strings
#         """
#         detected_list = []
        
#         # Convert to RGB (EasyOCR expects RGB, OpenCV gives BGR)
#         if len(input_img.shape) == 3 and input_img.shape[2] == 3:
#             image_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#         else:
#             image_rgb = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
        
#         # Perform OCR
#         kwargs = {
#             'detail': 1,
#             'paragraph': False,
#             'text_threshold': text_threshold
#         }
#         if expected_str:
#             kwargs['allowlist'] = expected_str
            
#         result = self._reader.readtext(image_rgb, **kwargs)
        
#         for (bbox, text, conf) in result:
#             detected_list.append(text.strip())
        
#         return detected_list