
############################
# Open CV
############################

WINDOW_NAME = "Frames"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 400
#Frame processing settings
FRAME_DELAY_MS = 1  # Delay between frames in milliseconds

############################
# Yolo model
############################

YOLO_MODEL_PATH = "models/y11n_int8_openvino_model"
#Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.85

#output directory for detected frames
OUTPUT_DIR = "yolo_matches"

#matched frames are saved in OUTPUT_DIR if True
SAVE_FRAME = False


############################
# OCR model
############################



OCR_MODEL_TYPE = "paddleOCR"  # Options: "easyOCR", "paddleOCR"
OCR_LANGUAGES = "en"  # Languages for OCR model
OCR_USE_GPU = False  # Whether to use GPU for OCR processing    

############################
# Frame selection
############################

# Number of frames to analyze for selection
FRAME_SELECTION_COUNT = 5
# Delay between frames to select (in frames)
FRAME_SELECTION_INTERVAL = 3


############################
# Frame formatting
############################
FRAME_BORDER_SIZE = 1  # Border size around frames

FRAME_WIDTH = 600
FRAME_HEIGHT = 60
FRAME_HEIGHT_OPTIONS=[60,50,70,55,83]

## Other options for frame heights
# FRAME_WIDTH = 1000
# FRAME_HEIGHT_OPTIONS = [90,100,110,120,135]

GB_BlURR_KSIZES = [3,7]  # Gaussian blur size for preprocessing
ERODE_KVAL = 2  # Erosion kernel size for preprocessing