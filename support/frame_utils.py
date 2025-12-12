import os
from pathlib import Path
import cv2
import numpy as np
import config


def save_frame(file_path: str, frame) -> bool:
    
    if config.SAVE_FRAME:
        try:
            # Create directory hierarchy if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)

            # Save the frame as JPG
            success = cv2.imwrite(file_path, frame)
            return success

        except Exception as e:
            print(f"Error saving frame: {e}")
            return False

    return False


def resize_frame(frame, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
    h,w,_ = frame.shape
    if h!=height or w!=width:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    else:
        return frame


def enhance_blacks(img, threshold=110):
    
    # Create mask and apply
    mask = img < threshold
    enhanced_img = img.copy()
    enhanced_img[mask] = 0
    if config.SAVE_FRAME:
        cv2.imwrite("enahanced_img.jpg", enhanced_img)

    return enhanced_img


def black_to_binary(inputImg):
    _, binary_img = cv2.threshold(inputImg, 110, 255, cv2.THRESH_BINARY)
    if config.SAVE_FRAME:
        cv2.imwrite("Black_to_binaryImg.jpg", binary_img)

    return binary_img


def apply_erosion(frame, kval=config.ERODE_KVAL):
    

    kernel = np.ones((kval, kval), np.uint8)
    erosion = cv2.erode(frame, kernel, iterations=1)
    if config.SAVE_FRAME:
        cv2.imwrite("eroded.jpg", erosion)

    return erosion


def apply_blurr(frame, size=3):
    gbframe = cv2.GaussianBlur(frame, (size, size), 1)
    if config.SAVE_FRAME:
        cv2.imwrite(f"gb{size}x{size}.jpg", gbframe)

    return gbframe
