import cv2
from pathlib import Path

import numpy as np
import config
from support import frame_utils
from support.compare_utils import normalize_text, strip_matched_fragments
from support.easyocr_model import EasyOCRModel
from support.paddleocr_model import PaddleOCRModel




def detect_and_verify(frame,expected_text=[], save_img=False):

    eroded_frame = frame_utils.apply_erosion(frame,kval=2,save_img=save_img)
    
    f_width = config.FRAME_WIDTH
    f_height_options = config.FRAME_HEIGHT_OPTIONS  
    kernel_sizes = config.GB_BlURR_KSIZES

    expected_norm_list = [
        normalize_text(expected_str) for expected_str in expected_text
    ]
    remaining_norm_list = expected_norm_list.copy()
    for f_height in f_height_options:
        resized = frame_utils.resize_frame(eroded_frame,f_width,f_height)
        for ksize in kernel_sizes:
            blurred = frame_utils.apply_blurr(resized, ksize, save_img=True)
            if config.OCR_MODEL_TYPE == "paddleOCR":    
                detected_list = PaddleOCRModel().read_text(blurred)
            elif config.OCR_MODEL_TYPE == "easyOCR":
               detected_list = EasyOCRModel().read_text(blurred)
            else:
                raise RuntimeError("Unsupported OCR model type: {}".format(config.OCR_MODEL_TYPE))

            detected_norm_list = [normalize_text(detected_str) for detected_str in detected_list]
            # print(f"img-size {img_height}x{img_width} at {size}x{size}: d_Actual: {detected_list} d_Normal: {detected_norm_list} R_Normal: {remaining_norm_list}")

            print(f"img-size {f_width}x{f_height} at {ksize}x{ksize}: d_Normal: {detected_norm_list}")
            remaining_norm_list = strip_matched_fragments(remaining_norm_list, detected_norm_list)

            if not remaining_norm_list:
                print(f" >>>  Full Detection succeeded at {ksize}x{ksize}")
                return True

    print(f">>> Detection failed. Remaining text: {remaining_norm_list!r}")
    return False


if __name__ == "__main__":
    save_img = False
    # path = Path('2.jpg')
    # path = Path('C:/Users/piyu_/Desktop/VIDEO/frames/Cropped/Filtered/cropped')
    passcount = 0
    failcount = 0
    path = Path("D:/yoloData - Copy/picked_frames_V1")
    if path.is_dir():
        jpg_paths = path.glob("*.jpg")
        for jpg_path in jpg_paths:
            print(f"Processing image: {jpg_path.name}")
            if detect_and_verify(cv2.imread(str(jpg_path)), save_img):
                passcount += 1
            else:
                failcount += 1
    elif path.is_file():
        detect_and_verify(cv2.imread(str(path)), save_img)
    else:
        print("It's neither (doesn't exist, symlink to nowhere, socket, etc.)")
    print(f"Total Passed: {passcount}, Total Failed: {failcount}")
