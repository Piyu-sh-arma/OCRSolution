
import config
from support import frame_utils
from support.compare_utils import normalize_text, strip_matched_fragments
from support.paddleocr_model import PaddleOCRModel




def detect_and_verify(frame,expected_text=[], save_img=False):

    eroded_frame = frame_utils.apply_erosion(frame,kval=2)
    
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
            blurred = frame_utils.apply_blurr(resized, ksize)
            
            #Run text detection on frame
            detected_list = PaddleOCRModel().read_text(blurred)
            
            detected_norm_list = [normalize_text(detected_str) for detected_str in detected_list]

            remaining_norm_list = strip_matched_fragments(remaining_norm_list, detected_norm_list)
            
            # print(f"img-size {f_width}x{f_height} ksize {ksize}x{ksize}: d_Normal: {detected_norm_list}, R_Normal: {remaining_norm_list}")

            if not remaining_norm_list:
                print(f" >>>  Full Detection succeeded at {ksize}x{ksize}")
                return True

    print(f">>> Detection failed. Remaining text: {remaining_norm_list!r}")
    return False