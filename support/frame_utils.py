import os
from pathlib import Path
import cv2
import numpy as np
import config


def save_frame(file_path: str, frame) -> bool:
    """
    Save a frame as a JPG image, creating folder hierarchy if it doesn't exist.

    Args:
        file_path: Complete file path where the image should be saved (e.g., '/path/to/images/frame.jpg')
        frame: The frame/image data (numpy array for cv2)

    Returns:
        bool: True if successful, False otherwise
    """
    if config.SAVE_MATCHED_FRAMES:
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
    """
    Resize a frame to specified dimensions.
    Args:
        frame: Input image/frame to be resized.
        width (int): Target width in pixels. Defaults to config.FRAME_WIDTH.
        height (int): Target height in pixels. Defaults to config.FRAME_HEIGHT.
    Returns:
        ndarray: Resized frame with dimensions (width, height).
    Example:
        resized_frame = resize_frame(frame, 640, 480)
    """


    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def enhance_blacks(img, threshold=110, save_img=False):
    """
    Enhance black regions in an image by setting pixels below a threshold to black.
    This function creates a mask to identify all pixels with intensity values below
    the specified threshold and sets them to 0 (black), effectively darkening the
    darker regions of the image.
    Args:
        img (numpy.ndarray): Input image array (grayscale or single channel).
        threshold (int, optional): Intensity threshold value. Pixels with values
            below this threshold will be set to 0. Defaults to 110.
        save_img (bool, optional): If True, saves the enhanced image to disk as
            "enahanced_img.jpg". Defaults to False.
    Returns:
        numpy.ndarray: Enhanced image with darkened regions, same shape and dtype
            as the input image.
    Example:
        >>> import cv2
        >>> img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
        >>> enhanced = enhance_blacks(img, threshold=100, save_img=True)
    """

    # Create mask and apply
    mask = img < threshold
    enhanced_img = img.copy()
    enhanced_img[mask] = 0
    if save_img:
        cv2.imwrite("enahanced_img.jpg", enhanced_img)

    return enhanced_img


def black_to_binary(inputImg):
    _, binary_img = cv2.threshold(inputImg, 110, 255, cv2.THRESH_BINARY)
    cv2.imwrite("Black_to_binaryImg.jpg", binary_img)

    return binary_img


def apply_erosion(frame, kval=config.ERODE_KVAL, save_img=False):
    """
    Apply morphological erosion to an image frame.
    This function performs erosion operation on the input frame using a 
    square kernel of specified size. Erosion is useful for removing small 
    noise or reducing object sizes in binary images.
    Args:
        frame (np.ndarray): Input image frame to apply erosion on.
        kval (int, optional): Size of the erosion kernel (kval x kval). 
            Defaults to config.ERODE_KVAL.
        save_img (bool, optional): If True, saves the eroded image as 
            "eroded.jpg". Defaults to False.
    Returns:
        np.ndarray: Eroded image frame.
    Example:
        >>> eroded_frame = apply_erosion(frame, kval=5, save_img=True)
    """

    kernel = np.ones((kval, kval), np.uint8)
    erosion = cv2.erode(frame, kernel, iterations=1)
    if save_img:
        cv2.imwrite("eroded.jpg", erosion)

    return erosion


def apply_blurr(frame, size=3, save_img=False):
    """
    Apply Gaussian blur to an image frame.
    Args:
        frame (numpy.ndarray): Input image frame to be blurred.
        size (int, optional): Kernel size for the Gaussian blur. 
                            Must be an odd number. Defaults to 3.
        save_img (bool, optional): If True, saves the blurred image to disk 
                                  as 'gbSIZExSIZE.jpg'. Defaults to False.
    Returns:
        numpy.ndarray: The blurred image frame.
    Example:
        >>> blurred_frame = apply_blurr(frame, size=5, save_img=True)
    """

    gbframe = cv2.GaussianBlur(frame, (size, size), 1)
    if save_img:
        cv2.imwrite(f"gb{size}x{size}.jpg", gbframe)

    return gbframe
