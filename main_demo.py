import cv2
import concurrent.futures

import config
from support.frame_selector import FrameSelector
from support.frame_utils import resize_frame, save_frame
from support.paddleocr_model import PaddleOCRModel
from support.yolo_model import YoloObjDetectionModel
from text_recognition_rnd import detect_and_verify
from timing_decorator import timing_decorator   


def scan_frame_for_text(frame_data, expected_text=[]):
    frame_num, frame = frame_data
    frame = resize_frame(frame)
    file_path = f"{config.OUTPUT_DIR}/{frame_num}.jpg"
    save_frame(file_path, frame)
    return detect_and_verify(frame, expected_text)
    

@timing_decorator
def process_video(
    video_path="",
    expected_text=[],
    conf_threshold=config.CONFIDENCE_THRESHOLD,
    frame_delay_ms=config.FRAME_DELAY_MS,
    show_frames=True,
):
    checked_frames_count = 0
    failed_frames_count = 0
    
    # Initialize executor for non-blocking OCR
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    ocr_futures = []
    # cv2.setNumThreads(10)
    # print(f">>> Number of threads: {cv2.getNumThreads()}")
    # Load model once
    model = YoloObjDetectionModel().get_model()

    if model is None:
        raise ValueError("Failed to load YOLO model")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get the FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f">>> Frames per second (FPS): {fps}")

    # WINDOW_NORMAL flag is required to allow resizing
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    frame_counter = 0
    detected_frames = []

    frame_selector = FrameSelector()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            
            # Check for completed OCR tasks
            # We iterate over a copy or use a list comprehension to filter
            done_futures = [f for f in ocr_futures if f.done()]
            for f in done_futures:
                ocr_futures.remove(f)
                try:
                    success = f.result()
                    checked_frames_count += 1
                    if not success:
                        failed_frames_count += 1
                except Exception as e:
                    print(f"Error in OCR task: {e}")
                    checked_frames_count += 1 # Count even if errored? Or just log
                    failed_frames_count += 1

            # Resize frame for faster processing
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
                stream=False,
            )

            # Process detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Extract ROI from original frame (better quality)
                x, y, w, h = boxes.xywh[0].int().tolist()

                # Extract ROI with bounds checking
                y1 = max(0, y - h // 2)
                y2 = min(frame.shape[0], y + h // 2)
                x1 = max(0, x - w // 2)
                x2 = min(frame.shape[1], x + w // 2)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:  # Ensure valid ROIq
                    resized_roi = resize_frame(roi)
                    detected_frames.append((frame_counter, resized_roi))
                    save_frame(f"detected_frames/{frame_counter}.jpg", resized_roi)
            else:
                # Process accumulated frames when detection ends
                if detected_frames:
                    selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
                    # Submit OCR task to background thread
                    # print(f"Selected 1 frames from {len(detected_frames)} detections.")
                    future = executor.submit(scan_frame_for_text, selected_frame_data, expected_text=expected_text)
                    ocr_futures.append(future)
                    detected_frames = []

            # Display frames if enabled
            if show_frames:
                annotated_frame = results[0].plot()
                cv2.imshow(config.WINDOW_NAME, annotated_frame)

                if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                    break
        # Process any remaining frames
        if detected_frames:
            selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
            # print(f"Selected 1 frames from {len(detected_frames)} detections.")
            future = executor.submit(scan_frame_for_text, selected_frame_data, expected_text=expected_text)
            ocr_futures.append(future)

        # Wait for all remaining futures to complete
        for f in concurrent.futures.as_completed(ocr_futures):
            try:
                success = f.result()
                checked_frames_count += 1
                if not success:
                    failed_frames_count += 1
            except Exception as e:
                print(f"Error in OCR task: {e}")
                failed_frames_count += 1

    finally:
        # Cleanup
        executor.shutdown(wait=True)
        cap.release()
        if show_frames:
            cv2.destroyAllWindows()

    print(f"Processing complete.\n Total frames: {frame_counter}")
    print(f" Checked frames: {checked_frames_count}\n Failed frames: {failed_frames_count}")




def main():

    # expected_str_list = ["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "330"]
    # expected_str_list = ["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"]
    # expected_str_list = ["70g+5g*", "Rs.0.14/g", "MFG.","11/25", "5330B095J3", "1355"]

    # initialize ocr model first
    print(">>> Initializing PaddleOCR model...")
    PaddleOCRModel()
    print(">>> PaddleOCR model initialized.")

    print(">>> Initializing Yolo Object detection model...")
    model = YoloObjDetectionModel().get_model()
    print(">>> PaddleOCR Yolo Object detection model initialized.")

    print(">>> Starting video processing...")
    process_video(
        video_path="D:/TestVideos/Videos/temp/8.mp4",
        expected_text=["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"],
        conf_threshold=0.8,
        show_frames=True,
    )

if __name__ == "__main__":
    # cProfile.run('main()')
    main()
 