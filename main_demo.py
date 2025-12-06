import cv2
import config
from support.frame_selector import FrameSelector
from support.frame_utils import resize_frame, save_frame
from support.paddleocr_model import PaddleOCRModel
from support.yolo_model import YoloObjDetectionModel
from text_recognition_rnd import detect_and_verify


def scan_frame_for_text(frame_data, expected_text=[]):
    frame_num, frame = frame_data
    frame = resize_frame(frame)
    file_path = f"{config.OUTPUT_DIR}/{frame_num}.jpg"
    save_frame(file_path, frame)
    return detect_and_verify(frame, expected_text)
    


def process_video(
    video_path="",
    expected_text=[],
    conf_threshold=config.CONFIDENCE_THRESHOLD,
    frame_delay_ms=config.FRAME_DELAY_MS,
    show_frames=True,
):
    checked_frames_count = 0
    failed_frames_count = 0

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
                    # print(f"Selected 1 frames from {len(detected_frames)} detections.")
                    checked_frames_count += 1
                    if not scan_frame_for_text(selected_frame_data, expected_text=expected_text):
                        failed_frames_count += 1
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
            checked_frames_count += 1
            if not scan_frame_for_text(selected_frame_data, expected_text=expected_text):
                failed_frames_count += 1

    finally:
        # Cleanup
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
        frame_delay_ms=1,
        show_frames=True,
    )

if __name__ == "__main__":
    # cProfile.run('main()')
    main()
 