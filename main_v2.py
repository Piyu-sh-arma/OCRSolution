import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
from support.frame_selector import FrameSelector
from support.frame_utils import resize_frame, save_frame
from support.model_manager import ModelManager
from support.paddleocr_model import PaddleOCRModel
from support.yolo_model import YoloObjDetectionModel
from text_recognition_rnd import detect_and_verify
from timing_decorator import timing_decorator

def scan_frame_for_text_optimized(frame_data, expected_text=[], save_to_disk=False):
    """
    Optimized version - avoids redundant operations
    
    Args:
        frame_data: Tuple of (frame_num, frame) - frame should already be resized
        expected_text: List of expected text strings
        save_to_disk: Whether to save frame to disk (optional)
    """
    frame_num, frame = frame_data
    
    # Frame is already resized, no need to resize again
    # Only save if explicitly requested
    if save_to_disk:
        file_path = f"{config.OUTPUT_DIR}/{frame_num}.jpg"
        save_frame(file_path, frame)
    
    return detect_and_verify(frame, expected_text)


def process_detection_group(group_data):
    """
    Process a single detection group - can be run in parallel
    
    Args:
        group_data: Tuple of (group_id, detected_frames, expected_text, save_frames)
    
    Returns:
        Tuple of (frame_num, result)
    """
    group_id, detected_frames, expected_text, save_frames = group_data
    
    # Select best frame from group
    frame_selector = FrameSelector()
    selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
    frame_num = selected_frame_data[0]
    
    print(f">>>>> Group {group_id}: Selected frame {frame_num} from {len(detected_frames)} detections")
    
    # Process the selected frame
    result = scan_frame_for_text_optimized(
        selected_frame_data, 
        expected_text=expected_text,
        save_to_disk=save_frames
    )
    
    # Clear memory
    detected_frames.clear()
    
    return (frame_num, result)


@timing_decorator
def process_video_optimized(
    video_path="",
    expected_text=[],
    conf_threshold=config.CONFIDENCE_THRESHOLD,
    frame_delay_ms=config.FRAME_DELAY_MS,
    show_frames=True,
    save_frames=True,
    parallel_processing=False,
    max_workers=4,
):
    """
    Optimized video processing with optional parallel OCR processing
    
    Args:
        video_path: Path to video file
        expected_text: List of expected text strings to verify
        conf_threshold: YOLO confidence threshold
        frame_delay_ms: Delay between frames when showing
        show_frames: Whether to display frames during processing
        save_frames: Whether to save processed frames to disk
        parallel_processing: Whether to process OCR in parallel (True) or sequential (False)
        max_workers: Number of parallel workers for OCR processing
    """
    checked_frames_count = 0
    failed_frames_count = 0

    # Get pre-loaded model from ModelManager
    model_mgr = ModelManager()
    model = model_mgr.get_yolo_model()

    if model is None:
        raise ValueError("Failed to load YOLO model")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> Video properties: FPS={fps}, Total frames={total_frames}")

    # Setup display window if needed
    if show_frames:
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    frame_counter = 0
    detected_frames = []
    all_detection_groups = []  # Store all groups for optional parallel processing
    
    frame_selector = FrameSelector()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            
            # Run YOLO detection on original frame
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
                stream=False,
                device="intel:cpu"
            )

            # Process detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Extract ROI from original frame
                x, y, w, h = boxes.xywh[0].int().tolist()

                # Extract ROI with bounds checking
                y1 = max(0, y - h // 2)
                y2 = min(frame.shape[0], y + h // 2)
                x1 = max(0, x - w // 2)
                x2 = min(frame.shape[1], x + w // 2)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Resize once and store
                    resized_roi = resize_frame(roi)
                    detected_frames.append((frame_counter, resized_roi))
                    
                    # Only save detection frames if explicitly requested
                    if save_frames:
                        save_frame(f"detected_frames/{frame_counter}.jpg", resized_roi)
            else:
                # Process accumulated frames when detection ends
                if detected_frames:
                    if parallel_processing:
                        # Store for parallel processing later
                        all_detection_groups.append(detected_frames.copy())
                        detected_frames = []
                    else:
                        # Sequential processing (like original)
                        selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
                        print(f">>>>> Selected frame {selected_frame_data[0]} for detection")
                        checked_frames_count += 1
                        
                        if not scan_frame_for_text_optimized(
                            selected_frame_data, 
                            expected_text=expected_text,
                            save_to_disk=save_frames
                        ):
                            failed_frames_count += 1
                        
                        detected_frames = []

            # Display frames if enabled
            if show_frames:
                annotated_frame = results[0].plot()
                cv2.imshow(config.WINDOW_NAME, annotated_frame)

                if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                    break

    finally:
        # Process any remaining frames
        if detected_frames:
            if parallel_processing:
                all_detection_groups.append(detected_frames.copy())
            else:
                selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
                print(f">>>>> Selected frame {selected_frame_data[0]} for detection")
                checked_frames_count += 1
                if not scan_frame_for_text_optimized(
                    selected_frame_data, 
                    expected_text=expected_text,
                    save_to_disk=save_frames
                ):
                    failed_frames_count += 1
        
        # Cleanup
        cap.release()
        if show_frames:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"Frame detection complete.")
    print(f"  Total frames scanned: {frame_counter}")
    print(f"  Detection groups found: {len(all_detection_groups) if parallel_processing else checked_frames_count}")
    print(f"{'='*60}")

    # Parallel OCR processing if enabled
    if parallel_processing and all_detection_groups:
        print(f"\n>>> Starting parallel OCR processing with {max_workers} workers...")
        
        # Prepare data for parallel processing
        group_data_list = [
            (idx + 1, group, expected_text, save_frames) 
            for idx, group in enumerate(all_detection_groups)
        ]
        
        # Process groups in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_detection_group, group_data): group_data[0]
                for group_data in group_data_list
            }
            
            for future in as_completed(futures):
                group_id = futures[future]
                try:
                    frame_num, result = future.result()
                    results.append((frame_num, result))
                    checked_frames_count += 1
                    if not result:
                        failed_frames_count += 1
                    
                    status = "✓ SUCCESS" if result else "✗ FAILED"
                    print(f"  {status} - Frame {frame_num} (Group {group_id})")
                    
                except Exception as exc:
                    print(f"  ✗ Group {group_id} generated an exception: {exc}")
                    failed_frames_count += 1
        
        # Sort results by frame number
        results.sort(key=lambda x: x[0])
        
        print(f"\n{'='*60}")
        print(f"Parallel OCR processing complete.")
        print(f"  Processed groups: {len(results)}")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"  Total frames: {frame_counter}")
    print(f"  Checked frames: {checked_frames_count}")
    print(f"  Failed frames: {failed_frames_count}")
    print(f"  Success rate: {(checked_frames_count - failed_frames_count) / checked_frames_count * 100:.1f}%" if checked_frames_count > 0 else "  Success rate: N/A")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("OPTIMIZED VIDEO PROCESSING")
    print("="*60 + "\n")
    
    # Initialize and warm up models once
    model_mgr = ModelManager()
    # Create a dummy frame for warmup
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    model_mgr.warmup_models(dummy_frame)
    
    print("\n" + "="*60)
    print("Starting video processing...")
    print("="*60 + "\n")
    
    # OPTION 1: Sequential processing (original behavior, optimized)
    print(">>> Mode: Sequential processing (optimized)")
    process_video_optimized(
        video_path="D:/TestVideos/Videos/temp/8.mp4",
        expected_text=["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"],
        conf_threshold=0.8,
        frame_delay_ms=1,
        show_frames=True,
        save_frames=True,
        parallel_processing=False  # Sequential like original
    )
    
    # OPTION 2: Parallel OCR processing (faster)
    # Uncomment below to try parallel processing
    # print("\n" + "="*60)
    # print(">>> Mode: Parallel OCR processing")
    # process_video_optimized(
    #     video_path="D:/TestVideos/Videos/temp/8.mp4",
    #     expected_text=["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"],
    #     conf_threshold=0.8,
    #     frame_delay_ms=1,
    #     show_frames=True,
    #     save_frames=True,
    #     parallel_processing=True,  # Enable parallel OCR
    #     max_workers=4  # Number of parallel workers
    # )


if __name__ == "__main__":
    main()