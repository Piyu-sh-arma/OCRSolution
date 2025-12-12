import threading
from queue import Queue
import cv2

import config
from support.frame_selector import FrameSelector
from support.frame_utils import resize_frame, save_frame
from support.yolo_model import YoloObjDetectionModel
from text_recognition_rnd import detect_and_verify


class ContinuousPipelineProcessor:
    def __init__(self, queue_size=50, gap_threshold=3, max_group_size=5):
        """
        Queue continuously receives detected frames from producer
        Consumer processes them as they arrive
        
        Args:
            queue_size: Maximum number of frames in queue
            gap_threshold: Number of consecutive frames without detection before finalizing a group
            max_group_size: Maximum number of consecutive detections before forcing finalization
        """
        self.frame_queue = Queue(maxsize=queue_size)
        self.results = []
        self.gap_threshold = gap_threshold
        self.max_group_size = max_group_size
        
    def detect_and_find_best_frames(
        self,
        video_path,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        frame_delay_ms=config.FRAME_DELAY_MS,
        show_frames=True,
    ):
        """
        PRODUCER: Continuously detects frames and pushes to queue
        Runs in separate thread, keeps feeding frames to consumer
        """
        model = YoloObjDetectionModel().get_model()
        if model is None:
            raise ValueError("Failed to load YOLO model")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f">>> Video FPS: {fps}, Total frames: {total_frames}")

        if show_frames:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        frame_counter = 0
        detected_frames = []
        frames_without_detection = 0
        frame_selector = FrameSelector()
        groups_pushed = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f">>> End of video reached at frame {frame_counter}")
                    break

                frame_counter += 1
                
                # Run YOLO detection
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False,
                    stream=False,
                    device="intel:cpu"
                )

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Reset gap counter when detection occurs
                    frames_without_detection = 0
                    
                    # Extract ROI
                    x, y, w, h = boxes.xywh[0].int().tolist()
                    y1 = max(0, y - h // 2)
                    y2 = min(frame.shape[0], y + h // 2)
                    x1 = max(0, x - w // 2)
                    x2 = min(frame.shape[1], x + w // 2)

                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        resized_roi = resize_frame(roi)
                        detected_frames.append((frame_counter, resized_roi))
                        save_frame(f"detected_frames/{frame_counter}.jpg", resized_roi)
                        
                        # Check if we've reached max group size - force finalization
                        if len(detected_frames) >= self.max_group_size:
                            groups_pushed += 1
                            print(f"[Producer] Group #{groups_pushed}: Max size reached! Selecting best from {len(detected_frames)} detections (frames {detected_frames[0][0]}-{detected_frames[-1][0]})...")
                            selected_frame_data = frame_selector.find_with_best_contrast(
                                detected_frames
                            )
                            print(f"[Producer] ✓ Pushing frame {selected_frame_data[0]} to queue (group #{groups_pushed})")
                            self.frame_queue.put(selected_frame_data)
                            
                            # Reset for next detection sequence
                            detected_frames = []
                            frames_without_detection = 0
                else:
                    # Increment gap counter when no detection
                    if detected_frames:  # Only count gap if we have detections
                        frames_without_detection += 1
                    
                    # Only finalize frame group if gap threshold is reached
                    if detected_frames and frames_without_detection >= self.gap_threshold:
                        groups_pushed += 1
                        print(f"[Producer] Group #{groups_pushed}: Selecting best from {len(detected_frames)} detections (frames {detected_frames[0][0]}-{detected_frames[-1][0]})...")
                        selected_frame_data = frame_selector.find_with_best_contrast(
                            detected_frames
                        )
                        
                        # PUSH TO QUEUE - Consumer will process it
                        print(f"[Producer] ✓ Pushing frame {selected_frame_data[0]} to queue (group #{groups_pushed})")
                        self.frame_queue.put(selected_frame_data)
                        
                        # Reset for next detection sequence
                        detected_frames = []
                        frames_without_detection = 0

                # Display frames
                if show_frames:
                    annotated_frame = results[0].plot()
                    cv2.imshow(config.WINDOW_NAME, annotated_frame)
                    if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                        print(">>> User pressed 'q' - stopping video processing")
                        break

        finally:
            # Process any remaining frames (end of video or interrupted)
            if detected_frames:
                groups_pushed += 1
                print(f"[Producer] Final Group #{groups_pushed}: Selecting best from {len(detected_frames)} detections (frames {detected_frames[0][0]}-{detected_frames[-1][0]})...")
                selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
                print(f"[Producer] ✓ Pushing final frame {selected_frame_data[0]} to queue (group #{groups_pushed})")
                self.frame_queue.put(selected_frame_data)
                
            # Signal completion to consumer
            self.frame_queue.put(None)
            print(f"[Producer] ✓ Finished - Pushed {groups_pushed} frame groups to queue")
            print(f"[Producer] ✓ Sent completion signal to consumer")
            
            cap.release()
            if show_frames:
                cv2.destroyAllWindows()

        print(f"[Producer] Video processing complete. Total frames scanned: {frame_counter}")

    def find_text_in_best_frame(self, expected_text=[]):
        """
        CONSUMER: Continuously checks queue and processes incoming frames
        Runs in separate thread, processes frames as they arrive
        """
        processed_count = 0
        failed_count = 0
        
        print("[Consumer] ✓ Started - waiting for frames from queue...")
        
        while True:
            # BLOCKING CALL - waits until frame is available
            frame_data = self.frame_queue.get()
            
            # Check for completion signal
            if frame_data is None:
                print("[Consumer] ✓ Received completion signal - stopping")
                break
            
            frame_num, frame = frame_data
            print(f"[Consumer] → Processing frame {frame_num} for text extraction...")
            
            # Process the frame
            frame = resize_frame(frame)
            file_path = f"{config.OUTPUT_DIR}/{frame_num}.jpg"
            save_frame(file_path, frame)
            
            # Extract text
            result = detect_and_verify(frame, expected_text)
            self.results.append((frame_num, result))
            
            processed_count += 1
            if not result:
                failed_count += 1
                
            status = "✓ SUCCESS" if result else "✗ FAILED"
            print(f"[Consumer] {status} - Frame {frame_num} complete (processed: {processed_count}, failed: {failed_count})")
        
        print(f"\n[Consumer] Finished:")
        print(f"  Processed: {processed_count} frames")
        print(f"  Failed: {failed_count} frames")
        print(f"  Success rate: {processed_count - failed_count}/{processed_count}")

    def pipeline_process(self, video_path="", expected_text=[], conf_threshold=config.CONFIDENCE_THRESHOLD, show_frames=True):
        """
        Start both producer and consumer threads
        They run concurrently:
        - Producer: Detects frames → pushes to queue
        - Consumer: Pulls from queue → extracts text
        """
        print("=" * 60)
        print("Starting continuous pipeline processing...")
        print(f"Video: {video_path}")
        print(f"Gap threshold: {self.gap_threshold} frames")
        print(f"Max group size: {self.max_group_size} detections")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Expected text: {expected_text}")
        print("=" * 60)
        
        # Stage 1: Producer thread
        producer = threading.Thread(
            target=self.detect_and_find_best_frames,
            args=(video_path, conf_threshold, config.FRAME_DELAY_MS, show_frames),
            name="FrameDetector"
        )
        
        # Stage 2: Consumer thread
        consumer = threading.Thread(
            target=self.find_text_in_best_frame,
            args=(expected_text,),
            name="TextExtractor"
        )
        
        # Start both threads
        print("✓ Starting producer and consumer threads...")
        producer.start()
        consumer.start()
        print("✓ Both threads running in parallel\n")
        
        # Wait for both to complete
        producer.join()
        print("\n✓ Producer thread completed")
        
        consumer.join()
        print("✓ Consumer thread completed")
        
        print("\n" + "=" * 60)
        print(f"Pipeline complete!")
        print(f"  Total frames processed: {len(self.results)}")
        success_count = sum(1 for _, result in self.results if result)
        print(f"  Success: {success_count}/{len(self.results)}")
        if len(self.results) > 0:
            print(f"  Success rate: {success_count/len(self.results)*100:.1f}%")
        print("=" * 60)
        
        return self.results


# ALTERNATIVE: Process multiple videos continuously
class MultiVideoPipelineProcessor(ContinuousPipelineProcessor):
    """
    Extended version for processing multiple videos
    Producer keeps pushing frames from all videos
    Consumer processes them all continuously
    """
    
    def process_multiple_videos(self, video_paths, expected_texts_list, conf_threshold=config.CONFIDENCE_THRESHOLD):
        """
        Process multiple videos through the pipeline
        
        Args:
            video_paths: List of video file paths
            expected_texts_list: List of expected text strings to verify
            conf_threshold: YOLO confidence threshold
        """
        def producer():
            for idx, video_path in enumerate(video_paths, 1):
                print(f"\n{'='*60}")
                print(f"[Producer] Starting video {idx}/{len(video_paths)}: {video_path}")
                print(f"{'='*60}")
                self.detect_and_find_best_frames(
                    video_path, 
                    conf_threshold=conf_threshold,
                    show_frames=False
                )
            # Signal completion after all videos
            self.frame_queue.put(None)
            print(f"\n[Producer] ✓ All {len(video_paths)} videos processed")
        
        def consumer():
            self.find_text_in_best_frame(expected_texts_list)
        
        print("=" * 60)
        print(f"Starting multi-video pipeline...")
        print(f"  Videos to process: {len(video_paths)}")
        print(f"  Gap threshold: {self.gap_threshold} frames")
        print(f"  Max group size: {self.max_group_size} detections")
        print(f"  Confidence threshold: {conf_threshold}")
        print("=" * 60)
        
        producer_thread = threading.Thread(target=producer, name="MultiVideoProducer")
        consumer_thread = threading.Thread(target=consumer, name="MultiVideoConsumer")
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        print("\n" + "=" * 60)
        print(f"Multi-video pipeline complete!")
        print(f"  Total frames processed: {len(self.results)}")
        success_count = sum(1 for _, result in self.results if result)
        print(f"  Success: {success_count}/{len(self.results)}")
        if len(self.results) > 0:
            print(f"  Success rate: {success_count/len(self.results)*100:.1f}%")
        print("=" * 60)
        
        return self.results


# Example usage
if __name__ == "__main__":
    from support.paddleocr_model import PaddleOCRModel
    
    # Initialize models first
    print(">>> Initializing PaddleOCR model...")
    PaddleOCRModel()
    print(">>> PaddleOCR model initialized.")
    
    print(">>> Initializing YOLO Object detection model...")
    YoloObjDetectionModel().get_model()
    print(">>> YOLO Object detection model initialized.\n")
    
    # Single video processing with gap threshold
    print("="*60)
    print("SINGLE VIDEO PROCESSING")
    print("="*60 + "\n")
    
    
    processor = ContinuousPipelineProcessor(
        queue_size=25, 
        gap_threshold=3,  # Wait for 3 frames without detection before finalizing
        max_group_size=3  # Force finalization after 5 consecutive detections
    )
    
    results = processor.pipeline_process(
        video_path="D:/TestVideos/Videos/temp/8.mp4",
        expected_text=["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"],
        conf_threshold=0.8,
        show_frames=True
    )
    
    print("\n" + "="*60)
    print("DETAILED RESULTS:")
    print("="*60)
    for frame_num, text_result in results:
        status = "✓ PASS" if text_result else "✗ FAIL"
        print(f"  {status} - Frame {frame_num}: {text_result}")
    print("="*60)
    
    # # OPTIONAL: Multiple videos processing
    # print("\n\n" + "="*60)
    # print("MULTIPLE VIDEO PROCESSING")
    # print("="*60 + "\n")
    # 
    # multi_processor = MultiVideoPipelineProcessor(
    #     queue_size=100,
    #     gap_threshold=3,
    #     max_group_size=5
    # )
    # 
    # results = multi_processor.process_multiple_videos(
    #     video_paths=[
    #         "D:/TestVideos/Videos/temp/8.mp4",
    #         "D:/TestVideos/Videos/temp/9.mp4",
    #         "D:/TestVideos/Videos/temp/10.mp4"
    #     ],
    #     expected_texts_list=["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"],
    #     conf_threshold=0.8
    # )
    # 
    # print("\n" + "="*60)
    # print("DETAILED RESULTS (ALL VIDEOS):")
    # print("="*60)
    # for frame_num, text_result in results:
    #     status = "✓ PASS" if text_result else "✗ FAIL"
    #     print(f"  {status} - Frame {frame_num}: {text_result}")
    # print("="*60)