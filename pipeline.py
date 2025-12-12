import threading
from queue import Queue
import cv2

import config
from support.frame_selector import FrameSelector
from support.frame_utils import resize_frame, save_frame
from support.yolo_model import YoloObjDetectionModel
from text_recognition_rnd import detect_and_verify


class ContinuousPipelineProcessor:
    def __init__(self, queue_size=50):
        """
        Queue continuously receives detected frames from producer
        Consumer processes them as they arrive
        """
        self.frame_queue = Queue(maxsize=queue_size)
        self.results = []
        
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
        print(f">>> Frames per second (FPS): {fps}")

        if show_frames:
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
                else:
                    # When detection ends, select best frame and push to queue
                    if detected_frames:
                        print(f"Selecting best from {len(detected_frames)} detections...")
                        selected_frame_data = frame_selector.find_with_best_contrast(
                            detected_frames
                        )
                        
                        # PUSH TO QUEUE - Consumer will process it
                        print(f"✓ Pushing frame {selected_frame_data[0]} to queue")
                        self.frame_queue.put(selected_frame_data)
                        detected_frames = []

                # Display frames
                if show_frames:
                    annotated_frame = results[0].plot()
                    cv2.imshow(config.WINDOW_NAME, annotated_frame)
                    if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                        break

        finally:
            # Process any remaining frames
            if detected_frames:
                selected_frame_data = frame_selector.find_with_best_contrast(detected_frames)
                print(f"✓ Pushing final frame {selected_frame_data[0]} to queue")
                self.frame_queue.put(selected_frame_data)
                
            # Signal completion to consumer
            self.frame_queue.put(None)
            print("✓ Producer finished - sent completion signal")
            
            cap.release()
            if show_frames:
                cv2.destroyAllWindows()

        print(f"Processing complete. Total frames: {frame_counter}")

    def find_text_in_best_frame(self, expected_text=[]):
        """
        CONSUMER: Continuously checks queue and processes incoming frames
        Runs in separate thread, processes frames as they arrive
        """
        processed_count = 0
        
        while True:
            # BLOCKING CALL - waits until frame is available
            frame_data = self.frame_queue.get()
            
            # Check for completion signal
            if frame_data is None:
                print("✓ Consumer received completion signal - stopping")
                break
            
            frame_num, frame = frame_data
            print(f"→ Processing frame {frame_num} for text extraction...")
            
            # Process the frame
            frame = resize_frame(frame)
            file_path = f"{config.OUTPUT_DIR}/{frame_num}.jpg"
            save_frame(file_path, frame)
            
            # Extract text
            result = detect_and_verify(frame, expected_text)
            self.results.append((frame_num, result))
            
            processed_count += 1
            print(f"✓ Completed text extraction for frame {frame_num} ({processed_count} total)")
        
        print(f"Consumer finished. Processed {processed_count} frames")

    def pipeline_process(self, video_path="", expected_text=[]):
        """
        Start both producer and consumer threads
        They run concurrently:
        - Producer: Detects frames → pushes to queue
        - Consumer: Pulls from queue → extracts text
        """
        print("=" * 60)
        print("Starting continuous pipeline processing...")
        print("=" * 60)
        
        # Stage 1: Producer thread
        producer = threading.Thread(
            target=self.detect_and_find_best_frames,
            args=(video_path,),
            name="FrameDetector"
        )
        
        # Stage 2: Consumer thread
        consumer = threading.Thread(
            target=self.find_text_in_best_frame,
            args=(expected_text,),
            name="TextExtractor"
        )
        
        # Start both threads
        producer.start()
        consumer.start()
        
        print("✓ Both threads started - working in parallel")
        
        # Wait for both to complete
        producer.join()
        print("✓ Producer thread completed")
        
        consumer.join()
        print("✓ Consumer thread completed")
        
        print("=" * 60)
        print(f"Pipeline complete! Processed {len(self.results)} frames")
        print("=" * 60)
        
        return self.results


# ALTERNATIVE: Process multiple videos continuously
class MultiVideoPipelineProcessor(ContinuousPipelineProcessor):
    """
    Extended version for processing multiple videos
    Producer keeps pushing frames from all videos
    Consumer processes them all continuously
    """
    
    def process_multiple_videos(self, video_paths, expected_texts_list):
        """
        Process multiple videos through the pipeline
        """
        def producer():
            for video_path in video_paths:
                print(f"\n>>> Starting video: {video_path}")
                self.detect_and_find_best_frames(video_path, show_frames=False)
            # Signal completion after all videos
            self.frame_queue.put(None)
        
        def consumer():
            self.find_text_in_best_frame(expected_texts_list)
        
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        return self.results

# expected_str_list = ["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "330"]
# expected_str_list = ["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"]
# expected_str_list = ["70g+5g*", "Rs.0.14/g", "MFG.","11/25", "5330B095J3", "1355"]

# Example usage
if __name__ == "__main__":
    # Single video processing
    processor = ContinuousPipelineProcessor(queue_size=25)
    results = processor.pipeline_process(
        video_path="D:/TestVideos/Videos/temp/8.mp4",
        expected_text= ["70g+10g*", "Rs.0.14/g", "MFG.","12/25", "5338B095J3", "325"]
    )
    
    print("\nResults:")
    for frame_num, text_result in results:
        print(f"Frame {frame_num}: {text_result}")
    
    # # Multiple videos
    # multi_processor = MultiVideoPipelineProcessor(queue_size=100)
    # results = multi_processor.process_multiple_videos(
    #     video_paths=["video1.mp4", "video2.mp4", "video3.mp4"],
    #     expected_texts_list=["text1", "text2"]
    # )