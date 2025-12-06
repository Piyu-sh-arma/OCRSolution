import numpy as np


class FrameSelector:
    """A class for selecting frames from video or image sequences."""

    def __init__(self):
        """Initialize the FrameSelector."""
        pass

    def find_with_best_contrast(self, frames):
        max_contrast = 0
        max_frame_idx = -1

        for idx, (_, frame) in enumerate(frames):
            contrast = frame.std()
            if contrast > max_contrast:
                max_contrast = contrast
                max_frame_idx = idx

        return frames[max_frame_idx]

    def find_frame_with_max_black(self, frames, black_threshold=50):

        max_ratio = -1
        max_frame_idx = -1

        for idx, (_,frame) in enumerate(frames):
            # Count black pixels (values <= threshold)
            black_pixels = np.sum(frame <= black_threshold)

            # Count grey pixels (values > threshold)
            grey_pixels = np.sum(frame > black_threshold)

            # Calculate ratio (handle division by zero)
            if grey_pixels > 0:
                ratio = black_pixels / grey_pixels
            else:
                # If all pixels are black, ratio is infinite
                ratio = float("inf") if black_pixels > 0 else 0

            # Update maximum
            if ratio > max_ratio:
                max_ratio = ratio
                max_frame_idx = idx

        return frames[max_frame_idx]
