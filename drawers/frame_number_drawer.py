# frame_number_drawer.py

"""
Draws the current frame number on each video frame — useful for QA'ing
tracker/keypoint output against the source video during review.

Referenced by the original main.py but never implemented (see
DISCOVERY_REPORT.md).
"""

import cv2


class FrameNumberDrawer:
    def __init__(self, position=(10, 30), color=(255, 255, 255)):
        self.position = position
        self.color = color

    def draw_frame(self, frame, frame_num):
        """Draws the frame number on one frame. Mutates and returns it."""
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            self.position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.color,
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw(self, video_frames):
        """
        Draws frame numbers across a whole list of frames, returning a new
        list. For long videos prefer `draw_frame` in a streaming loop (see
        main.py).
        """
        return [self.draw_frame(frame.copy(), frame_num) for frame_num, frame in enumerate(video_frames)]
