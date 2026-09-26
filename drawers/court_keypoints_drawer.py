import cv2


class CourtKeypointsDrawer:
    """
    Draws court keypoints (e.g. lines, markers) on video frames.

    Rewritten to consume plain (N, 2) numpy arrays directly — the format
    CourtKeypointDetector.get_court_keypoints() actually returns — instead
    of a supervision.KeyPoints-like object with a `.xy` attribute, which is
    what this file previously (and incorrectly) assumed. Drops the
    `supervision` dependency for this drawer.
    """

    def __init__(self, keypoint_color=(44, 44, 255), radius: int = 6):
        self.keypoint_color = keypoint_color
        self.radius = radius

    def draw_frame(self, frame, keypoints):
        """
        Annotates one frame's keypoints (and their index). Mutates and
        returns `frame` in place.
        """
        for kp_idx, (x, y) in enumerate(keypoints):
            if x <= 0 and y <= 0:
                continue  # undetected / invalidated keypoint

            point = (int(x), int(y))
            cv2.circle(frame, point, self.radius, self.keypoint_color, -1)
            cv2.putText(
                frame,
                str(kp_idx),
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def draw(self, frames, court_keypoints):
        """
        Annotates keypoints across a whole list of frames, returning a new
        list. For long videos prefer `draw_frame` in a streaming loop (see
        main.py).
        """
        return [
            self.draw_frame(frame.copy(), court_keypoints[index])
            for index, frame in enumerate(frames)
        ]
