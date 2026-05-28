import time

import cv2


class FPS:
    """Smoothed frames-per-second counter."""

    def __init__(self, smoothing=0.9):
        self.prev = time.time()
        self.fps = 0.0
        self.smoothing = smoothing

    def tick(self):
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * inst
        return self.fps


def draw_hud(frame, lines, color=(0, 255, 0)):
    """Draw a few status lines in the top-left corner."""
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30


def open_camera(source):
    """Open a camera index or a video file. Raises if it can't."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source {source!r}. "
            "Check the camera index or file path."
        )
    return cap
