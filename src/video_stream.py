import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO add documentation and comments
class VideoStream:
    def __init__(self, source=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logging.error(f"Error: Unable to open video source {source}")
        else:
            # Set the resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logging.error("Warning: FPS is zero, defaulting to 30")
            fps = 30  # Default to 30 FPS if unable to retrieve
        return fps

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()