import time
from os.path import split

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QScrollArea
from src.ui.config_panel import ConfigPanel
from src.ui.video_panel import VideoPanel
import cv2
import os
import sys
import yaml


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the configuration file
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Setup properties
        self.fps = 0
        self.native_fps = 0
        self.confidence = 50
        self.__res = None
        self.__device_id = None
        self.__codec = None
        self.__nth_frame = 1
        self.__bbox_max = 100

        # Get the available classes
        self.__class_details = self.config['class_details']
        self.__multi_color_classes = False
        self.__omit_classes = []

        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set up the layout for the central widget
        self.layout = QHBoxLayout(self.central_widget)
        self.scroll_area = QScrollArea()

        # Create and add the ConfigPanel to the layout
        self.config_panel = ConfigPanel(self)
        self.video_panel = VideoPanel(self)

        # Set default values in config.
        self.config_panel.set_fps(1)
        self.config_panel.set_confidence(50)

        self.scroll_area.setWidget(self.config_panel)
        self.scroll_area.setWidgetResizable(True)

        self.layout.addWidget(self.video_panel)
        self.layout.addWidget(self.scroll_area)

        # Set the stretch factors for the layout
        self.layout.setStretch(0, 7)  # VideoPanel takes 70% of the space
        self.layout.setStretch(1, 3)  # ConfigPanel takes 30% of the space

    def set_confidence(self, value):
        """Set the confidence threshold value."""
        # TODO: Implement the update to the detection worker
        self.confidence = value

    # UI Value Setters and Getters #

    # Sets the resolution multiplier
    def set_video_file(self, file_path):
        """Set the video file path."""
        self.video_file = file_path
        self.video_panel.load_video_file(file_path)
        # TODO Add this shit
        print(f'Video file: {file_path}')

    def set_resolution(self, res, fps):
        """Set the resolution multiplier value."""
        update_res = False
        update_fps = True
        if res is not self.__res:
            update_res = True
        if fps is not self.fps:
            update_fps = True

        self.__res = res
        self.fps = fps

        if res and fps and self.__codec and self.__device_id != -1 and self.__device_id:
            if self.video_panel.get_video_stream() is not None:
                # changes resolution and fps for the current file.
                if update_fps:
                    self.video_panel.get_video_stream().set(cv2.CAP_PROP_FPS, fps)
                if update_res:
                    width, height = map(int, res.split('x'))
                    self.video_panel.get_video_stream().set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.video_panel.get_video_stream().set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            else:
                # set a new video device
                print(
                    f'new video device with: {res} @ {fps} on device id {self.__device_id} with vcodec {self.__codec}')


    # Gets the list of available classes
    def get_available_classes(self):
        """Get the list of available classes."""
        if not self.__class_details:
            return []

        if self.__multi_color_classes:
            return [f"{details['class']}: ({details['name']})" for details in self.__class_details.values()]
        return [details['class'] for details in self.__class_details.values()]

    def set_multi_color_classes(self, value):
        """Set the multi-color classes value."""
        self.__multi_color_classes = value
        print(f"Multi-color classes: {value}")

    def update_omitted_classes(self, classes):
        """Update the omitted classes."""
        self.__omit_classes = classes

    def set_nth_frame(self, value):
        """Set the nth frame value."""
        self.__nth_frame = value
        print(f"Nth frame: {value}")

    def set_bounding_box_max(self, value):
        """Set the bounding box max value."""
        self.__bbox_max = value
        print(f"Bounding box max: {value}")

    def set_codec(self, codec):
        """Set the codec."""
        self.__codec = codec
        print(f"Codec: {codec}")

    def set_video_device(self, device):
        """Set the video device."""
        self.__device_id = device

        if device == -1:
            # Clear the video device settings.
            self.set_codec(None)
            self.set_resolution(None, None)
            print("Video device removed.")
