import numpy as np
import cv2
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl

class OpenGLVideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.image = None
        self.bounding_boxes = []

    def initializeGL(self):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0, 0, 0, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)

    def resizeGL(self, width, height):
        if self.image is not None:
            video_width, video_height = self.image.shape[1], self.image.shape[0]
            video_aspect = video_width / video_height
            widget_aspect = width / height

            if widget_aspect > video_aspect:
                scaled_width = int(height * video_aspect)
                scaled_height = height
            else:
                scaled_width = width
                scaled_height = int(width / video_aspect)

            x_offset = (width - scaled_width) // 2
            y_offset = (height - scaled_height) // 2
            gl.glViewport(x_offset, y_offset, scaled_width, scaled_height)
        else:
            gl.glViewport(0, 0, width, height)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        self.draw_texture_and_bounding_boxes()
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL Error: {error}")

    def upload_frame_to_opengl(self, frame):
        self.image = frame
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, frame_rgba.shape[1], frame_rgba.shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, frame_rgba)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, frame_rgba.shape[1], frame_rgba.shape[0], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, frame_rgba)

    def draw_texture_and_bounding_boxes(self):
        if self.texture_id is None or self.image is None:
            return

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        video_width, video_height = self.image.shape[1], self.image.shape[0]
        widget_width, widget_height = self.width(), self.height()
        video_aspect = video_width / video_height
        widget_aspect = widget_width / widget_height

        if widget_aspect > video_aspect:
            scale = widget_height / video_height
            scaled_width = video_width * scale
            scaled_height = widget_height
            x_offset = (widget_width - scaled_width) / 2
            y_offset = 0
        else:
            scale = widget_width / video_width
            scaled_width = widget_width
            scaled_height = video_height * scale
            x_offset = 0
            y_offset = (widget_height - scaled_height) / 2

        gl.glViewport(int(x_offset), int(y_offset), int(scaled_width), int(scaled_height))

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1); gl.glVertex3f(-1, -1, 0)
        gl.glTexCoord2f(1, 1); gl.glVertex3f(1, -1, 0)
        gl.glTexCoord2f(1, 0); gl.glVertex3f(1, 1, 0)
        gl.glTexCoord2f(0, 0); gl.glVertex3f(-1, 1, 0)
        gl.glEnd()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_DEPTH_TEST)

        if self.bounding_boxes:
            for box, score, class_id in self.bounding_boxes:
                x1, y1, x2, y2 = box
                x1 = x_offset + (x1 / video_width) * scaled_width
                y1 = y_offset + (y1 / video_height) * scaled_height
                x2 = x_offset + (x2 / video_width) * scaled_width
                y2 = y_offset + (y2 / video_height) * scaled_height

                x1_ndc = (x1 / widget_width) * 2 - 1
                y1_ndc = 1 - (y1 / widget_height) * 2
                x2_ndc = (x2 / widget_width) * 2 - 1
                y2_ndc = 1 - (y2 / widget_height) * 2

                gl.glColor4f(1.0, 0.0, 0.0, 0.5)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex2f(x1_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y2_ndc)
                gl.glVertex2f(x1_ndc, y2_ndc)
                gl.glEnd()

                gl.glColor4f(1.0, 0.0, 0.0, 1.0)
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex2f(x1_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y1_ndc)
                gl.glVertex2f(x2_ndc, y2_ndc)
                gl.glVertex2f(x1_ndc, y2_ndc)
                gl.glEnd()

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL Error: {error}")