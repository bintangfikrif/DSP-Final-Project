# import the libraries
import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def extract_rppg_signal(frame, bbox):
    x, y, w, h = bbox
    face_roi = frame[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None
    yuv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    return np.mean(y_channel)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time rPPG & Respiration GUI")
        self.resize(900, 900)

        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # rPPG value display
        self.rppg_label = QLabel("rPPG: -")
        self.rppg_label.setStyleSheet("font-size: 20px; color: orange;")

        # Matplotlib plots
        self.fig, (self.ax_rppg, self.ax_resp) = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.rppg_signal = []
        self.resp_signal = []
        self.fps = 35
        self.frame_buffer_limit = 30 * self.fps

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.rppg_label)
        vbox.addWidget(self.canvas)
        self.setLayout(vbox)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Detection for rPPG
        face_results = self.face_detector.process(rgb_frame)
        rppg_value = None
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x, y = max(0, x), max(0, y)
                w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rppg_value = extract_rppg_signal(frame, (x, y, w, h))
                if rppg_value is not None:
                    self.rppg_signal.append(rppg_value)
                    if len(self.rppg_signal) > self.frame_buffer_limit:
                        self.rppg_signal.pop(0)
                break

        # Pose Detection for Respiration
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            h_img, w_img, _ = frame.shape
            rs = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ls = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            x1_r, y1_r = int(rs.x * w_img), int(rs.y * h_img)
            x1_l, y1_l = int(ls.x * w_img), int(ls.y * h_img)
            left = max(min(x1_r, x1_l) - 20, 0)
            top = max(min(y1_r, y1_l) - 65, 0)
            right = min(max(x1_r, x1_l) + 20, w_img)
            bottom = min(max(y1_r, y1_l) + 20, h_img)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            avg_y_shoulder = np.mean([y1_r, y1_l])
            self.resp_signal.append(-avg_y_shoulder)
            if len(self.resp_signal) > self.frame_buffer_limit:
                self.resp_signal.pop(0)

        if rppg_value is not None:
            self.rppg_label.setText(f"rPPG (Y mean): {rppg_value:.2f}")

        self.ax_rppg.clear()
        self.ax_rppg.plot(self.rppg_signal, color='orange')
        self.ax_rppg.set_title("rPPG Signal")
        self.ax_resp.clear()
        self.ax_resp.plot(self.resp_signal, color='green')
        self.ax_resp.set_title("Respiration Signal")
        self.canvas.draw()

        rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_show.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_show.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        self.pose.close()
        self.face_detector.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())