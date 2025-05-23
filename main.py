# import the libraries
import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import scipy.signal as signal
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

        # HR and Respiration rate display
        self.hr_label = QLabel("HR: - BPM")
        self.hr_label.setStyleSheet("font-size: 20px; color: red; font-weight: bold;")

        self.rr_label = QLabel("RR: - Breaths/min")
        self.rr_label.setStyleSheet("font-size: 20px; color: blue; font-weight: bold;")

        # Matplotlib plots
        self.fig, (self.ax_rppg, self.ax_resp) = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.rppg_signal = []
        self.resp_signal = []
        self.fps = 35
        self.frame_buffer_limit = 30 * self.fps

        # rPPG filter parameters (0.75 Hz to 4 Hz for heart rate)
        # Normalizing frequencies by Nyquist frequency (fs/2)
        self.rppg_lowcut = 0.75
        self.rppg_highcut = 4.0
        self.rppg_nyquist = 0.5 * self.fps
        self.rppg_b, self.rppg_a = signal.butter(3, [self.rppg_lowcut / self.rppg_nyquist, self.rppg_highcut / self.rppg_nyquist], btype='band')

        # Respiration filter parameters (0.1 Hz to 0.5 Hz for respiration rate)
        self.resp_lowcut = 0.1
        self.resp_highcut = 0.5
        self.resp_nyquist = 0.5 * self.fps
        self.resp_b, self.resp_a = signal.butter(3, [self.resp_lowcut / self.resp_nyquist, self.resp_highcut / self.resp_nyquist], btype='band')

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.rppg_label)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.hr_label)
        vbox.addWidget(self.rr_label)
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
                
                # Define forehead ROI as a portion of the detected face bounding box
                # Example: Top 30% of the face, centered horizontally
                forehead_x = int(x + w * 0.25) # Start from 25% of face width from left
                forehead_y = int(y + h * 0.05) # Start from 5% of face height from top
                forehead_w = int(w * 0.5) # Take 50% of face width
                forehead_h = int(h * 0.25) # Take 25% of face height

                # Ensure ROI stays within frame boundaries
                forehead_x = max(0, forehead_x)
                forehead_y = max(0, forehead_y)
                forehead_w = min(frame.shape[1] - forehead_x, forehead_w)
                forehead_h = min(frame.shape[0] - forehead_y, forehead_h)

                # Draw rectangle for the new forehead ROI
                cv2.rectangle(frame, (forehead_x, forehead_y), (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 255), 2) # Yellow color

                # Use the forehead ROI for rPPG signal extraction
                rppg_value = extract_rppg_signal(frame, (forehead_x, forehead_y, forehead_w, forehead_h))

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

        # Apply filter only if enough data points are available
        min_signal_length = int(2 * self.fps) # Roughly 2 seconds of data

        if len(self.rppg_signal) > min_signal_length:
            filtered_rppg_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal).tolist()
        else:
            filtered_rppg_signal = self.rppg_signal # Use raw signal if not enough data

        if len(self.resp_signal) > min_signal_length:
            filtered_resp_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal).tolist()
        else:
            filtered_resp_signal = self.resp_signal # Use raw signal if not enough data

        current_hr = -1.0 # Initialize with an invalid value
        current_rr = -1.0 # Initialize with an invalid value

        if len(filtered_rppg_signal) > min_signal_length:
            # Calculate HR using FFT
            rppg_fft = np.abs(np.fft.fft(filtered_rppg_signal))
            freqs = np.fft.fftfreq(len(filtered_rppg_signal), 1/self.fps)
            # Find the peak in the relevant HR frequency range (0.75 Hz to 4 Hz)
            rppg_valid_indices = np.where((freqs >= self.rppg_lowcut) & (freqs <= self.rppg_highcut))
            if len(rppg_valid_indices[0]) > 0:
                dominant_rppg_freq_idx = rppg_valid_indices[0][np.argmax(rppg_fft[rppg_valid_indices])]
                dominant_rppg_freq = freqs[dominant_rppg_freq_idx]
                current_hr = dominant_rppg_freq * 60 # Convert Hz to BPM

        if len(filtered_resp_signal) > min_signal_length:
            # Calculate RR using FFT
            resp_fft = np.abs(np.fft.fft(filtered_resp_signal))
            resp_freqs = np.fft.fftfreq(len(filtered_resp_signal), 1/self.fps)
            # Find the peak in the relevant RR frequency range (0.1 Hz to 0.5 Hz)
            resp_valid_indices = np.where((resp_freqs >= self.resp_lowcut) & (resp_freqs <= self.resp_highcut))
            if len(resp_valid_indices[0]) > 0:
                dominant_resp_freq_idx = resp_valid_indices[0][np.argmax(resp_fft[resp_valid_indices])]
                dominant_resp_freq = resp_freqs[dominant_resp_freq_idx]
                current_rr = dominant_resp_freq * 60 # Convert Hz to Breaths/min

        self.ax_rppg.clear()
        self.ax_rppg.plot(filtered_rppg_signal, color='orange')
        self.ax_rppg.set_title("rPPG Signal")
        self.ax_resp.clear()
        self.ax_resp.plot(filtered_resp_signal, color='green')
        self.ax_resp.set_title("Respiration Signal")
        self.canvas.draw()

        if current_hr > 0:
            self.hr_label.setText(f"HR: {current_hr:.1f} BPM")
        else:
            self.hr_label.setText("HR: - BPM")

        if current_rr > 0:
            self.rr_label.setText(f"RR: {current_rr:.1f} Breaths/min")
        else:
            self.rr_label.setText("RR: - Breaths/min")

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