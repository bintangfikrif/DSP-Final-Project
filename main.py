# main.py (Modified)
import sys
import cv2
import numpy as np
import mediapipe as mp
import scipy.signal as signal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from gui import HealthTrackerUI 
from signal_processing import extract_rppg_signal, butter_bandpass, calculate_rate_from_fft

class MainWindow(QMainWindow): 
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Health Tracker - Engine")

        # Create and set the central widget from the new UI class
        self.ui = HealthTrackerUI()
        
        # Create a container widget and layout to hold the UI
        self.setCentralWidget(self.ui)
        self.setMinimumSize(1000, 600) 

        # --- Application Logic Setup ---
        self.mp_face_detection = mp.solutions.face_detection #
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) #
        self.mp_pose = mp.solutions.pose #
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5) #

        self.fps = 30 # Adjusted from 35
        self.min_signal_length = int(2 * self.fps) #

        self.rppg_lowcut = 0.75 #
        self.rppg_highcut = 4.0 #
        self.rppg_b, self.rppg_a = butter_bandpass(self.rppg_lowcut, self.rppg_highcut, self.fps) #

        self.resp_lowcut = 0.1 #
        self.resp_highcut = 0.5 #
        self.resp_b, self.resp_a = butter_bandpass(self.resp_lowcut, self.resp_highcut, self.fps) #

        self.rppg_signal = [] #
        self.resp_signal = [] #
        self.frame_buffer_limit = int(10 * self.fps) # Buffer for 10 seconds

        # --- UI Element References (from self.ui) ---
        self.video_label = self.ui.video_label
        self.hr_label = self.ui.hr_value_label
        self.rr_label = self.ui.rr_value_label
        self.ax_rppg = self.ui.ax_rppg
        self.canvas_rppg = self.ui.hr_canvas
        self.ax_resp = self.ui.ax_resp
        self.canvas_resp = self.ui.rr_canvas

        # Video capture and timer
        self.cap = None # Initialize later
        self.timer = QTimer(self) #
        self.timer.timeout.connect(self.update_frame) #

        # Connect UI buttons to methods
        self.ui.start_button.clicked.connect(self.start_processing)
        self.ui.end_button.clicked.connect(self.end_processing)
        self.ui.end_button.setEnabled(False)

        self.ui.video_label.setText("Press START to begin Camera Feed")

    def start_processing(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0) #
        
        if not self.cap.isOpened():
            self.ui.video_label.setText("Error: Cannot open webcam!")
            self.cap = None
            return

        self.timer.start(int(1000.0 / self.fps))
        self.ui.start_button.setEnabled(False)
        self.ui.end_button.setEnabled(True)
        self.rppg_signal.clear()
        self.resp_signal.clear()

    def end_processing(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release() #
            self.cap = None
        self.ui.start_button.setEnabled(True)
        self.ui.end_button.setEnabled(False)
        self.video_label.setText("Camera Feed Ended. Press START.")
        self.video_label.setStyleSheet(self.ui.styleSheet() + " QLabel#VideoLabel { background-color: black; }") # Reset style
        self.hr_label.setText("-- BPM") #
        self.rr_label.setText("-- Breaths/min") #
        self.ax_rppg.clear() #
        self.ax_resp.clear() #
        self.ui._apply_styles() # Re-apply styles to clear plot titles etc.
        self.canvas_rppg.draw() #
        self.canvas_resp.draw() #

    def _preprocess_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None, None
        ret, frame = self.cap.read() #
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1) #
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #
        return frame, rgb_frame

    def _process_rppg_signal(self, frame, rgb_frame): #
        face_results = self.face_detector.process(rgb_frame) #
        rppg_value = None
        if face_results.detections: #
            detection = face_results.detections[0] 
            bbox = detection.location_data.relative_bounding_box #
            ih, iw, _ = frame.shape #
            x = int(bbox.xmin * iw); y = int(bbox.ymin * ih) #
            w = int(bbox.width * iw); h = int(bbox.height * ih) #
            x, y = max(0, x), max(0, y) #
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h) #
            
            forehead_x = int(x + w * 0.25); forehead_y = int(y + h * 0.05) #
            forehead_w = int(w * 0.5); forehead_h = int(h * 0.25) #

            forehead_x = max(0, forehead_x); forehead_y = max(0, forehead_y) #
            forehead_w = min(frame.shape[1] - forehead_x, forehead_w) #
            forehead_h = min(frame.shape[0] - forehead_y, forehead_h) #

            cv2.rectangle(frame, (forehead_x, forehead_y), (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 255), 2) #
            rppg_value = extract_rppg_signal(frame, (forehead_x, forehead_y, forehead_w, forehead_h)) #

            if rppg_value is not None:
                self.rppg_signal.append(rppg_value) #
                if len(self.rppg_signal) > self.frame_buffer_limit: self.rppg_signal.pop(0) #
        return rppg_value

    def _process_respiration_signal(self, frame, rgb_frame): #
        pose_results = self.pose.process(rgb_frame) #
        if pose_results.pose_landmarks: #
            h_img, w_img, _ = frame.shape #
            rs = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] #
            ls = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER] #
            x1_r, y1_r = int(rs.x * w_img), int(rs.y * h_img) #
            x1_l, y1_l = int(ls.x * w_img), int(ls.y * h_img) #
            # Chest ROI slightly adjusted
            chest_center_y = int(np.mean([y1_r, y1_l]))
            chest_center_x = int(np.mean([x1_r, x1_l]))
            shoulder_width = abs(x1_r - x1_l)
            
            roi_w = int(shoulder_width * 0.8)
            roi_h = int(shoulder_width * 0.5) 

            left = max(chest_center_x - roi_w // 2, 0)
            top = max(chest_center_y - roi_h // 2 -10, 0) 
            right = min(chest_center_x + roi_w // 2, w_img)
            bottom = min(chest_center_y + roi_h // 2 + 10, h_img)

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2) #
            avg_y_shoulder = np.mean([y1_r, y1_l]) 
            self.resp_signal.append(-avg_y_shoulder) 
            if len(self.resp_signal) > self.frame_buffer_limit: self.resp_signal.pop(0) #

    def _filter_and_calculate_rates(self): #
        filtered_rppg_signal = self.rppg_signal #
        if len(self.rppg_signal) > self.min_signal_length: #
            filtered_rppg_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal).tolist() #

        filtered_resp_signal = self.resp_signal #
        if len(self.resp_signal) > self.min_signal_length: #
            filtered_resp_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal).tolist() #

        current_hr = calculate_rate_from_fft(filtered_rppg_signal, self.fps, self.rppg_lowcut, self.rppg_highcut) #
        current_rr = calculate_rate_from_fft(filtered_resp_signal, self.fps, self.resp_lowcut, self.resp_highcut) #
        return filtered_rppg_signal, filtered_resp_signal, current_hr, current_rr

    def _update_gui_plots_and_labels(self, frame_processed, filtered_rppg, filtered_resp, hr, rr): #
        self.ax_rppg.clear() #
        self.ax_rppg.plot(filtered_rppg, color='#FF6B6B') # Reddish
        self.canvas_rppg.draw() #

        self.ax_resp.clear() #
        self.ax_resp.plot(filtered_resp, color='#6BCBFF') # Bluish
        self.canvas_resp.draw() #
        
        self.ui._apply_styles() # re-apply styles to ensure plot backgrounds and axes are correct after clear

        if hr > 0: self.hr_label.setText(f"{hr:.0f} BPM") #
        else: self.hr_label.setText("-- BPM") #

        if rr > 0: self.rr_label.setText(f"{rr:.0f} Breaths/min") #
        else: self.rr_label.setText("-- Breaths/min") #

        display_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB) #
        h, w, ch = display_frame.shape #
        bytes_per_line = ch * w #
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888) #
        
        # Scale pixmap to fit video_label while keeping aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)) #


    def update_frame(self): #
        frame, rgb_frame = self._preprocess_frame()
        if frame is None:
            return

        self._process_rppg_signal(frame, rgb_frame) # rppg_value is not directly used in this version of GUI
        self._process_respiration_signal(frame, rgb_frame)
        
        filtered_rppg, filtered_resp, current_hr, current_rr = self._filter_and_calculate_rates()
        
        self._update_gui_plots_and_labels(frame, filtered_rppg, filtered_resp, current_hr, current_rr)

    def closeEvent(self, event): #
        self.end_processing() # Ensure resources are released
        if hasattr(self, 'pose') and self.pose: self.pose.close() #
        if hasattr(self, 'face_detector') and self.face_detector: self.face_detector.close() #
        event.accept() #

if __name__ == "__main__":
    app = QApplication(sys.argv) #
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) #