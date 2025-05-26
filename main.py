# main.py (Modified to integrate MediaPipe Tasks models)
import sys
import cv2
import numpy as np
import mediapipe as mp # Keep for mp.Image
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import scipy.signal as signal
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from gui import HealthTrackerUI
from signal_processing import extract_rppg_signal, butter_bandpass, calculate_rate_from_fft

# Helper function to draw landmarks (optional, but good for visualization)
def draw_landmarks_on_image(rgb_image, detection_result):
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = mp_vision.PoseLandmarkerResult(pose_landmarks=pose_landmarks).pose_landmarks[0] # Convert to list of NormalizedLandmark
        
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto, # Expects a list of NormalizedLandmark
            mp.solutions.pose.POSE_CONNECTIONS, # Use standard pose connections
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )
    return annotated_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Health Tracker - Engine")

        self.ui = HealthTrackerUI()
        self.setCentralWidget(self.ui)
        self.setMinimumSize(1000, 600)
        self.face_model_path = "models/blaze_face_short_range.tflite" 
        self.pose_model_path = "models/pose_landmarker.task"

        try:
            # Face Detector Initialization
            face_base_options = mp_python.BaseOptions(model_asset_path=self.face_model_path)
            face_options = mp_vision.FaceDetectorOptions(
                base_options=face_base_options,
                running_mode=mp_vision.RunningMode.IMAGE, # Process individual images
                min_detection_confidence=0.5
            )
            self.face_detector = mp_vision.FaceDetector.create_from_options(face_options)

            # Pose Landmarker Initialization
            pose_base_options = mp_python.BaseOptions(model_asset_path=self.pose_model_path)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5 # if pose model supports tracking
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            print("MediaPipe models loaded successfully.")

        except Exception as e:
            print(f"Error loading MediaPipe models: {e}")
            # You might want to disable processing or show an error in the UI
            self.face_detector = None
            self.pose_landmarker = None
            self.ui.video_label.setText(f"Error loading models: {e}\nPlease ensure model files are correctly placed in 'models' folder.")

        self.fps = 35
        self.min_signal_length = int(2 * self.fps)

        self.rppg_lowcut = 0.75 
        self.rppg_highcut = 3.5 
        self.rppg_b, self.rppg_a = butter_bandpass(self.rppg_lowcut, self.rppg_highcut, self.fps) 

        self.resp_lowcut = 0.1 
        self.resp_highcut = 0.7 
        self.resp_b, self.resp_a = butter_bandpass(self.resp_lowcut, self.resp_highcut, self.fps) 

        self.rppg_signal = [] 
        self.resp_signal = [] 
        self.frame_buffer_limit = int(10 * self.fps)

        self.video_label = self.ui.video_label
        self.hr_label = self.ui.hr_value_label
        self.rr_label = self.ui.rr_value_label
        self.ax_rppg = self.ui.ax_rppg
        self.canvas_rppg = self.ui.hr_canvas
        self.ax_resp = self.ui.ax_resp
        self.canvas_resp = self.ui.rr_canvas

        self.cap = None
        self.timer = QTimer(self) 
        self.timer.timeout.connect(self.update_frame) 

        self.ui.start_button.clicked.connect(self.start_processing)
        self.ui.end_button.clicked.connect(self.end_processing)
        self.ui.end_button.setEnabled(False)

        self.ui.video_label.setText("Press START to begin Camera Feed")

    def start_processing(self):
        if self.face_detector is None or self.pose_landmarker is None:
            self.ui.video_label.setText("Models not loaded. Cannot start.")
            print("Attempted to start processing but models are not loaded.")
            return

        if self.cap is None:
            self.cap = cv2.VideoCapture(0) 
        
        if not self.cap.isOpened():
            self.ui.video_label.setText("Error: Cannot open webcam!")
            self.cap = None
            return

        self.timer.start(int(1000.0 / self.fps))
        self.ui.start_button.setEnabled(False)
        self.ui.end_button.setEnabled(True)
        self.rppg_signal.clear()
        self.resp_signal.clear()
        print("Processing started.")

    def end_processing(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release() 
            self.cap = None
        self.ui.start_button.setEnabled(True)
        self.ui.end_button.setEnabled(False)
        self.video_label.setText("Camera Feed Ended. Press START.")
        self.video_label.setStyleSheet(self.ui.styleSheet() + " QLabel#VideoLabel { background-color: black; }")
        self.hr_label.setText("-- BPM") 
        self.rr_label.setText("-- Breaths/min") 
        self.ax_rppg.clear() 
        self.ax_resp.clear() 
        self.ui._apply_styles() # Re-apply styles to reset plot titles
        self.canvas_rppg.draw() 
        self.canvas_resp.draw() 
        print("Processing ended.")

    def _preprocess_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        ret, frame = self.cap.read() 
        if not ret:
            return None, None, None
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # Convert to mp.Image
        return frame, rgb_frame, mp_image

    def _process_rppg_signal(self, frame_to_draw_on, mp_image_input): #
        if self.face_detector is None:
            return None

        face_detector_result = self.face_detector.detect(mp_image_input)
        rppg_value = None

        if face_detector_result.detections:
            for detection in face_detector_result.detections: # Iterate through all detections
                bbox = detection.bounding_box
                
                # Get image dimensions from mp_image_input as bbox is normalized by it
                ih, iw = mp_image_input.height, mp_image_input.width

                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)
                
                # Ensure ROI is within frame boundaries (using frame_to_draw_on's shape for safety)
                frame_h, frame_w, _ = frame_to_draw_on.shape
                x = max(0, min(x, frame_w -1))
                y = max(0, min(y, frame_h -1))
                w = max(0, min(w, frame_w - x))
                h = max(0, min(h, frame_h - y))

                if w > 0 and h > 0 :
                    # Define forehead ROI relative to the detected face bounding box
                    forehead_x = int(x + w * 0.15) # Adjust ratios as needed
                    forehead_y = int(y + h * 0.05)
                    forehead_w = int(w * 0.7)
                    forehead_h = int(h * 0.25)

                    # Ensure forehead ROI is within frame boundaries
                    forehead_x = max(0, min(forehead_x, frame_w -1))
                    forehead_y = max(0, min(forehead_y, frame_h -1))
                    forehead_w = max(0, min(forehead_w, frame_w - forehead_x))
                    forehead_h = max(0, min(forehead_h, frame_h - forehead_y))

                    if forehead_w > 0 and forehead_h > 0:
                        cv2.rectangle(frame_to_draw_on, (forehead_x, forehead_y), 
                                      (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 255), 2)
                        
                        # Extract rPPG signal from the BGR frame (frame_to_draw_on)
                        rppg_value = extract_rppg_signal(frame_to_draw_on, 
                                                         (forehead_x, forehead_y, forehead_w, forehead_h))

                        if rppg_value is not None:
                            self.rppg_signal.append(rppg_value) 
                            if len(self.rppg_signal) > self.frame_buffer_limit: self.rppg_signal.pop(0) 
                break # Process only the first detected face for simplicity
        return rppg_value


    def _process_respiration_signal(self, frame_to_draw_on, mp_image_input): #
        if self.pose_landmarker is None:
            return

        pose_landmarker_result = self.pose_landmarker.detect(mp_image_input)
        
        if pose_landmarker_result.pose_landmarks:
            # Draw landmarks on the frame for visualization
            landmarks = pose_landmarker_result.pose_landmarks[0]
            
            h_img, w_img, _ = frame_to_draw_on.shape 

            try:
                # Access left and right shoulder landmarks
                ls = landmarks[11]
                rs = landmarks[12]

                y1_l = int(ls.y * h_img)
                y1_r = int(rs.y * h_img)
                
                # Simple visualization of shoulder points
                cv2.circle(frame_to_draw_on, (int(rs.x * w_img), y1_r), 5, (255,0,0), -1)
                cv2.circle(frame_to_draw_on, (int(ls.x * w_img), y1_l), 5, (0,255,0), -1)

                avg_y_shoulder = np.mean([y1_r, y1_l]) 
                self.resp_signal.append(-avg_y_shoulder) 
                if len(self.resp_signal) > self.frame_buffer_limit: self.resp_signal.pop(0) 
            except IndexError:
                print("Error accessing shoulder landmarks. Check model output or landmark indices.")
            except AttributeError: 
                 print("Error: Landmark object does not have x or y attributes.")

    def _filter_and_calculate_rates(self): 
        filtered_rppg_signal = self.rppg_signal 
        if len(self.rppg_signal) > self.min_signal_length: 
            filtered_rppg_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal).tolist() 

        filtered_resp_signal = self.resp_signal 
        if len(self.resp_signal) > self.min_signal_length: 
            filtered_resp_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal).tolist() 

        current_hr = calculate_rate_from_fft(filtered_rppg_signal, self.fps, self.rppg_lowcut, self.rppg_highcut) 
        current_rr = calculate_rate_from_fft(filtered_resp_signal, self.fps, self.resp_lowcut, self.resp_highcut) 
        return filtered_rppg_signal, filtered_resp_signal, current_hr, current_rr

    def _update_gui_plots_and_labels(self, frame_processed, filtered_rppg, filtered_resp, hr, rr): # (No changes needed here, but ensure plot colors are fine)
        self.ax_rppg.clear() 
        self.ax_rppg.plot(filtered_rppg, color='#FF6B6B') 
        self.canvas_rppg.draw() 

        self.ax_resp.clear() 
        self.ax_resp.plot(filtered_resp, color='#6BCBFF') 
        self.canvas_resp.draw() 
        
        self.ui._apply_styles() 

        if hr > 0: self.hr_label.setText(f"{hr:.0f} BPM") 
        else: self.hr_label.setText("-- BPM") 

        if rr > 0: self.rr_label.setText(f"{rr:.0f} Breaths/min") 
        else: self.rr_label.setText("-- Breaths/min") 

        display_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB) 
        h, w, ch = display_frame.shape 
        bytes_per_line = ch * w 
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888) 
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)) 


    def update_frame(self): 
        original_frame, rgb_frame, mp_image = self._preprocess_frame() # Get mp.Image here
        if original_frame is None or mp_image is None:
            return

        frame_to_display = original_frame.copy()

        self._process_rppg_signal(frame_to_display, mp_image) 
        self._process_respiration_signal(frame_to_display, mp_image)
        
        filtered_rppg, filtered_resp, current_hr, current_rr = self._filter_and_calculate_rates()
        
        self._update_gui_plots_and_labels(frame_to_display, filtered_rppg, filtered_resp, current_hr, current_rr)

    def closeEvent(self, event): 
        self.end_processing() 
        print("Application closing.")
        event.accept() 

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())