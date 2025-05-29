import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import scipy.signal as signal
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import os

try:
    from utils.gui import HealthTrackerUI
    from utils.signal_processing import extract_rppg_signal, butter_bandpass, calculate_rate_from_fft
except ImportError as e:
    print(f"Penting: Gagal mengimpor modul dari folder 'utils'. Pastikan file ada dan benar: {e}")
    print("Harap buat file 'utils/gui.py' dan 'utils/signal_processing.py' sesuai kebutuhan.")
    sys.exit(1)
    
# Cek apakah modul MediaPipe sudah terinstal
def draw_landmarks_on_image(rgb_image, detection_result):
    from mediapipe.python.solutions import drawing_utils as mp_drawing 
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto_list = pose_landmarks

        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto_list, 
            mp.solutions.pose.POSE_CONNECTIONS if hasattr(mp.solutions, 'pose') else None, # Cek apakah mp.solutions.pose ada
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )
    return annotated_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Health Tracker - Engine (Optimized)")

        # Menggunakan UI dari utils.gui
        self.ui = HealthTrackerUI() 
        self.setCentralWidget(self.ui)
        self.setMinimumSize(1000, 600) 

        self.face_model_path = "models/blaze_face_short_range.tflite" 
        self.pose_model_path = "models/pose_landmarker.task"

        self.face_detector = None
        self.pose_landmarker = None
        try:
            if not os.path.exists(self.face_model_path):
                raise FileNotFoundError(f"File model wajah tidak ditemukan: {self.face_model_path}")
            face_base_options = mp_python.BaseOptions(model_asset_path=self.face_model_path)
            face_options = mp_vision.FaceDetectorOptions(
                base_options=face_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.5
            )
            self.face_detector = mp_vision.FaceDetector.create_from_options(face_options)

            if not os.path.exists(self.pose_model_path):
                 raise FileNotFoundError(f"File model pose tidak ditemukan: {self.pose_model_path}")
            pose_base_options = mp_python.BaseOptions(model_asset_path=self.pose_model_path)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5 
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            print("Model MediaPipe berhasil dimuat.")

        except Exception as e:
            print(f"Error saat memuat model MediaPipe: {e}")
            if hasattr(self.ui, 'video_label') and self.ui.video_label:
                self.ui.video_label.setText(f"Error memuat model: {e}\nPastikan file model ada di folder 'models'.")
            else:
                print("UI video_label tidak tersedia untuk menampilkan pesan error model.")

        self.fps = 30 
        self.min_signal_length = int(2 * self.fps)

        self.rppg_lowcut = 0.75 
        self.rppg_highcut = 4.0 
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

        self.rppg_line, = self.ax_rppg.plot([], [], color='#FF6B6B')
        self.resp_line, = self.ax_resp.plot([], [], color='#6BCBFF')

        self.cap = None
        self.timer = QTimer(self) 
        self.timer.timeout.connect(self.update_frame)

        self.inference_interval = 3
        self.frame_count_for_inference = 0
        self.last_face_detection_result = None
        self.last_pose_detection_result = None

        self.process_interval = self.fps // 2
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        self.last_filtered_rppg = []
        self.last_filtered_resp = []

        self.ui.start_button.clicked.connect(self.start_processing)
        self.ui.end_button.clicked.connect(self.end_processing)
        self.ui.end_button.setEnabled(False)

        self.video_label.setText("Tekan START untuk memulai feed kamera")

    def start_processing(self):
        if self.face_detector is None or self.pose_landmarker is None:
            self.ui.video_label.setText("Model tidak termuat. Proses tidak dapat dimulai.")
            print("Percobaan memulai proses namun model tidak termuat.")
            return

        if self.cap is None:
            self.cap = cv2.VideoCapture(0) 
        
        if not self.cap.isOpened():
            self.ui.video_label.setText("Error: Tidak dapat membuka webcam!")
            self.cap = None
            return

        self.timer.start(int(1000.0 / self.fps))
        self.ui.start_button.setEnabled(False)
        self.ui.end_button.setEnabled(True) 
        
        self.rppg_signal.clear()
        self.resp_signal.clear()
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        self.last_filtered_rppg = []
        self.last_filtered_resp = []
        
        self.frame_count_for_inference = 0
        self.last_face_detection_result = None
        self.last_pose_detection_result = None
        
        # Placeholder frame awal untuk GUI
        placeholder_height = self.video_label.height() if self.video_label.height() > 10 else 480
        placeholder_width = self.video_label.width() if self.video_label.width() > 10 else 640
        blank_frame = np.zeros((placeholder_height, placeholder_width, 3), dtype=np.uint8)

        self._update_gui_plots_and_labels(
            blank_frame, [], [], 0.0, 0.0, force_plot_update=True
        )
        if hasattr(self.ui, '_apply_styles') and callable(self.ui._apply_styles):
            self.ui._apply_styles()
        print("Proses dimulai.")

    def end_processing(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.ui.start_button.setEnabled(True)
        self.ui.end_button.setEnabled(False)
        
        self.video_label.setText("Feed Kamera Berakhir. Tekan START.")
        current_stylesheet = self.video_label.styleSheet()
        if "QLabel#VideoLabel" in current_stylesheet: # Hanya jika ada style spesifik
             self.video_label.setStyleSheet(current_stylesheet.split("QLabel#VideoLabel")[0] + " QLabel#VideoLabel { background-color: black; color:white; qproperty-alignment: AlignCenter; }")
        else: # Fallback jika tidak ada style spesifik
             self.video_label.setStyleSheet("background-color: black; color:white; qproperty-alignment: AlignCenter;")


        self.hr_label.setText("-- BPM") 
        self.rr_label.setText("-- Breaths/min") 
        
        self.last_filtered_rppg = []
        self.last_filtered_resp = []
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        
        self.rppg_line.set_data([], [])
        self.resp_line.set_data([], [])
        
        if hasattr(self.ui, '_apply_styles') and callable(self.ui._apply_styles):
            self.ui._apply_styles() 
        self.canvas_rppg.draw_idle() 
        self.canvas_resp.draw_idle() 
        print("Proses dihentikan.")

    def _preprocess_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        ret, frame = self.cap.read() 
        if not ret:
            return None, None, None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return frame, rgb_frame, mp_image

    def _process_rppg_signal(self, frame_for_drawing_and_signal, face_result_to_use):
        if self.face_detector is None or face_result_to_use is None:
            return

        if face_result_to_use.detections:
            detection = face_result_to_use.detections[0] 
            bbox = detection.bounding_box
            frame_h, frame_w, _ = frame_for_drawing_and_signal.shape

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)
            
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(0, min(w, frame_w - x))
            h = max(0, min(h, frame_h - y))

            if w > 0 and h > 0 :
                forehead_x = int(x + w * 0.25) 
                forehead_y = int(y + h * 0.05)
                forehead_w = int(w * 0.5)
                forehead_h = int(h * 0.20)

                forehead_x = max(0, min(forehead_x, frame_w - 1))
                forehead_y = max(0, min(forehead_y, frame_h - 1))
                forehead_w = max(0, min(forehead_w, frame_w - forehead_x))
                forehead_h = max(0, min(forehead_h, frame_h - forehead_y))

                if forehead_w > 0 and forehead_h > 0:
                    cv2.rectangle(frame_for_drawing_and_signal, (forehead_x, forehead_y), 
                                  (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 255), 1)
                    
                    # Memanggil fungsi dari utils.signal_processing
                    rppg_value = extract_rppg_signal(frame_for_drawing_and_signal, 
                                                     (forehead_x, forehead_y, forehead_w, forehead_h))

                    if rppg_value is not None:
                        self.rppg_signal.append(rppg_value) 
                        if len(self.rppg_signal) > self.frame_buffer_limit:
                            self.rppg_signal.pop(0) 

    def _process_respiration_signal(self, frame_for_drawing, pose_result_to_use):
        if self.pose_landmarker is None or pose_result_to_use is None:
            return

        if pose_result_to_use.pose_landmarks:
            landmarks = pose_result_to_use.pose_landmarks[0]
            h_img, w_img, _ = frame_for_drawing.shape

            try:
                if len(landmarks) > max(11,12) and \
                   hasattr(landmarks[11], 'visibility') and landmarks[11].visibility > 0.5 and \
                   hasattr(landmarks[12], 'visibility') and landmarks[12].visibility > 0.5:
                    rs_landmark = landmarks[12] 
                    ls_landmark = landmarks[11]

                    y1_r = int(rs_landmark.y * h_img)
                    y1_l = int(ls_landmark.y * h_img)
                    
                    cv2.circle(frame_for_drawing, (int(rs_landmark.x * w_img), y1_r), 3, (255,0,0), -1) 
                    cv2.circle(frame_for_drawing, (int(ls_landmark.x * w_img), y1_l), 3, (0,255,0), -1) 

                    avg_y_shoulder = np.mean([y1_r, y1_l]) 
                    self.resp_signal.append(-avg_y_shoulder)
                    if len(self.resp_signal) > self.frame_buffer_limit:
                        self.resp_signal.pop(0) 
            except (IndexError, AttributeError) as e:
                print(f"Peringatan: Gagal memproses landmark bahu: {e}")
                pass

    def _filter_and_calculate_rates(self):
        filtered_rppg_signal = []
        current_hr = 0.0
        if len(self.rppg_signal) >= self.min_signal_length:
            try:
                if not (np.array_equal(self.rppg_b, [1]) and np.array_equal(self.rppg_a, [1])):
                    padlen_rppg = min(self.min_signal_length - 1, len(self.rppg_signal) - 1)
                    if padlen_rppg > 0 :
                        filtered_rppg_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal, padlen=padlen_rppg).tolist()
                    else: 
                        filtered_rppg_signal = list(self.rppg_signal)
                else:
                    filtered_rppg_signal = list(self.rppg_signal)
                current_hr = calculate_rate_from_fft(filtered_rppg_signal, self.fps, self.rppg_lowcut, self.rppg_highcut) 
            except ValueError: 
                filtered_rppg_signal = list(self.rppg_signal)
        else:
            filtered_rppg_signal = list(self.rppg_signal)

        filtered_resp_signal = []
        current_rr = 0.0
        if len(self.resp_signal) >= self.min_signal_length:
            try:
                if not (np.array_equal(self.resp_b, [1]) and np.array_equal(self.resp_a, [1])): 
                    padlen_resp = min(self.min_signal_length - 1, len(self.resp_signal) - 1)
                    if padlen_resp > 0:
                        filtered_resp_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal, padlen=padlen_resp).tolist()
                    else:
                        filtered_resp_signal = list(self.resp_signal)
                else:
                    filtered_resp_signal = list(self.resp_signal)
                current_rr = calculate_rate_from_fft(filtered_resp_signal, self.fps, self.resp_lowcut, self.resp_highcut) 
            except ValueError:
                filtered_resp_signal = list(self.resp_signal)
        else:
            filtered_resp_signal = list(self.resp_signal)
            
        return filtered_rppg_signal, filtered_resp_signal, current_hr, current_rr

    def _update_gui_plots_and_labels(self, frame_processed, filtered_rppg, filtered_resp, hr, rr, force_plot_update=False):
        if filtered_rppg or force_plot_update:
            self.rppg_line.set_ydata(filtered_rppg)
            self.rppg_line.set_xdata(range(len(filtered_rppg)))
            self.ax_rppg.relim()
            self.ax_rppg.autoscale_view(True,True,True)
            self.canvas_rppg.draw_idle() 

        if filtered_resp or force_plot_update:
            self.resp_line.set_ydata(filtered_resp)
            self.resp_line.set_xdata(range(len(filtered_resp)))
            self.ax_resp.relim()
            self.ax_resp.autoscale_view(True,True,True)
            self.canvas_resp.draw_idle() 
        
        if force_plot_update and hasattr(self.ui, '_apply_styles') and callable(self.ui._apply_styles):
             self.ui._apply_styles()

        self.hr_label.setText(f"{hr:.0f} BPM" if hr > 0 else "-- BPM") 
        self.rr_label.setText(f"{rr:.0f} Breaths/min" if rr > 0 else "-- Breaths/min") 

        if frame_processed is not None and frame_processed.size > 0 :
            try:
                display_frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                h, w, ch = display_frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(display_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except cv2.error:
                self.video_label.setText("Error Frame")
        elif frame_processed is None: 
            pass
        else: # Jika frame_processed adalah array kosong atau invalid
            self.video_label.setText("Processing...")

    def update_frame(self): 
        original_frame, rgb_frame, mp_image = self._preprocess_frame() 
        
        if original_frame is None:
            self._update_gui_plots_and_labels(None, 
                                              self.last_filtered_rppg, self.last_filtered_resp, 
                                              self.last_processed_hr, self.last_processed_rr)
            return

        frame_to_display = original_frame.copy()

        run_inference_this_frame = (self.frame_count_for_inference % self.inference_interval == 0)
        self.frame_count_for_inference += 1

        if run_inference_this_frame and mp_image:
            if self.face_detector:
                try:
                    self.last_face_detection_result = self.face_detector.detect(mp_image)
                except Exception:
                    self.last_face_detection_result = None
            if self.pose_landmarker:
                try:
                    self.last_pose_detection_result = self.pose_landmarker.detect(mp_image)
                except Exception:
                    self.last_pose_detection_result = None
        
        if self.last_face_detection_result:
             self._process_rppg_signal(frame_to_display, self.last_face_detection_result)
        if self.last_pose_detection_result:
             self._process_respiration_signal(frame_to_display, self.last_pose_detection_result)
        
        self.frames_since_last_process += 1
        
        plot_data_updated_this_cycle = False
        if self.frames_since_last_process >= self.process_interval:
            self.frames_since_last_process = 0
            if len(self.rppg_signal) >= self.min_signal_length or \
               len(self.resp_signal) >= self.min_signal_length:
                
                filtered_rppg, filtered_resp, current_hr, current_rr = self._filter_and_calculate_rates()
                
                self.last_filtered_rppg = filtered_rppg
                self.last_filtered_resp = filtered_resp
                self.last_processed_hr = current_hr if current_hr is not None else 0.0
                self.last_processed_rr = current_rr if current_rr is not None else 0.0
                plot_data_updated_this_cycle = True
        
        self._update_gui_plots_and_labels(
            frame_to_display,
            self.last_filtered_rppg,
            self.last_filtered_resp,
            self.last_processed_hr,
            self.last_processed_rr,
            force_plot_update=plot_data_updated_this_cycle
        )

    def closeEvent(self, event): 
        self.end_processing()
        print("Aplikasi ditutup.")
        event.accept() 

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    
    if not os.path.exists("models"):
        try:
            os.makedirs("models")
            print("Folder 'models' dibuat. Harap letakkan file model MediaPipe di dalamnya.")
        except OSError as e:
            print(f"Gagal membuat folder 'models': {e}")
            sys.exit(1)
            
    window = MainWindow()
    
    # Hanya jalankan aplikasi jika model berhasil dimuat
    if window.face_detector and window.pose_landmarker:
        window.show()
        sys.exit(app.exec_())
    else:
        print("Gagal memuat model MediaPipe. Aplikasi tidak dapat dijalankan sepenuhnya.")
        window.show()