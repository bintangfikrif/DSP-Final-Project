import sys
import cv2
import numpy as np
import mediapipe as mp # Untuk mp.Image
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import scipy.signal as signal
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Import kelas UI dan fungsi pemrosesan sinyal dari folder utils
from utils.gui import HealthTrackerUI 
from utils.signal_processing import extract_rppg_signal, butter_bandpass, calculate_rate_from_fft

def draw_landmarks_on_image(rgb_image, detection_result):
    from mediapipe.python.solutions import drawing_utils as mp_drawing # Import utilitas menggambar dari MediaPipe
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Iterasi melalui setiap set landmark pose yang terdeteksi
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        # Konversi format landmark untuk kompatibilitas dengan fungsi drawing_utils
        pose_landmarks_proto = mp_vision.PoseLandmarkerResult(pose_landmarks=pose_landmarks).pose_landmarks[0] 
        
        # Gambar landmark dan koneksinya pada gambar
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks_proto, # Membutuhkan list dari NormalizedLandmark
            mp.solutions.pose.POSE_CONNECTIONS, # Gunakan koneksi pose standar
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), # Spesifikasi titik landmark
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)  # Spesifikasi garis koneksi
        )
    return annotated_image

class MainWindow(QMainWindow):
    """
    Kelas utama untuk aplikasi Realtime Health Tracker.
    Mengelola logika aplikasi, pemrosesan video, dan interaksi dengan GUI.
    """
    def __init__(self):
        """
        Konstruktor untuk MainWindow.
        Menginisialisasi UI, model MediaPipe, parameter sinyal, dan timer.
        """
        super().__init__()
        self.setWindowTitle("Realtime Health Tracker - Engine")

        # Membuat instance dari UI yang telah didesain
        self.ui = HealthTrackerUI() 
        self.setCentralWidget(self.ui) # Menetapkan UI sebagai widget sentral dari QMainWindow
        self.setMinimumSize(1000, 600) 

        # --- Path Model MediaPipe Tasks ---
        self.face_model_path = "models/blaze_face_short_range.tflite" 
        self.pose_model_path = "models/pose_landmarker.task" 

        # --- Inisialisasi MediaPipe Tasks ---
        try:
            # Inisialisasi Face Detector
            face_base_options = mp_python.BaseOptions(model_asset_path=self.face_model_path)
            face_options = mp_vision.FaceDetectorOptions(
                base_options=face_base_options,
                running_mode=mp_vision.RunningMode.IMAGE, # Proses gambar individual
                min_detection_confidence=0.5 # Tingkat kepercayaan minimum untuk deteksi
            )
            self.face_detector = mp_vision.FaceDetector.create_from_options(face_options)

            # Inisialisasi Pose Landmarker
            pose_base_options = mp_python.BaseOptions(model_asset_path=self.pose_model_path)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=mp_vision.RunningMode.IMAGE, # Proses gambar individual
                num_poses=1, # Deteksi satu pose saja
                min_pose_detection_confidence=0.5, # Tingkat kepercayaan minimum
                min_tracking_confidence=0.5 
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            print("Model MediaPipe berhasil dimuat.")

        except Exception as e:
            print(f"Error saat memuat model MediaPipe: {e}")
            self.face_detector = None
            self.pose_landmarker = None
            self.ui.video_label.setText(f"Error memuat model: {e}\nPastikan file model ada di folder 'models'.")

        # --- Parameter Sinyal ---
        self.fps = 30 # Frame per detik untuk pemrosesan
        self.min_signal_length = int(2 * self.fps) # Panjang sinyal minimum untuk analisis FFT

        # Parameter filter bandpass untuk rPPG (detak jantung)
        self.rppg_lowcut = 0.75 
        self.rppg_highcut = 4.0 
        self.rppg_b, self.rppg_a = butter_bandpass(self.rppg_lowcut, self.rppg_highcut, self.fps) 

        # Parameter filter bandpass untuk respirasi
        self.resp_lowcut = 0.1 
        self.resp_highcut = 0.7 
        self.resp_b, self.resp_a = butter_bandpass(self.resp_lowcut, self.resp_highcut, self.fps) 

        # Buffer untuk menyimpan sinyal rPPG dan respirasi
        self.rppg_signal = [] 
        self.resp_signal = [] 
        self.frame_buffer_limit = int(10 * self.fps) # Buffer untuk 10 detik sinyal

        # --- Referensi Elemen UI (dari self.ui) ---
        self.video_label = self.ui.video_label
        self.hr_label = self.ui.hr_value_label
        self.rr_label = self.ui.rr_value_label
        self.ax_rppg = self.ui.ax_rppg
        self.canvas_rppg = self.ui.hr_canvas
        self.ax_resp = self.ui.ax_resp
        self.canvas_resp = self.ui.rr_canvas

        # --- Inisialisasi Kamera dan Timer ---
        self.cap = None # Objek VideoCapture, diinisialisasi saat start
        self.timer = QTimer(self) 
        self.timer.timeout.connect(self.update_frame) # Hubungkan timeout timer ke metode update_frame

        # --- Variabel untuk Buffering/Smoothing Sinyal ---
        self.process_interval = self.fps // 2
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        self.last_filtered_rppg = []
        self.last_filtered_resp = []

        # --- Hubungkan Tombol UI ke Metode ---
        self.ui.start_button.clicked.connect(self.start_processing)
        self.ui.end_button.clicked.connect(self.end_processing)
        self.ui.end_button.setEnabled(False) # Tombol "END" dinonaktifkan pada awalnya

        self.ui.video_label.setText("Tekan START untuk memulai feed kamera")

    def start_processing(self):
        """
        Memulai proses akuisisi video dan analisis sinyal.
        Menginisialisasi kamera dan memulai timer.
        """
        if self.face_detector is None or self.pose_landmarker is None:
            self.ui.video_label.setText("Model tidak termuat. Proses tidak dapat dimulai.")
            print("Percobaan memulai proses namun model tidak termuat.")
            return

        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # Inisialisasi kamera webcam
        
        if not self.cap.isOpened():
            self.ui.video_label.setText("Error: Tidak dapat membuka webcam!")
            self.cap = None
            return

        self.timer.start(int(1000.0 / self.fps)) # Memulai timer sesuai fps
        self.ui.start_button.setEnabled(False) # Nonaktifkan tombol "START"
        self.ui.end_button.setEnabled(True)   # Aktifkan tombol "END"
        
        # Reset buffer dan state untuk smoothing
        self.rppg_signal.clear()
        self.resp_signal.clear()
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        self.last_filtered_rppg = []
        self.last_filtered_resp = []
        
        # Clear plots and labels initially
        self._update_gui_plots_and_labels(
            np.zeros((self.video_label.height() if self.video_label.height() > 10 else 480, 
                      self.video_label.width() if self.video_label.width() > 10 else 640, 3), dtype=np.uint8), # Placeholder
            self.last_filtered_rppg,
            self.last_filtered_resp,
            self.last_processed_hr,
            self.last_processed_rr
        )
        print("Proses dimulai.")

    def end_processing(self):
        """
        Menghentikan proses akuisisi video dan analisis sinyal.
        Menghentikan timer dan melepaskan kamera.
        """
        self.timer.stop() # Hentikan timer
        if self.cap is not None:
            self.cap.release() # Lepaskan kamera
            self.cap = None
        self.ui.start_button.setEnabled(True)  # Aktifkan tombol "START"
        self.ui.end_button.setEnabled(False) # Nonaktifkan tombol "END"
        
        # Reset tampilan UI
        self.video_label.setText("Feed Kamera Berakhir. Tekan START.")
        self.video_label.setStyleSheet(self.ui.styleSheet() + " QLabel#VideoLabel { background-color: black; }") 
        self.hr_label.setText("-- BPM") 
        self.rr_label.setText("-- Breaths/min") 
        
        # Clear last processed data as well 
        self.last_filtered_rppg = []
        self.last_filtered_resp = []
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        
        self.ax_rppg.clear() 
        self.ax_resp.clear() 
        self.ui._apply_styles() # Terapkan kembali style untuk mereset judul plot
        self.canvas_rppg.draw() 
        self.canvas_resp.draw() 
        print("Proses dihentikan.")

    def _preprocess_frame(self):
        """
        Membaca frame dari kamera, melakukan flip, dan konversi ke format RGB serta MediaPipe Image.
        Mengembalikan:
            frame (numpy.ndarray): Frame asli BGR.
            rgb_frame (numpy.ndarray): Frame dalam format RGB.
            mp_image (mediapipe.Image): Frame dalam format MediaPipe Image.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        ret, frame = self.cap.read() 
        if not ret:
            return None, None, None
        frame = cv2.flip(frame, 1) # Flip frame secara horizontal
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Konversi BGR ke RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) # Konversi ke format MediaPipe Image
        return frame, rgb_frame, mp_image

    def _process_rppg_signal(self, frame_to_draw_on, mp_image_input):
        """
        Memproses frame untuk ekstraksi sinyal rPPG menggunakan FaceDetector dari MediaPipe Tasks.
        Args:
            frame_to_draw_on (numpy.ndarray): Frame BGR untuk menggambar ROI.
            mp_image_input (mediapipe.Image): Frame input untuk deteksi wajah.
        Returns:
            float or None: Nilai rPPG yang diekstraksi, atau None jika tidak ada wajah terdeteksi.
        """
        if self.face_detector is None:
            return None

        face_detector_result = self.face_detector.detect(mp_image_input) # Deteksi wajah
        rppg_value = None

        if face_detector_result.detections:
            for detection in face_detector_result.detections: # Iterasi melalui semua deteksi wajah
                bbox = detection.bounding_box
                
                ih, iw = mp_image_input.height, mp_image_input.width

                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)
                
                frame_h, frame_w, _ = frame_to_draw_on.shape
                x = max(0, min(x, frame_w -1))
                y = max(0, min(y, frame_h -1))
                w = max(0, min(w, frame_w - x))
                h = max(0, min(h, frame_h - y))

                if w > 0 and h > 0 :
                    forehead_x = int(x + w * 0.15) 
                    forehead_y = int(y + h * 0.05)
                    forehead_w = int(w * 0.7)
                    forehead_h = int(h * 0.25)

                    forehead_x = max(0, min(forehead_x, frame_w -1))
                    forehead_y = max(0, min(forehead_y, frame_h -1))
                    forehead_w = max(0, min(forehead_w, frame_w - forehead_x))
                    forehead_h = max(0, min(forehead_h, frame_h - forehead_y))

                    if forehead_w > 0 and forehead_h > 0:
                        cv2.rectangle(frame_to_draw_on, (forehead_x, forehead_y), 
                                      (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 255), 2)
                        
                        rppg_value = extract_rppg_signal(frame_to_draw_on, 
                                                         (forehead_x, forehead_y, forehead_w, forehead_h))

                        if rppg_value is not None:
                            self.rppg_signal.append(rppg_value) 
                            if len(self.rppg_signal) > self.frame_buffer_limit: # Jaga ukuran buffer
                                self.rppg_signal.pop(0) 
                break 
        return rppg_value


    def _process_respiration_signal(self, frame_to_draw_on, mp_image_input):
        """
        Memproses frame untuk ekstraksi sinyal respirasi menggunakan PoseLandmarker dari MediaPipe Tasks.
        Fokus pada landmark bahu kiri (11) dan kanan (12).
        Args:
            frame_to_draw_on (numpy.ndarray): Frame BGR untuk menggambar landmark.
            mp_image_input (mediapipe.Image): Frame input untuk deteksi pose.
        """
        if self.pose_landmarker is None:
            return

        pose_landmarker_result = self.pose_landmarker.detect(mp_image_input) # Deteksi pose
        
        if pose_landmarker_result.pose_landmarks:
            landmarks = pose_landmarker_result.pose_landmarks[0]
            
            h_img, w_img, _ = frame_to_draw_on.shape

            try:
                rs_landmark = landmarks[12] # Bahu kanan
                ls_landmark = landmarks[11] # Bahu kiri

                y1_r = int(rs_landmark.y * h_img)
                y1_l = int(ls_landmark.y * h_img)
                
                cv2.circle(frame_to_draw_on, (int(rs_landmark.x * w_img), y1_r), 5, (255,0,0), -1) 
                cv2.circle(frame_to_draw_on, (int(ls_landmark.x * w_img), y1_l), 5, (0,255,0), -1) 

                avg_y_shoulder = np.mean([y1_r, y1_l]) 
                self.resp_signal.append(-avg_y_shoulder)
                if len(self.resp_signal) > self.frame_buffer_limit: # Jaga ukuran buffer
                    self.resp_signal.pop(0) 
            except IndexError:
                print("Error mengakses landmark bahu. Periksa output model atau indeks landmark.")
            except AttributeError: 
                 print("Error: Objek landmark tidak memiliki atribut x atau y.")


    def _filter_and_calculate_rates(self):
        """
        Memfilter sinyal rPPG dan respirasi menggunakan filter bandpass Butterworth.
        Kemudian menghitung HR dan RR menggunakan FFT.
        Returns:
            tuple: (filtered_rppg_signal, filtered_resp_signal, current_hr, current_rr)
        """
        filtered_rppg_signal = self.rppg_signal 
        if len(self.rppg_signal) >= self.min_signal_length: # Geq min_signal_length for stability
            # Terapkan filter bandpass ke sinyal rPPG
            filtered_rppg_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal).tolist() 
        else: # If signal is too short, return a copy of the original to avoid modifying it if it's self.rppg_signal
            filtered_rppg_signal = list(self.rppg_signal)


        filtered_resp_signal = self.resp_signal 
        if len(self.resp_signal) >= self.min_signal_length: # Geq min_signal_length for stability
            # Terapkan filter bandpass ke sinyal respirasi
            filtered_resp_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal).tolist() 
        else:
            filtered_resp_signal = list(self.resp_signal)

        current_hr = calculate_rate_from_fft(filtered_rppg_signal, self.fps, self.rppg_lowcut, self.rppg_highcut) 
        current_rr = calculate_rate_from_fft(filtered_resp_signal, self.fps, self.resp_lowcut, self.resp_highcut) 
        return filtered_rppg_signal, filtered_resp_signal, current_hr, current_rr

    def _update_gui_plots_and_labels(self, frame_processed, filtered_rppg, filtered_resp, hr, rr):
        """
        Memperbarui elemen-elemen GUI: plot sinyal, label HR dan RR, serta tampilan video.
        Args:
            frame_processed (numpy.ndarray): Frame yang telah diproses (dengan ROI/landmark) untuk ditampilkan.
            filtered_rppg (list): Sinyal rPPG yang telah difilter.
            filtered_resp (list): Sinyal respirasi yang telah difilter.
            hr (float): Nilai detak jantung saat ini.
            rr (float): Nilai laju pernapasan saat ini.
        """
        # Perbarui plot rPPG
        self.ax_rppg.clear() 
        if filtered_rppg: # Only plot if there's data
            self.ax_rppg.plot(filtered_rppg, color='#FF6B6B')
        self.canvas_rppg.draw() 

        # Perbarui plot respirasi
        self.ax_resp.clear() 
        if filtered_resp: # Only plot if there's data
            self.ax_resp.plot(filtered_resp, color='#6BCBFF')
        self.canvas_resp.draw() 
        
        self.ui._apply_styles() 

        # Perbarui label HR
        if hr > 0: 
            self.hr_label.setText(f"{hr:.0f} BPM") 
        else: 
            self.hr_label.setText("-- BPM") 

        # Perbarui label RR
        if rr > 0: 
            self.rr_label.setText(f"{rr:.0f} Breaths/min") 
        else: 
            self.rr_label.setText("-- Breaths/min") 

        # Tampilkan frame video yang telah diproses
        if frame_processed is not None and frame_processed.size > 0 :
            display_frame = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
             # Fallback or clear if frame_processed is not valid
            self.video_label.setText("Processing...") # Or clear pixmap
            self.video_label.setStyleSheet("QLabel#VideoLabel { background-color: black; }")

    def update_frame(self): 
        """
        Metode utama yang dipanggil oleh QTimer untuk setiap frame.
        Mengkoordinasikan pra-pemrosesan frame, ekstraksi sinyal, filtering,
        perhitungan laju, dan pembaruan GUI.
        """
        original_frame, rgb_frame, mp_image = self._preprocess_frame() 
        if original_frame is None or mp_image is None:
            blank_frame_for_display = np.zeros((self.video_label.height() if self.video_label.height() > 10 else 480, 
                                               self.video_label.width() if self.video_label.width() > 10 else 640, 3), dtype=np.uint8)
            self._update_gui_plots_and_labels(blank_frame_for_display, 
                                             self.last_filtered_rppg, self.last_filtered_resp, 
                                             self.last_processed_hr, self.last_processed_rr)
            return

        frame_to_display = original_frame.copy()

        # Selalu ekstrak sinyal untuk menjaga buffer tetap update
        self._process_rppg_signal(frame_to_display, mp_image)
        self._process_respiration_signal(frame_to_display, mp_image)
        self.frames_since_last_process += 1

        # Proses sinyal dan update rate hanya pada interval yang ditentukan
        if self.frames_since_last_process >= self.process_interval:
            self.frames_since_last_process = 0
            
            # Lakukan filtering dan perhitungan rate
            if len(self.rppg_signal) >= self.min_signal_length and len(self.resp_signal) >= self.min_signal_length:
                filtered_rppg, filtered_resp, current_hr, current_rr = self._filter_and_calculate_rates()
                
                # Simpan hasil proses terakhir
                self.last_filtered_rppg = filtered_rppg
                self.last_filtered_resp = filtered_resp
                self.last_processed_hr = current_hr
                self.last_processed_rr = current_rr

        # Selalu update GUI:
        self._update_gui_plots_and_labels(
            frame_to_display,
            self.last_filtered_rppg,
            self.last_filtered_resp,
            self.last_processed_hr,
            self.last_processed_rr
        )

    def closeEvent(self, event): 
        """
        Menangani event penutupan jendela aplikasi.
        Memastikan semua sumber daya dilepaskan dengan benar.
        """
        self.end_processing() # Hentikan proses dan lepaskan kamera
        print("Aplikasi ditutup.")
        event.accept() 

if __name__ == "__main__":
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())