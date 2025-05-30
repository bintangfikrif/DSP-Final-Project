import sys
import cv2
import numpy as np
import mediapipe as mp 
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import os

try:
    from utils.gui import HealthTrackerUI
    from utils.signal_processing import HealthAnalyzer # Kelas utama untuk pemrosesan sinyal
except ImportError as e:
    print(f"Penting: Gagal mengimpor modul dari folder 'utils'. Pastikan file ada dan benar: {e}")
    print("Harap buat file 'utils/gui.py' dan 'utils/signal_processing.py' sesuai kebutuhan.")
    sys.exit(1)

def draw_landmarks_on_image(rgb_image, detection_result):
    """Menggambar landmark pose pada gambar untuk debugging."""
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS if hasattr(mp.solutions, 'pose') else None,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )
    return annotated_image

class MainWindow(QMainWindow):
    """
    Kelas utama window aplikasi yang mengatur GUI, input video,
    dan orkestrasi pemrosesan sinyal menggunakan HealthAnalyzer.
    """
    def __init__(self):
        """
        Inisialisasi MainWindow, UI, HealthAnalyzer, dan parameter aplikasi.
        """
        super().__init__()
        self.setWindowTitle("Realtime rPPG & Respiration Rate Tracker by BEE Team")
        self.ui = HealthTrackerUI() # Inisialisasi Antarmuka Pengguna
        self.setCentralWidget(self.ui)
        self.setMinimumSize(1000, 600)

        # Konfigurasi path model dan FPS
        self.face_model_path_config = "models/blaze_face_short_range.tflite"
        self.pose_model_path_config = "models/pose_landmarker.task"
        self.fps_config = 30 # FPS target untuk kamera dan pemrosesan

        self.analyzer = None 
        try:
            # Inisialisasi HealthAnalyzer dengan konfigurasi yang ditentukan
            self.analyzer = HealthAnalyzer(
                face_model_path=self.face_model_path_config,
                pose_model_path=self.pose_model_path_config,
                fps=self.fps_config
            )
            if not self.analyzer.has_models():
                 raise RuntimeError("Model MediaPipe gagal dimuat di HealthAnalyzer.")
            print("HealthAnalyzer berhasil diinisialisasi dengan model.")
        except Exception as e:
            print(f"Error saat inisialisasi HealthAnalyzer: {e}")
            if hasattr(self.ui, 'video_label') and self.ui.video_label:
                self.ui.video_label.setText(f"Error init Analyzer: {e}\nPastikan file model ada.")
            # self.analyzer akan tetap None, start_processing akan dicegah

        # Referensi ke elemen UI untuk kemudahan akses
        self.video_label = self.ui.video_label
        self.hr_label = self.ui.hr_value_label
        self.rr_label = self.ui.rr_value_label
        self.ax_rppg = self.ui.ax_rppg
        self.canvas_rppg = self.ui.hr_canvas
        self.ax_resp = self.ui.ax_resp
        self.canvas_resp = self.ui.rr_canvas

        # Inisialisasi garis plot untuk sinyal rPPG dan pernapasan
        self.rppg_line, = self.ax_rppg.plot([], [], color='#FF6B6B')
        self.resp_line, = self.ax_resp.plot([], [], color='#6BCBFF')

        self.cap = None # Objek VideoCapture
        self.timer = QTimer(self) # Timer untuk memicu update_frame secara periodik
        self.timer.timeout.connect(self.update_frame)

        # Kontrol frekuensi inferensi model (setiap N frame)
        self.inference_interval = 3
        self.frame_count_for_inference = 0
        self.last_face_detection_result = None # Menyimpan hasil deteksi wajah terakhir
        self.last_pose_detection_result = None # Menyimpan hasil deteksi pose terakhir

        # Kontrol frekuensi pemrosesan sinyal (filtering & FFT, setiap M frame)
        self.process_interval = self.fps_config // 2 # Setengah detik
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0 # Menyimpan nilai HR terakhir yang valid
        self.last_processed_rr = 0.0 # Menyimpan nilai RR terakhir yang valid
        self.last_filtered_rppg = [] # Menyimpan data plot rPPG terakhir
        self.last_filtered_resp = [] # Menyimpan data plot pernapasan terakhir

        # Hubungkan tombol Start/End ke metode terkait
        self.ui.start_button.clicked.connect(self.start_processing)
        self.ui.end_button.clicked.connect(self.end_processing)
        self.ui.end_button.setEnabled(False) # Awalnya tombol End nonaktif
        self.video_label.setText("Tekan START untuk memulai feed kamera")

    def start_processing(self):
        """
        Memulai proses pengambilan video dan analisis sinyal.
        Mengaktifkan kamera, timer, dan mereset state.
        """
        # Cek apakah HealthAnalyzer dan modelnya siap
        if self.analyzer is None or not self.analyzer.has_models():
            self.ui.video_label.setText("Analyzer/Model tidak termuat. Proses tidak dapat dimulai.")
            print("Percobaan memulai proses namun Analyzer/model tidak termuat.")
            return

        # Inisialisasi VideoCapture jika belum ada
        if self.cap is None:
            self.cap = cv2.VideoCapture(0) # Buka webcam default
        if not self.cap.isOpened():
            self.ui.video_label.setText("Error: Tidak dapat membuka webcam!")
            self.cap = None
            return

        self.timer.start(int(1000.0 / self.fps_config)) # Mulai timer sesuai FPS
        self.ui.start_button.setEnabled(False) # Nonaktifkan tombol Start
        self.ui.end_button.setEnabled(True)   # Aktifkan tombol End
        
        if self.analyzer: # Bersihkan buffer sinyal di HealthAnalyzer
            self.analyzer.clear_buffers()

        # Reset state variabel
        self.frames_since_last_process = 0
        self.last_processed_hr = 0.0
        self.last_processed_rr = 0.0
        self.last_filtered_rppg = []
        self.last_filtered_resp = []
        self.frame_count_for_inference = 0
        self.last_face_detection_result = None
        self.last_pose_detection_result = None
        
        # Tampilkan frame kosong sebagai placeholder awal di GUI
        placeholder_height = self.video_label.height() if self.video_label.height() > 10 else 480
        placeholder_width = self.video_label.width() if self.video_label.width() > 10 else 640
        blank_frame = np.zeros((placeholder_height, placeholder_width, 3), dtype=np.uint8)
        self._update_gui_plots_and_labels(blank_frame, [], [], 0.0, 0.0, force_plot_update=True)
        if hasattr(self.ui, '_apply_styles'): self.ui._apply_styles() # Terapkan style jika ada
        print("Proses dimulai.")

    def end_processing(self):
        """
        Menghentikan proses pengambilan video dan analisis sinyal.
        Melepaskan kamera dan menghentikan timer.
        """
        self.timer.stop() # Hentikan timer
        if self.cap is not None:
            self.cap.release() # Lepaskan resource kamera
            self.cap = None
        self.ui.start_button.setEnabled(True) # Aktifkan tombol Start
        self.ui.end_button.setEnabled(False)  # Nonaktifkan tombol End
        
        self.video_label.setText("Feed Kamera Berakhir. Tekan START.")
        # (Styling video_label lainnya tetap sama)
        self.hr_label.setText("-- BPM") # Reset label HR
        self.rr_label.setText("-- Breaths/min") # Reset label RR
        
        # Reset data plot terakhir
        self.last_filtered_rppg, self.last_filtered_resp = [], []
        self.last_processed_hr, self.last_processed_rr = 0.0, 0.0
        self.rppg_line.set_data([], []); self.resp_line.set_data([], []) # Kosongkan plot
        
        if hasattr(self.ui, '_apply_styles'): self.ui._apply_styles()
        self.canvas_rppg.draw_idle(); self.canvas_resp.draw_idle() # Perbarui canvas plot
        print("Proses dihentikan.")

    def _preprocess_frame(self):
        """
        Membaca frame dari kamera, melakukan flipping, konversi warna,
        dan menyiapkan mp.Image untuk MediaPipe.
        Returns:
            tuple: (frame BGR asli, frame RGB untuk MediaPipe, objek mp.Image) atau (None, None, None) jika gagal.
        """
        if self.cap is None or not self.cap.isOpened(): return None, None, None
        ret, frame = self.cap.read() # Baca frame
        if not ret: return None, None, None # Jika gagal baca frame

        frame = cv2.flip(frame, 1) # Flip horizontal agar seperti cermin
        rgb_frame_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Konversi ke RGB untuk MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_for_mp) # Buat objek mp.Image
        return frame, rgb_frame_for_mp, mp_image

    def _update_gui_plots_and_labels(self, frame_processed, filtered_rppg, filtered_resp, hr, rr, force_plot_update=False):
        """
        Memperbarui elemen GUI: video feed, plot sinyal, dan label nilai HR/RR.
        Args:
            frame_processed (numpy.ndarray): Frame video (BGR) yang akan ditampilkan.
            filtered_rppg (list): Data sinyal rPPG terfilter untuk diplot.
            filtered_resp (list): Data sinyal pernapasan terfilter untuk diplot.
            hr (float): Nilai Heart Rate (BPM).
            rr (float): Nilai Respiration Rate (Breaths/min).
            force_plot_update (bool): Paksa pembaruan plot meskipun data kosong.
        """
        # Update plot rPPG
        if filtered_rppg or force_plot_update:
            self.rppg_line.set_ydata(filtered_rppg)
            self.rppg_line.set_xdata(range(len(filtered_rppg)))
            self.ax_rppg.relim(); self.ax_rppg.autoscale_view(True,True,True)
            self.canvas_rppg.draw_idle()

        # Update plot pernapasan
        if filtered_resp or force_plot_update:
            self.resp_line.set_ydata(filtered_resp)
            self.resp_line.set_xdata(range(len(filtered_resp)))
            self.ax_resp.relim(); self.ax_resp.autoscale_view(True,True,True)
            self.canvas_resp.draw_idle()
        
        if force_plot_update and hasattr(self.ui, '_apply_styles'): self.ui._apply_styles()

        # Update label nilai HR dan RR
        self.hr_label.setText(f"{hr:.0f} BPM" if hr > 0 else "-- BPM")
        self.rr_label.setText(f"{rr:.0f} Breaths/min" if rr > 0 else "-- Breaths/min")

        # Update video feed
        if frame_processed is not None and frame_processed.size > 0 :
            try:
                # Konversi frame BGR ke RGB untuk QImage
                display_frame_for_qt = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                h, w, ch = display_frame_for_qt.shape
                bytes_per_line = ch * w
                qt_image = QImage(display_frame_for_qt.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                # Skalakan pixmap agar sesuai dengan ukuran label video
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except cv2.error as e_cv:
                print(f"Error konversi frame untuk GUI: {e_cv}")
                self.video_label.setText("Error Frame Conversion")
            except Exception as e_gui:
                print(f"Error updating video label: {e_gui}")
                self.video_label.setText("Error GUI Display")
        elif frame_processed is None: 
            pass # Jangan lakukan apa-apa jika frame tidak ada (mis., kamera berhenti)
        else: # Jika frame_processed adalah array kosong
            self.video_label.setText("Processing...")


    def update_frame(self):
        """
        Metode utama yang dipanggil oleh QTimer secara periodik.
        Mengambil frame, melakukan inferensi, memproses sinyal, dan memperbarui GUI.
        """
        if self.analyzer is None: # Cek jika analyzer gagal diinisialisasi
            if not self.timer.isActive(): # Set pesan error sekali jika timer tidak aktif
                self.video_label.setText("Health Analyzer tidak termuat. Aplikasi tidak dapat berfungsi.")
            return

        # 1. Dapatkan frame dari kamera dan lakukan pra-pemrosesan
        original_frame_bgr, _, mp_image = self._preprocess_frame() # rgb_frame_for_mp tidak dipakai langsung di sini
        
        if original_frame_bgr is None: # Jika gagal mendapatkan frame
            self._update_gui_plots_and_labels(None, self.last_filtered_rppg, self.last_filtered_resp,
                                              self.last_processed_hr, self.last_processed_rr)
            return

        # Buat salinan frame untuk digambari ROI (format BGR)
        frame_to_display_with_roi = original_frame_bgr.copy()

        # 2. Lakukan inferensi model secara berkala (tidak setiap frame)
        run_inference_this_frame = (self.frame_count_for_inference % self.inference_interval == 0)
        self.frame_count_for_inference += 1

        if run_inference_this_frame and mp_image:
            # Simpan hasil deteksi untuk digunakan pada frame berikutnya jika tidak ada inferensi baru
            self.last_face_detection_result = self.analyzer.detect_faces(mp_image)
            self.last_pose_detection_result = self.analyzer.detect_pose(mp_image)
        
        # 3. Ekstrak sinyal mentah rPPG dan pernapasan menggunakan hasil deteksi terakhir
        # HealthAnalyzer akan menggambar ROI pada frame_to_display_with_roi
        if self.last_face_detection_result:
            self.analyzer.process_rppg_from_face(frame_to_display_with_roi, self.last_face_detection_result)
        if self.last_pose_detection_result:
            self.analyzer.process_respiration_from_pose(frame_to_display_with_roi, self.last_pose_detection_result)
        
        # 4. Proses sinyal (filter & FFT) secara berkala
        self.frames_since_last_process += 1
        plot_data_updated_this_cycle = False # Flag untuk menandai apakah plot perlu di-update

        if self.frames_since_last_process >= self.process_interval:
            self.frames_since_last_process = 0
            
            # Proses sinyal rPPG dan hitung HR
            if len(self.analyzer.rppg_signal_buffer) >= self.analyzer.min_signal_length:
                filtered_rppg, current_hr = self.analyzer.filter_and_calculate_hr()
                self.last_filtered_rppg = filtered_rppg
                self.last_processed_hr = current_hr
                plot_data_updated_this_cycle = True
            else: # Jika sinyal belum cukup, tampilkan buffer mentah
                self.last_filtered_rppg = list(self.analyzer.rppg_signal_buffer)

            # Proses sinyal pernapasan dan hitung RR
            if len(self.analyzer.resp_signal_buffer) >= self.analyzer.min_signal_length:
                filtered_resp, current_rr = self.analyzer.filter_and_calculate_rr()
                self.last_filtered_resp = filtered_resp
                self.last_processed_rr = current_rr
                plot_data_updated_this_cycle = True
            else: # Jika sinyal belum cukup, tampilkan buffer mentah
                self.last_filtered_resp = list(self.analyzer.resp_signal_buffer)
        
        # 5. Update GUI dengan frame yang sudah digambari ROI dan data sinyal terbaru
        self._update_gui_plots_and_labels(
            frame_to_display_with_roi, # Frame BGR dengan ROI
            self.last_filtered_rppg,
            self.last_filtered_resp,
            self.last_processed_hr,
            self.last_processed_rr,
            force_plot_update=plot_data_updated_this_cycle # Paksa update plot jika data baru diproses
        )

    def closeEvent(self, event):
        """
        Dipanggil ketika window aplikasi ditutup.
        Memastikan proses dihentikan dengan benar.
        """
        self.end_processing()
        print("Aplikasi ditutup.")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Buat folder 'models' jika belum ada
    if not os.path.exists("models"):
        try:
            os.makedirs("models")
            print("Folder 'models' dibuat. Harap letakkan file model MediaPipe di dalamnya.")
        except OSError as e:
            print(f"Gagal membuat folder 'models': {e}")
            sys.exit(1)
            
    window = MainWindow() # Buat instance MainWindow
    
    # Hanya jalankan aplikasi jika HealthAnalyzer dan modelnya berhasil dimuat
    if window.analyzer and window.analyzer.has_models():
        window.show() # Tampilkan window
        sys.exit(app.exec_()) # Jalankan event loop aplikasi
    else:
        print("Gagal memuat HealthAnalyzer atau modelnya. Aplikasi mungkin tidak berfungsi penuh.")
        window.show()