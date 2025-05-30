import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import scipy.signal as signal
import os

def extract_rppg_signal(frame, roi):
    """
    Mengekstraksi nilai rata-rata channel hijau dari Region of Interest (ROI) pada sebuah frame.
    Args:
        frame (numpy.ndarray): Frame video input dalam format BGR.
        roi (tuple): Tuple berisi (x, y, w, h) yang mendefinisikan ROI.
    Returns:
        float or None: Nilai rata-rata channel hijau di ROI, atau None jika ROI tidak valid.
    """
    x, y, w, h = roi
    if w > 0 and h > 0:
        roi_frame = frame[y:y+h, x:x+w]
        mean_green = cv2.mean(roi_frame)[1] # Channel hijau (indeks 1 di BGR)
        return mean_green
    return None

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Mendesain koefisien filter bandpass Butterworth.
    Args:
        lowcut (float): Frekuensi cut-off bawah (Hz).
        highcut (float): Frekuensi cut-off atas (Hz).
        fs (float): Frekuensi sampling sinyal (Hz).
        order (int): Orde filter.
    Returns:
        tuple: Koefisien numerator (b) dan denominator (a) dari filter.
    """
    nyq = 0.5 * fs # Frekuensi Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def calculate_rate_from_fft(signal_values, fs, lowcut_hz, highcut_hz):
    """
    Menghitung laju (detak jantung/pernapasan) dari sinyal menggunakan FFT.
    Args:
        signal_values (list/numpy.ndarray): Nilai sinyal.
        fs (float): Frekuensi sampling (Hz).
        lowcut_hz (float): Batas bawah rentang frekuensi yang diminati (Hz).
        highcut_hz (float): Batas atas rentang frekuensi yang diminati (Hz).
    Returns:
        float: Laju yang dihitung dalam satuan per menit (mis., BPM).
    """
    if len(signal_values) < 20: # Membutuhkan panjang sinyal yang cukup untuk analisis FFT
        return 0

    N = len(signal_values)
    yf = np.fft.fft(signal_values)
    xf = np.fft.fftfreq(N, 1 / fs)

    # Ambil hanya spektrum positif dan normalisasi amplitudo
    xf_positive = xf[:N//2]
    yf_positive = 2.0/N * np.abs(yf[0:N//2])

    # Cari indeks frekuensi dalam rentang yang valid
    valid_indices = np.where((xf_positive >= lowcut_hz) & (xf_positive <= highcut_hz))[0]

    if len(valid_indices) == 0: # Tidak ada frekuensi dalam rentang valid
        return 0

    valid_yf = yf_positive[valid_indices]
    valid_xf = xf_positive[valid_indices]

    if len(valid_yf) == 0: # Pengecekan tambahan
        return 0

    # Temukan frekuensi dengan amplitudo terbesar (puncak dominan)
    dominant_peak_index_in_valid = np.argmax(valid_yf)
    dominant_frequency_hz = valid_xf[dominant_peak_index_in_valid]

    rate_per_minute = dominant_frequency_hz * 60 # Konversi Hz ke per menit
    return rate_per_minute


class HealthAnalyzer:
    """
    Kelas untuk menganalisis sinyal biologis dari input video, termasuk deteksi wajah,
    pose, ekstraksi sinyal rPPG dan pernapasan, serta perhitungan laju terkait.
    """
    def __init__(self, face_model_path="models/blaze_face_short_range.tflite",
                 pose_model_path="models/pose_landmarker.task",
                 fps=30,
                 rppg_lowcut=0.75, rppg_highcut=4.0,
                 resp_lowcut=0.1, resp_highcut=0.7,
                 min_signal_length_factor=2,
                 frame_buffer_factor=10):
        """
        Inisialisasi HealthAnalyzer.
        Args:
            face_model_path (str): Path ke model deteksi wajah MediaPipe.
            pose_model_path (str): Path ke model deteksi pose MediaPipe.
            fps (int): Frame per second dari input video.
            rppg_lowcut (float), rppg_highcut (float): Batas frekuensi untuk sinyal rPPG (Hz).
            resp_lowcut (float), resp_highcut (float): Batas frekuensi untuk sinyal pernapasan (Hz).
            min_signal_length_factor (int): Pengali FPS untuk menentukan panjang sinyal minimum.
            frame_buffer_factor (int): Pengali FPS untuk menentukan batas buffer sinyal.
        """
        self.fps = fps
        self.min_signal_length = int(min_signal_length_factor * self.fps)
        self.frame_buffer_limit = int(frame_buffer_factor * self.fps)

        self.rppg_lowcut = rppg_lowcut
        self.rppg_highcut = rppg_highcut
        self.resp_lowcut = resp_lowcut
        self.resp_highcut = resp_highcut

        self.face_detector = None
        self.pose_landmarker = None
        self._load_models(face_model_path, pose_model_path) # Muat model MediaPipe

        # Desain koefisien filter untuk rPPG dan pernapasan
        self.rppg_b, self.rppg_a = butter_bandpass(self.rppg_lowcut, self.rppg_highcut, self.fps)
        self.resp_b, self.resp_a = butter_bandpass(self.resp_lowcut, self.resp_highcut, self.fps)

        # Buffer untuk menyimpan sinyal mentah yang diekstrak
        self.rppg_signal_buffer = []
        self.resp_signal_buffer = []

    def _load_models(self, face_model_path, pose_model_path):
        """
        Memuat model deteksi wajah dan pose MediaPipe dari path yang diberikan.
        """
        try:
            # Inisialisasi FaceDetector
            if not os.path.exists(face_model_path):
                raise FileNotFoundError(f"File model wajah tidak ditemukan: {face_model_path}")
            face_base_options = mp_python.BaseOptions(model_asset_path=face_model_path)
            face_options = mp_vision.FaceDetectorOptions(
                base_options=face_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.5
            )
            self.face_detector = mp_vision.FaceDetector.create_from_options(face_options)

            # Inisialisasi PoseLandmarker
            if not os.path.exists(pose_model_path):
                 raise FileNotFoundError(f"File model pose tidak ditemukan: {pose_model_path}")
            pose_base_options = mp_python.BaseOptions(model_asset_path=pose_model_path)
            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
            print("Model MediaPipe di HealthAnalyzer berhasil dimuat.")
        except Exception as e:
            print(f"Error saat memuat model MediaPipe di HealthAnalyzer: {e}")
            raise e

    def detect_faces(self, mp_image):
        """
        Melakukan deteksi wajah pada gambar MediaPipe.
        Args:
            mp_image (mp.Image): Gambar dalam format MediaPipe.
        Returns:
            Hasil deteksi wajah dari MediaPipe atau None jika gagal.
        """
        if self.face_detector:
            try:
                return self.face_detector.detect(mp_image)
            except Exception as e:
                print(f"Error deteksi wajah: {e}")
        return None

    def detect_pose(self, mp_image):
        """
        Melakukan deteksi pose pada gambar MediaPipe.
        Args:
            mp_image (mp.Image): Gambar dalam format MediaPipe.
        Returns:
            Hasil deteksi pose dari MediaPipe atau None jika gagal.
        """
        if self.pose_landmarker:
            try:
                return self.pose_landmarker.detect(mp_image)
            except Exception as e:
                print(f"Error deteksi pose: {e}")
        return None

    def process_rppg_from_face(self, frame_for_signal, face_detection_result):
        """
        Mengekstrak sinyal rPPG dari ROI wajah (dahi, pipi kiri, pipi kanan),
        menggambar ROI pada frame, dan menambahkan rata-rata sinyal ke buffer.
        Args:
            frame_for_signal (numpy.ndarray): Frame video (BGR) untuk ekstraksi dan penggambaran.
            face_detection_result: Hasil deteksi wajah dari MediaPipe.
        Returns:
            float or None: Nilai rPPG mentah rata-rata yang baru diekstrak, atau None jika gagal.
        """
        if face_detection_result is None or not face_detection_result.detections:
            return None

        # Dapatkan bounding box utama wajah
        detection = face_detection_result.detections[0]
        bbox = detection.bounding_box
        frame_h, frame_w, _ = frame_for_signal.shape
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        # Validasi bounding box utama
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(0, min(w, frame_w - x))
        h = max(0, min(h, frame_h - y))

        if not (w > 0 and h > 0):
            return None

        extracted_signals = []
        # Definisi, validasi, ekstraksi, dan penggambaran untuk ROI Dahi
        fh_x, fh_y, fh_w, fh_h = int(x+w*0.25), int(y+h*0.05), int(w*0.5), int(h*0.20)
        fh_x,fh_y = max(0,min(fh_x,frame_w-1)),max(0,min(fh_y,frame_h-1))
        fh_w,fh_h = max(0,min(fh_w,frame_w-fh_x)),max(0,min(fh_h,frame_h-fh_y))
        if fh_w > 0 and fh_h > 0:
            val = extract_rppg_signal(frame_for_signal, (fh_x, fh_y, fh_w, fh_h))
            if val is not None: extracted_signals.append(val)
            cv2.rectangle(frame_for_signal, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (0, 255, 255), 1) # Cyan

        # Definisi, validasi, ekstraksi, dan penggambaran untuk ROI Pipi Kiri (kanan subjek)
        lc_x, lc_y, lc_w, lc_h = int(x+w*0.60), int(y+h*0.40), int(w*0.30), int(h*0.30)
        lc_x,lc_y = max(0,min(lc_x,frame_w-1)),max(0,min(lc_y,frame_h-1))
        lc_w,lc_h = max(0,min(lc_w,frame_w-lc_x)),max(0,min(lc_h,frame_h-lc_y))
        if lc_w > 0 and lc_h > 0:
            val = extract_rppg_signal(frame_for_signal, (lc_x, lc_y, lc_w, lc_h))
            if val is not None: extracted_signals.append(val)
            cv2.rectangle(frame_for_signal, (lc_x, lc_y), (lc_x + lc_w, lc_y + lc_h), (255, 0, 255), 1) # Magenta

        # Definisi, validasi, ekstraksi, dan penggambaran untuk ROI Pipi Kanan (kiri subjek)
        rc_x, rc_y, rc_w, rc_h = int(x+w*0.10), int(y+h*0.40), int(w*0.30), int(h*0.30)
        rc_x,rc_y = max(0,min(rc_x,frame_w-1)),max(0,min(rc_y,frame_h-1))
        rc_w,rc_h = max(0,min(rc_w,frame_w-rc_x)),max(0,min(rc_h,frame_h-rc_y))
        if rc_w > 0 and rc_h > 0:
            val = extract_rppg_signal(frame_for_signal, (rc_x, rc_y, rc_w, rc_h))
            if val is not None: extracted_signals.append(val)
            cv2.rectangle(frame_for_signal, (rc_x, rc_y), (rc_x + rc_w, rc_y + rc_h), (255, 255, 0), 1) # Kuning

        # Rata-ratakan sinyal yang berhasil diekstrak dan tambahkan ke buffer
        if extracted_signals:
            avg_rppg_value = np.mean(extracted_signals)
            self.rppg_signal_buffer.append(avg_rppg_value)
            if len(self.rppg_signal_buffer) > self.frame_buffer_limit:
                self.rppg_signal_buffer.pop(0)
            return avg_rppg_value
        return None


    def process_respiration_from_pose(self, frame_for_signal, pose_detection_result):
        """
        Mengekstrak sinyal pernapasan dari gerakan bahu, menggambar landmark pada frame,
        dan menambahkan sinyal ke buffer.
        Args:
            frame_for_signal (numpy.ndarray): Frame video (BGR) untuk ekstraksi dan penggambaran.
            pose_detection_result: Hasil deteksi pose dari MediaPipe.
        Returns:
            float or None: Nilai sinyal pernapasan mentah yang baru diekstrak, atau None jika gagal.
        """
        if pose_detection_result is None or not pose_detection_result.pose_landmarks:
            return None

        landmarks = pose_detection_result.pose_landmarks[0]
        h_img, w_img, _ = frame_for_signal.shape
        try:
            # Pastikan landmark bahu (indeks 11 dan 12) terdeteksi dengan baik
            if len(landmarks) > max(11,12) and \
               hasattr(landmarks[11], 'visibility') and landmarks[11].visibility > 0.5 and \
               hasattr(landmarks[12], 'visibility') and landmarks[12].visibility > 0.5:
                rs_landmark = landmarks[12] # Bahu kanan
                ls_landmark = landmarks[11] # Bahu kiri
                y1_r, y1_l = int(rs_landmark.y * h_img), int(ls_landmark.y * h_img)

                # Gambar lingkaran pada landmark bahu untuk visualisasi
                cv2.circle(frame_for_signal, (int(rs_landmark.x*w_img), y1_r), 3, (255,0,0), -1) # Biru
                cv2.circle(frame_for_signal, (int(ls_landmark.x*w_img), y1_l), 3, (0,255,0), -1) # Hijau

                avg_y_shoulder = np.mean([y1_r, y1_l])
                self.resp_signal_buffer.append(-avg_y_shoulder) # Inversi agar puncak napas positif
                if len(self.resp_signal_buffer) > self.frame_buffer_limit:
                    self.resp_signal_buffer.pop(0)
                return -avg_y_shoulder
        except (IndexError, AttributeError) as e:
            print(f"Peringatan: Gagal memproses landmark bahu di HealthAnalyzer: {e}")
        return None

    def filter_and_calculate_hr(self):
        """
        Memfilter sinyal rPPG yang ada di buffer dan menghitung Heart Rate (HR).
        Returns:
            tuple: (list sinyal rPPG terfilter, float nilai HR dalam BPM).
        """
        if len(self.rppg_signal_buffer) < self.min_signal_length:
            return list(self.rppg_signal_buffer), 0.0 # Kembalikan buffer mentah jika terlalu pendek
        try:
            # Tentukan panjang padding untuk filtfilt, hindari error jika sinyal terlalu pendek
            padlen = min(self.min_signal_length -1, len(self.rppg_signal_buffer)-1)
            if padlen <=0: # Tidak cukup data untuk filtfilt yang stabil
                 filtered_signal = list(self.rppg_signal_buffer)
            else:
                 filtered_signal = signal.filtfilt(self.rppg_b, self.rppg_a, self.rppg_signal_buffer, padlen=padlen).tolist()

            hr = calculate_rate_from_fft(filtered_signal, self.fps, self.rppg_lowcut, self.rppg_highcut)
            return filtered_signal, hr
        except ValueError: # Jika terjadi error saat filtering/FFT
            return list(self.rppg_signal_buffer), 0.0

    def filter_and_calculate_rr(self):
        """
        Memfilter sinyal pernapasan yang ada di buffer dan menghitung Respiration Rate (RR).
        Returns:
            tuple: (list sinyal pernapasan terfilter, float nilai RR dalam Breaths/min).
        """
        if len(self.resp_signal_buffer) < self.min_signal_length:
            return list(self.resp_signal_buffer), 0.0 # Kembalikan buffer mentah jika terlalu pendek
        try:
            padlen = min(self.min_signal_length -1, len(self.resp_signal_buffer)-1)
            if padlen <= 0: # Tidak cukup data untuk filtfilt yang stabil
                filtered_signal = list(self.resp_signal_buffer)
            else:
                filtered_signal = signal.filtfilt(self.resp_b, self.resp_a, self.resp_signal_buffer, padlen=padlen).tolist()

            rr = calculate_rate_from_fft(filtered_signal, self.fps, self.resp_lowcut, self.resp_highcut)
            return filtered_signal, rr
        except ValueError: # Jika terjadi error saat filtering/FFT
            return list(self.resp_signal_buffer), 0.0

    def clear_buffers(self):
        """
        Menghapus semua data dari buffer sinyal rPPG dan pernapasan.
        """
        self.rppg_signal_buffer.clear()
        self.resp_signal_buffer.clear()

    def has_models(self):
        """
        Memeriksa apakah model MediaPipe telah berhasil dimuat.
        Returns:
            bool: True jika kedua model (wajah dan pose) telah dimuat, False jika tidak.
        """
        return self.face_detector is not None and self.pose_landmarker is not None