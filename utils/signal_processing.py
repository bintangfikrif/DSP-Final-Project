import cv2
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq

def extract_rppg_signal(frame, roi):
    """
    Mengekstraksi nilai rata-rata channel hijau dari Region of Interest (ROI) pada sebuah frame.
    Ini adalah pendekatan sederhana untuk mendapatkan sinyal rPPG mentah.
    Args:
        frame (numpy.ndarray): Frame video input dalam format BGR.
        roi (tuple): Tuple berisi (x, y, w, h) yang mendefinisikan ROI.
                     x: koordinat x sudut kiri atas ROI.
                     y: koordinat y sudut kiri atas ROI.
                     w: lebar ROI.
                     h: tinggi ROI.
    Returns:
        float or None: Nilai rata-rata channel hijau di ROI, atau None jika ROI tidak valid.
    """
    x, y, w, h = roi # Unpack koordinat dan dimensi ROI
    if w > 0 and h > 0: # Pastikan ROI memiliki dimensi yang valid
        roi_frame = frame[y:y+h, x:x+w] # Potong frame untuk mendapatkan area ROI
        # Hitung nilai rata-rata untuk setiap channel warna (BGR) di ROI
        # Kemudian ambil nilai rata-rata dari channel Hijau (indeks 1 pada BGR)
        # Channel hijau sering digunakan dalam rPPG karena lebih sensitif terhadap perubahan volume darah
        mean_green = cv2.mean(roi_frame)[1] 
        return mean_green
    return None 

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Mendesain koefisien filter bandpass Butterworth.
    Args:
        lowcut (float): Frekuensi cut-off bawah (dalam Hz).
        highcut (float): Frekuensi cut-off atas (dalam Hz).
        fs (float): Frekuensi sampling sinyal (dalam Hz).
        order (int, optional): Orde filter. Defaultnya adalah 5.
                               Orde filter yang lebih tinggi memberikan roll-off yang lebih tajam.
    Returns:
        tuple: (b, a) yang merupakan koefisien numerator (b) dan denominator (a) dari filter.
    """
    nyq = 0.5 * fs # Frekuensi Nyquist, yaitu setengah dari frekuensi sampling
    low = lowcut / nyq 
    high = highcut / nyq 
    b, a = signal.butter(order, [low, high], btype='band') 
    return b, a


def calculate_rate_from_fft(signal_values, fs, lowcut_hz, highcut_hz):
    """
    Menghitung laju (seperti detak jantung atau laju pernapasan) dari sebuah sinyal
    menggunakan Fast Fourier Transform (FFT).
    Args:
        signal_values (list or numpy.ndarray): Array nilai sinyal.
        fs (float): Frekuensi sampling sinyal (dalam Hz).
        lowcut_hz (float): Frekuensi cut-off bawah dari rentang yang diminati (dalam Hz).
        highcut_hz (float): Frekuensi cut-off atas dari rentang yang diminati (dalam Hz).
    Returns:
        float: Laju yang dihitung dalam satuan per menit (misalnya, BPM untuk detak jantung).
               Mengembalikan 0 jika sinyal terlalu pendek atau tidak ada puncak dominan yang ditemukan.
    """
    if len(signal_values) < 20: # Membutuhkan panjang sinyal yang cukup untuk analisis FFT yang berarti
        return 0 

    N = len(signal_values) # Jumlah sampel dalam sinyal
    
    # Lakukan FFT pada sinyal
    yf = fft(signal_values)
    # Hitung frekuensi yang sesuai untuk setiap komponen FFT
    xf = fftfreq(N, 1 / fs)

    # Ambil hanya bagian positif dari spektrum frekuensi 
    xf_positive = xf[:N//2]
    yf_positive = 2.0/N * np.abs(yf[0:N//2]) # Normalisasi amplitudo

    # Cari indeks frekuensi yang berada dalam rentang [lowcut_hz, highcut_hz]
    valid_indices = np.where((xf_positive >= lowcut_hz) & (xf_positive <= highcut_hz))[0]

    if len(valid_indices) == 0: # Jika tidak ada frekuensi dalam rentang yang valid
        return 0 # Kembalikan 0 karena tidak ada puncak yang dapat dianalisis

    # Dapatkan amplitudo dan frekuensi dalam rentang yang valid
    valid_yf = yf_positive[valid_indices]
    valid_xf = xf_positive[valid_indices]

    if len(valid_yf) == 0: # Pengecekan tambahan jika valid_yf kosong
        return 0

    # Temukan indeks dari puncak dominan (amplitudo terbesar) dalam rentang frekuensi yang valid
    dominant_peak_index_in_valid = np.argmax(valid_yf)
    # Dapatkan frekuensi yang sesuai dengan puncak dominan tersebut
    dominant_frequency_hz = valid_xf[dominant_peak_index_in_valid]

    # Konversi frekuensi dominan dari Hz ke satuan per menit (misalnya, BPM)
    rate_per_minute = dominant_frequency_hz * 60
    
    return rate_per_minute