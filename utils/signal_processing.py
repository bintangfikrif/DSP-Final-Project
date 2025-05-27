import cv2
import numpy as np
import scipy.signal as signal

def extract_rppg_signal(frame, bbox):
    x, y, w, h = bbox
    face_roi = frame[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None
    yuv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    return np.mean(y_channel)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def calculate_rate_from_fft(signal_data, fs, low_hz, high_hz):
    if not signal_data or len(signal_data) < fs : # Ensure enough data for a meaningful FFT (e.g., at least 1 sec)
        return -1.0

    fft_data = np.abs(np.fft.fft(signal_data))
    freqs = np.fft.fftfreq(len(signal_data), 1/fs)

    valid_indices = np.where((freqs >= low_hz) & (freqs <= high_hz))

    if len(valid_indices[0]) > 0:
        dominant_freq_idx = valid_indices[0][np.argmax(fft_data[valid_indices])]
        dominant_freq = freqs[dominant_freq_idx]
        return dominant_freq * 60  # Convert Hz to BPM/BreathsPM
    else:
        return -1.0