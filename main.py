# Final Project
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mediapipe as mp

# Setup
resp_signal = []
fps = 35
time_window = 60
frame_buffer_limit = time_window * fps
frame_buffer = []
STANDARD_SIZE = (640, 480)

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fungsi konversi Matplotlib ke OpenCV image
def plot_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))  # RGBA
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return image_rgb


# Ekstraksi sinyal rPPG (jika dibutuhkan)
def extract_rppg_signal(frame):
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y_channel = yuv_frame[:, :, 0]
    return np.mean(y_channel)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Kamera tidak bisa dibuka.")

    # Setup plot
    fig, ax_resp = plt.subplots(figsize=(8, 4))
    ax_resp.set_title("Real-Time Respiration Signal")
    ax_resp.set_xlabel("Frame")
    ax_resp.set_ylabel("Signal Amplitude")
    line_resp, = ax_resp.plot([], [], color='green')

    def update_plot():
        line_resp.set_data(range(len(resp_signal)), resp_signal)
        ax_resp.relim()
        ax_resp.autoscale_view()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        frame = cv2.resize(frame, STANDARD_SIZE)
        frame_buffer.append(frame)
        if len(frame_buffer) > frame_buffer_limit:
            frame_buffer.pop(0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            h, w, _ = frame.shape

            def get_landmark_coords(landmark):
                return int(landmark.x * w), int(landmark.y * h)

            try:
                rs = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                ls = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

                x1_r, y1_r = get_landmark_coords(rs)
                x1_l, y1_l = get_landmark_coords(ls)

                def clamp(val, minval, maxval):
                    return max(minval, min(val, maxval))

                # Calculate bounding box coordinates
                left = clamp(min(x1_r, x1_l) - 20, 0, w)
                top = clamp(min(y1_r, y1_l) - 65, 0, h)
                right = clamp(max(x1_r, x1_l) + 20, 0, w)
                bottom = clamp(max(y1_r, y1_l) + 20, 0, h)

                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                # Calculate average y-coordinate of shoulders
                avg_y_shoulder = np.mean([y1_r, y1_l])
                resp_signal.append(-avg_y_shoulder)
                if len(resp_signal) > frame_buffer_limit:
                    resp_signal.pop(0)

                update_plot()

            except IndexError:
                print("Landmark tidak lengkap.")

        plot_image = plot_to_image(fig)
        plot_image = cv2.resize(plot_image, (frame.shape[1], plot_image.shape[0]))

        combined_image = np.vstack((frame, plot_image))
        cv2.imshow("Respiration Signal and Webcam", combined_image)

        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Respiration Signal and Webcam", cv2.WND_PROP_VISIBLE) < 1:
            break

except Exception as e:
    print(f"Terjadi error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
