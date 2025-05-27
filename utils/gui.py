import os 
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QApplication, QSizePolicy) 
from PyQt5.QtGui import QFont, QPixmap 
from PyQt5.QtCore import Qt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
import matplotlib.pyplot as plt 

class HealthTrackerUI(QWidget):
    """
    Kelas untuk mendefinisikan dan mengatur elemen-elemen Antarmuka Grafis Pengguna (GUI)
    untuk aplikasi Realtime Health Tracker.
    """
    def __init__(self, parent=None):
        """
        Konstruktor untuk HealthTrackerUI.
        Args:
            parent (QWidget, optional): Widget induk dari UI ini. Defaultnya None.
        """
        super().__init__(parent)
        self._init_ui() # Panggil metode untuk inisialisasi elemen UI
        self._apply_styles() # Panggil metode untuk menerapkan stylesheet

    def _init_ui(self):
        """
        Menginisialisasi semua elemen UI dan layoutnya.
        """
        # --- Layout Utama Vertikal ---
        self.main_layout = QVBoxLayout(self) 

        # --- Label Judul Aplikasi ---
        self.title_label = QLabel("Realtime rPPG and Respiration Rate Tracker by BEE Team")
        self.title_label.setAlignment(Qt.AlignCenter) 
        self.main_layout.addWidget(self.title_label) 

        # --- Layout Konten (Horizontal: Panel Kiri dan Kanan) ---
        self.content_layout = QHBoxLayout() 
        self.main_layout.addLayout(self.content_layout) 

        # --- Panel Kiri (Video Feed dan Tombol Kontrol) ---
        left_pane_widget = QWidget() 
        left_pane_layout = QVBoxLayout(left_pane_widget) 

        self.video_label = QLabel("KAMERA FEED") 
        self.video_label.setAlignment(Qt.AlignCenter) 
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        left_pane_layout.addWidget(self.video_label, 1) 

        button_layout = QHBoxLayout() # Layout horizontal untuk tombol START dan END
        self.start_button = QPushButton("START") 
        self.end_button = QPushButton("END") 
        button_layout.addWidget(self.start_button) 
        button_layout.addWidget(self.end_button) 
        left_pane_layout.addLayout(button_layout) 
        self.content_layout.addWidget(left_pane_widget, 2) 

        # --- Panel Kanan (Plot Sinyal HR dan RR) ---
        right_pane_widget = QWidget() 
        right_pane_layout = QVBoxLayout(right_pane_widget) 

        # --- Bagian Detak Jantung (Heart Rate) ---
        hr_group = QGroupBox("Heart's Rate") 
        hr_group_layout = QVBoxLayout(hr_group) 

        hr_info_layout = QHBoxLayout() 
        self.hr_icon_label = QLabel() 

        # Dapatkan path direktori tempat file gui.py saat ini berada
        current_dir_hr = os.path.dirname(os.path.abspath(__file__))
        # Bangun path ke ikon heart-icon.png
        icon_path_heart = os.path.join(current_dir_hr, "..", "icon", "heart-icon.png")
        if os.path.exists(icon_path_heart):
            self.hr_icon_label.setPixmap(QPixmap(icon_path_heart).scaled(30,30,Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print(f"Ikon hati tidak ditemukan di: {icon_path_heart}") 
        
        self.hr_value_label = QLabel("-- BPM") # Label untuk menampilkan nilai BPM
        hr_info_layout.addWidget(self.hr_icon_label) 
        hr_info_layout.addStretch() 
        hr_info_layout.addWidget(self.hr_value_label) 
        hr_group_layout.addLayout(hr_info_layout) # Tambahkan layout info HR ke GroupBox HR

        self.hr_fig, self.ax_rppg = plt.subplots() # Buat figure dan axes baru untuk plot rPPG
        self.hr_canvas = FigureCanvas(self.hr_fig) # Buat canvas Matplotlib untuk plot rPPG
        hr_group_layout.addWidget(self.hr_canvas) # Tambahkan canvas plot rPPG ke GroupBox HR
        right_pane_layout.addWidget(hr_group) 

        # --- Bagian Laju Pernapasan (Respiration Rate) ---
        rr_group = QGroupBox("Respiration Rate") 
        rr_group_layout = QVBoxLayout(rr_group) 

        rr_info_layout = QHBoxLayout() # Layout horizontal untuk ikon RR dan nilai Breaths/min
        self.rr_icon_label = QLabel() # Label untuk ikon pernapasan (placeholder)
        current_dir_rr = os.path.dirname(os.path.abspath(__file__))
        icon_path_lungs = os.path.join(current_dir_rr, "..", "icon", "lungs-icon.png")
        if os.path.exists(icon_path_lungs):
            self.rr_icon_label.setPixmap(QPixmap(icon_path_lungs).scaled(30,30,Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print(f"Ikon paru-paru tidak ditemukan di: {icon_path_lungs}") 
            
        self.rr_value_label = QLabel("-- Breaths/min") # Label untuk menampilkan nilai laju pernapasan
        rr_info_layout.addWidget(self.rr_icon_label) 
        rr_info_layout.addStretch() 
        rr_info_layout.addWidget(self.rr_value_label) 
        rr_group_layout.addLayout(rr_info_layout) # Tambahkan layout info RR ke GroupBox RR

        self.rr_fig, self.ax_resp = plt.subplots() # Buat figure dan axes baru untuk plot respirasi
        self.rr_canvas = FigureCanvas(self.rr_fig) # Buat canvas Matplotlib untuk plot respirasi
        rr_group_layout.addWidget(self.rr_canvas) # Tambahkan canvas plot respirasi ke GroupBox RR
        right_pane_layout.addWidget(rr_group) 

        self.content_layout.addWidget(right_pane_widget, 1) # Tambahkan panel kanan ke layout konten, stretch factor 1

    def _apply_styles(self):
        """
        Menerapkan Qt StyleSheets untuk mengatur tampilan visual elemen-elemen UI.
        """
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E; /* Latar belakang gelap keseluruhan */
                color: #E0E0E0; /* Warna teks terang */
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #66BB6A; /* Warna judul kehijauan */
                margin-bottom: 15px;
                margin-top: 10px;
            }
            QLabel#VideoLabel {
                background-color: black; /* Latar belakang hitam untuk video feed */
                border: 1px solid #444444;
            }
            QPushButton {
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 15px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton#StartButton {
                background-color: #4CAF50; /* Warna hijau untuk tombol START */
            }
            QPushButton#StartButton:disabled {
                background-color: #388E3C; /* Warna hijau lebih gelap saat dinonaktifkan */
            }
            QPushButton#EndButton {
                background-color: #FF9800; /* Warna oranye untuk tombol END */
            }
            QPushButton#EndButton:disabled {
                background-color: #F57C00; /* Warna oranye lebih gelap saat dinonaktifkan */
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                background-color: #2A2A2A; /* Latar belakang untuk bagian (GroupBox) */
                border: 1px solid #444444;
                border-radius: 8px;
                margin-top: 1ex; /* Spasi untuk judul GroupBox */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Posisi judul GroupBox di tengah atas */
                padding: 0 10px;
                color: #66BB6A; /* Warna judul GroupBox */
            }
            QLabel#ValueLabel { /* Style umum untuk label nilai HR dan RR */
                font-size: 20px;
                font-weight: bold;
                color: #FFFFFF; /* Warna putih terang untuk nilai */
                padding: 5px;
            }
        """)
        # Menetapkan nama objek agar bisa di-style secara spesifik menggunakan ID selector
        self.title_label.setObjectName("TitleLabel")
        self.video_label.setObjectName("VideoLabel")
        self.start_button.setObjectName("StartButton")
        self.end_button.setObjectName("EndButton")

        # Style spesifik untuk label nilai HR dan RR 
        self.hr_value_label.setObjectName("ValueLabel")
        self.hr_value_label.setStyleSheet("color: #FF6B6B;") 
        self.rr_value_label.setObjectName("ValueLabel")
        self.rr_value_label.setStyleSheet("color: #6BCBFF;")

        # Style untuk plot Matplotlib agar sesuai dengan tema gelap
        for fig, ax in [(self.hr_fig, self.ax_rppg), (self.rr_fig, self.ax_resp)]:
            fig.patch.set_facecolor('#2A2A2A') 
            ax.set_facecolor('#2A2A2A') 
            ax.tick_params(axis='x', colors='white') 
            ax.tick_params(axis='y', colors='white') 
            for spine in ax.spines.values():
                spine.set_edgecolor('white') 
            ax.set_title("") 

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv) 
    ui = HealthTrackerUI() 
    ui.setGeometry(50, 50, 1200, 700) 
    ui.show() 
    sys.exit(app.exec_()) 