# health_tracker_ui.py
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QGroupBox, QApplication,QSizePolicy)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class HealthTrackerUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._apply_styles()

    def _init_ui(self):
        # --- Main Vertical Layout ---
        self.main_layout = QVBoxLayout(self) 

        # --- Title Label ---
        self.title_label = QLabel("Realtime Health Tracker")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # --- Content Layout (Horizontal: Left and Right Panes) ---
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # --- Left Pane ---
        left_pane_widget = QWidget()
        left_pane_layout = QVBoxLayout(left_pane_widget)

        self.video_label = QLabel("CAMERA FEED") # Placeholder for video
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_pane_layout.addWidget(self.video_label, 1) 

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.end_button = QPushButton("END")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.end_button)
        left_pane_layout.addLayout(button_layout)
        self.content_layout.addWidget(left_pane_widget, 2) 

        # --- Right Pane ---
        right_pane_widget = QWidget()
        right_pane_layout = QVBoxLayout(right_pane_widget)

        # Heart Rate Section
        hr_group = QGroupBox("Heart's Rate")
        hr_group_layout = QVBoxLayout(hr_group) 

        hr_info_layout = QHBoxLayout()
        self.hr_icon_label = QLabel() # Placeholder for icon
        self.hr_icon_label.setPixmap(QPixmap("icon/heart-icon.png").scaled(30,30,Qt.KeepAspectRatio))
        self.hr_value_label = QLabel("-- BPM")
        hr_info_layout.addWidget(self.hr_icon_label)
        hr_info_layout.addStretch()
        hr_info_layout.addWidget(self.hr_value_label)
        hr_group_layout.addLayout(hr_info_layout)

        self.hr_fig, self.ax_rppg = plt.subplots()
        self.hr_canvas = FigureCanvas(self.hr_fig)
        hr_group_layout.addWidget(self.hr_canvas)
        right_pane_layout.addWidget(hr_group)

        # Respiration Rate Section
        rr_group = QGroupBox("Respiration Rate")
        rr_group_layout = QVBoxLayout(rr_group) # Layout for the group box

        rr_info_layout = QHBoxLayout()
        self.rr_icon_label = QLabel() # Placeholder for icon
        self.rr_icon_label.setPixmap(QPixmap("icon/lungs-icon.png").scaled(30,30,Qt.KeepAspectRatio))
        self.rr_value_label = QLabel("-- Breaths/min")
        rr_info_layout.addWidget(self.rr_icon_label)
        rr_info_layout.addStretch()
        rr_info_layout.addWidget(self.rr_value_label)
        rr_group_layout.addLayout(rr_info_layout)

        self.rr_fig, self.ax_resp = plt.subplots()
        self.rr_canvas = FigureCanvas(self.rr_fig)
        rr_group_layout.addWidget(self.rr_canvas)
        right_pane_layout.addWidget(rr_group)

        self.content_layout.addWidget(right_pane_widget, 1) # Right pane takes 1/3rd

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E; /* Overall dark background */
                color: #E0E0E0; /* Light text color */
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #66BB6A; /* Greenish title */
                margin-bottom: 15px;
                margin-top: 10px;
            }
            QLabel#VideoLabel {
                background-color: black;
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
                background-color: #4CAF50; /* Green */
            }
            QPushButton#StartButton:disabled {
                background-color: #388E3C; /* Darker Green when disabled */
            }
            QPushButton#EndButton {
                background-color: #FF9800; /* Orange */
            }
            QPushButton#EndButton:disabled {
                background-color: #F57C00; /* Darker Orange when disabled */
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #E0E0E0;
                background-color: #2A2A2A; /* Background for sections */
                border: 1px solid #444444;
                border-radius: 8px;
                margin-top: 1ex; /* Space for title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Center the title */
                padding: 0 10px;
                color: #66BB6A; /* Title color for group boxes */
            }
            QLabel#ValueLabel { /* Common style for HR and RR value labels */
                font-size: 20px;
                font-weight: bold;
                color: #FFFFFF; /* Bright white for values */
                padding: 5px;
            }
        """)
        self.title_label.setObjectName("TitleLabel")
        self.video_label.setObjectName("VideoLabel")
        self.start_button.setObjectName("StartButton")
        self.end_button.setObjectName("EndButton")

        # Specific styling for value labels for easier reference
        self.hr_value_label.setObjectName("ValueLabel")
        self.hr_value_label.setStyleSheet("color: #FF6B6B;") # Reddish for HR
        self.rr_value_label.setObjectName("ValueLabel")
        self.rr_value_label.setStyleSheet("color: #6BCBFF;") # Bluish for RR

        # Style plots
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