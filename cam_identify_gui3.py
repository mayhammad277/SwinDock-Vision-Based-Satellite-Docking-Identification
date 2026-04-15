import sys
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

# ---------------- Config ----------------
class_names = ["Docking Side", "Non-Docking"]
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    ignore_mismatched_sizes=True
)
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.load_state_dict(torch.load("/home/student/swin_idetify/swin_epoch_8.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image_pil):
    image = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image).logits
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_class].item()
    return pred_class, conf

# ---------------- Main App ----------------
class SpaceHUD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spacecraft Identification HUD")
        self.setGeometry(100, 50, 1600, 900)

        # Dark sci-fi style
        self.setStyleSheet("""
            QWidget {
                background-image: url('/home/student/swin_idetify/star_2.jpg');
                background-position: center;
                background-repeat: no-repeat;
                background-color: black;
            }
        """)

        # Title
        title_label = QLabel("SATELLITE DOCKING IDENTIFIER")
        title_label.setFont(QFont("Orbitron", 28, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: #00ffea;
            text-shadow: 0 0 20px #00ffea;
        """)

        # Video Display
        self.video_label = QLabel()
        self.video_label.setFixedSize(1000, 650)
        self.video_label.setStyleSheet("""
            border: 4px solid #00ffea;
            border-radius: 12px;
            background-color: rgba(0,0,0,0.6);
        """)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Prediction result
        self.result_label = QLabel("Prediction: --")
        self.result_label.setFont(QFont("Orbitron", 20, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #00ff88;")

        # Preview panel
        preview_layout = QHBoxLayout()
        self.preview_labels = []
        self.preview_texts = []
        for _ in range(5):
            panel = QVBoxLayout()
            img_label = QLabel()
            img_label.setFixedSize(160, 100)
            img_label.setStyleSheet("""
                border: 2px solid #00ffea;
                border-radius: 6px;
                background-color: rgba(0,0,0,0.6);
            """)
            text_label = QLabel("")
            text_label.setStyleSheet("color: white; font-size: 12px;")
            text_label.setAlignment(Qt.AlignCenter)
            panel.addWidget(img_label)
            panel.addWidget(text_label)
            preview_layout.addLayout(panel)
            self.preview_labels.append(img_label)
            self.preview_texts.append(text_label)

        # Buttons
        btn_identify = QPushButton("▶ Identify Live")
        btn_identify.setStyleSheet(self.button_style())
        btn_identify.clicked.connect(self.start_identify)

        btn_5shots = QPushButton("📸 5 Shots")
        btn_5shots.setStyleSheet(self.button_style())
        btn_5shots.clicked.connect(self.sample_five_images)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_identify)
        btn_layout.addWidget(btn_5shots)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        layout.addLayout(preview_layout)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # Video capture
        self.cap = cv2.VideoCapture("/home/student/swin_idetify/WhatsApp Unknown 2025-06-19 at 11.43.06 AM/WhatsApp Video 2025-06-19 at 11.35.08 AM.mp4")
        self.skip_frames = 5
        self.frame_counter = 0

        # Timer for video
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def button_style(self):
        return """
            QPushButton {
                background-color: rgba(0,255,234,0.1);
                color: #00ffea;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 20px;
                border: 2px solid #00ffea;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(0,255,234,0.25);
                color: white;
            }
        """

    def start_identify(self):
        self.timer.start(100)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        self.frame_counter += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        if self.frame_counter % self.skip_frames == 0:
            pred_class, conf = predict_image(img_pil)
            color = "#00ff88" if pred_class == 0 else "#ff6600"
            self.result_label.setText(f"{class_names[pred_class]} ({conf:.2f})")
            self.result_label.setStyleSheet(f"color: {color};")

        img_resized = cv2.resize(rgb, (1000, 650))
        qt_img = QImage(img_resized.data, img_resized.shape[1], img_resized.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def sample_five_images(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // 6
        sampled_indices = [step * (i + 1) for i in range(5)]

        for i, idx in enumerate(sampled_indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            pred_class, conf = predict_image(img_pil)

            img_resized = cv2.resize(rgb, (160, 100))
            qt_img = QImage(img_resized.data, img_resized.shape[1], img_resized.shape[0], QImage.Format_RGB888)
            self.preview_labels[i].setPixmap(QPixmap.fromImage(qt_img))
            color = "#00ff88" if pred_class == 0 else "#ff6600"
            self.preview_texts[i].setText(f"{class_names[pred_class]} ({conf:.2f})")
            self.preview_texts[i].setStyleSheet(f"color: {color};")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SpaceHUD()
    win.show()
    sys.exit(app.exec_())

