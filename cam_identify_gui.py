import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import SwinForImageClassification
import threading
import time

# Configurations
class_names = ["Docking Side", "Non-Docking"]
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224", 
    ignore_mismatched_sizes=True
)
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.load_state_dict(torch.load("/home/student/swin_idetify/swin_epoch_8.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Predict function
def predict_image(model, image, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image).logits
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    return predicted_class, confidence

# GUI Setup
root = tk.Tk()
root.title("Spacecraft Identification System")
root.geometry("900x700")
root.configure(bg="#1e1e2f")

# Title
title = ttk.Label(root, text="Satellite Docking Classifier", font=("Helvetica", 24, "bold"), background="#1e1e2f", foreground="white")
title.pack(pady=20)

# Video Display
video_label = tk.Label(root)
video_label.pack()

# Classification Result
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 16), bg="#1e1e2f", fg="lightgreen")
result_label.pack(pady=10)

# Camera
cap = cv2.VideoCapture("/home/student/swin_idetify/WhatsApp Unknown 2025-06-19 at 11.43.06 AM/WhatsApp Video 2025-06-19 at 11.35.08 AM.mp4")
cap.set(cv2.CAP_PROP_FPS, 1)
skip_frames = 5
frame_counter = 0

# Identify function
def identify():
    def loop():
        global frame_counter
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)

            if frame_counter % skip_frames == 0:
                try:
                    pred_class, conf = predict_image(model, img_pil, transform, device)
                    result_text = f"{class_names[pred_class]} ({conf:.2f})"
                    result_var.set(result_text)
                except Exception as e:
                    result_var.set(f"Error: {str(e)}")

            img_pil = img_pil.resize((640, 480))
            img_tk = ImageTk.PhotoImage(img_pil)
            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)

            time.sleep(0.1)
    threading.Thread(target=loop, daemon=True).start()

# Pose Estimation placeholder
def pose_estimation():
    result_var.set("Pose estimation not implemented yet.")

# Buttons
btn_frame = tk.Frame(root, bg="#1e1e2f")
btn_frame.pack(pady=20)

btn1 = ttk.Button(btn_frame, text="🛰️ Identify", command=identify)
btn1.grid(row=0, column=0, padx=20)

btn2 = ttk.Button(btn_frame, text="📐 Pose Estimation", command=pose_estimation)
btn2.grid(row=0, column=1, padx=20)

# Start GUI
root.mainloop()

