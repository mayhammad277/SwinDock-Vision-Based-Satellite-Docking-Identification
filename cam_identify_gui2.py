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
import numpy as np

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
root.geometry("1800x1800")

# Load starry background
bg_img = Image.open("/home/student/swin_idetify/star_2.jpg").resize((1800, 1800))
bg_tk = ImageTk.PhotoImage(bg_img)
bg_label = tk.Label(root, image=bg_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Title
title = tk.Label(root, text="Satellite Docking Classifier", font=("Helvetica", 28, "bold"),
                 bg="#000000", fg="white", pady=10)
title.pack(pady=20)

# Video Display
video_label = tk.Label(root, bd=4, relief="solid")
video_label.pack()

# Classification Result
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 18, "bold"),
                        bg="#000000", fg="lightgreen")
result_label.pack(pady=10)

# Frame previews
preview_frame = tk.Frame(root, bg="#000000")
preview_frame.pack(pady=20)

preview_labels = []
preview_texts = []
for i in range(5):
    img_label = tk.Label(preview_frame, bg="black", bd=2, relief="ridge")
    img_label.grid(row=0, column=i, padx=10)
    text_label = tk.Label(preview_frame, text="", font=("Helvetica", 12), bg="black", fg="white")
    text_label.grid(row=1, column=i)
    preview_labels.append(img_label)
    preview_texts.append(text_label)

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

# Function to sample 5 frames
def sample_five_images():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // 6
    sampled_indices = [step * (i + 1) for i in range(5)]

    for i, idx in enumerate(sampled_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        pred_class, conf = predict_image(model, img_pil, transform, device)
        resized = img_pil.resize((150, 100))
        img_tk = ImageTk.PhotoImage(resized)

        preview_labels[i].configure(image=img_tk)
        preview_labels[i].image = img_tk
        preview_texts[i].configure(text=f"{class_names[pred_class]} ({conf:.2f})",
                                   fg="lightgreen" if pred_class == 0 else "orange")

# Pose Estimation placeholder
def pose_estimation():
    result_var.set("Pose estimation not implemented yet.")

# Buttons
# Buttons
style = ttk.Style()
style.theme_use("clam")

# Base style for all buttons
style.configure(
    "Space.TButton",
    font=("Helvetica", 16, "bold"),
    padding=10,
    background="#1f1f2e",
    foreground="white",
    borderwidth=0
)
# Hover effect
style.map(
    "Space.TButton",
    background=[("active", "#3b3b52")],
    foreground=[("active", "lightblue")]
)

btn_frame = tk.Frame(root, bg="#000000")
btn_frame.pack(pady=20)

btn1 = ttk.Button(btn_frame, text="Identify Live", style="Space.TButton", command=identify)
btn1.grid(row=0, column=0, padx=40)

btn2 = ttk.Button(btn_frame, text="5 Shots", style="Space.TButton", command=sample_five_images)
btn2.grid(row=0, column=1, padx=40)

btn3 = ttk.Button(btn_frame, text="Pose Estimation", style="Space.TButton")
btn3.grid(row=0, column=2, padx=40)
root.mainloop()


