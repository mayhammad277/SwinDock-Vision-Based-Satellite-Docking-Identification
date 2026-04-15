import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import SwinForImageClassification
from PIL import Image
import time

def select_camera():
    """Try to find available cameras and let user select one"""
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
            cap.release()
        index += 1
    
    print(f"Available cameras: {cameras}")
    if not cameras:
        raise RuntimeError("No cameras found!")
    
    # Select first USB camera (typically /dev/video1 if /dev/video0 is built-in)
    selected_cam = cameras[-1] if len(cameras) > 1 else cameras[0]
    print(f"Selecting camera index {selected_cam}")
    return selected_cam

# Initialize webcam with USB camera selection
#camera_index = select_camera()
cap = cv2.VideoCapture("/home/student/swin_idetify/WhatsApp Unknown 2025-06-19 at 11.43.06 AM/WhatsApp Video 2025-06-19 at 11.35.08 AM.mp4")

# Additional camera configuration
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Try higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 0.1)  # Try to set FPS

# Verify camera settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

# Configuration
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Docking Side", "Non-Docking"]

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load Swin model
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224", 
    ignore_mismatched_sizes=True
)
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.to(device)

# Load model weights (update this path)
model_save_path = "/home/student/swin_idetify/swin_epoch_8.pth"
try:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def predict_image(model, image, transform, device):
    """Predict class and confidence from a PIL Image"""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image).logits
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    return predicted_class, confidence

frame_counter = 0
skip_frames = 10  # Process every 10th frame
processing_time = 0
avg_fps = 0

try:
    while True:
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error")
            break
            
        # Display frame
        cv2.imshow('USB Webcam Feed', frame)
        
        # Process every nth frame
        if frame_counter % skip_frames == 0:
            try:
                # Convert and predict
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                inference_start = time.time()
                predicted_class, confidence = predict_image(model, pil_image, transform, device)
                processing_time = time.time() - inference_start
                
                # Display results
                result_text = f"{class_names[predicted_class]} ({confidence:.2f})"
                time.sleep(0.2)
                fps_text = f"FPS: {avg_fps:.1f}" if avg_fps > 0 else ""
                cv2.putText(frame, result_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, fps_text, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow('USB Webcam Feed', frame)
                
            except Exception as e:
                print(f"Processing error: {e}")
        
        # Calculate FPS
        frame_time = time.time() - start_time
        avg_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        frame_counter += 1
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")
