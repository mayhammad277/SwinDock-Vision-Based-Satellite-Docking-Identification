# SwinDock-Vision-Based-Satellite-Docking-Identification
A real-time computer vision pipeline utilizing Swin Transformers and NaViT for identifying satellite docking surfaces. Features a PyQt5/Tkinter GUI for live inference, automated fine-tuning scripts, and high-performance video-based classification.


# 🛰️ SwinDock: Vision-Based Satellite Docking Identification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Model-Swin--Transformer-orange)](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**SwinDock** is an end-to-end deep learning framework designed to classify satellite surfaces into "Docking Side" and "Non-Docking" categories. By leveraging **Swin Transformers** for hierarchical feature extraction and **NaViT (Native Resolution ViT)** for flexible image processing, this project provides a robust solution for autonomous space docking simulations.

## ✨ Key Features
* **Hybrid Architectures:** Implementation of Swin Transformer (Tiny) and NaViT models.
* **Real-Time Inference:** Live video processing with automated frame sampling and result visualization.
* **Modern GUI:** A sophisticated **PyQt5-based Dashboard** for live monitoring, confidence metrics, and state tracking.
* **Training Pipeline:** Includes scripts for data augmentation (MixUp), fine-tuning, and weight management.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/SwinDock-Vision.git](https://github.com/yourusername/SwinDock-Vision.git)
   cd SwinDock-Vision


Set up the environment & Dependencies :
  ```
  conda env create -f poet_2_env.yml
  conda activate torch21
  pip install navit-torch transformers PyQt5 opencv-python pillow

  ```


🚀 Quick Start
Launch the Live Dashboard
- To run the primary PyQt5 identification interface:
  ```Python
  python cam_identify_gui3.py

  ```

Fine-Tune the Model
- To start training on your custom satellite dataset:
  ```Python
  python finetune_navvit_swin.py
  ```

  
🧠 Model Pipeline
- Hierarchical Feature Extraction :
The project uses the Swin Transformer Tiny as its backbone, which employs shifted windowing to capture local and global features efficiently.


```Python
from transformers import SwinForImageClassification

# Load model with custom classification head
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
  ```

- Flexible Resolution with NaViT:
For processing satellite imagery at native aspect ratios without distortion:


```Python
from navit.main import NaViT

model = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6
)
```

- - 📂 Project Structure
.
- ├── cam_identify_gui3.py    # Main PyQt5 Live Inference Dashboard
- ├── cam_inf_idetify.py      # Lightweight OpenCV inference script
- ├── finetune_navvit_swin.py # Model training and fine-tuning logic
- ├── navvit_swin.py          # NaViT architecture implementation
- ├── poet_2_env.yml          # Conda environment configuration
- └── assets/                 # Backgrounds and sample images (star_background.jpg)



📊 Dashboard Preview

- Live Feed: Real-time analysis of the docking target.

- Identification Status: Green/Orange color-coded indicators for "Docking Side" detection.

- Confidence Meter: Probability score for the current classification.

- Historical Samples: Auto-captures 5 snapshots for mission review.


<div align="center">
  <h3>🎥 Project Demo</h3>
  <video src="https://github.com/mayhammad277/SwinDock-Vision-Based-Satellite-Docking-Identification/raw/refs/heads/main/Screencast%20from%2026.06.2025%2015_03_53.webm" width="100%" controls></video>
</div>



<img width="1299" height="954" alt="image" src="https://github.com/user-attachments/assets/cfdfd894-c308-4a91-8f10-d5a28bb574c8" />

















📝 License
Distributed under the MIT License. See LICENSE for more information.






🤝 Contributing
```
Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
