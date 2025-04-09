# Image-Classification

# 🧠 Image Classification with PyTorch

This project provides an image classification pipeline using PyTorch. You can train your own models and make predictions from the command line.

---

## 🚀 Features

- Train image classification models using PyTorch
- Use pretrained models (e.g., ResNet)
- Predict a single image via CLI
- TensorBoard logging support
- Custom dataset support (organized by class folders)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Duuccc/Image-Classification.git
cd Image-Classification

###2. Create virtual environments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

###3. install essential library
pip install -r requirements.txt

###4. predict image
python going_modular/predict.py --image "path/to/your/image"

