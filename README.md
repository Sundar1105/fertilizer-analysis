# 🩺 Kidney Disease Detection using Deep Learning (AlexNet)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

## 📌 Project Overview

This project uses a **pretrained AlexNet** model to classify CT kidney scans into **four categories**:
- 🟢 **Normal** (5,077 images)
- 🔵 **Cyst** (3,709 images)
- 🟡 **Stone** (1,377 images)
- 🔴 **Tumor** (2,283 images)

> **Total Dataset Size:** 12,446 CT scan images

## 🎯 Model Performance

After training for 5 epochs, the model achieved:
- **Accuracy:** ~99.94%
- **Precision:** ~99.94%
- **F2 Score:** ~99.94%
- **Validation Accuracy:** 100% on final epoch

## 🏗️ Model Architecture

- **Base Model:** AlexNet (pretrained on ImageNet)
- **Transfer Learning:** Frozen feature extraction layers
- **Custom Classifier:** Modified final layer for 4 classes
- **Input Size:** 224×224 pixels
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (learning rate: 0.0001)

## 📁 Dataset Structure
Dataset/
├── Cyst/ (3,709 images)
├── Normal/ (5,077 images)
├── Stone/ (1,377 images)
└── Tumor/ (2,283 images)


## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sundar1105/kidney-disease-detection.git
cd kidney-disease-detection

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install requirements.txt

4. Download Dataset
Download from Kaggle CT Kidney Dataset and place in Dataset/ folder.

5. Run training
Open and run Kidney.ipynb Jupyter notebook

📊 Training Results
Epoch	Train Loss	Train Accuracy	Validation Accuracy
1	0.3235	87.03%	98.19%
2	0.1013	96.06%	99.34%
3	0.0630	97.88%	99.94%
4	0.0493	98.23%	99.94%
5	0.0359	98.66%	100.00%
📈 Visualizations
Confusion Matrix

Training Loss Curve

Accuracy Trends (Train vs Validation)

🖼️ Single Image Prediction
# Load trained model and predict
python predict.py --image path/to/ct_scan.jpg

@article{islam2022vision,
  title={Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography},
  author={Islam, Md Nazmul and Hasan, Mehedi and Hossain, Md Kabir and Alam, Md Golam Rabiul and Uddin, Mohammad Zia and Soylu, Ahmet},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={1--14},
  year={2022},
  publisher={Nature Publishing Group}
}

📝 License
This project is for educational and research purposes.