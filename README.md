# Brain Tumor Detection using DenseNet121

Deep learning model to classify brain MRI images into 4 categories:
- **Glioma**
- **Meningioma**
- **No Tumor**  
- **Pituitary**

## Dataset
This project uses the **Brain Tumor MRI Dataset** from Kaggle:
- [Dataset Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


### Dataset Setup
1. Download dataset from Kaggle
2. Extract to project directory:


File Structure:

brain-tumor-detection/
├── Train/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
└── Test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/


## Model Architecture
- **Base**: DenseNet121 (pretrained on ImageNet)
- **Custom Head**: Dropout (0.4) → Linear(512) → ReLU → BatchNorm → Dropout (0.3) → Linear(4)
- **Training Strategy**: Transfer learning with warmup fine-tuning

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 76.75% |
| Macro AUC | 0.9327 |

### Per-class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|-----------|
| Glioma | 0.6573 | 0.8200 | 0.7297 |
| Meningioma | 0.7834 | 0.4250 | 0.5511 |
| No Tumor | 0.8668 | 0.9600 | 0.9110 |
| Pituitary | 0.7846 | 0.8650 | 0.8228 |

## Requirements
```bash
pip install -r requirements.txt

@misc{brain_tumor_mri_dataset,
  author = {Masoud Nickparvar},
  title = {Brain Tumor MRI Dataset},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
}