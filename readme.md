# Fertilizer Analysis & Prediction System 🌱

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-Classifier-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project uses **Random Forest Classifier** to recommend the most suitable fertilizer based on soil parameters and crop type. The model analyzes soil nutrients, pH levels, and crop requirements to provide accurate fertilizer recommendations.

## 🎯 Features

- **Input Parameters**:
  - Nitrogen (N), Phosphorus (P), Potassium (K) levels
  - Soil pH
  - Soil color
  - Crop type

- **Output**: Recommended fertilizer from 19 different fertilizer types

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~77%
- **Dataset Size**: 4,283 samples
- **Fertilizer Types**: 19 categories including:
  - Urea, DAP, MOP, SSP
  - 10:10:10 NPK, 19:19:19 NPK
  - Ammonium Sulphate, Ferrous Sulphate
  - Magnesium Sulphate, White Potash, and more

## 📁 Project Structure

Fertilizer-analysis/
├── Untitled.ipynb # Main Jupyter notebook with model training
├── Fertilizer.csv # Dataset (soil parameters + crop types)
├── confusion_matrix.png # Model performance visualization
├── .gitignore # Git ignore rules
└── README.md # Project documentation


## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Sundar1105/Fertilizer-analysis.git
cd Fertilizer-analysis

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Launch Jupyter Notebook
jupyter notebook Untitled.ipynb


## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Fertilizer-analysis.git
cd Fertilizer-analysis

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Launch Jupyter Notebook
jupyter notebook Untitled.ipynb

import pandas as pd
import joblib

# Load the trained model
model = joblib.load('fertilizer_model.joblib')

# Prepare input data
sample = pd.DataFrame({
    'Nitrogen': [50],
    'Phosphorus': [40],
    'Potassium': [30],
    'pH': [6.5],
    'Soil_color': ['Black'],
    'Crop': ['Wheat']
})

# Predict fertilizer
prediction = model.predict(sample)
print(f"Recommended Fertilizer: {prediction[0]}")

📈 Model Features
Feature	Description	Range
Nitrogen	Nitrogen content in soil	0-100+
Phosphorus	Phosphorus content	0-100+
Potassium	Potassium content	0-100+
pH	Soil pH level	3.0-9.0
Soil_color	Color of soil	Black, Red, Sandy, etc.
Crop	Crop type	Wheat, Rice, Maize, etc.
🔧 Dependencies
pandas - Data manipulation

numpy - Numerical operations

matplotlib & seaborn - Visualization

scikit-learn - Machine learning (Random Forest, preprocessing)

joblib - Model persistence

📊 Classification Report

Accuracy: 0.77
Weighted Avg Precision: 0.78
Weighted Avg Recall: 0.77
Weighted Avg F1-Score: 0.77

🔄 Model Pipeline
Preprocessing: OneHotEncoder for categorical features

Model: RandomForestClassifier with balanced class weights

Training: 80/20 train-test split with stratification

Evaluation: Classification report and confusion matrix

💾 Output Files
After running the notebook:

fertilizer_model.joblib - Trained Random Forest model

confusion_matrix.png - Visualization of model performance

📝 Notes
The model handles 19 different fertilizer types

Uses class_weight='balanced' to handle imbalanced classes

Categorical features (Soil_color, Crop) are one-hot encoded

Model achieves robust performance across all fertilizer categories

🤝 Contributing
Feel free to submit issues and enhancement requests!

📄 License
MIT License - Free for academic and commercial use




