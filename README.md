# 🌾 Crop Recommendation & Yield Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains two machine learning models for agricultural applications using **XGBoost**:

| Model | Type | Accuracy/R² | Output |
|-------|------|-------------|--------|
| **Crop Recommendation** | Multi-class Classification | ~99.5% | Recommended crop (22 types) |
| **Crop Yield Prediction** | Regression | ~0.913 R² | Yield (tons/hectare) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Crop.git
cd Crop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r req.txt

Dataset Setup
Place your datasets in the datasets/ folder:

Crop_recommendation.csv - For crop recommendation 

crop_Yield.csv - For yield prediction 

Run Notebooks

jupyter notebook crop.ipynb      # Crop recommendation model
jupyter notebook yield.ipynb     # Yield prediction model

📊 Model Details
Crop Recommendation Model (crop.ipynb)
Features:

Nitrogen (N), Phosphorus (P), Potassium (K)

Temperature, Humidity, pH, Rainfall

Output: One of 22 crops including rice, maize, wheat, cotton, sugarcane, etc.

Evaluation:

Accuracy: 0.9954
Precision/Recall/F1: >0.98 for all crops

Yield Prediction Model (yield.ipynb)
Features:

Region, Soil_Type, Crop, Weather_Condition

Additional agronomic parameters

Evaluation:

R² Score: 0.9127
Mean Absolute Error: 0.4002 tons/ha

💻 Usage Examples
Crop Recommendation

import pickle
import numpy as np

# Load model
with open('XGBoostcrop.pkl', 'rb') as f:
    model, le = pickle.load(f)

# Predict
sample = np.array([[90, 42, 43, 20.87, 82.0, 6.50, 202.9]])
crop = le.inverse_transform(model.predict(sample))
print(f"Recommended: {crop[0]}")  # Output: rice

Yield Prediction

import pickle
import pandas as pd

# Load model
with open('XGBoost_Yield.pkl', 'rb') as f:
    model = pickle.load(f)
with open('LabelEncoders_yield.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare input (encode categoricals)
sample = pd.DataFrame([['North', 'Loamy', 'Wheat', 'Sunny']], 
                      columns=['Region', 'Soil_Type', 'Crop', 'Weather_Condition'])
for col in sample.columns:
    if col in encoders:
        sample[col] = encoders[col].transform(sample[col])

yield_pred = model.predict(sample)[0]
print(f"Predicted Yield: {yield_pred:.2f} tons/ha")

📁 Project Structure

Crop/
├── crop.ipynb                 # Classification model notebook
├── yield.ipynb                # Regression model notebook
├── datasets/                  # Place CSV files here
│   └── .gitkeep
├── .gitignore                 # Git ignore rules
├── README.md                  # Documentation
├── req.txt                    # Dependencies
└── (generated) *.pkl          # Saved models (auto-generated)

📦 Dependencies

numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
jupyter==1.0.0

🔄 Model Output Files

After running notebooks:

XGBoostcrop.pkl - Crop classifier with label encoder

XGBoost_Yield.pkl - Yield regressor

LabelEncoders_yield.pkl - Categorical encoders for yield model

🤝 Contributing
Fork the repository

Create feature branch (git checkout -b feature/amazing)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing)

Open Pull Request

📄 License
MIT License - See LICENSE file for details

👨‍🌾 Acknowledgements
Dataset sources (update with your data sources)

XGBoost developers

Agricultural research partners