## Web Application

The project includes a complete Flask web application with:

- **User Authentication**: Registration and login system
- **MRI Upload**: Upload brain MRI images for analysis
- **Real-time Prediction**: DenseNet121 model for tumor classification
- **Grad-CAM Visualization**: Heatmap overlay showing tumor location
- **Tumor Information**: Detailed info about each tumor type including:
  - Symptoms, causes, treatment options
  - Prevention tips and prognosis
- **User Dashboard**: View prediction history
- **Profile Management**: Update personal information

### Run the Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

# Open browser and navigate to
http://localhost:5000


Dataset Link:
url = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
