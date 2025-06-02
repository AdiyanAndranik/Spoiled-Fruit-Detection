# Spoiled Fruit Detection

## Overview
Spoiled Fruit Detection is a computer vision project that classifies fruits as **Fresh** or **Rotten** using a TensorFlow-based CNN, enhanced by fractional codes and the Potentials Method. A Flask web app provides real-time predictions, ideal for agricultural quality control.

![Application Screenshot](images/app_image1.png)
![Application Screenshot](images/app_image2.png)

## Features
- Classifies fruits as Fresh or Rotten with high accuracy.
- Uses fractional codes and Potentials Method for feature extraction.
- Flask app for user-friendly image uploads and predictions.
- Visualizes features.

## Technologies
- **Python**: TensorFlow, Keras, OpenCV, Flask
- **Tools**: Jupyter Notebook, Git

## Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/AdiyanAndranik/Spoiled-Fruit-Detection.git
   cd spoiled-fruit-detection
   ```

2. **Set Up Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Train Model**:
   ```bash
   python train_model.py
   ```

4. **Run Flask App**:
   ```bash
   python app.py
   ```
   - Visit `http://localhost` to upload images.


## Usage
- **Train**: Run `train_model.py` to train the model.
- **Predict**: Use the Flask app to upload fruit images for classification.

## Dataset
Uses a 2.1 GB dataset of fruit images (not included).

