# Spoiled Fruit Detection

## Overview
Spoiled Fruit Detection is a computer vision project that classifies fruits as **Fresh** or **Rotten** using a TensorFlow-based CNN, enhanced by fractional codes and the Potentials Method. A Flask web app provides real-time predictions, ideal for agricultural quality control. Showcased in my [Upwork portfolio](https://www.upwork.com/freelancers/~your-profile-link).

![Upload Interface](images/app_image1.png)
![Prediction Results](images/app_image2.png)

## Features
- Classifies fruits as Fresh or Rotten with high accuracy.
- Uses fractional codes and Potentials Method for feature extraction.
- Flask app for user-friendly image uploads and predictions.
- Visualizes HSV histograms and GLCM features.

## Technologies
- **Python**: TensorFlow, Keras, OpenCV, Flask
- **Dataset**: 2.1 GB fruit images (Fresh/Rotten)
- **Tools**: Jupyter Notebook, Git

## How to Run?
### STEPS:

Clone the repository
```bash
Project repo: https://github.com/AdiyanAndranik/spoiled-fruit-detection.git
```

### STEP 01 - Create a virtual environment
```bash
python -m venv venv
```
For Windows:
```bash
.\venv\Scripts\activate
```
For Linux/Mac:
```bash
source venv/bin/activate
```

### STEP 02 - Install requirements
```bash
pip install -r requirements.txt
```

### STEP 03 - Add dataset
- Place fruit images in `data/Fresh/` and `data/Rotten/` (2.1 GB dataset, contact me for access).
- Update dataset path in `config.py` if needed.

### STEP 04 - Run the Flask app
```bash
python app.py
```
Now, open:
```bash
http://localhost:5000
```

### STEP 05 - Train the model
- Navigate to:
  ```bash
  http://localhost:5000/train
  ```
- This triggers the training pipeline and returns "Training Successful!!" upon completion.

## Usage
- **Train**: Visit `http://localhost:5000/train` to train the model via the Flask app.
- **Predict**: Use the Flask app at `http://localhost:5000` to upload fruit images for classification.

## Dataset
Uses a 2.1 GB dataset of fruit images (not included). Contact me for access or organize your dataset in `data/Fresh/` and `data/Rotten/`.

## License
MIT License. See [LICENSE](LICENSE).

## Contact
- GitHub: [AdiyanAndranik](https://github.com/AdiyanAndranik)
- Upwork: [Your Upwork Profile](https://www.upwork.com/freelancers/~your-profile-link)