# Bone Age Prediction Model

## Project Overview
This project leverages machine learning techniques to automate the assessment of pediatric bone age using hand X-ray images. Accurate bone age prediction is essential for diagnosing growth disorders, monitoring development, and addressing endocrine issues in children. Traditional methods are time-consuming and prone to variability, making automation a valuable tool.

We developed and evaluated various models, including custom convolutional neural networks (CNNs), traditional machine learning approaches, and pretrained architectures, to classify bone age as above or below 100 months (~8.3 years).

## Dataset
- **Source**: [RSNA Bone Age Dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age/data)
- **Details**: 
  - Grayscale X-ray images of pediatric hands (digital and scanned formats).
  - Metadata includes bone age (in months) and gender.
- **Preprocessing**:
  - Images resized to 224x224 pixels.
  - Data augmentation applied (rotation, brightness adjustment, flipping, etc.).
  - Normalized pixel values to [0,1].
  - Converted images into tensors for compatibility with machine learning frameworks.

## Models
### **1. Custom Binary CNN**
- Tailored for task-specific feature extraction in X-ray data.
- Architecture: 4 convolutional blocks, dropout regularization, global average pooling.
### **2. Custom Linear Regression CNN**
- Utilizes same architecture as binary CNN implementation; final Dense layer modified for appropriate output.
- Uses continuous bone age predictions as an alternative to binary classification.
- Architecture: 4 convolutional blocks, dropout regularization, global average pooling.

### **3. Traditional Models**
- **Support Vector Machine (SVM)**:
  - Utilizing PCA for dimensionality reduction and hyperparameter tuning for the SVM model.
- **Random Forest**:
  - Hyperparameter tuning using GridSearchCV to find best fit for the model.

### **4. Pretrained Architectures**
- **MobileNetV2**: Within both the `bin_cnn_train.py` and `lin_cnn_train.py` files, there is a function defined to create a model using MobileNetV2, which is commented out but ready for use.
  - You can also replace MobileNetV2 with any accepted models from `tensorflow.keras.applications`

## Repository Contents
- `~`: Contains scripts for all models and scripts.
- `output/`: Associated images and text files for results from the specified model directories inside.
- `model/`: Pretrained and saved models for reproducibility.

## Reproducibility
1. Download the RSNA Bone Age Dataset from Kaggle and place it in the appropriate directory.
2. Run the preprocessing scripts to prepare the data.
3. Use the model training scripts to train models from scratch or load pretrained models from the `model/` directory.
4. Evaluate the models using the provided evaluation scripts.

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib
- Pillow