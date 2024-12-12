"""
Script responsible for loading images and labels from CSV files and creating datasets for training and testing.
"""

import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from preprocessing import create_tf_dataset, preprocess_image_for_sklearn, process_split, create_tf_dataset_regression
from sklearn.model_selection import train_test_split

# Used for Binary CNN models
def load_images_from_csv(csv_path, image_dir, threshold=100, limit=None, batch_size=8, img_size=(224, 224)):
    """
    Load image paths and create binary labels based on bone age threshold.

    Args:
        csv_path (str): Path to the CSV file (for train or test).
        image_dir (str): Directory where the images are stored.
        threshold (int): Bone age threshold for binary classification.
        limit (int): Number of images to load (for testing purposes).
        batch_size (int): Batch size for training dataset.
        img_size (tuple): Target size for resizing images.

    Returns:
        tf.data.Dataset: A dataset of preprocessed images and their corresponding binary labels.
    """
    # Load the CSV
    data = pd.read_csv(csv_path)

    # Extract image paths based on 'id' column
    data['file_path'] = data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    
    # Create binary labels based on the bone age threshold
    data['label'] = data['boneage'].apply(lambda x: 1 if x > threshold else 0)

    # Apply limit if specified (for testing purposes)
    if limit is not None:
        data = data.head(limit)

    # Use the create_tf_dataset from preprocessing.py to generate a dataset
    dataset = create_tf_dataset(data['file_path'], data['label'], img_size=img_size, batch_size=batch_size)

    return dataset

# Used for Linear Regression CNN models
def load_images_for_regression(csv_path, image_dir, img_size=(224, 224), limit=None, batch_size=32):
    """
    Load image paths and continuous labels for CNN-based regression.

    Args:
        csv_path (str): Path to the CSV file (for train or test).
        image_dir (str): Directory where the images are stored.
        img_size (tuple): Target size for resizing images.
        limit (int): Number of images to load (for testing purposes).
        batch_size (int): Batch size for training dataset.

    Returns:
        tf.data.Dataset: A dataset of preprocessed images and their corresponding labels.
    """
    # Load the CSV
    data = pd.read_csv(csv_path)

    # Extract image paths based on 'id' column
    data['file_path'] = data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    
    # Use 'boneage' directly as the label for regression
    data['label'] = data['boneage']

    # Apply limit if specified
    if limit is not None:
        data = data.head(limit)

    # Use the create_tf_dataset_regression function to generate a dataset
    dataset = create_tf_dataset_regression(data['file_path'], data['label'], img_size=img_size, batch_size=batch_size)

    return dataset


# Used for sklearn models (SVM, Random Forest, etc.)
def load_images_for_sklearn(csv_path, image_dir, threshold=100, img_size=(224, 224), 
                            train_size=0.8, val_size=0.15, test_size=0.05, limit=None):
    """
    Load image paths, preprocess images for sklearn models, and split into train, val, and test sets.

    Args:
        csv_path (str): Path to the CSV file (for train or test).
        image_dir (str): Directory where the images are stored.
        threshold (int): Bone age threshold for binary classification.
        img_size (tuple): Target size for resizing images.
        train_size (float): Proportion of the dataset to use for training.
        val_size (float): Proportion of the dataset to use for validation.
        test_size (float): Proportion of the dataset to use for testing.
        limit (int, optional): Number of images to load. If None, load the entire dataset.

    Returns:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        X_test, y_test: Test features and labels.
    """
    # Load the CSV
    data = pd.read_csv(csv_path)

    # Extract image paths based on 'id' column
    data['file_path'] = data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    
    # Create binary labels based on the bone age threshold
    data['label'] = data['boneage'].apply(lambda x: 1 if x > threshold else 0)

    # Apply limit if specified
    if limit is not None:
        data = data.head(limit)

    # Shuffle and split the data into train, val, and test sets
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=val_size / (train_size + val_size), random_state=42)

    # Process each split
    X_train, y_train = process_split(train_data, img_size)
    X_val, y_val = process_split(val_data, img_size)
    X_test, y_test = process_split(test_data, img_size)

    return X_train, y_train, X_val, y_val, X_test, y_test