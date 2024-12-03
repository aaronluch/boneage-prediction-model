import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from preprocessing import create_tf_dataset

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