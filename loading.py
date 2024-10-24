import numpy as np
import pandas as pd
import random
import os
from preprocessing import preprocess_image

def load_images_from_csv(csv_path, image_dir, threshold=100, limit=None):
    """
    Load image paths and create binary labels based on bone age threshold.
    
    Args:
        csv_path (str): Path to the CSV file (for train or test).
        image_dir (str): Directory where the images are stored.
        threshold (int): Bone age threshold for binary classification.
        limit (int): Number of images to load (for testing purposes).
    
    Returns:
        (tuple): Numpy arrays of preprocessed images and their corresponding binary labels.
    """
    # load the CSV
    data = pd.read_csv(csv_path)
    
    # extract image paths based on 'id' column
    image_paths = data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.png")).tolist()
    
    # create binary labels based on the bone age threshold
    boneage_labels = np.array(data['boneage'].apply(lambda x: 1 if x > threshold else 0).tolist())

    if limit is not None:
        image_paths = image_paths[:limit]
        boneage_labels = boneage_labels[:limit]
    
    preprocessed_images = []
    
    # preprocess images with error handling
    for image_path in image_paths:
        try:
            img = preprocess_image(image_path)
            preprocessed_images.append(np.array(img))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    # convert to NumPy array once at the end for efficiency
    preprocessed_images = np.array(preprocessed_images)
    
    return preprocessed_images, boneage_labels
