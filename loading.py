import numpy as np
import pandas as pd
import random
import os
from preprocessing import preprocess_image

# CURRENTLY not needed as dataset pre-separated into training and testing sets
# def load_and_preprocess_images(csv_path, image_dir):
#     """
#     load image paths and labels from the csv, preprocess the images, and then split them into respective sets
    
#     Args:
#         csv_path (str): path to the the CSV file
#         image_dir (str): directory where the images are stored
    
#     Returns:
#         (tuple): arrays of preprocessed images and labels for training, validation, and testing.
#     """
#     # load the CSV file
#     data = pd.read_csv(csv_path)
    
#     # extract image paths and labels
#     image_paths = data['filename'].apply(lambda x: os.path.join(image_dir, x)).tolist()
#     labels = data['label'].tolist()  # adjust the column name if necessary (e.g., 'age' or 'is_within_10_years')
    
#     # combine paths and labels and shuffle together for random splitting
#     combined = list(zip(image_paths, labels))
#     random.shuffle(combined)
#     image_paths, labels = zip(*combined)
    
#     # calculate the split sizes
#     total_size = len(image_paths)
#     training_size = int(total_size * 0.65)
#     validation_size = int(total_size * 0.20)
#     testing_size = total_size - training_size - validation_size  # Ensuring all images are used

#     # split into training, validation, and testing sets
#     train_images = [preprocess_image(path) for path in image_paths[:training_size]]
#     train_labels = labels[:training_size]
    
#     val_images = [preprocess_image(path) for path in image_paths[training_size:training_size + validation_size]]
#     val_labels = labels[training_size:training_size + validation_size]
    
#     test_images = [preprocess_image(path) for path in image_paths[training_size + validation_size:]]
#     test_labels = labels[training_size + validation_size:]

#     # convert images to numpy arrays for TensorFlow compatibility
#     train_images = np.array([np.array(img) for img in train_images])
#     val_images = np.array([np.array(img) for img in val_images])
#     test_images = np.array([np.array(img) for img in test_images])

#     return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

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
    
    # preprocess images
    preprocessed_images = [preprocess_image(image_path) for image_path in image_paths]
    
    # convert to NumPy arrays for TensorFlow compatibility
    preprocessed_images = np.array([np.array(img) for img in preprocessed_images])
    
    return preprocessed_images, boneage_labels
