import numpy as np
import pandas as pd
import random
import os
from preprocessing import preprocess_image

def load_and_preprocess_images(csv_path, image_dir):
    """
    load image paths and labels from the csv, preprocess the images, and then split them into respective sets
    
    Args:
        csv_path (str): path to the the CSV file
        image_dir (str): directory where the images are stored
    
    Returns:
        (tuple): arrays of preprocessed images and labels for training, validation, and testing.
    """
    # load the CSV file
    data = pd.read_csv(csv_path)
    
    # extract image paths and labels
    image_paths = data['filename'].apply(lambda x: os.path.join(image_dir, x)).tolist()
    labels = data['label'].tolist()  # adjust the column name if necessary (e.g., 'age' or 'is_within_10_years')
    
    # combine paths and labels and shuffle together for random splitting
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    # calculate the split sizes
    total_size = len(image_paths)
    training_size = int(total_size * 0.65)
    validation_size = int(total_size * 0.20)
    testing_size = total_size - training_size - validation_size  # Ensuring all images are used

    # split into training, validation, and testing sets
    train_images = [preprocess_image(path) for path in image_paths[:training_size]]
    train_labels = labels[:training_size]
    
    val_images = [preprocess_image(path) for path in image_paths[training_size:training_size + validation_size]]
    val_labels = labels[training_size:training_size + validation_size]
    
    test_images = [preprocess_image(path) for path in image_paths[training_size + validation_size:]]
    test_labels = labels[training_size + validation_size:]

    # convert images to numpy arrays for TensorFlow compatibility
    train_images = np.array([np.array(img) for img in train_images])
    val_images = np.array([np.array(img) for img in val_images])
    test_images = np.array([np.array(img) for img in test_images])

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
