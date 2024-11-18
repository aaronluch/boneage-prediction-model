import numpy as np
import os
import tensorflow as tf
from preprocessing import preprocess_image

def data_generator(data, image_dir, batch_size, threshold=100):
    """
    A generator that yields batches of RGB images and labels.
    
    Args:
        data (DataFrame): DataFrame containing the data.
        image_dir (str): Directory where the images are stored.
        batch_size (int): The number of images to load in each batch.
        threshold (int): Bone age threshold for binary classification.

    Yields:
        (tuple): Batch of images and corresponding binary labels.
    """
    while True:  # Infinite loop to keep the generator running for epochs
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data.iloc[start:end]
            
            # Extract image paths and corresponding labels
            image_paths = batch_data['id'].apply(lambda x: os.path.join(image_dir, f"{x}.png")).tolist()
            boneage_labels = np.array(batch_data['boneage'].apply(lambda x: 1 if x > threshold else 0).tolist())
            
            # Preprocess the images
            images = [preprocess_image(image_path) for image_path in image_paths]
            
            # Convert the list of images to a NumPy array
            images = np.array(images)
            
            # Expand dimensions to add a channel dimension (for grayscale images)
            images = np.expand_dims(images, axis=-1)  # Now shape is (batch_size, height, width, 1)
            
            # Convert grayscale to RGB (required for VGG16)
            images = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images))
            images = images.numpy()  # Convert back to NumPy array
            
            # Yield the batch of images and labels
            yield images, boneage_labels
