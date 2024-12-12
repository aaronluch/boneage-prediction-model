"""
Script responsible for preprocessing images and labels for training and testing.
This script contains functions to load images, preprocess them, and encapsulate them into TensorFlow datasets for training and testing.
It is used in the loading.py script to create datasets for repsective tasks.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=5,
    shear_range=0.01,
    fill_mode='nearest',
    zoom_range=0.25
)

# Load an image from the file path and decode it to RGB
def load_image(file_path, target_size=(224, 224)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    return image

# Augment and normalize the image using ImageDataGenerator
def augment_image(image):
    image_array = image.numpy()
    image_array = np.expand_dims(image_array, axis=0)
    augmented_image_iter = datagen.flow(image_array, batch_size=1)
    augmented_image_array = next(augmented_image_iter)[0]
    augmented_image = augmented_image_array / 255.0
    # Convert back to tensor
    return tf.convert_to_tensor(augmented_image, dtype=tf.float32)

# normalizing the label
def normalize_label(label, min_value=0, max_value=228):
    return (label - min_value) / (max_value - min_value)

# preprocess image for cnn
def preprocess_image(file_path, label, img_size=(224, 224)):
    image = load_image(file_path, target_size=img_size)
    image = tf.py_function(func=augment_image, inp=[image], Tout=tf.float32)
    label = tf.py_function(func=normalize_label, inp=[label], Tout=tf.float32)
    return image, label

# Creates a dataset for regression
def create_tf_dataset_regression(file_paths, labels, img_size=(224, 224), batch_size=32):
    def preprocess(file_path, label):
        return preprocess_image(file_path, label, img_size)
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# Encapsulate the image and label into a dataset
@tf.function
def preprocess_image(file_path, label, img_size=(224, 224)):
    image = load_image(file_path, target_size=img_size)
    image = tf.py_function(func=augment_image, inp=[image], Tout=tf.float32)
    return image, label

# Data loading pipeline
def create_tf_dataset(file_paths, labels, img_size=(224, 224), batch_size=32):
    def preprocess(file_path, label):
        return preprocess_image(file_path, label, img_size)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# Convert an image to a flat tensor
def flatten_image(image):
    return tf.reshape(image, [-1])

# Preprocess an image: load, resize, normalize, and flatten for sklearn models (SVM, Random Forest)
def preprocess_image_for_sklearn(file_path, label, img_size=(224, 224)):
    # Load and resize the image
    image = Image.open(file_path).convert('RGB')  # Ensure RGB format
    image = image.resize(img_size)

    # Convert image to numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0

    # Flatten the image into a 1D vector
    flattened_image = image_array.flatten()

    return flattened_image, label

# Create a dataset as numpy arrays for SVM or Random Forest
def create_numpy_dataset(file_paths, labels, target_size=(224, 224)):
    images, labels_array = [], []
    for file_path, label in zip(file_paths, labels):
        # Preprocess each image
        flattened_image, label = preprocess_image_for_sklearn(file_path, label, target_size)
        images.append(flattened_image)
        labels_array.append(label)
    
    return np.array(images), np.array(labels_array)

# Process a data split into flattened tensors
def process_split(split_data, img_size=(224, 224)):
    images, labels = [], []
    for file_path, label in zip(split_data['file_path'], split_data['label']):
        # Preprocess each image
        flattened_image, label = preprocess_image_for_sklearn(file_path, label, img_size)
        images.append(flattened_image)
        labels.append(label)
    return np.array(images), np.array(labels)