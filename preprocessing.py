import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the ImageDataGenerator for augmentation
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
    # Use TensorFlow functions to load and decode the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    return image

# Augment and normalize the image using ImageDataGenerator
def augment_image(image):
    # Convert TensorFlow tensor to a NumPy array, add a batch dimension for ImageDataGenerator
    image_array = image.numpy()  # Convert to NumPy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Generate augmented image (Note: only one augmented image is generated here)
    augmented_image_iter = datagen.flow(image_array, batch_size=1)
    augmented_image_array = next(augmented_image_iter)[0]  # Remove batch dimension

    # Normalize the augmented image
    augmented_image = augmented_image_array / 255.0

    # Convert back to TensorFlow tensor
    return tf.convert_to_tensor(augmented_image, dtype=tf.float32)

# Encapsulate all preprocessing steps into one function
@tf.function
def preprocess_image(file_path, label, img_size=(224, 224)):
    image = load_image(file_path, target_size=img_size)  # Load and resize image
    image = tf.py_function(func=augment_image, inp=[image], Tout=tf.float32)  # Augment the image using a custom function
    return image, label

# Efficient data loading and preprocessing using tf.data
def create_tf_dataset(file_paths, labels, img_size=(224, 224), batch_size=4):
    def preprocess(file_path, label):
        return preprocess_image(file_path, label, img_size)

    # Create a dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Apply preprocessing to each element in the dataset
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset to improve training speed
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
