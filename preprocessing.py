from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# load an image from the file path
def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return image

# resize the image to the target size
def resize_image(image, target_size=(256,256)):
    return image.resize(target_size)

# normalize the image pixel values
def normalize_image(image):
    image_array = np.array(image)
    return image_array / 255.0

# augmenting (important) and converting the image to a tensor for TensorFlow
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.2,0.5],
        zoom_range=0.25,
        fill_mode='nearest',
        horizontal_flip=True
    )

    # Convert PIL image to numpy array and add channel dimension (grayscale)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension (required by datagen)
    image_array = np.expand_dims(image_array, axis=0)

    # Generate augmented image and squeeze back to 2D
    augmented_image = datagen.flow(image_array, batch_size=1)[0]
    augmented_image = augmented_image.squeeze()

    # Normalize pixel values to [0, 1] after augmentation
    augmented_image = augmented_image / 255.0

    return augmented_image

# finally, convert the image to a TensorFlow tensor
def convert_to_tensor(image):
    return tf.convert_to_tensor(image, dtype=tf.float32)

# encapsulate all the preprocessing steps into one function
def preprocess_image(image_path):
    image = load_image(image_path)       # Load and convert to grayscale
    image = resize_image(image)          # Resize to target dimensions
    image = augment_image(image)         # Augment the image
    image = convert_to_tensor(image)     # Convert to TensorFlow tensor
    return image