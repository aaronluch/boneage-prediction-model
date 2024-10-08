from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# load an image from the file path
def load_image(image_path):
    image = Image.open(image_path)
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
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    # convert PIL image to np array and add batch dimension
    image_array = np.array(image)
    augmented_image = datagen.flow(image_array, batch_size=1)[0].astype(np.float32)
    return Image.fromarray((augmented_image[0] * 255).astype(np.uint8))

# convert the image to grayscale
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# finally, convert the image to a TensorFlow tensor
def convert_to_tensor(image):
    return tf.convert_to_tensor(image)

# encapsulate all the preprocessing steps into one function
def preprocess_image(image):
    image = load_image(image)
    image = resize_image(image)
    image = augment_image(image)
    image = convert_to_grayscale(image)
    image = normalize_image(image)
    image = convert_to_tensor(image)
    return image