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
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.2,0.5],
        zoom_range=0.25,
        fill_mode='nearest',
        horizontal_flip=True
    )

    # convert PIL image to np array and add batch dimension
    image_array = np.array(image)

    # add channel dimension (grayscale)
    image_array = np.expand_dims(image_array, axis=-1)

    # add batch dimension (required by datagen)
    image_array = np.expand_dims(image_array, axis=0)

    # generate augmented image
    augmented_image = datagen.flow(image_array, batch_size=1)[0]

    # return to 2d
    augmented_image = augmented_image.squeeze()

    # normalize pixel values [0, 1] range
    augmented_image = augmented_image / 255.0

    return augmented_image

# finally, convert the image to a TensorFlow tensor
def convert_to_tensor(image):
    return tf.convert_to_tensor(image)

# encapsulate all the preprocessing steps into one function
def preprocess_image(image):
    image = load_image(image)
    image = resize_image(image)
    image = augment_image(image)
    image = normalize_image(image)
    image = convert_to_tensor(image)
    return image