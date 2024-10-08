from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # this is important, its in the kaggle dataset example code a lot

def load_image(image_path):
    image = Image.open(image_path)
    return image

def resize_image(image, target_size=(256,256)):
    return image.resize(target_size)

def normalize_image(image):
    image_array = np.array(image)
    return image_array / 255.0

# augment is just starter code bc the library is broken on my mac
# but it's important like said before
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

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def preprocess_image(image):
    image = load_image(image)
    image = resize_image(image)
    image = normalize_image(image)
    image = augment_image(image)
    image = convert_to_grayscale(image)
    return image