import tensorflow as tf # still broken thanks mac
import numpy as np
import preprocessing as prep
import os

# we also need to implement image correlation with the csv file
def load_and_preprocess_images(image_paths):
    images = []
    labels = []

    for image_path in image_paths:
        prep_image = prep(image_path)
        images.append(np.array(prep_image))
        labels.append(label)

    return np.array(images), np.array(labels)