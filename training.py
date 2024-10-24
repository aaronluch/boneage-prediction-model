import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from loading import load_images_from_csv

boneage_threshold = 100

# load the training images and labels (set a limit for testing purposes)
train_images, train_labels = load_images_from_csv(
    'data/boneage-training-dataset.csv', 
    'data/boneage-training-dataset/boneage-training-dataset', 
    threshold=boneage_threshold, limit=6000
)

# define the CNN model
def create_model(input_shape):
    model = Sequential()

    # First convolutional block with LeakyReLU and Batch Normalization
    model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Second convolutional block with LeakyReLU and Batch Normalization
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Third convolutional block + MaxPooling
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth convolutional block (fewer filters, no max pooling)
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # Fifth convolutional block + MaxPooling - making this last for now
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(alpha=0.1))

    # Last block: no max pooling, more filters
    # model.add(Conv2D(1024, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.1))

    # Global Average Pooling instead of Flatten
    model.add(GlobalAveragePooling2D())

    # Fully connected layers with regularization and reduced Dropout
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer with slightly increased learning rate
    optimizer = Adam(learning_rate=0.0005, decay = .00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# for debugging purposes
def debug_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# set input shape for the model
input_shape = (256, 256, 1) 

# create the model
model = create_model(input_shape)

# summary of the model architecture
model.summary()

# add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
model.fit(train_images, train_labels, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])