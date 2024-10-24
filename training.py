import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from loading import load_images_from_csv
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load the training images and labels (set a limit for testing purposes)
train_images, train_boneage, train_gender = load_images_from_csv(
    'data/boneage-training-dataset.csv', 
    'data/boneage-training-dataset/boneage-training-dataset', 
    limit=5000
)

# define the CNN model
def create_model(input_shape):
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second convolutional layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third convolutional layer
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # fourth convolutional layer (added for more complexity)
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten the output from the convolutional layers
    model.add(Flatten())

    # fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # output layer (binary classification: 0 or 1)
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# set input shape for the model
input_shape = (256, 256, 1) 

# create the model
model = create_model(input_shape)

# summary of the model architecture
model.summary()

# train the model
model.fit(train_images, train_gender, epochs=5, batch_size=32, validation_split=0.2)