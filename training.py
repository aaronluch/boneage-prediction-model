import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from loading import load_images_from_csv

# load the training images and labels (set a limit for testing purposes)
train_images, train_boneage, train_gender = load_images_from_csv(
    'data/boneage-training-dataset.csv', 
    'data/boneage-training-dataset/boneage-training-dataset', 
    limit=500
)

# define the CNN model
def create_model(input_shape):
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
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

    # fifth convolutional layer (added for more complexity)
    model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))

    # flatten the output from the convolutional layers
    model.add(Flatten())

    # fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # output layer (binary classification: 0 or 1)
    model.add(Dense(1, activation='sigmoid'))

    # optimizer for learning rate control
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
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
model.fit(train_images, train_gender, 
          epochs=5, 
          batch_size=32, 
          validation_split=0.3,
          callbacks=[early_stopping])