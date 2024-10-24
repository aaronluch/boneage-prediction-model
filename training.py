import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from loading import load_images_from_csv

# load the training images and labels (set a limit for testing purposes)
df = pd.read_csv('data/boneage-training-dataset.csv')

boneage_threshold = 100 # threshold (months) as decision boundary for binary classification
limit = len(df['id'])

# Load the training images and labels with the determined limit
train_images, train_labels = load_images_from_csv(
    'data/boneage-training-dataset.csv', 
    'data/boneage-training-dataset/boneage-training-dataset', 
    threshold=boneage_threshold, limit=limit
)

# setup train and test as X and y
X_train = train_images
y_train = train_labels

# split data manually to access validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# define the CNN model
def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # pool and flatten the output before the fully connected layers
    model.add(GlobalAveragePooling2D())

    # fully connected layers with regularization
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# set input shape for the model
input_shape = (256, 256, 1) 

# create the model
model = create_model(input_shape)

# summary of the model architecture
model.summary()

# add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[early_stopping], validation_data=(X_val, y_val))

# evaluate the model
y_pred_val = model.predict(X_val)
y_pred_bin = (y_pred_val >= 0.5).astype(int)

# print the last 100 predictions and actual values
val_size = len(y_val-50)
startind = val_size
for i in range(startind, 40):
    print(f"Validation Sample {i + 1}: Predicted: {y_pred_val[i][0]}, Actual: {y_val[i]}")

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()