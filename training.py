import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# First, split the dataset into training and validation sets (60% train, 40% val)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42, shuffle=True)

# Further split the validation data into validation and test sets (50% of the validation set for testing)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, shuffle=True)

# the data is split as follows:
# - X_train, y_train: 60% of the data
# - X_val, y_val: 20% of the data (validation set)
# - X_test, y_test: 20% of the data (test set)

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
    model.add(MaxPooling2D(pool_size=(2, 2))) # downsample the output

    model.add(Conv2D(16, kernel_size=(3, 3))) # reduce the number of filters after downsampling
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(512, kernel_size=(3, 3))) # increase the number of filters again
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    # pool and flatten the output before the fully connected layers
    model.add(GlobalAveragePooling2D())

    # fully connected layers with regularization
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer
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
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')
#callbacks=[early_stopping],

# introduce reduce lr on plateau to adjust learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# train the model
history = model.fit(X_train, y_train, 
                    epochs=50, batch_size=32, 
                    callbacks=[reduce_lr], 
                    validation_data=(X_val, y_val))

# Evaluate the model on the test set (final evaluation after training)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict on the test set
y_pred_test = model.predict(X_test)

# Convert the test predictions to binary format (0 or 1)
y_pred_test_bin = (y_pred_test >= 0.5).astype(int)

# Print the last 50 predictions and actual values for the test set
test_size = len(y_test)
startind = max(0, test_size - 50)
print("\nTest Set Predictions (last 50 samples):")
for i in range(startind, test_size):
    print(f"Test Sample {i + 1}: Predicted: {y_pred_test[i][0]}, Actual: {y_test[i]}")

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.legend(loc='upper right')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()