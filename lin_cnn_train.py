"""
lin_cnn_train.py

This script is used to train a Convolutional Neural Network (CNN) for linear regression of bone age from X-ray images.
The model predicts the bone age of a person from an X-ray image of their hand, in months.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from loading import load_images_for_regression
from keyboardinterrupt import KeyboardInterruptCallback

# Enable memory growth for the GPU to avoid full allocation at the beginning
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 32
dataset = load_images_for_regression(
    csv_path='data/boneage-training-dataset.csv',
    image_dir='data/boneage-training-dataset/boneage-training-dataset',
    limit=None, # Load the entire dataset
    batch_size=batch_size
)

# Split the dataset into training, validation, and test sets
train_size = 0.8
val_size = 0.15
test_size = 0.05

total_size = len(list(dataset))
train_count = int(total_size * train_size)
val_count = int(total_size * val_size)
test_count = total_size - train_count - val_count

# Shuffle and split the dataset
shuffled_dataset = dataset.shuffle(buffer_size=total_size, seed=42)
train_dataset = shuffled_dataset.take(train_count)
val_dataset = shuffled_dataset.skip(train_count).take(val_count)
test_dataset = shuffled_dataset.skip(train_count + val_count)

# Define the model
input_shape = (224, 224, 3)

def create_model(input_shape):
    model = Sequential([
        # conv block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # conv block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # conv block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # conv block 4
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        # fully connected layers
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear', kernel_regularizer=l2(0.01))
    ])

    # Compile the model with mixed precision policy
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Create the model
model = create_model(input_shape)

"""
If you're looking to use a pre-trained model like VGG16 or MobileNetV2, you can use the following
function to create the model instead of the custom CNN model above.

Just replace the model = create_model(input_shape) line with model = mobilenet_model(input_shape) or model = vgg16_model(input_shape).
# using mobilenetv2
def mobilenet_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


model = mobilenet_model(input_shape)
"""

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
keyboard_interrupt = KeyboardInterruptCallback()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stopping, keyboard_interrupt]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the model - commented out to avoid overwriting the model
# model.save('model/mobilenetV2_lin_cnn_model_adlucian.h5')

# Collect predictions and true labels from the test dataset
y_true_test = []
y_pred_test = []

for image_batch, label_batch in test_dataset:
    predictions = model.predict(image_batch)
    y_pred_test.extend(predictions.flatten())  # Flatten to 1D array
    y_true_test.extend(label_batch.numpy())

def display_random_predictions(test_dataset, model, num_samples=5):
    images = []
    true_labels = []
    pred_labels = []
    
    # Collect images and their predictions
    for image_batch, label_batch in test_dataset:
        predictions = model.predict(image_batch)
        images.extend(image_batch.numpy())
        true_labels.extend(label_batch.numpy())
        pred_labels.extend(predictions)
    
    # Randomly select num_samples
    selected_indices = random.sample(range(len(images)), num_samples)
    selected_images = [images[i] for i in selected_indices]
    selected_true_labels = [true_labels[i] for i in selected_indices]
    selected_pred_labels = [pred_labels[i] for i in selected_indices]
    
    # Plot the images with predictions and true labels
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(selected_images[i])
        plt.axis('off')
        plt.title(f"True: {selected_true_labels[i]}\nPred: {selected_pred_labels[i][0]:.2f}")
    plt.tight_layout()
    plt.show()

# Call the function
display_random_predictions(test_dataset, model, num_samples=5)

# Convert predictions and true labels to numpy arrays
y_pred_test = np.array(y_pred_test)
y_true_test = np.array(y_true_test)

# Calculate Mean Absolute Error and Mean Squared Error
mae = np.mean(np.abs(y_true_test - y_pred_test))
mse = np.mean((y_true_test - y_pred_test) ** 2)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# Plot Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_true_test, y_pred_test, alpha=0.6, edgecolor='k', label='Predictions')
plt.plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Bone Age')
plt.ylabel('Predicted Bone Age')
plt.title('Predicted vs. Actual Bone Age')
plt.legend()
plt.grid(True)

# Plot training and validation loss over epochs
plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()
