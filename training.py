import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from loading import load_images_from_csv

# Enable memory growth for the GPU to avoid full allocation at the beginning
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the training dataset paths and labels using `load_images_from_csv` from `loading.py`
batch_size = 32
dataset = load_images_from_csv(
    csv_path='data/boneage-training-dataset.csv',
    image_dir='data/boneage-training-dataset/boneage-training-dataset',
    threshold=100,
    limit=None,  # Load the entire dataset
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
    base_model = MobileNetV2(input_shape=input_shape)

    model = Sequential([
        base_model,
        Dense(1, activation='sigmoid')
    ])

    # Compile the model with mixed precision policy
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = create_model(input_shape)

# Summary of the model architecture
model.summary()

# Early stopping and learning rate scheduling callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

# Train the model using the `tf.data` datasets
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=400,
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Collect predictions and true labels from the test dataset
y_true_test = []
y_pred_test = []

for image_batch, label_batch in test_dataset:
    predictions = model.predict(image_batch)
    y_pred_test.extend(predictions)
    y_true_test.extend(label_batch.numpy())

# Convert predictions to binary values (0 or 1) based on a threshold of 0.5
y_pred_test = np.array(y_pred_test)
y_true_test = np.array(y_true_test)
y_pred_binary = (y_pred_test >= 0.5).astype(int)

# Calculate sensitivity and specificity using confusion_matrix
from sklearn.metrics import confusion_matrix

thresholds = np.arange(0, 1.01, 0.01)

# def calculate_sensitivity_specificity(y_true, y_pred, thresholds):
#     sensitivity = []
#     specificity = []

#     for threshold in thresholds:
#         y_pred_binary = (y_pred >= threshold).astype(int)

#         # Calculate confusion matrix
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

#         # Sensitivity (true positive rate)
#         sens = tp / (tp + fn) if (tp + fn) > 0 else 0
#         sensitivity.append(sens)

#         # Specificity (true negative rate)
#         spec = tn / (tn + fp) if (tn + fp) > 0 else 0
#         specificity.append(spec)

#     return sensitivity, specificity

# # Calculate sensitivity and specificity
# sensitivity, specificity = calculate_sensitivity_specificity(y_true_test, y_pred_test, thresholds)


# Plot training, validation, and test results
plt.figure(figsize=(16, 9))

# Plot loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot accuracy
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='green', linestyle='--', label='Test Accuracy')  # Dotted green line for test accuracy
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()
