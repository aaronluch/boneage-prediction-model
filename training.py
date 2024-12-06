import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from loading import load_images_from_csv
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from keyboardinterrupt import KeyboardInterruptCallback

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
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])

    # Compile the model with mixed precision policy
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = create_model(input_shape)
# Summary of the model architecture
model.summary()
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
keyboard_interrupt = KeyboardInterruptCallback()

# Train the model using the `tf.data` datasets
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
history.history['test_loss'] = test_loss * np.ones(len(history.history['loss']))
history.history['test_accuracy'] = test_accuracy * np.ones(len(history.history['accuracy']))

# Save the model
#model.save('model/model.h5')

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

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_test)

# calculate sensitivity and specificity
sensitivity = []
specificity = []
# Iterate through thresholds to calculate sensitivity (TPR) and specificity (TNR) to understand how the model balances positive and negative predictions at each threshold
for threshold in thresholds:
# The threshold is the cutoff point on a test score that determines the classification of a sample as positive or negative
    y_pred_binary = (y_pred_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_binary).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    
    sensitivity.append(sens)
    specificity.append(spec)

# Convert predictions to binary values (0 or 1) based on a threshold of 0.5
y_pred_test = np.array(y_pred_test)
y_true_test = np.array(y_true_test)
y_pred_binary = (y_pred_test >= 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_true_test, y_pred_binary)
print(f"F1 Score: {f1}")
# Calculate AUC-ROC
auc_score = roc_auc_score(y_true_test, y_pred_test)
print(f"AUC-ROC Score: {auc_score}")
# plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.text(0.6, 0.2, f"F1 Score = {f1:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend()

# confusion matrix
cm = confusion_matrix(y_true_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()
# Create a 2x2 grid for the confusion matrix
conf_matrix = np.array([[tn, fp],
                        [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest')

# Add titles and labels
ax.set_title("Confusion Matrix", fontsize=16)
ax.set_xlabel("Predicted Labels", fontsize=14)
ax.set_ylabel("True Labels", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticklabels(["Negative", "Positive"])

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{conf_matrix[i, j]}",
                ha="center", va="center", color="black", fontsize=12)

# Add color bar for reference
plt.colorbar(im, ax=ax)

# Plot Sensitivity vs. Specificity
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivity, label='Sensitivity (True Positive Rate)', color='blue')
plt.plot(thresholds, specificity, label='Specificity (True Negative Rate)', color='green')
plt.title('Sensitivity and Specificity')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.legend(loc='best')
plt.grid(True)

# Plot training, validation, and test results
plt.figure(figsize=(18, 9))
# Plot loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['test_loss'], label='Test Loss', linestyle='dashed', color='lime')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim(0,1)
plt.legend(loc='upper right')

# Plot accuracy
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['test_accuracy'], label='Test Accuracy', linestyle='dashed', color='lime')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim(0,1)
plt.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()