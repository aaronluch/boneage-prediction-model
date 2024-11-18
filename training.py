import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from loading import load_images_from_csv
from datagenerator import data_generator

print("start")

# params
csv_path = 'data/boneage-training-dataset.csv'
image_dir = 'data/boneage-training-dataset/boneage-training-dataset'
batch_size = 16
threshold = 100

# load and shuffle the data
data = pd.read_csv(csv_path).sample(frac=1, random_state=42)

limit = len(data)
print(limit)

# apply limit if specified
if limit is not None:
    data = data.head(limit)

# split the data into training, validation, and test sets
train_size = int(0.6 * len(data))
val_size = int(0.2 * len(data))
test_size = len(data) - train_size - val_size

# assign the data to the respective sets
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Create data generators for training, validation, and testing
train_generator = data_generator(train_data, image_dir, batch_size=batch_size, threshold=threshold)
val_generator = data_generator(val_data, image_dir, batch_size=batch_size, threshold=threshold)
test_generator = data_generator(test_data, image_dir, batch_size=batch_size, threshold=threshold)

# calculate the number of steps per epoch for training and validation
steps_per_epoch_train = len(train_data) // batch_size
steps_per_epoch_val = len(val_data) // batch_size
steps_per_epoch_test = len(test_data) // batch_size

# set input shape for the model
input_shape = (224, 224, 3) # 3 channels for VGG16 compatibility

print("loaded images")

# the data is split as follows:
# - X_train, y_train: 60% of the data
# - X_val, y_val: 20% of the data (validation set)
# - X_test, y_test: 20% of the data (test set)

print("about to create model")
# define the CNN model
def create_model(input_shape):
    # load base VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # create new model with sequential and VGG16 base
    model = Sequential()
    model.add(base_model)

    # due to hardware constraints, we're just going to purely use the base model

    # pool and flatten the output before the fully connected layers
    model.add(GlobalAveragePooling2D())

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer and compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# function to calculate sensitivty and specificity
def calculate_sensitivity_specificity(y_true, y_pred, thresholds):
    sensitivity = []
    specificity = []

    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)

        # calc confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0,1]).ravel()

        # sensitivity (true positive rate)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity.append(sens)

        # specificity (true negative rate)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    
    return sensitivity, specificity

# create the model
model = create_model(input_shape)

# summary of the model architecture
model.summary()

# add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')

# introduce reduce lr on plateau to adjust learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=50,
    callbacks=[reduce_lr, early_stopping],
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val
)

# Evaluate the model on the test set (final evaluation after training)
test_loss, test_accuracy = model.evaluate(test_generator, steps=steps_per_epoch_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Add test loss and accuracy to the history object
history.history['test_loss'] = test_loss * np.ones(len(history.history['loss']))
history.history['test_accuracy'] = test_accuracy * np.ones(len(history.history['accuracy']))

# Predict on the test set using the generator
y_pred_test = model.predict(test_generator, steps=steps_per_epoch_test)

# Gather true labels for the test set
# We need to manually gather these because we are using a generator
y_test = []
for _, labels in test_generator:
    y_test.extend(labels)
    if len(y_test) >= test_size:  # Ensure we only collect as many as there are in the test set
        y_test = y_test[:test_size]
        break
y_test = np.array(y_test)

# Define thresholds for sensitivity and specificity
thresholds = np.arange(0, 1.01, 0.01)

# Calculate sensitivity and specificity
sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_pred_test, thresholds)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)


# # save the history object
# with open('model/history200epochsNOED.pkl', 'wb') as file:
#     pickle.dump(history.history, file)

# # save test predictions
# with open('model/test_predictions200epochsNOED.pkl', 'wb') as file:
#     pickle.dump(y_pred_test, file)

# # save model with keras' save function
# model.save('model/boneage_model200epochsNOED.h5')

# Plot training & validation loss values
plt.figure(figsize=(16, 9))

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

# Plot sensitivity and specificity
plt.subplot(2, 2, 3)
plt.plot(thresholds, sensitivity, label='Sensitivity (True Positive Rate)', color='blue')
plt.plot(thresholds, specificity, label='Specificity (True Negative Rate)', color='green')
plt.title('Sensitivity and Specificity')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.legend(loc='best')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()