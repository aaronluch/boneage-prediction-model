import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

# set input shape for the model
input_shape = (256, 256, 1) 

# create the model
model = create_model(input_shape)

# summary of the model architecture
model.summary()

# add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')

# introduce reduce lr on plateau to adjust learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# train the model
history = model.fit(X_train, y_train, 
                    epochs=100, batch_size=32, 
                    callbacks=[reduce_lr, early_stopping], 
                    validation_data=(X_val, y_val))

# Evaluate the model on the test set (final evaluation after training)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# add test loss and accuracy to the history object
history.history['test_loss'] = test_loss * np.ones(len(history.history['loss']))
history.history['test_accuracy'] = test_accuracy * np.ones(len(history.history['accuracy']))

# Predict on the test set
y_pred_test = model.predict(X_test)

# define thresholds for sensitivity and specificity
thresholds = np.arange(0, 1.01, 0.01)

# calculate sensitivity and specificity
sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_pred_test, thresholds)

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