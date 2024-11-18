import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Input
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
print ("set limit")

# Load the training images and labels with the determined limit
train_images, train_labels = load_images_from_csv(
    'data/boneage-training-dataset.csv', 
    'data/boneage-training-dataset/boneage-training-dataset', 
    threshold=boneage_threshold, limit=limit
)
print("loaded images")

# convert images to rgb for vgg16
train_images = np.expand_dims(train_images, axis=-1)  # Add channel dimension for grayscale
train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
X_train = tf.image.grayscale_to_rgb(train_images_tensor).numpy()
y_train = train_labels
print("converted images to rgb")

# setup train and test as X and y
# X_train = train_images
# y_train = train_labels

# split the dataset into training and validation sets (60% train, 40% val)
X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), y_train, test_size=0.4, random_state=42, shuffle=True)

# Further split the validation data into validation and test sets (50% of the validation set for testing)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42, shuffle=True)
print("split data")

# set input shape for the model
input_shape = (128, 128, 3) # 3 channels for MobileNetV2 compatibility

print("loaded images")

# the data is split as follows:
# - X_train, y_train: 60% of the data
# - X_val, y_val: 20% of the data (validation set)
# - X_test, y_test: 20% of the data (test set)

print("about to create model")
# define the CNN model
def create_model(input_shape):
    # load base MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-10]:  # Freeze all but the last 20 layers
        layer.trainable = False


    # create new model with sequential and MobileNetV2 base
    model = Sequential()
    model.add(base_model)

    # due to hardware constraints, we're just going to purely use the base model

    # pool and flatten the output before the fully connected layers
    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

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
history = model.fit(X_train, y_train, 
                    epochs=20, batch_size=8, 
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