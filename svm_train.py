# SVM model training using sklearn
from loading import load_images_for_sklearn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import time
import pickle

# Load training, validation, and test datasets for sklearn models
train_csv = 'data/boneage-training-dataset.csv'
image_dir = 'data/boneage-training-dataset/boneage-training-dataset'
# split the data
X_train, y_train, X_val, y_val, X_test, y_test = load_images_for_sklearn(
    csv_path='data/boneage-training-dataset.csv',
    image_dir='data/boneage-training-dataset/boneage-training-dataset',
    threshold=100,
    train_size=0.8,
    val_size=0.15,
    test_size=0.05,
    limit=None
)

# PCA the data
pca = PCA(n_components=.95, random_state=42)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

print("Starting SVM Training...")
start_time = time.time()

# Train SVM
svm_model = make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear', class_weight='balanced', C=0.01))
svm_model.fit(X_train_reduced, y_train)

# save the model
with open('model/svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

# Evaluate SVM
y_pred = svm_model.predict(X_test_reduced)
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred))

print("SVM Training Complete.")
end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# graphing stuff
# Collect predictions and true labels from the test dataset
y_true_test = y_test  # True labels
y_pred_test = svm_model.predict_proba(X_test_reduced)[:, 1]  # Predicted probabilities for the positive class
print("Min Probability:", np.min(y_pred_test))
print("Max Probability:", np.max(y_pred_test))

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_test)

# Convert predictions to binary values
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold}")
y_pred_binary = (y_pred_test >= optimal_threshold).astype(int)
# print("Binary Predictions:", y_pred_binary)

# F1 Score
f1 = f1_score(y_true_test, y_pred_binary)
print(f"F1 Score: {f1}")

# AUC-ROC Score
auc_score = roc_auc_score(y_true_test, y_pred_test)
print(f"AUC-ROC Score: {auc_score}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.text(0.6, 0.2, f"F1 Score = {f1:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")
plt.legend()

# Confusion Matrix
cm = confusion_matrix(y_true_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()

# Create a 2x2 grid for the confusion matrix
conf_matrix = np.array([[tn, fp],
                        [fn, tp]])

fig, ax = plt.subplots(figsize=(10, 6))
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

# Show all plots
plt.show()