"""
This script trains a Random Forest model on the Bone Age dataset using the scikit-learn library.
It performs hyperparameter tuning using GridSearchCV and evaluates the model on the validation and test sets.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import GridSearchCV
from loading import load_images_for_sklearn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import time
import numpy as np

csv_path = 'data/boneage-training-dataset.csv'
image_dir = 'data/boneage-training-dataset/boneage-training-dataset'

X_train, y_train, X_val, y_val, X_test, y_test = load_images_for_sklearn(
    csv_path=csv_path,
    image_dir=image_dir,
    threshold=100,
    img_size=(224, 224),
    train_size=0.8,
    val_size=0.15,
    test_size=0.05,
    limit=None # Load the full dataset or set a limit for testing
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# start a timer to measure training time
start_time = time.time()
print("Training Random Forest...")

# pca the data
pca = PCA(n_components=.95, random_state=42)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
rf_model = grid_search
rf_model.fit(X_train_reduced, y_train)

# Get the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# end the timer
end_time = time.time()
print(f"Complete. Training Time: {end_time - start_time:.2f} seconds")

# Predict on the validation set
y_val_pred = rf_model.predict(X_val_reduced)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model on the test set
y_test_pred = rf_model.predict(X_test_reduced)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")

# Collect predictions and true labels from the test dataset
y_true_test = y_test  # True labels
y_pred_test = rf_model.predict_proba(X_test_reduced)[:, 1]  # Predicted probabilities for the positive class
print("Predicted Probabilities (y_pred_test):", y_pred_test[-20:])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true_test, y_pred_test)

# Determine the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold}")

# Convert probabilities to binary predictions using the optimal threshold
y_pred_binary = (y_pred_test >= optimal_threshold).astype(int)
print("Unique values in Binary Predictions:", np.unique(y_pred_binary))

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

for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{conf_matrix[i, j]}",
                ha="center", va="center", color="black", fontsize=12)

# Add color bar for reference
plt.colorbar(im, ax=ax)
plt.show()