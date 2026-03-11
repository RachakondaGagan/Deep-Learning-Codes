# Breast Cancer Classification using Threshold Activation Function

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# Convert labels to {-1, +1}
y = np.where(y == 0, -1, 1)

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 4: Threshold Function
# -----------------------------
def threshold(x):
    return 1 if x >= 0 else -1

# -----------------------------
# Step 5: Perceptron Training
# -----------------------------
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.01
epochs = 100

for _ in range(epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(X_train[i], weights) + bias
        y_pred = threshold(linear_output)

        if y_train[i] != y_pred:
            weights += learning_rate * y_train[i] * X_train[i]
            bias += learning_rate * y_train[i]

# -----------------------------
# Step 6: Testing
# -----------------------------
y_pred_test = []

for x in X_test:
    output = np.dot(x, weights) + bias
    y_pred_test.append(threshold(output))

# Convert list to numpy array
y_pred_test = np.array(y_pred_test)

# -----------------------------
# Step 7: Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred_test)
print("Threshold Activation Model Accuracy:", accuracy * 100)
