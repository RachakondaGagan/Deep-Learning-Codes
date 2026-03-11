# Ridge Regression (L2 Regularization)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge Model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Prediction
y_pred = ridge.predict(X_test)
y_pred = [1 if i >= 0.5 else 0 for i in y_pred]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Ridge Regression Accuracy:", accuracy * 100)
