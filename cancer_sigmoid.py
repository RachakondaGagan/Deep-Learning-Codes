from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model using Sigmoid
model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Sigmoid Model Accuracy:", accuracy * 100)