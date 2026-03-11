import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])
w = 0.0
lr = 0.01
batch_size = 2

for _ in range(1000):
    idx = np.random.choice(len(y), batch_size)
    grad = np.mean((w*X[idx] - y[idx]) * X[idx])
    w -= lr * grad

pred = w * X
error = np.mean((y - pred)**2)
acc = (1 - error / np.var(y)) * 100

print("Mini-Batch GD")
print("Weight:", w)
print("Error:", error)
print("Accuracy:", acc, "%")