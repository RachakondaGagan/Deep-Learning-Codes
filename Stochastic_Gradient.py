import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])
w = 0.0
lr = 0.01

for _ in range(1000):
    i = np.random.randint(len(y))
    grad = (w*X[i] - y[i]) * X[i]
    w -= lr * grad

pred = w * X
error = np.mean((y - pred)**2)
acc = (1 - error / np.var(y)) * 100

print("Stochastic GD")
print("Weight:", w)
print("Error:", error)
print("Accuracy:", acc, "%")