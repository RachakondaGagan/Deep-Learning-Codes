import numpy as np

X = np.array([1,2,3,4])
y = np.array([2,4,6,8])
w = 0.0
v = 0.0
lr = 0.01
beta = 0.9

for _ in range(1000):
    grad = np.mean((w*X - y) * X)
    v = beta * v + lr * grad
    w -= v

pred = w * X
error = np.mean((y - pred)**2)
acc = (1 - error / np.var(y)) * 100

print("Momentum GD")
print("Weight:", w)
print("Error:", error)
print("Accuracy:", acc, "%")