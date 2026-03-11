import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt

# Loading the dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

print("The size of train_x is:", train_x.shape)
print("The size of train_y is:", train_y.shape)
print("The size of test_x is:", test_x.shape)
print("The size of test_y is:", test_y.shape)

# Reshaping to 4D
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

# Normalization
train_x = train_x / 255.0
test_x = test_x / 255.0

# One hot encoding
train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

# Model
model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(5,5),
padding='valid', input_shape=(28,28,1),
activation='tanh'))

model.add(AvgPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(5,5),
padding='valid', activation='tanh'))

model.add(AvgPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))

model.add(Dense(10, activation='softmax'))

model.summary()

# Compile
model.compile(loss='categorical_crossentropy',
optimizer=Adam(),
metrics=['accuracy'])

# Train
model.fit(train_x, train_y,
batch_size=128,
epochs=1,
verbose=1,
validation_data=(test_x, test_y))

# Visualize filters
conv_layer = model.layers[0]
filters = conv_layer.get_weights()[0]

plt.figure(figsize=(10,10))
for i in range(6):
    plt.subplot(4,8,i+1)
    plt.imshow(filters[:,:,0,i], cmap='gray')
    plt.axis('off')

plt.show()

# Evaluate
score = model.evaluate(test_x, test_y)

print("Test Loss:", score[0])
print("Test accuracy:", score[1])