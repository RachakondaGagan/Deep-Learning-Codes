import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model

# Load dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

print("The size of train_x is:", train_x.shape)
print("The size of train_y is:", train_y.shape)
print("The size of test_x is:", test_x.shape)
print("The size of test_y is:", test_y.shape)

# Reshape
train_x = train_x.reshape(train_x.shape[0],28,28,1)
test_x = test_x.reshape(test_x.shape[0],28,28,1)

# Normalize
train_x = train_x/255.0
test_x = test_x/255.0

# One hot encoding
train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)

# VGG Model
model = Sequential()

# Block 1
model.add(Conv2D(32,(3,3),activation='relu',
padding='same',input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

# Block 2
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

# Block 3
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

# Fully connected
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

# Compile
model.compile(loss='categorical_crossentropy',
optimizer=Adam(),
metrics=['accuracy'])

# Train
model.fit(train_x,train_y,
batch_size=128,
epochs=1,
verbose=1,
validation_data=(test_x,test_y))

# Evaluate
score = model.evaluate(test_x,test_y)

print("Test Loss:",score[0])
print("Test accuracy:",score[1])