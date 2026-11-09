# Convolutional Neural Network -> type of neural network particularly designed to process grid-like data, especially images
# 4 impt components:
# 1. Input -> receives an image as numerical pixel values
# 2. Convolution -> detects patterns such as edges, textures and shapes
# 3. Pooling -> reduces spatial dimensions of feautre maps
# 4. Fully connected layer -> useas learned features ot make a final classification

# A convolutional layer uses small matrices called filters (or kernels) that move across an image.
# each layer learns to detect a particular pattern
# Max Pooling -> a method in pooling whereby we minimise the size of a matrix by taking the largest value from each region

# Conv2D(32, (3,3)) -> learns 32 filters, each with 3x3 kernel
# MaxPooling2D((2,2)) -> reduces feature-map dimensions
# Flatten() -> converts feature maps into a one-dimensional vecotr
# Dense(64) -> learns combination of extracted features
# Dense(10) -> produces scores for 10 digit clases
# Epochs -> mumber of complete passes through the training dataset
# Adam: An optimisation algorithm that adjusts network weights.
# Loss: Measures how far predictions are from the correct labels.
# Validation split: Holds back part of the training data to monitor performance during training.

# MNIST -> a sample dataset contains thousands of pictures of handwritten digits from 0 to 9
# image has 3 main dimensions: height, width, colour channels
# grayscale image -> 1 channel whil RGB -> 3 channels
# MNIST pictures are 28x28 pixels and grayscale
# hence, the shape is (28, 28, 1)

# Optimizer -> how the AI leanrs. adam adjusts the network's weights to reduce mistakes
# Loss -> how wrong the AI is.it measures how fat the predictions are from the correct answers
# Metrics -> how we measure performance. 

import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalise pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train[..., None]
X_test = X_test[..., None]

# Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),

    layers.Dense(10, activation="softmax")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2
)

# Evaluate
model.evaluate(X_test, y_test)
