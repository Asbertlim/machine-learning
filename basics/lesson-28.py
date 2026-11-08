# Neural Network -> machine learning mode made up of connected computational units called neurons
# Neural network has 3 layers
# input layer -> receives information from dataset
# hidden layer -> processes information and learns pattern
# output layer -> produces final prediction

# basic equation of how neuron works: z = w1x1 + w2x2 + b
# w = weights
# x = input features
# b = bias
# z = weighted sum before activation

# Activation function determines how a neuron tranforms its weigthed sum
# 2 important functions : ReLU (Rectified Linear Unit) and Sigmoid

# Neural networks learn through a process involving three main steps:
# 1. Forward propagation: Inputs travel through the network to generate a prediction.
# 2. Calculate the error (loss): The model compares its prediction with the correct answer using a loss function.
# 3.Backpropagation and optimisation: The network calculates how its weights contributed to the error, then an optimiser such as gradient descent adjusts them to reduce the loss.

# MLP -> Multi-Layer Perception
# hidden_layer_sizes=(8,4) -> first hidden layer: 8 neurons, second hidden layer: 4 neurons


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

data = {
    "distance": [10, 15, 20, 25, 30, 35, 40, 45,
                 50, 55, 60, 65, 70, 75, 80, 85],
    "battery": [90, 80, 75, 85, 60, 70, 55, 65,
                80, 50, 70, 45, 60, 40, 55, 35],
    "safe_to_move": [0, 0, 0, 0, 1, 1, 0, 1,
                     1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["distance", "battery"]]
y = df["safe_to_move"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16,8),
        activation="relu",
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42
    )
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
recall_0 = recall_score(y_test, predictions, pos_label=0)
recall_1 = recall_score(y_test, predictions, pos_label=1)

print("Actual test:", y_test)
print("Predictions:", predictions)
print("Accuracy:", accuracy)
print("Recall 0:", recall_0)
print("Recall 1:", recall_1)
