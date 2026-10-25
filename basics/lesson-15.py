# Self coded

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Features: [distance to obstacle (cm), battery percentage]
X = [
    [5, 90],
    [10, 70],
    [15, 50],
    [18, 85],
    [20, 60],
    [25, 5],
    [40, 8],
    [60, 10],
    [80, 5],
    [100, 9],
    [25, 80],
    [30, 90],
    [40, 70],
    [50, 60],
    [60, 85],
    [70, 50],
    [80, 95],
    [100, 70],
    [120, 40],
    [150, 100]
]

# Target labels: robot's action
y = [
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "STOP",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE",
    "MOVE"
]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2
)

new_readings = [
    [12, 80],
    [100, 5]
]

new_predictions = model.predict(new_readings)

print("New predictions:", new_predictions)

model = DecisionTreeClassifier(max_depth=3, random_state=2)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Prediction:", prediction)

accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)
