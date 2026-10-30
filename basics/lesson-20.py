# zip() allows us to iterate through the sensor readings and their corresponding predictions together.
# pipeline ensures that scaling is fitted using the training data and applied consistently when making predictions.

X = [
    [5, 90], [10, 70], [15, 50], [18, 85],
    [20, 60], [25, 5], [40, 8], [60, 10],
    [80, 5], [100, 9], [25, 80], [30, 90],
    [40, 70], [50, 60], [60, 85], [70, 50],
    [80, 95], [100, 70], [120, 40], [150, 100]
]

y = [
    "STOP", "STOP", "STOP", "STOP", "STOP",
    "STOP", "STOP", "STOP", "STOP", "STOP",
    "MOVE", "MOVE", "MOVE", "MOVE", "MOVE",
    "MOVE", "MOVE", "MOVE", "MOVE", "MOVE"
]

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Actual test:", y_test)
print("Predicted:", prediction)
print("Accuracy:", accuracy)
