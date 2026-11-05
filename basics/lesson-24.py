# generally predicts 0 or 1
# threshold = 0.3 -> more willing to predict class 1
# threshold = 0.5 -> default
# threshold = 0.7 -> more cautious to predict class 1
# ROC -> Receiver Operating Characteristic
# it shows how a classifier performs at different threshold by plotting:
# Trus Positive Rate -> te proportion of actual positives correctly identified. This is the same as recall.
# False Positive Rate (FPR) -> the proportion of actual negatives incorrectly classified as positive.
# curve closer to top left generally indicates better discrimination between two classes
# AUC -> Area Under Curve -> summarises ROC curve into a number
# e.g. 1.0 -> perfect separation. 0.5 -> no better thn random raking
# ROC (Receiver Operating Characteristic) is a graph that shows how well a classification model performs at different classification thresholds.
# A higher AUC generally means the model is better at distinguishing positive cases from negative cases.



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


data ={
    "distance": [100, 200, 300, 400, 150, 500, 250, 350],
    "duration": [10, 20, 30, 40, 15, 50, 25, 35],
    "starting_battery": [90, 80, 70, 60, 85, 50, 90, 65],
    "energy_consumed": [15, 25, 35, 50, 20, 45, 30, 40],
    "completed": [1, 1, 1, 0, 1, 0, 1, 0]
}

import pandas as pd

df = pd.DataFrame(data)

X = df[[
    "distance",
    "duration",
    "starting_battery"
]]

y = df["completed"]


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
acc = accuracy_score(y_test, prediction)


# Get probabilities for class 1
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, probabilities)
print("AUC:", auc)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Task

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data ={
    "distance": [100, 200, 300, 400, 150, 500, 250, 350],
    "duration": [10, 20, 30, 40, 15, 50, 25, 35],
    "starting_battery": [90, 80, 70, 60, 85, 50, 90, 65],
    "energy_consumed": [15, 25, 35, 50, 20, 45, 30, 40],
    "completed": [1, 1, 1, 0, 1, 0, 1, 0]
}

import pandas as pd

df = pd.DataFrame(data)

X = df[[
    "distance",
    "duration",
    "starting_battery"
]]

y = df["completed"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Create and train your model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 1. Obtain class 1 probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# 2. Calculate AUC
auc = roc_auc_score(y_test, probabilities)

print("AUC:", auc)

# 3. Apply a threshold of 0.3
threshold = 0.3
predictions = (probabilities >= threshold).astype(int)

print("Predictions:", predictions)
