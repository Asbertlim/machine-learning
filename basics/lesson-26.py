# Gradient Boosting -> ensemble learning technique that builds models sequentially, with each new model attempting to reduce the errors made by the existing ensemble.
# learning_rate -> controls how much each new tree contributes to the overall model

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

data = {
    "distance": [100, 200, 300, 400, 150, 500, 250, 350],
    "duration": [10, 20, 30, 40, 15, 50, 25, 35],
    "starting_battery": [90, 80, 70, 60, 85, 50, 90, 65],
    "completed": [1, 1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["distance", "duration", "starting_battery"]]
y = df["completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)


tree = DecisionTreeClassifier(max_depth=3, random_state=42)

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

models = {
    "Tree": tree,
    "Logistic": logistic,
    "Forest": forest,
    "Boosting": boosting
}

for model in models:
  models[model].fit(X_train, y_train)
  prediction = models[model].predict(X_test)
  accuracy = accuracy_score(y_test, prediction)
  recall_0 = recall_score(y_test, prediction, pos_label=0)
  recall_1 = recall_score(y_test, prediction, pos_label=1)
  print(f"{model}: Accuracy: {accuracy}, Recall: {recall_0} and {recall_1}")
