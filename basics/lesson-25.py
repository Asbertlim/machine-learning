# ensemble learning -> machine learning technique that combines predictions from multiple models to come up iwth a final prediction
# random forest -> ensemble learning algorithm that combines multiple Decision Trees
# n_estimator -> number of trees in the forest

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

# Model 1: Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Model 2: Logistic Regression
logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Model 3: Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

tree.fit(X_train, y_train)
logistic.fit(X_train, y_train)
forest.fit(X_train, y_train)

pred_tree = tree.predict(X_test)
pred_log = logistic.predict(X_test)
pred_for = forest.predict(X_test)

acc_tree = accuracy_score(y_test, pred_tree)
acc_log = accuracy_score(y_test, pred_log)
acc_for = accuracy_score(y_test, pred_for)

rec_tree_0 = recall_score(y_test, pred_tree, pos_label=0)
rec_tree_1 = recall_score(y_test, pred_tree, pos_label=1)

rec_log_0 = recall_score(y_test, pred_log, pos_label=0)
rec_log_1 = recall_score(y_test, pred_log, pos_label=1)

rec_for_0 = recall_score(y_test, pred_for, pos_label=0)
rec_for_1 = recall_score(y_test, pred_for, pos_label=1)

print("-----Comparison-----")
print("Tree Accuracy:", acc_tree)
print("Logistic Accuracy:", acc_log)
print("Forest Accuracy:", acc_for)

print()

print("Tree Recall:", rec_tree_0, rec_tree_1)
print("Logistic Recall:", rec_log_0, rec_log_1)
print("Forest Recall:", rec_for_0, rec_for_1)
