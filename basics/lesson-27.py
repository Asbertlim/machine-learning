# XGBoost (Extreme Gradient Boosting) -> takes the idea of Gradient Boosting further by adding several important improvements
# 1. Regularisation -> penalises over complex trees to help train models faster
# 2. Efficient computation -> uses optimised methods to trian model faster
# 3. Shrinkage -> controls how much each new tree contributes
# 4. Row and feature sampling -> can use subsets of training examples and features to improve efficiencyd reduce overfitting

# subsample = fraction of training rows used per tree
# colsample_bytree = fraction of feautres used per tree
# reg_lambda = L2 reguralisation length
# reg_alpha = L1 reguralisation length

import pandas as pd

data = {
    "distance": [100, 200, 300, 400, 150, 500, 250, 350,
                 120, 450, 280, 380, 180, 520, 320, 420],
    "duration": [10, 20, 30, 40, 15, 50, 25, 35,
                 12, 45, 28, 38, 18, 52, 32, 42],
    "starting_battery": [90, 80, 70, 60, 85, 50, 90, 65,
                         95, 55, 75, 62, 88, 48, 72, 58],
    "completed": [1, 1, 1, 0, 1, 0, 1, 0,
                  1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["distance", "duration", "starting_battery"]]
y = df["completed"]

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recall_failed = recall_score(y_test, predictions, pos_label=0)
recall_success = recall_score(y_test, predictions, pos_label=1)

print("Actual test:", y_test)
print("Predictions:", predictions)

print("Accuracy:", accuracy)
print("Recall Failed:", recall_failed)
print("Recall Success:", recall_success)
