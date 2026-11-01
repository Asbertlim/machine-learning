# Feature importance helps us understand which feature contributes the most to a prediction
# A feature importance score of 0.7 means that the feature accounts for 70% of the model's 
# total calculated impurity reduction, under scikit-learn's Decision Tree importance calculation.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    "distance": [100, 200, 300, 400, 150, 500, 250, 350, 120, 450],
    "duration": [10, 20, 30, 40, 15, 50, 25, 35, 12, 45],
    "battery": [90, 80, 70, 40, 85, 30, 90, 45, 95, 35],
    "completed": [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["distance", "duration", "battery"]]
y = df["completed"]

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X, y)

importance = model.feature_importances_

feature_importance = list(zip(X.columns, importance))

sorted_features = sorted(
    feature_importance,
    key=lambda x: x[1],
    reverse=True
)

print("Features ranked by importance:")

for feature, score in sorted_features:
    print(feature, round(score, 3))

print("\nMost important:", sorted_features[0][0])
print("Least important:", sorted_features[-1][0])
