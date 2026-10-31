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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

model1 = DecisionTreeClassifier(max_depth=3, random_state=42)
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, prediction1)

model2 = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, prediction2)


print("\nMODEL COMPARISON")

print("Decision Tree:", acc1)
print("Logistic Regression:", acc2)

if acc1 > acc2:
    print("Decision Tree performs better.")
elif acc2 > acc1:
    print("Logistic Regression performs better.")
else:
    print("Both models have equal accuracy.")
