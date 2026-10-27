# cross-validation is a method for evaluating a model across multiple training and validation splits
# a common method is K-fold cross-validation
# cv=5 -> this requests five-fold cross-validation
# scoring="accuracy" -> this tells scikit-learn to evaluate each round usimg accuracy

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X = [
    [5, 90],
    [10, 70],
    [15, 50],
    [25, 80],
    [30, 90],
    [40, 70],
    [60, 85],
    [80, 95],
    [100, 70],
    [120, 40]
]

y = [
    "STOP", "STOP", "STOP",
    "MOVE", "MOVE", "MOVE",
    "MOVE", "MOVE", "MOVE", "MOVE"
]

model = DecisionTreeClassifier(max_depth=3, random_state=42)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Individual score:", scores)
print("Average:", scores.mean())
