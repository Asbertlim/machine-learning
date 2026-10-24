# parameters = values the model learns from training data
# hyperparameters = setting you choose before training. e.g. max_depth=3
# Model tuning: Try different hyperparameter values and compare performance.
# Validation set: Used to select the model and its hyperparameters.
# Test set: Used for the final evaluation of the selected model.

# validation
# Features: [study hours per day, attendance percentage]
X = [
    [1, 50],
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 80],
    [6, 85],
    [7, 90],
    [8, 95]
]

# Target: whether the student passes
y = [
    "No",
    "No",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes"
]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=20
)

depths = [1, 2, 3, 5]

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=20
    )

    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Predict the validation labels
    predictions = model.predict(X_val)

    # 3. Calculate validation accuracy
    accuracy = accuracy_score(y_val, predictions)

    print("Depth:", depth, "| Validation accuracy:", accuracy)
