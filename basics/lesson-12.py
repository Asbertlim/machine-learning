# DecisionTreeClassifier imports the algorithm.
# max_depth=2 limits the tree's depth, helping prevent it from becoming unnecessarily complex.
# model.fit(X_train, y_train) teaches the model using training features and their correct labels.

# Features: [study hours, attendance percentage]
X = [
    [1, 50], [2, 60], [3, 65], [4, 70],
    [5, 80], [6, 85], [7, 90], [8, 95]
]

# Target: pass or fail
y = [
    "No", "No", "No", "Yes",
    "Yes", "Yes", "Yes", "Yes"
]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=2, random_state=42)

model.fit(X_train, y_train)

# Predict the results for the test students
predictions = model.predict(X_test)

print("Actual results:", y_test)
print("Predicted results:", predictions)

from sklearn.metrics import accuracy_score

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
