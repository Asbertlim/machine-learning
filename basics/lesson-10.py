# stratify=y helps preserve the proportion of Yes and No lables in the training and testing sets.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Prepare the data
X = [
    [1, 50], [2, 60], [3, 65], [4, 70],
    [5, 80], [6, 85], [7, 90], [8, 95],
    [2, 90], [6, 60], [4, 90], [3, 80]
]

y = [
    "No", "No", "No", "Yes",
    "Yes", "Yes", "Yes", "Yes",
    "No", "No", "Yes", "No"
]

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict the test examples
predictions = model.predict(X_test)

# 5. Evaluate the model
print("Actual:", y_test)
print("Predicted:", predictions)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(
    y_test, predictions, pos_label="Yes", zero_division=0
))
print("Recall:", recall_score(
    y_test, predictions, pos_label="Yes", zero_division=0
))

# 6. Predict a new example
new_student = [[5, 75]]
result = model.predict(new_student)

print("New student's predicted result:", result[0])
