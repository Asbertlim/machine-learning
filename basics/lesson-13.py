# impurities measure how mixed the labels are within a group
# e.g. Imagine a group of 10 students:
#  Group A: 5 pass, 5 fail → highly mixed.
#  Group B: 9 pass, 1 fail → less mixed.
#  Group C: 10 pass, 0 fail → completely pure.
#  Decision Tree prefers splits that make the resulting groups purer, meaning each group contains more consistent labels.
# max_depth controls the maximum number of levels in a Decision Tree. A tree that's too shallow may be underfit, too deep -> overfit

X = [
    [1, 50], [2, 60], [3, 65], [4, 70],
    [5, 80], [6, 85], [7, 90], [8, 95]
]

y = ["No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes"]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=20
)


# Model A: shallow tree
model_a = DecisionTreeClassifier(max_depth=1, random_state=20)
model_a.fit(X_train, y_train)
pred_a = model_a.predict(X_test)

# Model B: deeper tree
model_b = DecisionTreeClassifier(max_depth=3, random_state=20)
model_b.fit(X_train, y_train)
pred_b = model_b.predict(X_test)

# Evaluate both models
print("Shallow tree:", accuracy_score(y_test, pred_a))
print("Deeper tree:", accuracy_score(y_test, pred_b))


train_acc_a = accuracy_score(y_train, model_a.predict(X_train))
test_acc_a = accuracy_score(y_test, model_a.predict(X_test))
train_acc_b = accuracy_score(y_train, model_b.predict(X_train))
test_acc_b = accuracy_score(y_test, model_b.predict(X_test))


print(train_acc_a)
print(test_acc_a)
print(train_acc_b)
print(test_acc_b)
