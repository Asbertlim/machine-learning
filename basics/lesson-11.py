# Underfitting -> training performance and testing performance are low. 
# Overfitting -> training performance is high but testing performance is low. solution: more diverse data, more representative data

# use code from lesson 10
from sklearn.metrics import accuracy_score

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
