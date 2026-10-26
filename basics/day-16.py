# pos_label tells scikit-learn that the class it contians is the positive class
# precision = when the model predicts positive , how often is it right?
# recall = of all positive catches, how often did the model catch right?

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_actual = [
    "STOP", "STOP", "STOP", "STOP", "STOP",
    "MOVE", "MOVE", "MOVE", "MOVE", "MOVE"
]

y_predicted = [
    "STOP", "STOP", "STOP", "MOVE", "MOVE",
    "STOP", "MOVE", "MOVE", "MOVE", "MOVE"
]

accuracy = accuracy_score(y_actual, y_predicted)

precision = precision_score(
    y_actual, y_predicted, pos_label="MOVE"
)

recall = recall_score(
    y_actual, y_predicted, pos_label="MOVE"
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
