# K Nearest Neighbors (KNN) -> machine learning algorithm that predicts the label of a new data point by examining the labels of its nearest examples in the training data.
# n_neighbors=3 -> the model finds 3 nearest training examples to the new input, checks their labels, predicts the majority label
# pipeline -> lets us scale the training data correctly inside each cross-validation fold, helping prevent data leakage. 
# pipeline connects multiple data-processing and machine-learning steps into one workflow.
# StandardScaler -> preprocessing tool that standardises numerical features so that they have approximately:
  # Mean = 0
  # Standard deviation = 1



from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "DecisionTreeClassifier": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    "KNeighborsClassifier": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3)
    )
}
lst = []

for name, model in models.items():
  scores = cross_val_score(
      model,
      X,
      y,
      cv=5,
      scoring="accuracy"
  )

  print("Name:", name)
  print("Scores:", scores)
  print("Average:", scores.mean())
  lst.append(scores.mean())

print(max(lst))
