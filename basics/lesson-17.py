# feature engineering -> process of creating new features from existing data to help a machine learning model learns useful patterns

# data leakage -> occurs when information that should not be available to the model during training or
# evaluation influences its predictions, making its measured performance misleading.
# e.g. Student A learns concepts for Math exams. while Student B knows the answer key for the exam. (in the future, student A knows the math while B not)


distance = [10, 20, 15, 30, 25]
time = [2, 4, 5, 3, 5]

speed = []
for i in range(len(distance)):
  ratio = distance[i]/time[i]
  speed.append(ratio)

print(speed)
