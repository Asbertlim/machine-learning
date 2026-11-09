# Activation functions
Neuron recives information, performs calculations and produces an output.
Activation function acts like a decision gate that tranforms the neuron’s 
calculated value.

### A. ReLU — the simple gate
    f(x) = max(0, x)
    commonly used in hidden layers

### B. Sigmoid — the probability converter
    converts a number into a value between 0 and 1
    e.g. 0.9 = 90%
    cmmonly used for binary classification.

### C. Softmax — the class selector
    converts model’s outputs into probabilities that add yp to 100%
