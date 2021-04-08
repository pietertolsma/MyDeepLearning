import numpy as np

def logistic_sigmoid(x):
    return x / (1 + np.exp(-x))

def relu(x):
    x[x< 0] = 0
    return x