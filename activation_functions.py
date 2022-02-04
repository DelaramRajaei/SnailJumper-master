import numpy as np


def sigmoid(z):
    try:
        res = 1 / (1 + np.exp(-z))
    except OverflowError:
        res = 0
    return res


def relu(z):
    l1 = []
    for i in range(len(z)):
        l2 = []
        for j in range(len(z[i])):
            l2.append(z[i][j]) if z[i][j] >= 0 else l2.append(0)
        l1.append(l2)
    return l1
