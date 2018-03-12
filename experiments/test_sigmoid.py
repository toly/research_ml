import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative(func, x, d=0.00001):
    dy = func(x + d) - func(x)
    return dy / d


for i in range(-20, 20):
    x = float(i) / 10
    y = sigmoid(x)
    dy = derivative(sigmoid, x)
    dy_test = y * (1 - y)
    error = dy_test - dy
    print(x, y, dy, dy_test, error)
