import numpy as np
import pandas as pd
from gradient_descent_optimizer import GradientDescentOptimizer
import utils as util


def gradient_function_svm(X, Y, w, reg_param):
    y_cap = (X.dot(w) >= 0).astype(int).replace(to_replace=0, value=-1)
    temp = ((y_cap * Y) < 1).astype(int)
    mul = np.multiply(X.T, Y)
    mul = np.multiply(mul,temp)
    return (reg_param * w - mul.T).sum()/len(X)


class SVM():
    def __init__(self):
        self.w = []

    def fit(self, X, Y, step_size=0.01, reg_param=0.01, max_iter=500, tol=1e-7):
        self.w = np.zeros(len(X.columns))
        gd = GradientDescentOptimizer(step_size=step_size, reg_param=reg_param, max_iter=max_iter, tol=tol)
        self.w = gd.optimize(X, Y, self.w, gradient_function_svm)

    def predict(self, X):
        outputs = X.dot(self.w)
        #print(outputs)
        predictions = (outputs > 0).astype(int).replace(to_replace=0, value=-1)
        return predictions

    def get_accuracies(self, test_X, test_Y):
        predictions = self.predict(test_X)
        accurate = (predictions == test_Y).value_counts().get(True, 0)
        #print(test_Y.value_counts())
        #print(accurate)
        return accurate * 100 / len(test_Y)
