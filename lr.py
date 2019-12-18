import numpy as np
import pandas as pd
from scipy.special import expit, logit
from gradient_descent_optimizer import GradientDescentOptimizer
import utils as util


def gradient_function_lr(X, Y, w, reg_param):
    y_cap = expit(X.dot(w))
    temp = y_cap - Y
    mul = np.multiply(X.T, temp)
    return mul.T.sum() + reg_param * w


class LogisticRegressor():
    def __init__(self):
        self.w = []

    def fit(self, X, Y, step_size=0.01, reg_param=0.01, max_iter=500, tol=1e-7):
        self.w = np.zeros(len(X.columns))
        gd = GradientDescentOptimizer(step_size= step_size, reg_param=reg_param, max_iter=max_iter, tol=tol)
        self.w = gd.optimize(X, Y, self.w, gradient_function_lr)

    def predict(self, X):
        predictions = []
        probabilities = expit(X.dot(self.w))
        predictions = (probabilities>=0.5).astype(int)
        return predictions

    def get_accuracies(self, test_X, test_Y):
        predictions = self.predict(test_X)
        accurate = (predictions == test_Y).value_counts().get(True, 0)
        return accurate * 100 / len(test_Y)
