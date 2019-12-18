import numpy as np
import pandas as pd
import utils as util


class GradientDescentOptimizer:
    def __init__(self, step_size=0.01, reg_param=0.01, max_iter=500, tol=1e-7):
        self.step_size = step_size
        self.reg_param = reg_param
        self.max_iter = max_iter if util.final else 3
        self.tol = tol

    def optimize(self, X, Y, w, gradient_function):
        cur_iter = 0
        #print('w', w)
        while True:
            cur_iter = cur_iter + 1
            gradient = gradient_function(X, Y, w, self.reg_param)
            w = w - self.step_size * gradient
            #print('w', w.values)
            #print('gradient', gradient.values)

            #print(self.step_size*np.linalg.norm(gradient), cur_iter)
            if self.step_size * np.linalg.norm(gradient) < self.tol:
                print('breaking')
                break
            if cur_iter == self.max_iter:
                break

        return w.values