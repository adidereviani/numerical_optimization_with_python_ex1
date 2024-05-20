import math
import numpy as np

class Quadratic:

    def __init__(self, q=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.q = q
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def evaluate_function(self, x):
        if x.size == 2:
            f = x.T @ self.q @ x
            g = (self.q + self.q.T) @ x
            if self.hessian:
                h = self.q + self.q.T
                return f.item(), g, h
            return f.item(), g
        return np.einsum('ij,ji->i', x.T @ self.q, x)


class Rosenbrock:

    def __init__(self, max_iter=10000, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def evaluate_function(self, x):
        x1, x2 = x[0], x[1]

        if x.size == 2:
            f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
            grad_x1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
            grad_x2 = 200 * (x2 - x1 ** 2)
            hess_x1_x1 = -400 * x2 + 1200 * x1 ** 2 + 2
            hess_x2_x2 = 200
            hess_x1_x2 = -400 * x1
            hess_x2_x1 = hess_x1_x2

            grad = np.array([grad_x1, grad_x2]).T

            if self.hessian:
                hess = np.array([[hess_x1_x1, hess_x1_x2], [hess_x2_x1, hess_x2_x2]])
                return f, grad, hess
            else:
                return f, grad
        else:
            return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


class Linear:

    def __init__(self, a=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.a = a
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def evaluate_function(self, x):
        if x.size == 2:
            f = np.dot(self.a.T, x)
            g = self.a.T

            if self.hessian:
                h = np.zeros((2, 2))
                return f, g, h
            else:
                return f, g
        else:
            return np.dot(self.a.T, x)


class Exponential:

    def __init__(self, max_iter=100, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def evaluate_function(self, x=None):
        x1, x2 = x[0], x[1]
        if x.size == 2:
            f = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            grad_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)-math.exp(-x1-0.1)
            grad_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            hess_x1_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            hess_x2_x2 = 9*math.exp(x1+3*x2-0.1)+9*math.exp(x1-3*x2-0.1)
            hess_x1_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            hess_x2_x1 = hess_x1_x2
            g = np.array([grad_x1, grad_x2]).T
            if self.hessian:
                h11 = hess_x1_x1
                h12 = hess_x1_x2
                h21 = hess_x2_x1
                h22 = hess_x2_x2
                h = np.array([[h11, h12], [h21, h22]])
                return f, g, h
            else:
                return f, g
        else:
            return np.exp(x1+3*x2-0.1)+np.exp(x1-3*x2-0.1)+np.exp(-x1-0.1)


funcQ1 = Quadratic(q=np.array([[1, 0], [0, 1]]))
funcQ2 = Quadratic(q=np.array([[1, 0], [0, 100]]))
funcQ3 = Quadratic(q=np.array([[(3 ** 0.5) / 2, -0.5], [0.5, (3 ** 0.5) / 2]]).T.dot(np.array([[100, 0], [0, 1]])).dot(np.array([[(3 ** 0.5) / 2, -0.5], [0.5, (3 ** 0.5) / 2]])))
funcRosenbrock = Rosenbrock()
funcLinear = Linear(a=np.array([1, 1]))
funcExponential = Exponential()