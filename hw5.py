import numpy as np
from scipy.optimize import minimize


def softmax(parameters, X):
    # dopolnite ...
    pass


def cost(parameters, X, y, lambda_):
    # dopolnite

    pass


def grad(parameters, X, y, lambda_):
    # dopolnite ..
    pass


def bfgs(X, y, lambda_):
    # tukaj inicirajte parametere modela
    x0 = ...  # dopolnite ...

    # preostanek funkcije pustite kot je
    res = minimize(lambda pars, X=X, y=y, lambda_=lambda_: -cost(pars, X, y, lambda_),
                   x0,
                   method='L-BFGS-B',
                   jac=lambda pars, X=X, y=y, lambda_=lambda_: -grad(pars, X, y, lambda_),
                   tol=0.00001)
    return res.x


class SoftMaxLearner:

    def __init__(self, lambda_=0, intercept=True):
        self.intercept = intercept
        self.lambda_ = lambda_

    def __call__(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        pars = bfgs(X, y, self.lambda_)
        return SoftMaxClassifier(pars, self.intercept)


class SoftMaxClassifier:

    def __init__(self, parameters, intercept):
        self.parameters = parameters
        self.intercept = intercept

    def __call__(self, X):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        ypred = softmax(self.parameters, X)
        return ypred


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(SoftMaxLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = c(X)
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(SoftMaxLearner(lambda_=0.0), X, y)
    """
    # ... dopolnite
    pass


def CA(real, predictions):
    # ... dopolnite
    pass


def log_loss(real, predictions):
    # ... dopolnite
    pass


def create_final_predictions():
    # dopolnite..
    pass


if __name__ == "__main__":
    create_final_predictions()
