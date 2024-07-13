import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils.utils import vcol, vrow


class SVM:
    def __init__(self, DTR, LTR, C=None):
        if C is None:
            raise ValueError("C must be specified for SVM.")

        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.z = 2 * self.LTR - 1
        self.H = np.dot(DTR.T, DTR) * vrow(self.z) * vcol(self.z)
        self.alpha = None

    def __svm_obj__(self, alpha):
        return 0.5 * np.dot(alpha.T, np.dot(self.H, alpha)) - np.sum(alpha)

    def __svm_grad__(self, alpha):
        return np.dot(self.H, alpha) - np.ones(self.DTR.shape[1])

    def fit(self):
        alpha0 = np.zeros(self.DTR.shape[1])
        self.alpha, _, _ = fmin_l_bfgs_b(
            self.__svm_obj__,
            alpha0,
            self.__svm_grad__,
            approx_grad=False,
            factr=1.0,
            bounds=[(0, self.C) for _ in range(self.DTR.shape[1])],
        )

    def score(self, DTE):
        if self.alpha is None:
            raise ValueError("Model not trained yet.")

        w = np.dot(self.DTR, self.alpha * self.z)
        scores = np.dot(w.T, DTE)
        return scores

    def save(self, path):
        np.save(path + ".npy", self.alpha)

    def load(self, path):
        self.alpha = np.load(path)


class kernelSVM(SVM):
    def __init__(self, DTR, LTR, kernel, C=None, K=None, c=None, d=None, gamma=None):
        super().__init__(DTR, LTR, C)

        if K is None:
            raise ValueError("K must be specified for kernel SVM.")

        if kernel == "poly":
            if c is None or d is None:
                raise ValueError("c and d must be specified for polynomial kernel.")
            self.kernel = lambda X, Y: ((c + np.dot(X.T, Y)) ** d) + K**2
        elif kernel == "rbf":
            if gamma is None:
                raise ValueError("gamma must be specified for RBF kernel.")
            self.kernel = (
                lambda X, Y: np.exp(-gamma * self.__rbfKernelFunc__(X, Y)) + K**2
            )

        self.H = self.kernel(DTR, DTR) * vrow(self.z) * vcol(self.z)

    def __rbfKernelFunc__(self, D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)

        return Z

    def score(self, DTE):
        if self.alpha is None:
            raise ValueError("Model not trained yet.")

        K = self.kernel(self.DTR, DTE)
        scores = np.dot(self.alpha * self.z, K)

        return scores
