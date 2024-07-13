import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils.utils import vcol, vrow


class LogisticRegression:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.ZTR = 2 * self.LTR - 1  # Z = 2L - 1 -> Z \in {-1, 1} for L \in {0, 1}
        self.l = l
        self.grad = []
        self.v = None

    def __logreg_obj__(self, v):
        w, b = v[0:-1], v[-1]
        S = np.dot(vcol(w).T, self.DTR).ravel() + b  # scores: w^T x + b
        G = -self.ZTR / (1 + np.exp(self.ZTR * S))

        f = 0.5 * self.l * np.linalg.norm(np.dot(w, w)) + np.mean(
            np.logaddexp(0, -self.ZTR * S)
        )

        grad = np.hstack(
            [
                self.l * w + np.mean(G * self.DTR, axis=1),
                np.mean(G),
            ]
        )

        return f, grad

    def fit(self):
        v0 = np.random.rand(self.DTR.shape[0] + 1)
        self.v, _, _ = fmin_l_bfgs_b(self.__logreg_obj__, v0)

    def score(self, DTE):
        if self.v is None:
            raise ValueError("Model not trained yet.")

        w_star, b_star = self.v[:-1], self.v[-1]
        scores = (
            np.dot(vcol(w_star).T, DTE).ravel() + b_star
        )  # - np.log(emp_prior / (1 - emp_prior))
        return np.array(scores)

    def save(self, path):
        np.save(path + ".npy", self.v)

    def load(self, path):
        self.v = np.load(path)


class PWLogisticRegression(LogisticRegression):
    def __init__(self, DTR, LTR, l, prior):
        super().__init__(DTR, LTR, l)
        nt = len(self.LTR[self.LTR == 1])
        nf = len(self.LTR[self.LTR == 0])
        self.eps = [prior / nt if L == 1 else (1 - prior) / nf for L in self.LTR]

    def __logreg_obj__(self, v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, self.DTR).ravel() + b  # scores: w^T x + b
        G = -self.ZTR / (1 + np.exp(self.ZTR * S))

        f = 0.5 * self.l * np.linalg.norm(w) ** 2 + np.sum(
            np.logaddexp(0, -self.ZTR * S) * self.eps
        )

        grad = np.hstack(
            [
                self.l * w + np.sum(self.eps * G * self.DTR, axis=1),
                np.sum(self.eps * G),
            ]
        )

        return f, grad
