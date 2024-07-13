import numpy as np
from scipy.special import logsumexp
from utils.utils import vcol


class GMM:
    def __init__(self, DTR, LTR, n_comp_0, n_comp_1, cov_type, tol, alpha, psi):
        self.DTR = DTR
        self.LTR = LTR
        self.n_comp_0 = n_comp_0
        self.n_comp_1 = n_comp_1
        self.cov_type = cov_type
        self.tol = tol
        self.alpha = alpha
        self.psi = psi
        self.gmm_0 = None
        self.gmm_1 = None

    def __logpdf_GAU_ND__(self, D, mu, C):
        DC = D - mu
        M = D.shape[0]
        const = M * np.log(2 * np.pi)
        logdet = np.linalg.slogdet(C)[1]
        L = np.linalg.inv(C)

        if D.shape[1] == 1:
            v = (DC.T @ L @ DC).diagonal()
        else:
            v = (DC * np.dot(L, DC)).sum(0)

        logpdf = -0.5 * (const + logdet + v)

        return logpdf

    def __logpdf_joint__(self, D, gmm):
        S = []

        for w, mu, C in gmm:
            logpdf_conditional = self.__logpdf_GAU_ND__(D, mu, C)
            logpdf_joint = logpdf_conditional + np.log(w)
            S.append(logpdf_joint)

        S = np.vstack(S)
        return S

    def __logpdf_GMM__(self, D, gmm):
        logdens = logsumexp(self.__logpdf_joint__(D, gmm), axis=0)
        return logdens

    def __constrain_cov__(self, C):
        U, s, _ = np.linalg.svd(C)
        s[s < self.psi] = self.psi
        covNew = U @ (vcol(s) * U.T)

        return covNew

    def __EM__(self, DTR, gmm_params):
        N = DTR.shape[1]
        K = len(gmm_params)
        ll_curr = -np.inf
        gmm_params = np.array(gmm_params, dtype=object)

        while True:
            ll_prev = ll_curr

            # E-step
            log_joint = self.__logpdf_joint__(DTR, gmm_params)
            log_density = self.__logpdf_GMM__(DTR, gmm_params)
            gamma = np.exp(log_joint - log_density)

            ll_curr = log_density.mean()

            # M-step
            Z = np.sum(gamma, axis=1)

            for k in range(K):
                F = np.sum(gamma[k] * DTR, axis=1)
                S = np.dot(gamma[k] * DTR, DTR.T)

                mu = vcol(F / Z[k])
                C = S / Z[k] - np.dot(mu, mu.T)
                w = Z[k] / np.sum(Z)

                if self.cov_type.lower() == "diag":
                    C *= np.eye(C.shape[0])

                if self.psi is not None:
                    C = self.__constrain_cov__(C)

                gmm_params[k] = (w, mu, C)

            if self.cov_type.lower() == "tied":
                tied_C = sum(Z * gmm_params[:, 2]) / N

                if self.psi is not None:
                    tied_C = self.__constrain_cov__(tied_C)

                gmm_params[:, 2].fill(tied_C)

            if np.abs(ll_curr - ll_prev) < self.tol:
                return gmm_params

    def __LBG__(self, DTR, n_comp):
        mu = np.mean(DTR, axis=1).reshape(-1, 1)
        C = np.dot((DTR - mu), (DTR - mu).T) / DTR.shape[1]

        if self.cov_type.lower() == "diag":
            C = C * np.eye(C.shape[0])

        if self.psi is not None:
            gmm_params = [(1.0, mu, self.__constrain_cov__(C))]
        else:
            gmm_params = [(1.0, mu, C)]

        while len(gmm_params) < n_comp:
            gmm_new = []

            for w, mu, C in gmm_params:
                U, s, Vh = np.linalg.svd(C)
                d = U[:, 0:1] * s[0] ** 0.5 * self.alpha

                gmm_new.append((0.5 * w, mu - d, C))
                gmm_new.append((0.5 * w, mu + d, C))

            gmm_params = self.__EM__(DTR, gmm_new)

        return gmm_params

    def fit(self):
        self.gmm_0 = self.__LBG__(self.DTR[:, self.LTR == 0], self.n_comp_0)
        self.gmm_1 = self.__LBG__(self.DTR[:, self.LTR == 1], self.n_comp_1)

    def score(self, DTE):
        if self.gmm_0 is None or self.gmm_1 is None:
            raise ValueError("Model not trained yet.")

        logpdf_0 = self.__logpdf_GMM__(DTE, self.gmm_0)
        logpdf_1 = self.__logpdf_GMM__(DTE, self.gmm_1)

        llr = logpdf_1 - logpdf_0

        return llr

    def save(self, path):
        np.save(path + "_gmm_0.npy", self.gmm_0)
        np.save(path + "_gmm_1.npy", self.gmm_1)

    def load(self, path):
        self.gmm_0 = np.load(path, allow_pickle=True)
        self.gmm_1 = np.load(path, allow_pickle=True)
