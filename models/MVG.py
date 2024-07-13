import numpy as np
from scipy.special import logsumexp
from utils.utils import vrow


class MVG:
    def __init__(self, D, L):
        self.D = D
        self.L = L
        self.mean = None
        self.covariance = None
        self.num_classes = np.unique(L).size

    def __mean_cov_estimate__(self):
        mean_array = []
        cov_array = []

        for i in range(self.num_classes):
            # select data of each class
            D_class = self.D[:, self.L == i]
            # calculate mean of each class
            mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
            # calculate covariance matrix of each class
            cov_class = np.dot((D_class - mean_class), (D_class - mean_class).T) / (
                D_class.shape[1]
            )

            mean_array.append(mean_class)
            cov_array.append(cov_class)

        return np.array(mean_array), np.array(cov_array)

    def __logpdf_GAU_ND_fast__(self, X, mu, C):
        XC = X - mu
        M = X.shape[0]
        const = -0.5 * M * np.log(2 * np.pi)
        _, logdet = np.linalg.slogdet(C)
        L = np.linalg.inv(C)
        v = (XC * np.dot(L, XC)).sum(0)

        return const - (0.5 * logdet) - (0.5 * v)

    def __score_matrix__(self, D):
        S = []

        for i in range(self.num_classes):
            if self.covariance.ndim > 2:
                fcond = np.exp(
                    self.__logpdf_GAU_ND_fast__(D, self.mean[i], self.covariance[i])
                )
            else:
                # tied covariance model --> only one covariance matrix for all classes
                fcond = np.exp(
                    self.__logpdf_GAU_ND_fast__(D, self.mean[i], self.covariance)
                )

            S.append(vrow(fcond))

        return np.vstack(S)

    def fit(self):
        self.mean, self.covariance = self.__mean_cov_estimate__()

    def predict_prob(self, DTE, prior):
        # compute log score matrix for each sample of each class
        log_S_matrix = np.log(self.__score_matrix__(DTE))

        # compute the log joint distribution (each row of S_matrix (class-conditional probability) * each PRIOR probability)
        log_S_Joint = log_S_matrix + np.log(prior)

        log_S_marginal = vrow(logsumexp(log_S_Joint, axis=0))

        # compute posterior probability (log joint probability - log marginal densities)
        log_S_post = log_S_Joint - log_S_marginal

        return np.exp(log_S_post)

    # def predict(self, D, prior):
    #     return np.argmax(self.predict_prob(D, prior), axis=0)

    def score(self, DTE, prior):
        prob = self.predict_prob(DTE, prior)
        return np.log(prob[1] / prob[0])

    def log_likelihood(self, DTE):
        ll_array = []

        for i in range(self.num_classes):
            pdfGAU = self.__logpdf_GAU_ND_fast__(DTE, self.mean[i], self.covariance[i])
            ll_array.append(pdfGAU.sum())

        return np.array(ll_array)


class TiedMVG(MVG):
    def __init__(self, D, L):
        super().__init__(D, L)

    def __mean_cov_estimate__(self):
        mean_array = []
        class_cov = 0

        for i in range(self.num_classes):
            # select data of each class
            D_class = self.D[:, self.L == i]
            # calculate mean of each class
            mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
            # calculate covariance matrix of each class
            class_cov += np.dot((D_class - mean_class), (D_class - mean_class).T)

            mean_array.append(mean_class)

        # compute within class covariance
        within_class_cov = class_cov / (self.D.shape[1])

        return np.array(mean_array), within_class_cov

    def fit(self):
        self.mean, self.covariance = self.__mean_cov_estimate__()


class NaiveMVG(MVG):
    def __init__(self, D, L):
        super().__init__(D, L)

    def __mean_cov_estimate__(self):
        self.mean, self.covariance = super().__mean_cov_estimate__()

        for i in range(self.covariance.shape[0]):
            self.covariance[i] *= np.identity(self.covariance.shape[1])

        return self.mean, self.covariance

    def fit(self):
        self.mean, self.covariance = self.__mean_cov_estimate__()


class TiedNaiveMVG(MVG):
    def __init__(self, D, L):
        super().__init__(D, L)

    def __mean_cov_estimate__(self):
        mean_array = []
        class_cov = 0

        for i in range(self.num_classes):
            # select data of each class
            D_class = self.D[:, self.L == i]
            # calculate mean of each class
            mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
            # calculate covariance matrix of each class
            class_cov += np.dot((D_class - mean_class), (D_class - mean_class).T)

            mean_array.append(mean_class)

        # compute within class covariance
        within_class_cov = class_cov / (self.D.shape[1])
        within_class_cov *= np.identity(within_class_cov.shape[1])

        return np.array(mean_array), within_class_cov

    def fit(self):
        self.mean, self.covariance = self.__mean_cov_estimate__()
