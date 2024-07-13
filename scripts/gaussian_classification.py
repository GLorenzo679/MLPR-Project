import matplotlib.pyplot as plt
import numpy as np
from models.MVG import MVG, NaiveMVG, TiedMVG
from utils.dimensionalityReduction import LDA, PCA
from utils.utils import find_best_threshold, split_db_2to1, vcol, vrow


def gaussian_classification(D, L, preprocess_PCA=False, m=0):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    if preprocess_PCA:
        mu = DTR.mean(1)
        DTRc = DTR - mu.reshape(DTR.shape[0], 1)
        DVALc = DVAL - mu.reshape(DVAL.shape[0], 1)

        W_PCA = PCA(DTRc, m)
        DTR = np.dot(W_PCA.T, DTRc)
        DVAL = np.dot(W_PCA.T, DVALc)

    prior = 0.5
    threshold = 0  # due to uniform prior

    # classification using MVG
    mvg = MVG(DTR, LTR)
    mvg.fit()

    val_scores = mvg.score(DVAL, prior)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[val_scores >= threshold] = 1
    PVAL[val_scores < threshold] = 0

    error_rate = 100 * (np.sum(PVAL != LVAL) / LVAL.shape[0])

    print(f"Error rate (MVG): {error_rate:.2f} %")

    # classification using the tied covariance model
    tied_mvg = TiedMVG(DTR, LTR)
    tied_mvg.fit()

    val_scores = tied_mvg.score(DVAL, prior)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[val_scores >= threshold] = 1
    PVAL[val_scores < threshold] = 0

    error_rate = 100 * (np.sum(PVAL != LVAL) / LVAL.shape[0])

    print(f"Error rate (Tied MVG): {error_rate:.2f} %")

    # classification using the naive model
    naive_mvg = NaiveMVG(DTR, LTR)
    naive_mvg.fit()

    val_scores = naive_mvg.score(DVAL, prior)
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[val_scores >= threshold] = 1
    PVAL[val_scores < threshold] = 0

    error_rate = 100 * (np.sum(PVAL != LVAL) / LVAL.shape[0])

    print(f"Error rate (Naive MVG): {error_rate:.2f} %")

    # classification using LDA
    W_LDA = LDA(DTR, LTR, 1)
    DTR_LDA = np.dot(W_LDA.T, DTR)
    DVAL_LDA = np.dot(W_LDA.T, DVAL)

    lda_threshold = find_best_threshold(DTR_LDA, LTR, DVAL_LDA, LVAL)

    PVAL_LDA = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL_LDA[DVAL_LDA[0] >= lda_threshold] = 1
    PVAL_LDA[DVAL_LDA[0] < lda_threshold] = 0

    error_rate = 100 * (np.sum(PVAL_LDA != LVAL) / LVAL.shape[0])

    print(f"Error rate (LDA): {error_rate:.2f} %")


def correlation_analysis(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    C0 = np.cov(D0)
    C1 = np.cov(D1)

    corr0 = C0 / (vcol(C0.diagonal() ** 0.5) * vrow(C0.diagonal() ** 0.5))
    corr1 = C1 / (vcol(C1.diagonal() ** 0.5) * vrow(C1.diagonal() ** 0.5))

    print(f"Correlation matrix of class 0:\n{corr0}\n")
    print(f"Correlation matrix of class 1:\n{corr1}\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].matshow(corr0)
    ax[0].set_title("Correlation matrix of class 0")
    ax[0].set_xticks(range(D.shape[0]))
    ax[0].set_yticks(range(D.shape[0]))

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            ax[0].text(
                j,
                i,
                f"{corr0[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if i != j else "black",
            )

    ax[1].matshow(corr1)
    ax[1].set_title("Correlation matrix of class 1")
    ax[1].set_xticks(range(D.shape[0]))
    ax[1].set_yticks(range(D.shape[0]))

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            ax[1].text(
                j,
                i,
                f"{corr1[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if i != j else "black",
            )

    plt.show()
