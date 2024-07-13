import os

import matplotlib.pyplot as plt
import numpy as np
from utils.dimensionalityReduction import LDA, PCA
from utils.utils import find_best_threshold, split_db_2to1

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def dimensionality_reduction(
    D: np.ndarray, L: np.ndarray, save: bool = False, plot: bool = False
) -> None:
    mu = D.mean(1)
    W_PCA_6 = PCA(D - mu.reshape(D.shape[0], 1), 6)
    D_PCA_6 = np.dot(W_PCA_6.T, D)

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))

    for i in range(6):
        ax[i // 3, i % 3].hist(
            D_PCA_6[i, L == 0],
            bins=50,
            alpha=0.5,
            density=True,
            label="Class 0",
            color="r",
        )
        ax[i // 3, i % 3].hist(
            D_PCA_6[i, L == 1],
            bins=50,
            alpha=0.5,
            density=True,
            label="Class 1",
            color="b",
        )
        ax[i // 3, i % 3].legend()
        ax[i // 3, i % 3].set_title(f"PCA component {i + 1}")

    fig.suptitle("Histograms of the first 6 PCA components")

    if save:
        plt.savefig(f"{PATH}/report/plot/dim_red/PCA.png")
    elif plot:
        plt.show()

    W_LDA = LDA(D, L, 1)
    D_LDA = np.dot(W_LDA.T, D)

    plt.figure(figsize=(12, 8))
    plt.hist(
        D_LDA[0, L == 0], bins=50, alpha=0.5, density=True, label="Class 0", color="r"
    )
    plt.hist(
        D_LDA[0, L == 1], bins=50, alpha=0.5, density=True, label="Class 1", color="b"
    )
    plt.title("Histogram of the LDA component")
    plt.legend()

    if save:
        plt.savefig(f"{PATH}/report/plot/dim_red/LDA.png")
    elif plot:
        plt.show()

    # apply LDA as a classifier
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    W_LDA = LDA(DTR, LTR, 1)

    DTR_LDA = np.dot(W_LDA.T, DTR)
    DVAL_LDA = np.dot(W_LDA.T, DVAL)

    # Projected samples have only 1 dimension
    # threshold = (DTR_LDA[0, LTR == 0].mean() + DTR_LDA[0, LTR == 1].mean()) / 2.0
    threshold = find_best_threshold(DTR_LDA, LTR, DVAL_LDA, LVAL)

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)

    plt.hist(DTR_LDA[0, LTR == 0], bins=20, alpha=0.5, label="Class 0")
    plt.hist(DTR_LDA[0, LTR == 1], bins=20, alpha=0.5, label="Class 1")
    plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Histogram of the LDA component of the training set")
    plt.show()

    plt.hist(
        DVAL_LDA[0, LVAL == 0],
        bins=50,
        alpha=0.5,
        density=True,
        label="Class 0",
        color="r",
    )
    plt.hist(
        DVAL_LDA[0, LVAL == 1],
        bins=50,
        alpha=0.5,
        density=True,
        label="Class 1",
        color="b",
    )
    plt.axvline(threshold, color="r", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Histogram of the LDA component (Validation set)")
    if save:
        plt.savefig(f"{PATH}/report/plot/dim_red/LDA_threshold_opt.png")
    elif plot:
        plt.show()

    PVAL[DVAL_LDA[0] >= threshold] = 1
    PVAL[DVAL_LDA[0] < threshold] = 0

    error_rate = 100 * (PVAL != LVAL).sum() / LVAL.shape[0]
    print(f"Error rate (LDA): {error_rate:.2f} %")

    error_rates = []

    # combine PCA and LDA
    for i in range(1, 7):
        mu = DTR.mean(1)
        DTRc = DTR - mu.reshape(DTR.shape[0], 1)
        DVALc = DVAL - mu.reshape(DVAL.shape[0], 1)

        W_PCA = PCA(DTRc, i)
        DTR_PCA = np.dot(W_PCA.T, DTRc)
        DVAL_PCA = np.dot(W_PCA.T, DVALc)

        W_LDA = LDA(DTR_PCA, LTR, 1)
        DTR_LDA = np.dot(W_LDA.T, DTR_PCA)

        DVAL_LDA = np.dot(W_LDA.T, DVAL_PCA)
        # threshold = (DTR_LDA[0, LTR == 0].mean() + DTR_LDA[0, LTR == 1].mean()) / 2.0
        threshold = find_best_threshold(DTR_LDA, LTR, DVAL_LDA, LVAL)

        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_LDA[0] >= threshold] = 1
        PVAL[DVAL_LDA[0] < threshold] = 0

        error_rate = 100 * (PVAL != LVAL).sum() / LVAL.shape[0]

        print(f"Error rate (PCA={i}, LDA): {error_rate:.2f} %")

        error_rates.append(error_rate)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 7), [x for x in error_rates])
    plt.xlabel("Number of PCA components")
    plt.ylabel("Error rate (%)")
    plt.title("Error rate vs Number of PCA components")
    if save:
        plt.savefig(f"{PATH}/report/plot/dim_red/error_rate_opt.png")
    elif plot:
        plt.show()
