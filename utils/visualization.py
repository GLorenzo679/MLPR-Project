import os

import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def plot_histogram(
    D: np.ndarray, L: np.ndarray, save: bool = False, plot: bool = False
) -> None:
    """
    Plot histograms for each feature, separated by class.

    Parameters:
    ----------
    D (np.ndarray): The data matrix
    L (np.ndarray): The labels
    save (bool, optional): Whether to save the plots to files (default is False)
    plot (bool, optional): Whether to display the plots (default is False)
    """
    C0 = D[:, L == 0]
    C1 = D[:, L == 1]

    for i in range(D.shape[0]):
        plt.figure(figsize=(12, 6))
        plt.hist(
            C0[i, :],
            bins=50,
            alpha=0.5,
            density=True,
            label="Class 0 (counterfeit)",
            color="r",
        )
        plt.hist(
            C1[i, :],
            bins=50,
            alpha=0.5,
            density=True,
            label="Class 1 (genuine)",
            color="b",
        )
        plt.legend(loc="upper right")
        plt.title(f"Feature {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency (Normalized)")
        if save:
            plt.savefig(f"{PATH}/report/plot/features/feature_{i + 1}.png")
        elif plot:
            plt.show()


def plot_features(
    D: np.ndarray, L: np.ndarray, save: bool = False, plot: bool = False
) -> None:
    """
    Plot pairwise feature scatter plots and histograms for each feature.

    Parameters:
    ----------
    D (np.ndarray): The data matrix
    L (np.ndarray): The labels
    save (bool, optional): Whether to save the plots to files (default is False)
    """
    C0 = D[:, L == 0]
    C1 = D[:, L == 1]

    fig, ax = plt.subplots(D.shape[0], D.shape[0], figsize=(15, 15))
    plt.tight_layout()

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i == j:
                ax[i, j].hist(C0[i, :], bins=20, alpha=0.5, color="r")
                ax[i, j].hist(C1[i, :], bins=20, alpha=0.5, color="b")
            else:
                ax[i, j].scatter(C0[i, :], C0[j, :], alpha=0.5, color="r")
                ax[i, j].scatter(C1[i, :], C1[j, :], alpha=0.5, color="b")

    if save:
        plt.savefig(f"{PATH}/report/plot/features/features_all.png")
    elif plot:
        plt.show()


def plot_pairwise_features(
    D: np.ndarray, L: np.ndarray, save: bool = False, plot: bool = False
) -> None:
    """
    Plot histograms and scatter plots for pairs of features.

    Parameters:
    ----------
    D (np.ndarray): The data matrix
    L (np.ndarray): The labels
    save (bool, optional): Whether to save the plots to files (default is False)
    """
    C0 = D[:, L == 0]
    C1 = D[:, L == 1]

    for i in range(D.shape[0] // 2):
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        for j in range(2):
            for k in range(2):
                if j == k:
                    ax[j, k].hist(
                        C0[(i * 2) + j, :],
                        bins=20,
                        alpha=0.5,
                        color="r",
                    )
                    ax[j, k].hist(
                        C1[(i * 2) + j, :],
                        bins=20,
                        alpha=0.5,
                        color="b",
                    )
                    ax[j, k].set_xlabel(f"Feature {(i * 2) + j + 1}")
                    ax[j, k].set_ylabel("Frequency")
                else:
                    if j == 0:
                        label0 = "Class 0 (counterfeit)"
                        label1 = "Class 1 (genuine)"
                    else:
                        label0 = None
                        label1 = None

                    ax[j, k].scatter(
                        C0[(i * 2) + j, :],
                        C0[(i * 2) + k, :],
                        alpha=0.5,
                        color="r",
                        label=label0,
                    )
                    ax[j, k].scatter(
                        C1[(i * 2) + j, :],
                        C1[(i * 2) + k, :],
                        alpha=0.5,
                        color="b",
                        label=label1,
                    )
                    ax[j, k].set_xlabel(f"Feature {(i * 2) + j + 1}")
                    ax[j, k].set_ylabel(f"Feature {(i * 2) + k + 1}")

        fig.suptitle(f"Features {i * 2 + 1} and {i * 2 + 2}")
        fig.legend(loc="upper right")
        if save:
            plt.savefig(
                f"{PATH}/report/plot/features/features_{i * 2 + 1}_{i * 2 + 2}.png"
            )
        if plot:
            plt.show()
