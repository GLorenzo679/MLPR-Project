import os

import matplotlib.pyplot as plt
import numpy as np
from models.MVG import MVG
from utils.utils import vrow

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def gaussian_analysis(
    D: np.ndarray, L: np.ndarray, save: bool = False, plot: bool = False
) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(D.shape[0]):
        D_reduced = D[i, :]
        mvg = MVG(vrow(D_reduced), L)
        mvg.fit()

        for j in range(2):
            mu = mvg.mean[j]
            sigma = np.sqrt(mvg.covariance[j])

            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000).ravel()
            pdfGAU = np.exp(mvg.__logpdf_GAU_ND_fast__(vrow(x), mu, sigma))
            if i == 0:
                ax[i // 3, i % 3].hist(
                    D_reduced[L == j],
                    bins=50,
                    density=True,
                    alpha=0.5,
                    label=f"Class {j}",
                    color="r" if j == 0 else "b",
                )
                ax[i // 3, i % 3].plot(
                    x, pdfGAU, label=f"Class {j}", color="g" if j == 0 else "orange"
                )
            else:
                ax[i // 3, i % 3].hist(
                    D_reduced[L == j],
                    bins=50,
                    density=True,
                    alpha=0.5,
                    color="r" if j == 0 else "b",
                )
                ax[i // 3, i % 3].plot(x, pdfGAU, color="g" if j == 0 else "orange")

        ax[i // 3, i % 3].set_title(f"Feature {i + 1}")

    fig.suptitle("Gaussian PDFs of the features")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    if save:
        plt.savefig(
            f"{PATH}/report/plot/gauss_analysis/gaussian_analysis.png",
            bbox_inches="tight",
        )
    elif plot:
        plt.show()
