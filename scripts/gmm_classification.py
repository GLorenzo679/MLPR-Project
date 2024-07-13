import os

import matplotlib.pyplot as plt
import numpy as np
from models.GMM import GMM
from utils.evaluation import evaluate_model
from utils.utils import split_db_2to1

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

APP_PRIOR = 0.1


def gmm_classification(D, L, save=False, plot=False, verbose=False):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # part 1: evaluate the GMM model (full and diagonal)
    for cov_type in ["full", "diag", "tied"]:
        DCfs = []
        minDCfs = []

        if verbose:
            print("Running model for cov_type:", cov_type)

        for n_comp_0 in [1, 2, 4, 8, 16, 32]:
            for n_comp_1 in [1, 2, 4, 8, 16, 32]:
                if verbose:
                    print(f"\tWith components: {n_comp_0} and {n_comp_1}")
                model = GMM(
                    DTR,
                    LTR,
                    n_comp_0,
                    n_comp_1,
                    cov_type,
                    tol=1e-6,
                    alpha=0.1,
                    psi=0.01,
                )
                DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL, LVAL)
                DCfs.append(DCF)
                minDCfs.append(DCFmin)

        plt.figure(figsize=(12, 6))
        x = np.arange(36)
        width = 0.35
        plt.bar(x - width / 2, DCfs, width, label="DCF")
        plt.bar(x + width / 2, minDCfs, width, label="minDCF")
        plt.xticks(
            x,
            [
                f"{n_comp_0}-{n_comp_1}"
                for n_comp_0 in [1, 2, 4, 8, 16, 32]
                for n_comp_1 in [1, 2, 4, 8, 16, 32]
            ],
            rotation=90,
        )
        plt.xlabel("Model")
        plt.ylabel("DCF")
        plt.title(f"GMM with {cov_type} covariance type")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{PATH}/report/plot/GMM/gmm_{cov_type}.png")
        elif plot:
            plt.show()
