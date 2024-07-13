import os

import matplotlib.pyplot as plt
import numpy as np
from models.GMM import GMM
from models.LR import LogisticRegression
from models.SVM import kernelSVM
from utils.evaluation import evaluate_model
from utils.utils import expand_features, split_db_2to1

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# import sys


# def save_scores(DVAL, LVAL, model):
#     model.fit()
#     scores = model.score(DVAL)
#     np.save(
#         f"{PATH}/data/scores/{model.__class__.__name__}.npy",
#         (scores, LVAL.astype(int)),
#     )


def model_selection(D, L, save=False, plot=False, verbose=False):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    expanded_D = expand_features(D)
    (expanded_DTR, expanded_LTR), (expanded_DVAL, expanded_LVAL) = split_db_2to1(
        expanded_D, L
    )

    QLR = LogisticRegression(expanded_DTR, expanded_LTR, l=0.0316)
    SVM_RBF = kernelSVM(DTR, LTR, kernel="rbf", C=32, gamma=0.135, K=1)
    GMM_diag_8_32 = GMM(
        DTR,
        LTR,
        n_comp_0=8,
        n_comp_1=32,
        cov_type="diag",
        tol=1e-6,
        alpha=0.1,
        psi=0.01,
    )

    models = [QLR, SVM_RBF, GMM_diag_8_32]

    eff_prior_log_odds = np.linspace(-4, 4, 21)  # log(odds) = log(p/(1-p))
    eff_priors = 1 / (1 + np.exp(-eff_prior_log_odds))

    # for model in models:
    #     if model.__class__.__name__ == "LogisticRegression":
    #         save_scores(expanded_DVAL, expanded_LVAL, model)
    #     else:
    #         save_scores(DVAL, LVAL, model)

    # sys.exit(0)
    for model in models:
        DCfs = []
        minDCfs = []

        if verbose:
            print(f"\nModel: {model.__class__.__name__}")

        for ep in eff_priors:
            if verbose:
                idx_eff_prior = np.where(eff_priors == ep)[0][0]
                print(f"Effective prior {idx_eff_prior + 1}/{len(eff_priors)}")

            if model.__class__.__name__ == "LogisticRegression":
                DCFu, DCF, DCFmin = evaluate_model(
                    model, ep, expanded_DVAL, expanded_LVAL
                )
            else:
                DCFu, DCF, DCFmin = evaluate_model(model, ep, DVAL, LVAL)

            DCfs.append(DCF)
            minDCfs.append(DCFmin)

        idx_min_DCF = np.argmin(minDCfs)
        print(f"Model: {model.__class__.__name__}")
        print(f"Minimum DCF: {minDCfs[idx_min_DCF]}")
        print(f"Effective prior: {eff_priors[idx_min_DCF]}")
        print()

        plt.figure(figsize=(12, 6))
        plt.plot(eff_prior_log_odds, DCfs, label="DCF", color="r", linewidth=2)
        plt.plot(eff_prior_log_odds, minDCfs, label="minDCF", color="b", linewidth=2)
        plt.xlabel(r"$- \log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
        plt.ylabel("DCF")
        plt.xlim(-4, 4)
        plt.ylim(0, 1.02)

        if model.__class__.__name__ == "LogisticRegression":
            plt.title(
                "Bayes Error plot for Quadratic Logistic Regression "
                + r"$(\lambda$"
                + "=0.032)"
            )
        elif model.__class__.__name__ == "kernelSVM":
            plt.title(
                "Bayes Error plot for SVM (RBF kernel, C=32, "
                + r"$\gamma$"
                + "=0.14, K=1)"
            )
        elif model.__class__.__name__ == "GMM":
            plt.title("Bayes Error plot for GMM (8-32 components, diag cov)")

        plt.legend()
        plt.grid()
        if save:
            plt.savefig(
                f"{PATH}/report/plot/model_selection/{model.__class__.__name__}.png"
            )
        elif plot:
            plt.show()
