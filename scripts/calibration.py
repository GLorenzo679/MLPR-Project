import os

import matplotlib.pyplot as plt
import numpy as np
from utils.evaluation import compute_eval_metrics
from utils.kfold import kfold_calibration
from utils.utils import vrow

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def calibration(plot: bool = False, save: bool = False, verbose: bool = False):
    """
    Calibrates the models and plots the Bayes Error plot for each model.
    The calibration is done by using the kfold_calibration function.
    The Bayes Error plot is plotted for different training priors.

    Parameters:
    ----------
    plot (bool): If True, the plot will be displayed
    save (bool): If True, the plot will be saved
    verbose (bool): If True, the function will print the progress

    Returns:
    ----------
    None
    """

    scores_QLR, labels_QLR = np.load(PATH + "/data/scores/LogisticRegression.npy")
    scores_RBF_SVM, labels_RBF_SVM = np.load(PATH + "/data/scores/kernelSVM.npy")
    scores_GMM_8_32, labels_GMM_8_32 = np.load(PATH + "/data/scores/GMM.npy")

    scores_labels = [
        (scores_QLR, labels_QLR.astype(int)),
        (scores_RBF_SVM, labels_RBF_SVM.astype(int)),
        (scores_GMM_8_32, labels_GMM_8_32.astype(int)),
    ]

    model_names_short = ["QLR", "RBF_SVM", "GMM"]
    model_names = ["Quadratic Logistic Regression", "RBF SVM", "GMM (8-32 components)"]

    KFOLD_SPLITS = 5
    APP_PRIOR = 0.1

    training_priors = np.round(np.linspace(0.1, 0.9, 9), 1)

    eff_prior_log_odds = np.linspace(-4, 4, 21)  # log(odds) = log(p/(1-p))
    eff_priors = 1 / (1 + np.exp(-eff_prior_log_odds))

    for i, SL in enumerate(scores_labels):
        print(f"Model: {model_names[i]}")

        scores, labels = SL

        plt.figure(figsize=(12, 6))
        plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
        plt.ylabel("DCF")
        plt.xlim(-4, 4)
        plt.ylim(0, 1.02)

        for tp in training_priors:
            if verbose:
                print(f"Training prior {tp}")

            cal_scores = kfold_calibration(vrow(scores), labels, KFOLD_SPLITS, tp)

            if verbose:
                print(f"Results for training prior {tp} (app prior {APP_PRIOR}):")
                print(
                    f"minDCF: {compute_eval_metrics(scores.ravel(), labels, APP_PRIOR)[1]}"
                )
                print(
                    f"DCF: {compute_eval_metrics(scores.ravel(), labels, APP_PRIOR)[0]}"
                )
                print(
                    f"calDCF: {compute_eval_metrics(cal_scores.ravel(), labels, APP_PRIOR)[0]}"
                )
                print()

            uncal_DCFs = []
            cal_DCFs = []
            minDCfs = []

            for ep in eff_priors:
                uncal_DCF, DCFmin = compute_eval_metrics(scores.ravel(), labels, ep)
                calDCF, _ = compute_eval_metrics(cal_scores.ravel(), labels, ep)

                uncal_DCFs.append(uncal_DCF)
                cal_DCFs.append(calDCF)
                minDCfs.append(DCFmin)

            if tp == 0.1:
                plt.plot(
                    eff_prior_log_odds,
                    minDCfs,
                    label=f"minDCF",
                    linestyle="--",
                    color="r",
                )
                plt.plot(
                    eff_prior_log_odds,
                    uncal_DCFs,
                    label=f"DCF (tp={tp})",
                    linestyle=":",
                    # color="b",
                    linewidth=2,
                )

            plt.plot(
                eff_prior_log_odds,
                cal_DCFs,
                label=f"calDCF (tp={tp})",
                # color="b",
            )

        plt.title(f"Bayes Error plot for {model_names[i]}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.grid()

        if save:
            plt.savefig(
                f"{PATH}/report/plot/calibration/{model_names_short[i]}_different_tp.png"
            )
        elif plot:
            plt.show()
