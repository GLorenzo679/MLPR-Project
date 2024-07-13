import os

import matplotlib.pyplot as plt
import numpy as np
from utils.evaluation import compute_eval_metrics
from utils.kfold import kfold_calibration

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def fusion(plot: bool = False, save: bool = False, verbose: bool = False):
    scores_QLR, labels_QLR = np.load(PATH + "/data/scores/LogisticRegression.npy")
    scores_RBF_SVM, labels_RBF_SVM = np.load(PATH + "/data/scores/kernelSVM.npy")
    scores_GMM_8, labels_GMM_8 = np.load(PATH + "/data/scores/GMM.npy")

    scores_labels = [
        (scores_QLR, labels_QLR.astype(int)),
        (scores_RBF_SVM, labels_RBF_SVM.astype(int)),
        (scores_GMM_8, labels_GMM_8.astype(int)),
    ]

    model_names_short = ["QLR", "RBF_SVM", "GMM"]
    model_names = [
        "Quadratic Logistic Regression",
        "RBF SVM",
        "GMM diag (8-32 components)",
    ]
    model_mask = [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]

    KFOLD_SPLITS = 5
    APP_PRIOR = 0.1

    training_priors = np.round(np.linspace(0.1, 0.9, 9), 1)

    eff_prior_log_odds = np.linspace(-4, 4, 21)  # log(odds) = log(p/(1-p))
    eff_priors = 1 / (1 + np.exp(-eff_prior_log_odds))

    for mm in model_mask:
        masked_sl = [sl for sl, m in zip(scores_labels, mm) if m]
        masked_names = [name for name, m in zip(model_names, mm) if m]
        masked_mn_short = [mn_short for mn_short, m in zip(model_names_short, mm) if m]

        print(f"Model names: {' + '.join(masked_mn_short)}")

        scores = np.vstack([sl[0] for sl in masked_sl])
        labels = masked_sl[0][1]

        # DCF, minDCF = compute_eval_metrics(
        #     scores.ravel(), np.hstack([labels] * len(masked_sl)), APP_PRIOR
        # )
        # if verbose:
        # print(
        #     f"Results (no calibration) for application prior {APP_PRIOR} ({' + '.join(masked_mn_short)}):"
        # )
        # print(f"minDCF: {minDCF:.3f}")
        # print(f"DCF: {DCF:.3f}")
        # print()

        plt.figure(figsize=(12, 6))
        plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
        plt.ylabel("DCF")
        plt.xlim(-4, 4)
        plt.ylim(0, 1.02)

        for tp in training_priors:
            cal_scores = kfold_calibration(scores, labels, KFOLD_SPLITS, tp)

            if verbose:
                print(f"Results for training prior {tp} (app prior {APP_PRIOR}):")
                print(
                    f"minDCF: {compute_eval_metrics(cal_scores, labels, APP_PRIOR)[1]:.3f}"
                )
                print(
                    f"DCF: {compute_eval_metrics(scores.ravel(), np.hstack([labels] * len(masked_sl)), APP_PRIOR)[0]:.3f}"
                )
                print(
                    f"calDCF: {compute_eval_metrics(cal_scores, labels, APP_PRIOR)[0]:.3f}"
                )
                print()

            uncal_DCFs = []
            cal_DCFs = []
            minDCfs = []

            for ep in eff_priors:
                uncal_DCF, _ = compute_eval_metrics(
                    scores.ravel(), np.hstack([labels] * len(masked_sl)), ep
                )
                cal_DCF, cal_minDCF = compute_eval_metrics(cal_scores, labels, ep)

                uncal_DCFs.append(uncal_DCF)
                cal_DCFs.append(cal_DCF)
                minDCfs.append(cal_minDCF)

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

        plt.title(f"Bayes Error plot for {', '.join(masked_names)}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.grid()

        if save:
            plt.savefig(
                f"{PATH}/report/plot/fusion/{'_'.join(masked_mn_short)}_calibration.png"
            )
        elif plot:
            plt.show()
