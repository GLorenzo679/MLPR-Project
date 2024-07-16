import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from models.GMM import GMM
from models.LR import LogisticRegression, PWLogisticRegression
from models.SVM import kernelSVM
from utils.evaluation import compute_eval_metrics, evaluate_model
from utils.utils import expand_features, split_db_2to1, vrow

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def evaluation(
    train_D: np.ndarray,
    train_L: np.ndarray,
    eval_D: np.ndarray,
    eval_L: np.ndarray,
    plot: bool = False,
    save: bool = False,
    verbose: bool = False,
):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(train_D, train_L)

    expanded_D_train = expand_features(train_D)
    (expanded_DTR, expanded_LTR), (expanded_DVAL, expanded_LVAL) = split_db_2to1(
        expanded_D_train, train_L
    )
    expanded_D_eval = expand_features(eval_D)

    # QLR = LogisticRegression(expanded_DTR, expanded_LTR, l=0.0316)
    # SVM_RBF = kernelSVM(DTR, LTR, kernel="rbf", C=32, gamma=0.135, K=1)
    # GMM_diag_8_32 = GMM(
    #     DTR,
    #     LTR,
    #     n_comp_0=8,
    #     n_comp_1=32,
    #     cov_type="diag",
    #     tol=1e-6,
    #     alpha=0.1,
    #     psi=0.01,
    # )

    # models = [QLR, SVM_RBF, GMM_diag_8_32]

    # train the models and save the scores
    # for model in models:
    #     print(f"Training {model.__class__.__name__}")
    #     model.fit()

    # for model in models:
    #     if verbose:
    #         print(f"Evaluating with {model.__class__.__name__}")

    #     if model.__class__.__name__ == "LogisticRegression":
    #         scores = model.score(expanded_D_eval)
    #     else:
    #         scores = model.score(eval_D)

    #     np.save(
    #         f"{PATH}/data/eval_scores/{model.__class__.__name__}.npy",
    #         (scores, eval_L.astype(int)),
    #     )

    # sys.exit(0)

    scores_QLR, labels_QLR = np.load(PATH + "/data/eval_scores/LogisticRegression.npy")
    scores_RBF_SVM, labels_RBF_SVM = np.load(PATH + "/data/eval_scores/kernelSVM.npy")
    scores_GMM_8_32, labels_GMM_8_32 = np.load(PATH + "/data/eval_scores/GMM.npy")

    scores_labels = [
        (scores_QLR, labels_QLR.astype(int)),
        (scores_RBF_SVM, labels_RBF_SVM.astype(int)),
        (scores_GMM_8_32, labels_GMM_8_32.astype(int)),
    ]

    model_names_short = ["QLR", "RBF_SVM", "GMM"]
    model_names = ["Quadratic Logistic Regression", "RBF SVM", "GMM (8-32 components)"]
    model_mask = [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]

    APP_PRIOR = 0.1

    eff_prior_log_odds = np.linspace(-4, 4, 21)  # log(odds) = log(p/(1-p))
    eff_priors = 1 / (1 + np.exp(-eff_prior_log_odds))

    # Part 1: evaluate the 3-model fusion (delivered system)
    fused_scores = np.vstack([sl[0] for sl in scores_labels])
    labels = scores_labels[0][1]

    DCF, minDCF = compute_eval_metrics(
        fused_scores.ravel(), np.hstack([labels] * 3), APP_PRIOR
    )

    if verbose:
        print(f"Results for application prior {APP_PRIOR} (3-fused model):")
        print(f"minDCF: {minDCF}")
        print(f"DCF: {DCF}\n")

    plt.figure(figsize=(12, 6))
    plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
    plt.ylabel("DCF")
    plt.xlim(-4, 4)
    plt.ylim(0, 1.02)
    plt.title("Bayes Error plot for the delivered system (eval dataset)")

    DCFs = []
    minDCFs = []

    for ep in eff_priors:
        DCF, minDCF = compute_eval_metrics(
            fused_scores.ravel(), np.hstack([labels] * 3), ep
        )
        DCFs.append(DCF)
        minDCFs.append(minDCF)

    plt.plot(eff_prior_log_odds, DCFs, label="DCF", color="b")
    plt.plot(eff_prior_log_odds, minDCFs, label="minDCF", color="r", linestyle="--")
    plt.legend()
    plt.grid()

    if plot:
        plt.show()
    if save:
        plt.savefig(f"{PATH}/report/plot/eval/delivered.png")

    # Part 2: evaluate the 3 models separately and their fusion
    plt.figure(figsize=(12, 6))
    plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
    plt.ylabel("DCF")
    plt.xlim(-4, 4)
    plt.ylim(0, 1.02)
    plt.title(f"Bayes Error plots (eval dataset)")

    # Evaluate the models individually
    for idx in range(3):
        if verbose:
            print(f"Evaluating {model_names[idx]}")

        scores, labels = scores_labels[idx]

        DCF, minDCF = compute_eval_metrics(scores, eval_L, APP_PRIOR)

        if verbose:
            print(f"Results for application prior {APP_PRIOR} ({model_names[idx]}):")
            print(f"minDCF: {minDCF}")
            print(f"DCF: {DCF}\n")

        DCFs = []
        minDCFs = []

        for ep in eff_priors:
            DCF, minDCF = compute_eval_metrics(scores, eval_L, ep)
            DCFs.append(DCF)
            minDCFs.append(minDCF)

        plt.plot(eff_prior_log_odds, DCFs, label="DCF (" + model_names_short[idx] + ")")
        # plt.plot(eff_prior_log_odds, minDCFs, label="minDCF", color="r", linestyle="--")

    # Evaluate the fusion of the 3 models
    for mask in model_mask:
        masked_sl = [sl for sl, m in zip(scores_labels, mask) if m]
        masked_names = [name for name, m in zip(model_names, mask) if m]
        masked_mn_short = [
            mn_short for mn_short, m in zip(model_names_short, mask) if m
        ]

        print(f"Model names: {' + '.join(masked_mn_short)}")

        scores = np.vstack([sl[0] for sl in masked_sl])
        labels = masked_sl[0][1]

        DCF, minDCF = compute_eval_metrics(
            scores.ravel(), np.hstack([labels] * len(masked_sl)), APP_PRIOR
        )

        if verbose:
            print(
                f"Results for application prior {APP_PRIOR} ({' + '.join(masked_mn_short)}):"
            )
            print(f"minDCF: {minDCF}")
            print(f"DCF: {DCF}\n")

        DCFs = []
        minDCFs = []

        for ep in eff_priors:
            DCF, minDCF = compute_eval_metrics(
                scores.ravel(), np.hstack([labels] * len(masked_sl)), ep
            )
            DCFs.append(DCF)
            minDCFs.append(minDCF)

        plt.plot(
            eff_prior_log_odds, DCFs, label="DCF (" + " + ".join(masked_mn_short) + ")"
        )
        # plt.plot(eff_prior_log_odds, minDCFs, label="minDCF", color="r", linestyle="--")

    plt.legend()
    plt.grid()
    if plot:
        plt.show()
    if save:
        plt.savefig(f"{PATH}/report/plot/eval/DCF_eval.png")

    # Part 3: calibration on the 3 single models
    # best train priors found in model selection
    best_train_priors = [0.7, 0.1, 0.9]

    # Load the scores and labels of the validation set
    val_scores_QLR, cal_labels_QLR = np.load(
        PATH + "/data/eval_scores/val_LogisticRegression.npy"
    )
    val_scores_RBF_SVM, cal_labels_RBF_SVM = np.load(
        PATH + "/data/eval_scores/val_kernelSVM.npy"
    )
    val_scores_GMM_8_32, cal_labels_GMM_8_32 = np.load(
        PATH + "/data/eval_scores/val_GMM.npy"
    )

    val_scores_labels = [
        (val_scores_QLR, cal_labels_QLR.astype(int)),
        (val_scores_RBF_SVM, cal_labels_RBF_SVM.astype(int)),
        (val_scores_GMM_8_32, cal_labels_GMM_8_32.astype(int)),
    ]

    for idx in range(3):
        plt.figure(figsize=(12, 6))
        plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
        plt.ylabel("DCF")
        plt.xlim(-4, 4)
        plt.ylim(0, 1.02)

        if verbose:
            print(f"Calibrating {model_names[idx]}")

        val_scores, val_labels = val_scores_labels[idx]
        scores, labels = scores_labels[idx]
        train_prior = best_train_priors[idx]

        # fit the calibration model on the validation set scores
        PWL = PWLogisticRegression(vrow(val_scores), val_labels, 0, train_prior)
        PWL.fit()
        # use the calibration model to calibrate the scores of the evaluation set
        cal_scores = PWL.score(vrow(scores))
        cal_scores -= np.log(train_prior / (1 - train_prior))

        DCF, minDCF = compute_eval_metrics(cal_scores.ravel(), labels, APP_PRIOR)

        if verbose:
            print(f"Results for application prior {APP_PRIOR} ({model_names[idx]}):")
            print(f"minDCF: {minDCF}")
            print(f"calDCF: {DCF}\n")

        uncal_DCFs = []
        minDCFs = []
        cal_DCFs = []

        for ep in eff_priors:
            uncal_DCF, minDCF = compute_eval_metrics(scores, labels, ep)
            DCF, _ = compute_eval_metrics(cal_scores, labels, ep)
            uncal_DCFs.append(uncal_DCF)
            minDCFs.append(minDCF)
            cal_DCFs.append(DCF)

        plt.plot(
            eff_prior_log_odds,
            cal_DCFs,
            label="calDCF",
            color="b",
        )
        plt.plot(eff_prior_log_odds, minDCFs, label="minDCF", color="r", linestyle="--")
        plt.plot(eff_prior_log_odds, uncal_DCFs, label="DCF", color="b", linestyle=":")
        plt.legend()
        plt.grid()
        plt.title(f"Calibration for {model_names[idx]} (eval dataset)")

        if plot:
            plt.show()
        if save:
            plt.savefig(
                f"{PATH}/report/plot/eval/eval_calib_{model_names_short[idx]}.png"
            )

    best_train_priors = [0.9, 0.3, 0.7, 0.4]

    # calibration for the fusion of the 3 models
    for mask in model_mask:
        plt.figure(figsize=(12, 6))
        plt.xlabel(r"$\log \left( \frac{\tilde{\pi}}{1 - \tilde{\pi}} \right)$")
        plt.ylabel("DCF")
        plt.xlim(-4, 4)
        plt.ylim(0, 1.02)

        masked_val_sl = [sl for sl, m in zip(val_scores_labels, mask) if m]
        masked_sl = [sl for sl, m in zip(scores_labels, mask) if m]

        masked_mn_short = [
            mn_short for mn_short, m in zip(model_names_short, mask) if m
        ]
        train_priors = [best_train_priors[i] for i, m in enumerate(mask) if m]

        print(f"Model names: {' + '.join(masked_mn_short)}")

        val_scores = np.vstack([sl[0] for sl in masked_val_sl])
        val_labels = masked_val_sl[0][1]
        scores = np.vstack([sl[0] for sl in masked_sl])
        labels = masked_sl[0][1]

        # fit the calibration model on the validation set scores
        PWL = PWLogisticRegression(
            val_scores,
            val_labels,
            0,
            train_priors[0],
        )
        PWL.fit()
        # use the calibration model to calibrate the scores of the evaluation set
        cal_scores = PWL.score(scores)
        cal_scores -= np.log(train_priors[0] / (1 - train_priors[0]))

        DCF, minDCF = compute_eval_metrics(cal_scores.ravel(), labels, APP_PRIOR)

        if verbose:
            print(
                f"Results for application prior {APP_PRIOR} ({' + '.join(masked_mn_short)}):"
            )
            print(f"minDCF: {minDCF}")
            print(f"calDCF: {DCF}\n")

        uncal_DCFs = []
        minDCFs = []
        cal_DCFs = []

        for ep in eff_priors:
            uncal_DCF, _ = compute_eval_metrics(
                scores.ravel(), np.hstack([labels] * len(masked_sl)), ep
            )
            DCF, minDCF = compute_eval_metrics(cal_scores.ravel(), labels, ep)

            uncal_DCFs.append(uncal_DCF)
            minDCFs.append(minDCF)
            cal_DCFs.append(DCF)

        plt.plot(
            eff_prior_log_odds,
            cal_DCFs,
            label="calDCF",
            color="b",
        )
        plt.plot(eff_prior_log_odds, minDCFs, label="minDCF", color="r", linestyle="--")
        plt.plot(
            eff_prior_log_odds,
            uncal_DCFs,
            label="DCF",
            color="b",
            linestyle=":",
        )
        plt.legend()
        plt.grid()
        plt.title(f"Calibration for {' + '.join(masked_mn_short)} (eval dataset)")

        if plot:
            plt.show()
        if save:
            plt.savefig(
                f"{PATH}/report/plot/eval/eval_calib_{'_'.join(masked_mn_short)}.png"
            )

    # Part 4: analysis of the training strategy for GMM
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
                DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, eval_D, eval_L)
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
        plt.title(f"GMM with {cov_type} covariance type (eval dataset)")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{PATH}/report/plot/eval/GMM/gmm_{cov_type}.png")
        elif plot:
            plt.show()
