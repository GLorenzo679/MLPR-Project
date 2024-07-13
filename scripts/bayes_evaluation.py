import matplotlib.pyplot as plt
import numpy as np
from models.MVG import MVG, NaiveMVG, TiedMVG
from utils.dimensionalityReduction import PCA
from utils.evaluation import evaluate_model
from utils.utils import split_db_2to1

APP_PRIOR = 0.1


def bayes_decision_model_evaluation(D, L):
    # part 1: consider 5 applications and observe the effective priors
    applications = [
        (0.5, 1, 1),
        (0.9, 1, 1),
        (0.1, 1, 1),
        (0.5, 1, 9),
        (0.5, 9, 1),
    ]

    for i, (prior, Cfn, Cfp) in enumerate(applications):
        eff_prior = (prior * Cfn) / (prior * Cfn + ((1 - prior) * Cfp))
        print(
            f"Effective prior for application {i + 1} (prior={prior}, Cfn={Cfn}, Cfp={Cfp}): {eff_prior:.2f}"
        )

    # part 2: we consider only the first 3 applications
    # use the MVG model to evaluate the performance (with and without PCA)
    # use as evaluation metrics the DCFu, DCF, and minDCF
    effective_priors = [0.5, 0.9, 0.1]

    # MVG without PCA
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    models = [MVG(DTR, LTR), NaiveMVG(DTR, LTR), TiedMVG(DTR, LTR)]

    for eff_prior in effective_priors:
        print()
        print("-" * 80)
        print(f"Effective prior: {eff_prior}")

        for model in models:
            DCFu, DCF, DCFmin = evaluate_model(model, eff_prior, DVAL, LVAL)

            print(f"\nModel: {model.__class__.__name__}")
            print(f"DCF: {DCF:.5f}")
            print(f"Minimum Bayes risk (min DCF): {DCFmin:.5f}")
            print(f"Percentual difference: {(abs(DCF - DCFmin) / DCF) * 100:.5f}")

    # MVG with PCA
    for m in range(1, 7):
        print()
        print("-" * 80)
        print(f"PCA with {m} components")

        mu = DTR.mean(1)
        DTRc = DTR - mu.reshape(DTR.shape[0], 1)
        DVALc = DVAL - mu.reshape(DVAL.shape[0], 1)

        W_PCA = PCA(DTRc, m)
        DTR_PCA = np.dot(W_PCA.T, DTRc)
        DVAL_PCA = np.dot(W_PCA.T, DVALc)
        models = [MVG(DTR_PCA, LTR), NaiveMVG(DTR_PCA, LTR), TiedMVG(DTR_PCA, LTR)]

        for eff_prior in effective_priors:
            print()
            print("-" * 80)
            print(f"Effective prior: {eff_prior}")

            for model in models:
                DCFu, DCF, DCFmin = evaluate_model(model, eff_prior, DVAL_PCA, LVAL)

                print(f"\nModel: {model.__class__.__name__}")
                print(f"DCF: {DCF:.5f}")
                print(f"Minimum Bayes risk (min DCF): {DCFmin:.5f}")
                print(f"Percentual difference: {(abs(DCF - DCFmin) / DCF) * 100:.5f}")

    # part 3: consider the PCA setup that gave the best results for eff_prior = 0.1
    # PCA with 5 components is seen to be the best (2 components is the second best)
    # Compute the bayes error plot
    mu = DTR.mean(1)
    DTRc = DTR - mu.reshape(DTR.shape[0], 1)
    DVALc = DVAL - mu.reshape(DVAL.shape[0], 1)

    W_PCA = PCA(DTRc, 5)
    DTR_PCA = np.dot(W_PCA.T, DTRc)
    DVAL_PCA = np.dot(W_PCA.T, DVALc)

    models = [MVG(DTR_PCA, LTR), NaiveMVG(DTR_PCA, LTR), TiedMVG(DTR_PCA, LTR)]

    eff_prior_log_odds = np.linspace(-4, 4, 30)
    eff_prior = 1 / (1 + np.exp(-eff_prior_log_odds))

    DCFs = []
    minDCFs = []

    for model in models:
        DCFs.append([])
        minDCFs.append([])

        for ep in eff_prior:
            DCFu, DCF, DCFmin = evaluate_model(model, ep, DVAL_PCA, LVAL)

            DCFs[-1].append(DCF)
            minDCFs[-1].append(DCFmin)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    for i, model in enumerate(models):
        ax[i].plot(eff_prior_log_odds, DCFs[i], label="DCF", color="r", linewidth=2)
        ax[i].plot(
            eff_prior_log_odds, minDCFs[i], label="minDCF", color="b", linewidth=2
        )
        ax[i].set_xlabel("Effective prior log-odds")
        ax[i].set_ylabel("DCF")
        ax[i].set_title(f"{model.__class__.__name__}")
        ax[i].legend()

    plt.show()
