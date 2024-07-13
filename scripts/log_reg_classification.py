import os

import matplotlib.pyplot as plt
import numpy as np
from models.LR import LogisticRegression, PWLogisticRegression
from utils.dimensionalityReduction import PCA
from utils.evaluation import evaluate_model
from utils.utils import expand_features, split_db_2to1

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

APP_PRIOR = 0.1


def log_reg_classification(D, L, save=False, plot=False):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    lambdas = np.logspace(-4, 2, 13)

    # part 1: evaluate the logistic regression model with different lambdas
    DCfs = []
    minDCfs = []

    for l in lambdas:
        model = LogisticRegression(DTR, LTR, l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Logistic regression model evaluation")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/log_reg.png")
    elif plot:
        plt.show()

    # part 2: repeat the evaluation but keep only 1 in 50 training samples
    DCfs = []
    minDCfs = []

    for l in lambdas:
        model = LogisticRegression(DTR[:, ::50], LTR[::50], l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Logistic regression model evaluation (1 in 50 training samples)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/log_reg_1in50.png")
    elif plot:
        plt.show()

    # part 3: evaluate the prior weighted logistic regression model
    for pi in np.linspace(0.1, 0.9, 9):
        DCfs = []
        minDCfs = []

        for l in lambdas:
            model = PWLogisticRegression(DTR, LTR, l, pi)
            DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL, LVAL)
            DCfs.append(DCF)
            minDCfs.append(DCFmin)

        plt.figure(figsize=(12, 6))
        plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
        plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylim(0, 1.02)
        plt.ylabel("DCF")
        plt.title(
            f"Prior weighted logistic regression model evaluation (prior = {pi:.1f})"
        )
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{PATH}/report/plot/LR/prior_w_log_reg.png")
        elif plot:
            plt.show()

    # part 4: evaluate the quadratic logistic regression model
    DCfs = []
    minDCfs = []

    expanded_D = expand_features(D)
    (expanded_DTR, LTR), (expanded_DVAL, LVAL) = split_db_2to1(expanded_D, L)

    for l in lambdas:
        model = LogisticRegression(expanded_DTR, LTR, l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, expanded_DVAL, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Quadratic logistic regression model evaluation")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/quadratic_log_reg.png")
    elif plot:
        plt.show()

    # part 5: evaluate the logistic regression model with centered, z-norm, withening and pca on data

    # centered
    DTR_mean = np.mean(DTR, axis=1).reshape(-1, 1)
    DTR_centered = DTR - DTR_mean
    DVAL_centered = DVAL - DTR_mean

    DCfs = []
    minDCfs = []

    for l in lambdas:
        model = LogisticRegression(DTR_centered, LTR, l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL_centered, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Logistic regression model evaluation (centered data)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/log_reg_centered.png")
    elif plot:
        plt.show()

    # z-norm
    DTR_std = np.std(DTR, axis=1).reshape(-1, 1)
    DTR_znorm = (DTR - DTR_mean) / DTR_std
    DVAL_znorm = (DVAL - DTR_mean) / DTR_std

    DCfs = []
    minDCfs = []

    for l in lambdas:
        model = LogisticRegression(DTR_znorm, LTR, l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL_znorm, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Logistic regression model evaluation (z-norm data)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/log_reg_znorm.png")
    elif plot:
        plt.show()

    # withening
    DTR_cov = np.cov(DTR)
    _, DTR_eigvecs = np.linalg.eig(DTR_cov)
    DTR_withening = np.dot(DTR_eigvecs.T, DTR_znorm)
    DVAL_withening = np.dot(DTR_eigvecs.T, DVAL_znorm)

    DCfs = []
    minDCfs = []

    for l in lambdas:
        model = LogisticRegression(DTR_withening, LTR, l)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL_withening, LVAL)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Logistic regression model evaluation (withening data)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/LR/log_reg_withening.png")
    elif plot:
        plt.show()

    # pca
    for i in range(1, 7):
        mu = DTR.mean(1)
        DTRc = DTR - mu.reshape(DTR.shape[0], 1)
        DVALc = DVAL - mu.reshape(DVAL.shape[0], 1)

        W_PCA = PCA(DTRc, i)
        DTR_pca = np.dot(W_PCA.T, DTRc)
        DVAL_pca = np.dot(W_PCA.T, DVALc)

        DCfs = []
        minDCfs = []

        for l in lambdas:
            model = LogisticRegression(DTR_pca, LTR, l)
            DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DVAL_pca, LVAL)
            DCfs.append(DCF)
            minDCfs.append(DCFmin)

        plt.figure(figsize=(12, 6))
        plt.plot(lambdas, DCfs, label="DCF", color="r", linewidth=2)
        plt.plot(lambdas, minDCfs, label="minDCF", color="b", linewidth=2)
        plt.xscale("log")
        plt.xlabel("Lambda")
        plt.ylim(0, 1.02)
        plt.ylabel("DCF")
        plt.title(f"Logistic regression model evaluation (PCA {i} components)")
        plt.legend()
        plt.grid()
        if save:
            plt.savefig(f"{PATH}/report/plot/LR/log_reg_pca{i}.png")
        elif plot:
            plt.show()
