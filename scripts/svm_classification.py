import os

import matplotlib.pyplot as plt
import numpy as np
from models.SVM import SVM, kernelSVM
from utils.evaluation import evaluate_model
from utils.utils import split_db_2to1

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

APP_PRIOR = 0.1


def svm_classification(D, L, save=False, plot=False, verbose=False):
    # D = D[:, ::20]
    # L = L[::20]
    K = 1
    D_extended = np.vstack((D, K * np.ones(D.shape[1])))

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    (DTR_extended, _), (DTE_extended, _) = split_db_2to1(D_extended, L)
    Cs = np.logspace(-5, 0, 11)

    # part 1: evaluate the lienar SVM model with different Cs
    if verbose:
        print("Part 1: Linear SVM model evaluation")

    DCfs = []
    minDCfs = []

    for C in Cs:
        K = 1
        model = SVM(DTR_extended, LTR, C)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DTE_extended, LTE)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(Cs, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(Cs, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("SVM model evaluation")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/SVM/svm.png")
    elif plot:
        plt.show()

    # part 2: evaluate the lienar SVM model with different Cs and centered data
    if verbose:
        print("Part 2: Linear SVM model evaluation (centered data)")

    DCfs = []
    minDCfs = []

    # maybe do the centering wrt the extended data?
    DTR_mean = np.mean(DTR_extended, axis=1).reshape(-1, 1)
    DTR_c_ext = DTR_extended - DTR_mean
    DTE_c_ext = DTE_extended - DTR_mean
    # DTR_c_ext = np.vstack((DTR_c, K * np.ones(DTR_c.shape[1])))
    # DTE_c_ext = np.vstack((DTE_c, K * np.ones(DTE_c.shape[1])))

    for C in Cs:
        model = SVM(DTR_c_ext, LTR, C)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DTE_c_ext, LTE)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(Cs, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(Cs, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("SVM model evaluation (centered data)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/SVM/svm_centered.png")
    elif plot:
        plt.show()

    # part 3: evaluate the poly kernel SVM model with different Cs
    if verbose:
        print("Part 3: Polynomial kernel SVM model evaluation")

    DCfs = []
    minDCfs = []

    for C in Cs:
        model = kernelSVM(DTR, LTR, "poly", C, K=0, c=1, d=2)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DTE, LTE)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(Cs, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(Cs, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Polynomial kernel SVM model evaluation (c=1, d=2)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/SVM/svm_poly.png")
    elif plot:
        plt.show()

    # part 4: evaluate the rbf kernel SVM model with different Cs and gammas
    if verbose:
        print("Part 4: RBF kernel SVM model evaluation")

    gammas = np.exp(np.linspace(-4, -1, 4))
    Cs = np.logspace(-3, 2, 11)

    plt.figure(figsize=(12, 6))

    for g in gammas:
        DCfs = []
        minDCfs = []

        if verbose:
            print(f"Gamma: {g:.3f}")

        for C in Cs:
            if verbose:
                print(f"C: {C:.5f}")

            model = kernelSVM(DTR, LTR, "rbf", C, K=1, gamma=g)
            DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DTE, LTE)
            DCfs.append(DCF)
            minDCfs.append(DCFmin)
            print(DCFmin)

        plt.plot(Cs, DCfs, label=f"DCF (gamma={g:.2f})", linewidth=2)
        plt.plot(Cs, minDCfs, label=f"minDCF (gamma={g:.2f})", linewidth=2)

    plt.xlabel("C")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("RBF kernel SVM model evaluation")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/SVM/svm_rbf.png")
    elif plot:
        plt.show()

    # part 5: evaluate the poly kernel SVM model with different Cs (different configuration)
    if verbose:
        print("Part 5: Polynomial kernel SVM model evaluation (d=4)")

    DCfs = []
    minDCfs = []

    Cs = np.logspace(-5, 0, 11)

    for C in Cs:
        model = kernelSVM(DTR, LTR, "poly", C, K=0, c=1, d=4)
        DCFu, DCF, DCFmin = evaluate_model(model, APP_PRIOR, DTE, LTE)
        DCfs.append(DCF)
        minDCfs.append(DCFmin)

    plt.figure(figsize=(12, 6))
    plt.plot(Cs, DCfs, label="DCF", color="r", linewidth=2)
    plt.plot(Cs, minDCfs, label="minDCF", color="b", linewidth=2)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylim(0, 1.02)
    plt.ylabel("DCF")
    plt.title("Polynomial kernel SVM model evaluation (c=1, d=4)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{PATH}/report/plot/SVM/svm_poly_2.png")
    elif plot:
        plt.show()
