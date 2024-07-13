import os

from scripts.bayes_evaluation import bayes_decision_model_evaluation
from scripts.calibration import calibration
from scripts.dimensionality_reduction import dimensionality_reduction
from scripts.evaluation import evaluation
from scripts.feature_analysis import feature_analysis
from scripts.fusion import fusion
from scripts.gaussian_analysis import gaussian_analysis
from scripts.gaussian_classification import (
    correlation_analysis,
    gaussian_classification,
)
from scripts.gmm_classification import gmm_classification
from scripts.log_reg_classification import log_reg_classification
from scripts.model_selection import model_selection
from scripts.svm_classification import svm_classification
from utils.utils import load_dataset

PATH = os.path.abspath(os.path.dirname(__file__))


def main():
    D, L = load_dataset(f"{PATH}/data/trainData.txt")

    print("PART 1 (lab 2):\n")
    print("Feature analysis\n")
    feature_analysis(D, L)
    print("-" * 80)

    print("PART 2 (lab 3):\n")
    print("Dimensionality reduction\n")
    dimensionality_reduction(D, L)
    print("-" * 80)

    print("PART 3 (lab 4):\n")
    print("Gaussian analysis\n")
    gaussian_analysis(D, L, save=True)
    print("-" * 80)

    print("PART 4 (lab 5):\n")
    print("**Gaussian classification (all 6 features)**")
    gaussian_classification(D, L)
    print("Correlation analysis\n")
    correlation_analysis(D, L)
    print("\n**Gaussian classification (only the first 4 features)**")
    D_reduced = D[:4, :]
    gaussian_classification(D_reduced, L)
    print("\n**Gaussian classification (only features 1-2)**")
    D_reduced = D[:2, :]
    gaussian_classification(D_reduced, L)
    print("\n**Gaussian classification (only features 3-4)**")
    D_reduced = D[2:4, :]
    gaussian_classification(D_reduced, L)
    print("\n**Gaussian classification (PCA)**")
    for m in range(1, 7):
        print(f"\nPCA with {m} components")
        gaussian_classification(D, L, preprocess_PCA=True, m=m)
    print("-" * 80)

    print("PART 5 (lab 7):\n")
    print("Bayes decision model evaluation\n")
    bayes_decision_model_evaluation(D, L)
    print("-" * 80)

    print("PART 6 (lab 8):\n")
    print("Logistic regression classification\n")
    log_reg_classification(D, L, plot=True)
    print("-" * 80)

    print("PART 7 (lab 9):\n")
    print("SVM classification\n")
    svm_classification(D, L)
    print("-" * 80)

    print("PART 8 (lab 10):\n")
    print("GMM classification\n")
    gmm_classification(D, L, verbose=True)
    print("-" * 80)

    print("PART 9 (lab 10):\n")
    print("Model selection\n")
    model_selection(D, L, verbose=True)
    print("-" * 80)

    print("PART 10 (lab 11):\n")
    print("Calibration\n")
    calibration(verbose=True)
    print("-" * 80)

    print("PART 11 (lab 11):\n")
    print("Fusion\n")
    fusion(verbose=True)
    print("-" * 80)

    eval_D, eval_L = load_dataset(f"{PATH}/data/evalData.txt")
    print("PART 12 (lab 11):\n")
    print("Evaluation\n")
    evaluation(D, L, eval_D, eval_L, verbose=True)
    print("-" * 80)


if __name__ == "__main__":
    main()
