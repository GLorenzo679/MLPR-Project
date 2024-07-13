import os

import numpy as np

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for the given predictions and labels.

    Parameters:
    ----------
    predictions (np.ndarray): The predicted labels
    labels (np.ndarray): The true labels

    Returns:
    ----------
    np.ndarray: The confusion matrix
    """
    num_classes = np.unique(labels).size
    cm = np.zeros((num_classes, num_classes))

    for i in range(len(labels)):
        cm[predictions[i], labels[i]] += 1

    return cm


def optimal_bayes_decision_binary(
    llr: np.ndarray, prior: float, Cfn: float = 1, Cfp: float = 1
) -> np.ndarray:
    """
    Compute the optimal Bayes decision for binary classification.

    Parameters:
    ----------
    llr (np.ndarray): The log-likelihood ratios
    prior (float): The prior probability of the positive class
    Cfn (float, optional): The cost of false negative (default is 1)
    Cfp (float, optional): The cost of false positive (default is 1)

    Returns:
    ----------
    np.ndarray: The binary decisions
    """
    threshold = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    predictions = llr > threshold

    return predictions.astype(int)


def bayes_risk(cm: np.ndarray, prior: float, Cfn: float = 1, Cfp: float = 1) -> float:
    """
    Compute the unnormalized Bayes risk.

    Parameters:
    ----------
    cm (np.ndarray): The confusion matrix
    prior (float): The prior probability of the positive class
    Cfn (float, optional): The cost of false negative (default is 1)
    Cfp (float, optional): The cost of false positive (default is 1)

    Returns:
    ----------
    float: The unnormalized Bayes risk
    """
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    DCFu = prior * Cfn * FNR + (1 - prior) * Cfp * FPR

    return DCFu


def normalized_bayes_risk(
    cm: np.ndarray, prior: float, Cfn: float = 1, Cfp: float = 1
) -> float:
    """
    Compute the normalized Bayes risk.

    Parameters:
    ----------
    cm (np.ndarray): The confusion matrix
    prior (float): The prior probability of the positive class
    Cfn (float, optional): The cost of false negative (default is 1)
    Cfp (float, optional): The cost of false positive (default is 1)

    Returns:
    ----------
    float: The normalized Bayes risk
    """
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    B_dummy = min(prior * Cfn, (1 - prior) * Cfp)

    DCF = (prior * Cfn * FNR + (1 - prior) * Cfp * FPR) / B_dummy

    return DCF


def minimum_bayes_risk(
    llr: np.ndarray,
    labels: np.ndarray,
    eff_prior: float,
    thresholds: np.ndarray,
    Cfn: float = 1,
    Cfp: float = 1,
) -> float:
    """
    Compute the minimum Bayes risk.

    Parameters:
    ----------
    llr (np.ndarray): The log-likelihood ratios
    labels (np.ndarray): The true labels
    eff_prior (float): The effective prior probability of the positive class
    thresholds (np.ndarray): The decision thresholds
    Cfn (float, optional): The cost of false negative (default is 1)
    Cfp (float, optional): The cost of false positive (default is 1)

    Returns:
    ----------
    float: The minimum Bayes risk
    """
    DCFs = []

    for t in thresholds:
        predictions = (llr > t).astype(int)
        cm = confusion_matrix(predictions, labels)
        DCFs.append(normalized_bayes_risk(cm, eff_prior, Cfn, Cfp))

    return min(DCFs)


def minimum_bayes_risk_optimized(
    llr: np.ndarray,
    labels: np.ndarray,
    eff_prior: float,
    thresholds: np.ndarray,
    Cfn: float = 1,
    Cfp: float = 1,
) -> float:
    """
    Compute the optimized minimum Bayes risk.

    Parameters:
    ----------
    llr (np.ndarray): The log-likelihood ratios
    labels (np.ndarray): The true labels
    eff_prior (float): The effective prior probability of the positive class
    thresholds (np.ndarray): The decision thresholds
    Cfn (float, optional): The cost of false negative (default is 1)
    Cfp (float, optional): The cost of false positive (default is 1)

    Returns:
    ----------
    float: The optimized minimum Bayes risk
    """
    idx = np.argsort(llr)
    llr = llr[idx]
    labels = labels[idx]
    minDCF = np.inf

    cm = None

    for i in range(len(thresholds) - 1):
        if cm is None:
            cm = confusion_matrix(np.ones(len(labels), dtype=int), labels)
        else:
            if labels[i - 1] == 0:
                cm[0, 0] += 1
                cm[1, 0] -= 1
            else:
                cm[0, 1] += 1
                cm[1, 1] -= 1

        DCF = normalized_bayes_risk(cm, eff_prior, Cfn, Cfp)

        if DCF < minDCF:
            minDCF = DCF

    return minDCF


def evaluate_model(
    model: object, eff_prior: float, DVAL: np.ndarray, LVAL: np.ndarray
) -> tuple:
    """
    Evaluate a model using the minimum Bayes risk metric

    Parameters:
    ----------
    model (object): The model to evaluate
    eff_prior (float): The effective prior
    DVAL (np.ndarray): The validation data
    LVAL (np.ndarray): The validation labels

    Returns:
    ----------
    float: The unnormalized minimum Bayes risk
    float: The normalized minimum Bayes risk
    float: The optimized minimum Bayes risk
    """

    model.fit()

    if "MVG" in model.__class__.__name__:
        val_scores = model.score(DVAL, eff_prior)
    elif "LogisticRegression" in model.__class__.__name__:
        val_scores = model.score(DVAL)
        emp_prior = np.mean(model.LTR)
        val_scores -= np.log(emp_prior / (1 - emp_prior))
    elif "SVM" in model.__class__.__name__:
        val_scores = model.score(DVAL)
    elif "GMM" in model.__class__.__name__:
        val_scores = model.score(DVAL)

    thresholds = np.array([-np.infty, *val_scores, np.infty])
    PVAL = optimal_bayes_decision_binary(val_scores, eff_prior, 1, 1)

    cm = confusion_matrix(PVAL, LVAL)
    DCFu = bayes_risk(cm, eff_prior, 1, 1)
    DCF = normalized_bayes_risk(cm, eff_prior, 1, 1)
    DCFmin = minimum_bayes_risk_optimized(val_scores, LVAL, eff_prior, thresholds, 1, 1)

    return DCFu, DCF, DCFmin


def compute_eval_metrics(scores: np.ndarray, L: np.ndarray, prior: float) -> tuple:
    """
    Compute the evaluation metrics for the given scores and labels.

    Parameters:
    ----------
    scores (np.ndarray): The scores
    L (np.ndarray): The labels
    prior (float): The prior probability of the positive class

    Returns:
    ----------
    tuple: The DCF and DCFmin
    """

    thresholds = np.array([-np.infty, *scores, np.infty])
    DCFmin = minimum_bayes_risk_optimized(scores, L, prior, thresholds)

    th = -np.log(prior / (1 - prior))
    PVAL = np.array(scores > th, dtype=int)
    cm = confusion_matrix(PVAL, L)
    DCF = normalized_bayes_risk(cm, prior)

    return DCF, DCFmin
