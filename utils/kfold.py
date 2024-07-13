import numpy as np
from models.LR import PWLogisticRegression


def kfold_idxs(num_samples: int, K: int) -> list:
    """
    Generate the indices for the k-fold cross-validation.

    Parameters:
    ----------
    num_samples (int): The number of samples in the dataset
    K (int): The number of folds

    Returns:
    ----------
    list: A list of K tuples containing the train and test indices
    """
    idxs = np.arange(num_samples)
    np.random.shuffle(idxs)

    fold_size = num_samples // K
    idxs_folds = [idxs[i * fold_size : (i + 1) * fold_size] for i in range(K)]

    return [
        (np.concatenate(idxs_folds[:i] + idxs_folds[i + 1 :]), idxs_folds[i])
        for i in range(K)
    ]


def kfold_calibration(
    scores: np.ndarray, L: np.ndarray, K: int, train_prior: float
) -> np.ndarray:
    """
    Perform k-fold cross-validation for score calibration.

    Parameters:
    ----------
    scores (np.ndarray): The scores of the model
    L (np.ndarray): The labels of the model
    K (int): The number of folds
    train_prior (float): The training prior to use for training the PWL

    Returns:
    ----------
    np.ndarray: The calibrated scores
    """

    all_cal_scores = np.zeros(scores.shape[1])

    for train_idxs, val_idxs in kfold_idxs(scores.shape[1], K):
        scores_TR, LTR = scores[:, train_idxs], L[train_idxs]
        scores_VAL, _ = scores[:, val_idxs], L[val_idxs]

        PWL = PWLogisticRegression(scores_TR, LTR, 0, train_prior)
        PWL.fit()
        cal_scores = PWL.score(scores_VAL)
        cal_scores -= np.log(train_prior / (1 - train_prior))
        all_cal_scores[val_idxs] = cal_scores

    return all_cal_scores
