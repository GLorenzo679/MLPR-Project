import numpy as np


def load_dataset(path: str) -> tuple:
    """
    Load dataset from a given path.

    Parameters:
    ----------
    path (str): The path to the dataset file

    Returns:
    ----------
    tuple: A tuple containing the data matrix (D) and the labels (L)
    """
    data = np.loadtxt(path, delimiter=",")

    # we use the column vector convention
    D = data[:, :-1].T
    L = data[:, -1].T

    return D, L.astype(int)


def vcol(v: np.ndarray) -> np.ndarray:
    """
    Convert a vector into a column vector.

    Parameters:
    ----------
    v (np.ndarray): The input vector

    Returns:
    ----------
    np.ndarray: The column vector
    """
    return v.reshape(v.shape[0], 1)


def vrow(v: np.ndarray) -> np.ndarray:
    """
    Convert a vector into a row vector.

    Parameters:
    ----------
    v (np.ndarray): The input vector

    Returns:
    ----------
    np.ndarray: The row vector
    """
    return v.reshape(1, v.shape[0])


def split_db_2to1(D: np.ndarray, L: np.ndarray, seed: int = 0) -> tuple:
    """
    Split the dataset into two parts with a 2:1 ratio.

    Parameters:
    ----------
    D (np.ndarray): The data matrix
    L (np.ndarray): The labels
    seed (int, optional): The seed for random permutation (default is 0)

    Returns:
    ----------
    tuple: A tuple containing the training and validation datasets
    """
    nTrain = int(D.shape[1] * 2.0 / 3.0)

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def find_best_threshold(
    DTR_LDA: np.ndarray, LTR: np.ndarray, DVAL_LDA: np.ndarray, LVAL: np.ndarray
) -> float:
    """
    Find the best threshold for a classifier based on validation data.

    Parameters:
    ----------
    DTR_LDA (np.ndarray): The LDA transformed training data
    LTR (np.ndarray): The training labels
    DVAL_LDA (np.ndarray): The LDA transformed validation data
    LVAL (np.ndarray): The validation labels

    Returns:
    ----------
    float: The best threshold
    """
    thresholds = np.linspace(
        DTR_LDA[0, LTR == 0].min(), DTR_LDA[0, LTR == 1].max(), 1000
    )

    errors = []
    best_threshold = None

    for threshold in thresholds:
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_LDA[0] >= threshold] = 1
        PVAL[DVAL_LDA[0] < threshold] = 0

        errors.append(np.sum(PVAL != LVAL) / LVAL.shape[0])

    best_threshold = thresholds[np.argmin(errors)]

    return best_threshold


def sort_llrs_with_labels(llrs: np.ndarray, L: np.ndarray) -> tuple:
    """
    Sort log-likelihood ratios (LLRs) with corresponding labels.

    Parameters:
    ----------
    llrs (np.ndarray): The log-likelihood ratios
    L (np.ndarray): The labels

    Returns:
    ----------
    tuple: A tuple containing sorted LLRs and labels
    """
    tmp = np.vstack([llrs, L]).T
    idxs = np.argsort(tmp[:, 0])
    tmp = tmp[idxs]
    llrs = tmp[:, 0].reshape(llrs.shape)
    L = tmp[:, 1].reshape(L.shape).astype(int)

    return llrs, L


def expand_features(D: np.ndarray) -> np.ndarray:
    """
    Expand features by adding second-order polynomial features.

    Parameters:
    ----------
    D (np.ndarray): The input data matrix

    Returns:
    ----------
    np.ndarray: The expanded feature matrix
    """
    phi = []

    for i in range(D.shape[1]):
        phi_i = vcol(np.hstack((np.dot(vcol(D[:, i]), vcol(D[:, i]).T))))
        phi.append(phi_i)

    phi = np.hstack(phi)
    D_expanded = np.concatenate((phi, D), axis=0)

    return D_expanded
