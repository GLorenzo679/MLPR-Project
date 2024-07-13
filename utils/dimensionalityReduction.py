import numpy as np
from utils.utils import vcol, vrow


def PCA(D: np.ndarray, m: int) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) on the dataset.

    Parameters:
    ----------
    D (np.ndarray): The input data matrix with shape (n_features, n_samples)
    m (int): The number of principal components to retain

    Returns:
    ----------
    np.ndarray: The matrix of the top 'm' principal components
    """
    # remove the mean from all points
    Dc = D - D.mean(1).reshape(D.shape[0], 1)

    # calculate covariance matrix
    C = np.dot(Dc, Dc.T) / (D.shape[0])

    # compute eigenvalues and eigenvectors sorted
    s, U = np.linalg.eigh(C)

    # alternative method to calculate eigenvalues and eigenvectors
    # only possible because covariance matrix is semi-definite positive
    # U, s, Vh = np.linalg.svd(C)

    P = U[:, ::-1][:, 0:m]

    return P


def SbSw(D: np.ndarray, L: np.ndarray) -> tuple:
    """
    Compute the between-class (SB) and within-class (SW) scatter matrices.

    Parameters:
    ----------
    D (np.ndarray): The input data matrix with shape (n_features, n_samples)
    L (np.ndarray): The class labels

    Returns:
    ----------
    tuple: A tuple containing the between-class scatter matrix (SB) and the within-class scatter matrix (SW)
    """
    SW = 0
    SB = 0

    # calculate mean of all dataset
    mean = vcol(D.mean(1))

    for i in range(np.unique(L).shape[0]):
        # select data of each class
        D_class = D[:, L == i]
        # calculate mean of each class
        mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
        # calculate between class covariance matrix
        SB += D_class.shape[1] * np.dot((mean_class - mean), (mean_class - mean).T)
        # calculate the within class covariance matrix
        SW += np.dot((D_class - mean_class), (D_class - mean_class).T)

    SB /= D.shape[1]
    SW /= D.shape[1]

    return SB, SW


def LDA(D: np.ndarray, L: np.ndarray, m: int) -> np.ndarray:
    """
    Perform Linear Discriminant Analysis (LDA) on the dataset.

    Parameters:
    ----------
    D (np.ndarray): The input data matrix with shape (n_features, n_samples)
    L (np.ndarray): The class labels
    m (int): The number of linear discriminants to retain

    Returns:
    ----------
    np.ndarray: The matrix of the top 'm' linear discriminants
    """
    # compute convolution matrices
    SB, SW = SbSw(D, L)

    # Solving the eigenvalue problem by joint diagonalization
    U, s, _ = np.linalg.svd(SW)

    P1 = np.dot(U * vrow(1.0 / (s**0.5)), U.T)
    SBTilde = np.dot(P1, np.dot(SB, P1.T))
    U, _, _ = np.linalg.svd(SBTilde)

    P2 = U[:, 0:m]
    return -np.dot(P1.T, P2)
