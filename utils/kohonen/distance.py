import numpy as np
import math

def euclidean_distance(X_1: np.ndarray, X_2: np.ndarray) -> float:
    X_diff = X_1 - X_2
    X_diff = X_diff[:, None]
    if X_diff.shape[0] > X_diff.shape[1]:
        X_sqr_diff = X_diff.T.dot(X_diff)
    else:
        X_sqr_diff = X_diff.dot(X_diff.T)

    sqr_norm = X_sqr_diff.sum().item()

    return math.sqrt(sqr_norm)