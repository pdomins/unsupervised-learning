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

def inside_radius(point: np.ndarray, center: np.ndarray, ry: float, rx: float) -> bool:
    y_0 = center[0]
    x_0 = center[1]

    y = point[0]
    x = point[1]

    return ((x - x_0) / rx) ** 2 + ((y - y_0) / ry) ** 2 <= 1