import numpy as np
import math


def random_weight_init(X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[1]

    random_min = -1
    random_max = 1
    random_range = random_max - random_min

    neuron_weights = np.random.random((k, k, n)) * random_range + random_min
    return neuron_weights


def random_sample_weight_init(X: np.ndarray, k: int, replace: bool) -> np.ndarray:
    p = X.shape[0]
    n = X.shape[1]

    grid_size = k * k

    neuron_weights = np.zeros((k, k, n))

    neuron_choice = np.random.choice(p, grid_size, replace)
    neuron_choice = neuron_choice.reshape((k, k))
    for i in range(k):
        for j in range(k):
            neuron_weights[i, j] = X[neuron_choice[i, j]]

    return neuron_weights


def random_sample_weight_init_with_repos(X: np.ndarray, k: int) -> np.ndarray:
    return random_sample_weight_init(X, k, True)


def random_sample_weight_init_no_repos(X: np.ndarray, k: int) -> np.ndarray:
    return random_sample_weight_init(X, k, False)


def simple_weight_delta(X_p: np.ndarray, W_j: np.ndarray,
                        lr: float, d: float, R: float) -> np.ndarray:
    return lr * (X_p - W_j)


def exp_weight_delta(X_p: np.ndarray, W_j: np.ndarray,
                     lr: float, d: float, R: float) -> np.ndarray:
    V = math.exp(-2 * d / R)
    return V * lr * (X_p - W_j)
