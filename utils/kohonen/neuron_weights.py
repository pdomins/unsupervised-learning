import numpy as np

def random_weight_init(X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[1]

    random_min = -1
    random_max = 1
    random_range = random_max - random_min

    neuron_weights = np.random.random((k, k, n))*random_range + random_min
    return neuron_weights