from typing import Any
import numpy as np

def stochastic_picker(X: np.ndarray, memory: dict[str, Any]) -> tuple[np.ndarray, Any]:
    p = X.shape[0]
    i = np.random.choice(p)
    return X[i]