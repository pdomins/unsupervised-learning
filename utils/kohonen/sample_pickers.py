from typing import Any
import numpy as np


def stochastic_picker(X: np.ndarray, memory: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    p = X.shape[0]
    i = np.random.choice(p)
    return X[i], memory


def random_shuffle_picker(X: np.ndarray, memory: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    p = X.shape[0]

    if 'I' not in memory or memory['i'] >= p:
        memory['I'] = np.random.choice(p, size=p, replace=False)
        memory['i'] = 0

    I = memory['I']
    i = memory['i']

    aux = X[I[i]]

    memory['i'] += 1

    return aux, memory
