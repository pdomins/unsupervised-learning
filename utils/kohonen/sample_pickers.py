from typing import Any
import numpy as np


def stochastic_picker(X: np.ndarray, memory: dict[str, Any], 
                      random_state: np.random.Generator = None) -> tuple[np.ndarray, dict[str, Any]]:
    if random_state is None:
        random_state = np.random.default_rng()

    p = X.shape[0]
    i = random_state.choice(p)
    return X[i], memory


def random_shuffle_picker(X: np.ndarray, memory: dict[str, Any], 
                          random_state: np.random.Generator = None) -> tuple[np.ndarray, dict[str, Any]]:
    p = X.shape[0]

    if 'I' not in memory or memory['i'] >= p:
        
        if random_state is None:
            random_state = np.random.default_rng()

        memory['I'] = random_state.choice(p, size=p, replace=False)
        memory['i'] = 0

    I = memory['I']
    i = memory['i']

    aux = X[I[i]]

    memory['i'] += 1

    return aux, memory
