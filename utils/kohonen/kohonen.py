from utils.kohonen.distance import euclidean_distance
import numpy as np

def build_network_positions(k: int) -> np.ndarray:
    k_vals = np.arange(k)
    
    xx, yy = np.meshgrid(k_vals, k_vals)

    neuron_positions = np.zeros((k, k, 2))
    neuron_positions[:,:,0] = yy
    neuron_positions[:,:,1] = xx
    
    return neuron_positions

def obtain_winning_neuron_idx(X_p: np.ndarray, neuron_weights: np.ndarray, k: int) -> tuple[int, int]:
    k_i = None
    k_j = None
    min_dist = None
    for i in range(k):
        for j in range(k):
            W_j = neuron_weights[i,j]
            dist = euclidean_distance(X_p, W_j)

            if min_dist is None or dist < min_dist:
                min_dist = dist
                k_i = i
                k_j = j

    return (k_i, k_j)