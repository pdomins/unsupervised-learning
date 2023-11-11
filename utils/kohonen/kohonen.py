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

def obtain_neighbour_neurons_idxs(k_i: int, k_j: int, neuron_positions: np.ndarray, k: int, curr_iter: int, iters: int, direct_scale: float, diagonal_scale: float, neighbour_radius_function: Callable[[int, int, int, dict[str, Any]], tuple[float, dict[str, Any]]], neighbour_memory: dict[str, Any]) -> tuple[list[tuple[int, int]], list[float], float, dict[str, Any]]:
    idxs = []
    dists = []
    winner = neuron_positions[k_i, k_j]

    unscaled_radius, neighbour_memory = neighbour_radius_function(curr_iter, iters, k, neighbour_memory)
    
    direct_radius = direct_scale*unscaled_radius
    diagonal_radius = diagonal_scale*unscaled_radius

    direct_row = k_i % 2

    for i in range(k):
        for j in range(k):
            if not (i == k_i and j == k_j):
                row_type = i % 2
                neighbour = neuron_positions[i, j]
                dist = euclidean_distance(neighbour, winner)

                if row_type == direct_row and dist <= direct_radius:
                    idxs.append((i, j))
                    dists.append(dist)
                elif row_type != direct_row and dist <= diagonal_radius:
                    idxs.append((i, j))
                    dists.append(dist)

    return (idxs, dists, unscaled_radius, neighbour_memory)