from utils.distance import euclidean_distance
from utils.kohonen.neuron_weights import simple_weight_delta, exp_weight_delta
from typing import Any, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
import math


@dataclass
class KohonenNet:
    cols: list[str]
    k: int
    neuron_positions: np.ndarray
    neuron_weights: np.ndarray
    grid_type: str
    sum_dists: list[float]
    act_mat_sigmas: list[float] 
    init_neuron_weights: np.ndarray

    def predict(self, sample: pd.Series) -> tuple[int, int]:
        sample = sample[self.cols]
        X_p = sample.to_numpy()
        return self._predict(X_p)
    
    def _predict(self, X_p: np.ndarray) -> tuple[int, int]:
        i, j, _ = obtain_winning_neuron_idx(X_p, self.neuron_weights, self.k)
        return (i, j)
    
    def activations_mat(self, df: pd.DataFrame) -> np.ndarray:
        df = df[self.cols]
        predictions = df.apply(self.predict, axis=1)
        
        act_mat = np.zeros((self.k, self.k), dtype=int)
        for prediction in predictions:
            i, j = prediction
            act_mat[i, j] += 1

        return act_mat
    
    def _activations_mat(self, X: np.ndarray) -> np.ndarray:
        predictions = np.apply_along_axis(lambda X_p : self._predict(X_p), axis=1, arr=X)

        act_mat = np.zeros((self.k, self.k), dtype=int)
        for prediction in predictions:
            i, j = prediction
            act_mat[i, j] += 1

        return act_mat
    
    def activations_map(self, df: pd.DataFrame) -> np.ndarray:
        df = df[self.cols]
        predictions = df.apply(self.predict, axis=1)
        
        act_map = dict()
        for key in predictions.keys():
            if predictions[key] not in act_map:
                act_map[predictions[key]] = []
            act_map[predictions[key]].append(key)

        return act_map
    
    def u_mat(self) -> np.ndarray:
        GRID_TYPES = obtain_grid_types_data()
        scales = GRID_TYPES[self.grid_type]["scales"]
        direct_radius = scales["direct"]
        diagonal_radius = scales["diagonal"]

        neighbour_count = np.zeros((self.k, self.k))
        neighbour_sum = np.zeros((self.k, self.k))

        for k_i in range(self.k):
            direct_row = k_i % 2
            for k_j in range(self.k):
                winner = self.neuron_weights[k_i, k_j]

                for i in range(self.k):
                    for j in range(self.k):
                        if not (i == k_i and j == k_j):
                            neighbour = self.neuron_weights[i, j]

                            row_type = i % 2
                            weight_dist = euclidean_distance(neighbour, winner)
                            pos_dist = euclidean_distance(self.neuron_positions[k_i, k_j], self.neuron_positions[i, j])

                            if weight_dist <= np.finfo(float).eps:
                                weight_dist = 0

                            if (row_type == direct_row and pos_dist <= direct_radius) or \
                                    (row_type != direct_row and pos_dist <= diagonal_radius):
                                neighbour_sum[k_i, k_j] += weight_dist
                                neighbour_count[k_i, k_j] += 1

        neighbour_sum[neighbour_sum < np.finfo(float).eps] = 0
        neighbour_sum = np.divide(neighbour_sum, neighbour_count, where=(neighbour_sum > 0))
        neighbour_sum[neighbour_sum < np.finfo(float).eps] = 0
        return neighbour_sum


def build_network_positions(k: int, even_displacements: float = 0, odd_displacements: float = 0) -> np.ndarray:
    k_vals = np.arange(k)

    xx, yy = np.meshgrid(k_vals, k_vals)

    xx = xx.astype(float)
    yy = yy.astype(float)

    xx[::2] += even_displacements
    xx[::-2] += odd_displacements

    neuron_positions = np.zeros((k, k, 2))
    neuron_positions[:, :, 0] = yy
    neuron_positions[:, :, 1] = xx

    return neuron_positions


def obtain_winning_neuron_idx(X_p: np.ndarray, neuron_weights: np.ndarray, k: int) -> tuple[int, int, float]:
    k_i = None
    k_j = None
    min_dist = None
    for i in range(k):
        for j in range(k):
            W_j = neuron_weights[i, j]
            dist = euclidean_distance(X_p, W_j)

            if dist <= np.finfo(float).eps:
                dist = 0

            if min_dist is None or dist < min_dist:
                min_dist = dist
                k_i = i
                k_j = j

    return (k_i, k_j, min_dist)


def obtain_neighbour_neurons_idxs(k_i: int, k_j: int, neuron_positions: np.ndarray, k: int, curr_iter: int, iters: int,
                                  direct_scale: float, diagonal_scale: float, neighbour_radius_function: Callable[
            [int, int, int, dict[str, Any]], tuple[float, dict[str, Any]]], neighbour_memory: dict[str, Any]) -> tuple[
    list[tuple[int, int]], list[float], float, dict[str, Any]]:
    idxs = []
    dists = []
    winner = neuron_positions[k_i, k_j]

    unscaled_radius, neighbour_memory = neighbour_radius_function(curr_iter, iters, k, neighbour_memory)

    direct_radius = direct_scale * unscaled_radius
    diagonal_radius = diagonal_scale * unscaled_radius

    direct_row = k_i % 2

    for i in range(k):
        for j in range(k):
            if not (i == k_i and j == k_j):
                row_type = i % 2
                neighbour = neuron_positions[i, j]
                dist = euclidean_distance(neighbour, winner)

                if dist <= np.finfo(float).eps:
                    dist = 0

                if row_type == direct_row and dist <= direct_radius:
                    idxs.append((i, j))
                    dists.append(dist)
                elif row_type != direct_row and dist <= diagonal_radius:
                    idxs.append((i, j))
                    dists.append(dist)

    return (idxs, dists, unscaled_radius, neighbour_memory)


def update_neuron(X_p: np.ndarray, neuron_positions: np.ndarray, i: int, j: int,
                  lr: float, d: float, R: float,
                  weight_delta_function: Callable[[np.ndarray, np.ndarray, float, float, float], np.ndarray]) -> None:
    neuron_positions[i, j] = neuron_positions[i, j] + weight_delta_function(X_p, neuron_positions[i, j], lr, d, R)


def update_winner(X_p: np.ndarray, neuron_positions: np.ndarray, i: int, j: int,
                  lr: float) -> None:
    update_neuron(X_p, neuron_positions, i, j, lr, None, None, simple_weight_delta)


def update_neighbour(X_p: np.ndarray, neuron_positions: np.ndarray, i: int, j: int,
                     lr: float, d: float, R: float) -> None:
    update_neuron(X_p, neuron_positions, i, j, lr, d, R, exp_weight_delta)


def update_neighbours(X_p: np.ndarray, neuron_positions: np.ndarray, neighbour_idxs: list[tuple[int, int]],
                      lr: float, D: list[float], R: float, k_i: int, direct_scale: float,
                      diagonal_scale: float) -> None:
    direct_row = k_i % 2

    for neighbour_number in range(len(neighbour_idxs)):
        i, j = neighbour_idxs[neighbour_number]
        d = D[neighbour_number]

        if i % 2 == direct_row:
            scaled_R = R * direct_scale
        else:
            scaled_R = R * diagonal_scale

        update_neighbour(X_p, neuron_positions, i, j, lr, d, scaled_R)


def obtain_grid_types_data() -> dict[str, Any]:
    GRID_TYPES = {
        "rectangular": {
            "scales": {
                "direct": 1,
                "diagonal": 1
            },
            "displacements": {
                "even": 0,
                "odd": 0
            }
        },
        "hexagonal": {
            "scales": {
                "direct": 1,
                "diagonal": math.sqrt(4 / 3)
            },
            "displacements": {
                "even": 0,
                "odd": 0.5
            }
        }
    }

    return GRID_TYPES

from utils.kohonen.neuron_weights import random_weight_init, random_sample_weight_init_with_repos, random_sample_weight_init_no_repos
from utils.kohonen.sample_pickers import stochastic_picker, random_shuffle_picker

def build_kohonen_net(df: pd.DataFrame, cols: list[str], k: int, iters: int,
                      weight_init_f: str, sample_picker_f: str,
                      neighbour_radius_function: Callable[
                          [int, int, int, dict[str, Any]], tuple[float, dict[str, Any]]],
                      learning_rate_function: Callable[[int], float],
                      grid_type: str,
                      random_state: np.random.Generator = None, 
                      save_dist_sum: bool = False,
                      save_dist_offset: int = 1) -> KohonenNet:
    
    if random_state is None:
        random_state = np.random.default_rng()

    KOHONEN_PARAMS = {
        "weight_init": {
            "random": lambda X, k : random_weight_init(X, k, random_state),
            "sample with repos": lambda X, k : random_sample_weight_init_with_repos(X, k, random_state),
            "sample no repos": lambda X, k : random_sample_weight_init_no_repos(X, k, random_state)
        },
        "sample_picker": {
            "stochastic": lambda X, memory : stochastic_picker(X, memory, random_state),
            "random shuffle": lambda X, memory : random_shuffle_picker(X, memory, random_state)
        }
    }
    df = df[cols]
    X = df.to_numpy()
    return _build_kohonen_net(X, cols, k, iters, KOHONEN_PARAMS["weight_init"][weight_init_f], 
                              KOHONEN_PARAMS["sample_picker"][sample_picker_f],
                              neighbour_radius_function, learning_rate_function, grid_type, save_dist_sum,
                              save_dist_offset)

def sumdist2winning(X: np.ndarray, neuron_weights: np.ndarray, k: int) -> float:
    dist_sum = 0
    for p in range(X.shape[0]):
        X_p = X[p]
        _, _, dist = obtain_winning_neuron_idx(X_p, neuron_weights, k)
        dist_sum += dist
    return dist_sum

def _build_kohonen_net(X: np.ndarray, cols: list[str], k: int, iters: int,
                      weight_init_function: Callable[[np.ndarray, int], np.ndarray],
                      sample_picker_function: Callable[[np.ndarray, dict[str, Any]], tuple[np.ndarray, dict[str, Any]]],
                      neighbour_radius_function: Callable[
                          [int, int, int, dict[str, Any]], tuple[float, dict[str, Any]]],
                      learning_rate_function: Callable[[int], float],
                      grid_type: str, save_dist_sum: bool = False, save_dist_offset: int = 1) -> KohonenNet:
    GRID_TYPES = obtain_grid_types_data()

    if grid_type not in GRID_TYPES:
        raise ValueError("invalid value for grid_type '{}'".format(grid_type))

    direct_scale = GRID_TYPES[grid_type]["scales"]["direct"]
    diagonal_scale = GRID_TYPES[grid_type]["scales"]["diagonal"]

    even_displacements = GRID_TYPES[grid_type]["displacements"]["even"]
    odd_displacements = GRID_TYPES[grid_type]["displacements"]["odd"]

    neuron_positions = build_network_positions(k, even_displacements, odd_displacements)
    neuron_weights = weight_init_function(X, k)
    init_neuron_weights = neuron_weights.copy()

    picker_mem = dict()
    neighbour_memory = dict()

    sum_dists = None
    act_mat_sigmas = None
    if save_dist_sum:
        sum_dists = []
        act_mat_sigmas = []
        kohonen_net = KohonenNet(cols, k, neuron_positions, neuron_weights, grid_type, sum_dists, act_mat_sigmas, init_neuron_weights) 

    curr_iter = 0 

    for curr_iter in range(iters):
        X_p, picker_mem = sample_picker_function(X, picker_mem)

        k_i, k_j, _ = obtain_winning_neuron_idx(X_p, neuron_weights, k)

        neighbour_idxs, neighbour_dists, radius, neighbour_memory = obtain_neighbour_neurons_idxs(k_i, k_j,
                                                                                                  neuron_positions, k,
                                                                                                  curr_iter, iters,
                                                                                                  direct_scale,
                                                                                                  diagonal_scale,
                                                                                                  neighbour_radius_function,
                                                                                                  neighbour_memory)

        lr = learning_rate_function(curr_iter)

        update_winner(X_p, neuron_weights, k_i, k_j, lr)

        update_neighbours(X_p, neuron_weights, neighbour_idxs, lr, neighbour_dists, radius, k_i, direct_scale,
                          diagonal_scale)
        
        if save_dist_sum and curr_iter % save_dist_offset == 0:
            sum_dist = sumdist2winning(X, neuron_weights, k)
            sum_dists.append((curr_iter, sum_dist))

            act_mat = kohonen_net._activations_mat(X)
            act_mat_sigmas.append((curr_iter, act_mat.std()))

    if save_dist_sum:
        sum_dist = sumdist2winning(X, neuron_weights, k)
        sum_dists.append((curr_iter, sum_dist))

        act_mat = kohonen_net._activations_mat(X)
        act_mat_sigmas.append((curr_iter, act_mat.std()))

    return KohonenNet(cols, k, neuron_positions, neuron_weights, grid_type, sum_dists, act_mat_sigmas, init_neuron_weights)
