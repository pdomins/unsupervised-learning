from typing import Any
import math

def radius_prop2iters_neighbours(curr_iter: int, iters: int, k: int, memory: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if 'init_radius' not in memory:
        memory['init_radius'] = math.sqrt(2)*k

    return (((1 - memory['init_radius']) / iters)*curr_iter + memory['init_radius'], memory)

def relu_like_neighbours(curr_iter: int, iters: int, k: int, memory: dict[str, Any], alpha: float, beta: float) -> tuple[float, dict[str, Any]]:
    if 'init_radius' not in memory:
        memory['init_radius'] = math.sqrt(2)*k

    if 'rect_iter_bound' not in memory:
        memory['rect_iter_bound'] = iters - alpha

    if curr_iter <= beta:
        return memory['init_radius']

    if curr_iter >= memory['rect_iter_bound']:
        return 1

    return (((1 - memory['init_radius']) / (iters - alpha - beta))*(curr_iter - beta) + memory['init_radius'], memory)

def exp_neighbours(curr_iter: int, iters: int, k: int, memory: dict[str, Any], gamma: float, delta: float) -> tuple[float, Any]:
    return (delta*math.exp(-gamma*curr_iter) + 1, None)

def exp_neighbour_influence_function(d: float, R: float) -> float:
    return math.e(-2*d/R)