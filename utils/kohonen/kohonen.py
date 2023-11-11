import numpy as np

def build_network_positions(k: int) -> np.ndarray:
    k_vals = np.arange(k)
    
    xx, yy = np.meshgrid(k_vals, k_vals)

    neuron_positions = np.zeros((k, k, 2))
    neuron_positions[:,:,0] = yy
    neuron_positions[:,:,1] = xx
    
    return neuron_positions