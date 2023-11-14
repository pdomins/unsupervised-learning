from utils.kohonen.kohonen import KohonenNet
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def plot_neurons(kohonen_net: KohonenNet, x: np.ndarray, y: np.ndarray, 
                 w: np.ndarray, format: str) -> None:
    cdict = {
        'red':   ((0.0, 0.5, 0.5),
                  (1.0, 1.0, 1.0)),

        'green': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 1.0)),
            
        'blue':  ((0.0, 0.5, 0.5),
                  (1.0, 1.0, 1.0))
    }

    LavanderHaze = colors.LinearSegmentedColormap('LavanderHaze', cdict).reversed()

    min_w = w.min()
    max_w = w.max()
    range_w = max_w - min_w
    c = ((w - min_w) / range_w) * 255
    c = c.astype(int)
    
    ax = plt.gca()
    for i in range(kohonen_net.k*kohonen_net.k):
        circle = plt.Circle((x[i], y[i]), 0.45, edgecolor="k", 
                            linewidth=1, facecolor=LavanderHaze(c[i]))
        ax.add_patch(circle)
        ax.annotate(format.format(w[i]), xy=(x[i], y[i]), 
                    textcoords="data", ha="center", va="center")

    ax.set_aspect('equal')
    ax.autoscale_view()

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    sm = plt.cm.ScalarMappable(cmap=LavanderHaze, 
                               norm=plt.Normalize(vmin=min_w, vmax=max_w))
    
    cb = plt.colorbar(mappable=sm, ax=ax)
    cb.outline.set_color('black')
    cb.outline.set_linewidth(1)

    plt.title("Matriz U")
    plt.axis('off')