from utils.kohonen.kohonen import KohonenNet
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

def plot_u_mat(kohonen_net: KohonenNet, u_mat: np.ndarray, title: str = "Matriz U", format: str = "{:.2g}") -> None:
    plot_mat(kohonen_net, u_mat, title, format)

def plot_mat(kohonen_net: KohonenNet, mat: np.ndarray, title: str, format: str = "{}") -> None:
    x = []
    y = []
    w = []

    for i in range(kohonen_net.k):
        for j in range(kohonen_net.k):
            y.append(kohonen_net.neuron_positions[i, j, 0])
            x.append(kohonen_net.neuron_positions[i, j, 1])
            w.append(mat[i, j])
    
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    plot_neurons(kohonen_net, x, y, w, title, format)

def plot_neurons(kohonen_net: KohonenNet, x: np.ndarray, y: np.ndarray, 
                 w: np.ndarray, title: str, format: str = "{}") -> None:

    min_w = w.min()
    max_w = w.max()
    range_w = max_w - min_w
    if range_w == 0:
        range_w = 1
    c = ((w - min_w) / range_w)
    
    ax = plt.gca()
    for i in range(kohonen_net.k*kohonen_net.k):
        circle = plt.Circle((x[i], y[i]), 0.45, edgecolor="k", 
                            linewidth=1, facecolor=sns.cubehelix_palette(as_cmap=True, rot=.2, gamma=.5)(c[i]))
        ax.add_patch(circle)
        if c[i] >= (2/3):
            ax.annotate(format.format(w[i]), xy=(x[i], y[i]), 
                        textcoords="data", ha="center", va="center", color="white")
        else:
            ax.annotate(format.format(w[i]), xy=(x[i], y[i]), 
                        textcoords="data", ha="center", va="center")

    ax.set_aspect('equal')
    ax.autoscale_view()

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    sm = plt.cm.ScalarMappable(cmap=sns.cubehelix_palette(as_cmap=True, rot=.2, gamma=.5), 
                               norm=plt.Normalize(vmin=min_w, vmax=max_w))
    
    cb = plt.colorbar(mappable=sm, ax=ax)
    cb.outline.set_color('black')
    cb.outline.set_linewidth(1)

    plt.title(title)
    plt.axis('off')