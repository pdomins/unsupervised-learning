import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PCAData:
    feature_names: list[str]
    pca_df_scaled: pd.DataFrame
    xs: np.ndarray
    ys: np.ndarray


def pca(df: pd.DataFrame, cols: list[str]) -> PCAData:
    X = df[cols].to_numpy()
    x_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    pca_df_scaled = pca_df.copy()

    scaler_df = pca_df[['PC1', 'PC2']]
    scaler = 1 / (scaler_df.max() - scaler_df.min())

    for index in scaler.index:
        pca_df_scaled[index] *= scaler[index]

    loadings = pca.components_
    
    xs = loadings[0]
    ys = loadings[1]

    return PCAData(cols, pca_df_scaled, xs, ys)

def plot_pca(pca_data: PCAData) -> None:
    colors = sns.color_palette("hls", len(pca_data.feature_names))
    sns.lmplot(
        x='PC1', 
        y='PC2', 
        data=pca_data.pca_df_scaled, 
        fit_reg=False,
        scatter_kws={"color" : "lightblue"}
    )
    for i, varnames in enumerate(pca_data.feature_names):
        plt.scatter(pca_data.xs[i], pca_data.ys[i], s=200, label=varnames, c=colors[i])
        plt.arrow(
            0, 0, # coordinates of arrow base
            pca_data.xs[i], # length of the arrow along x
            pca_data.ys[i], # length of the arrow along y
            color=colors[i], 
            head_width=0.01
            )
        #plt.text(xs[i], ys[i], varnames)

    xticks = np.linspace(-0.8, 0.8, num=5)
    yticks = np.linspace(-0.8, 0.8, num=5)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('2D Loading plot')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")