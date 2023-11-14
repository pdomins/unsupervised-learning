import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


# def get_initial_matrix(df: pd.DataFrame):
#     rows = df.shape[0]
#     matrix = np.zeros((rows, rows))
#     for i in range(rows):
#         for j in range(i + 1, rows):
#             array1 = df.iloc[i].to_numpy()
#             array2 = df.iloc[j].to_numpy()
#
#             distance = euclidean_distance(array1, array2)
#             matrix[i, j] = distance
#             matrix[j, i] = distance
#
#     return matrix

def show_dendrogram(df: pd.DataFrame):
    data_array = df.to_numpy()
    distances = pdist(data_array, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')

    plt.figure(figsize=(15, 7))
    dendrogram(linkage_matrix)
    plt.title('Dendrograma')
    plt.ylabel('Distancia')
    plt.xticks([])
    plt.show()


def get_initial_matrix(df: pd.DataFrame):
    data_array = df.to_numpy()
    distances = pdist(data_array, metric='euclidean')  # Calculate pairwise distances
    matrix = squareform(distances)  # Convert the distances to a square matrix form
    return matrix


def run(df: pd.DataFrame):
    print("run hierarchical clustering")
