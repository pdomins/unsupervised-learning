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
    linkage_matrix = linkage(data_array, method='ward', metric='euclidean')

    plt.figure(figsize=(15, 7))
    dendrogram(linkage_matrix, labels=df.index, orientation='top')

    plt.title('Dendrogram')
    plt.ylabel('Distance')
    plt.xticks([])

    dendrogram_info = {}
    for i, d in enumerate(linkage_matrix):
        cluster_id = i + len(data_array) + 1
        dendrogram_info[cluster_id] = d[:2]

    # for cluster_id, (x, y) in dendrogram_info.items():
    #     plt.annotate(f'Cluster {cluster_id}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
    #                  fontsize=8, color='red')
    #
    # print(dendrogram_info)

    plt.show()


def get_initial_matrix(df: pd.DataFrame):
    data_array = df.to_numpy()
    distances = pdist(data_array, metric='euclidean')  # Calculate pairwise distances
    matrix = squareform(distances)  # Convert the distances to a square matrix form
    return matrix


def run(df: pd.DataFrame):
    print("run hierarchical clustering")
