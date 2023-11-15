import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

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
from sklearn.metrics import confusion_matrix


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


def plot_conf_matrix(text_df: pd.DataFrame):
    to_predict = text_df['genres']
    predictions = text_df['predicted_genre']
    class_labels = text_df['genres'].unique()

    cm = confusion_matrix(to_predict, predictions, labels=class_labels, normalize='true')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=sns.cubehelix_palette(as_cmap=True, rot=.2, gamma=.5))
    plt.title(f"Confusion Matrix for k = 3")
    plt.colorbar()

    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = cm.max() / 2.
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, "{:.2f}%".format(cm[i, j] * 100),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/confusion_matrix_k3.png", bbox_inches='tight', dpi=1200)
    plt.close()
