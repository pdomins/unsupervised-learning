import numpy as np
from scipy.spatial.distance import euclidean
import math


def create_cluster(element, genres):
    genre = element[-1]
    cluster = {
        "elements": [element],
        "genres": {genre: 0 for genre in genres},
        "centroid": np.array(element[:-1], dtype=float),  # only itself
    }
    cluster["genres"][genre] = 1
    return cluster


def merge_genres(x, y):
    return x + y


def add_to_cluster(cluster, other_cluster):
    cluster["elements"].extend(other_cluster["elements"])

    cluster['genres'] = cluster['genres'].apply(lambda x: merge_genres(x, other_cluster['genres']))

    cluster_elements = [subarray[:-1] for subarray in cluster["elements"]]  # el ultimo es genero
    cluster["centroid"] = np.mean(cluster_elements, axis=0)


def calculate_distance(cluster1, cluster2):
    return euclidean(cluster1["centroid"], cluster2["centroid"])


def run_hierarchical_clustering(k, observations, genres):
    clusters = [create_cluster(observation, genres) for observation in observations]
    n_clusters = len(clusters)

    distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if np.array_equal(clusters[i]["centroid"], clusters[j]["centroid"]):
                # Handle overlapping points
                distances[i][j] = distances[j][i] = 0
            else:
                distances[i][j] = distances[j][i] = calculate_distance(clusters[i], clusters[j])

    np.fill_diagonal(distances, 0)

    while n_clusters > k:
        min_distance_idxs = np.where(distances == np.min(distances[distances > 0]))
        first_cluster, second_cluster = min(min_distance_idxs[0]), max(min_distance_idxs[0])

        add_to_cluster(clusters[first_cluster], clusters[second_cluster])

        new_distances = np.array([calculate_distance(cluster, clusters[first_cluster]) for cluster in clusters])

        distances[first_cluster] = new_distances
        distances[:, first_cluster] = new_distances
        distances = np.delete(distances, second_cluster, axis=0)
        distances = np.delete(distances, second_cluster, axis=1)
        n_clusters -= 1

    return clusters
