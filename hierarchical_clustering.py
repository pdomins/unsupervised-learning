import numpy as np
import pandas as pd

from utils.hierarchical_clustering.clusters import show_dendrogram
from utils.hierarchical_clustering.hierarchical_clustering import run_hierarchical_clustering
from utils.initial_data_handle import clean_and_filter_df, handle_non_numerical_data, standardize_dataframe, \
    filter_genres


def main():
    df = pd.read_csv("data/movie_data.csv", delimiter=';', encoding='utf-8')
    df = clean_and_filter_df(df)
    df = filter_genres(df)
    copy_df = df.copy()
    df, genre_mapping = handle_non_numerical_data(df)
    df = standardize_dataframe(df)

    k = 3
    genres = np.unique(df['genres'].values)
    result_clusters = run_hierarchical_clustering(k, df.to_numpy(), genres)

    k = 4
    genres = np.unique(df['genres'].values)
    result_clusters = run_hierarchical_clustering(k, df.to_numpy(), genres)

    # cluster_mapping = show_dendrogram(df, copy_df)
    # # for cluster_label, data_points in cluster_mapping.items():
    # #     print(f'Data points in Cluster {cluster_label}: {data_points}')
    #
    # genre_counts = {}
    # for cluster_label in list(cluster_mapping.keys())[-10:]:
    #     data_points = cluster_mapping[cluster_label]
    #     # print(f'Data points in Cluster {cluster_label}: {data_points}')
    #     cluster_genre_counts = {}
    #
    #     for point in data_points:
    #         genres = point['genres']
    #
    #         # Count genres for the current data point
    #         for genre in genres:
    #             cluster_genre_counts[genre] = cluster_genre_counts.get(genre, 0) + 1
    #
    #     # Store the genre counts for the current cluster
    #     genre_counts[cluster_label] = cluster_genre_counts
    #
    # # Print the genre counts for each cluster
    # for cluster_label, counts in genre_counts.items():
    #     print(f'Cluster {cluster_label} - Genre Counts: {counts}')


if __name__ == '__main__':
    main()
