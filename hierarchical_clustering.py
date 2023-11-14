import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.hierarchical_clustering.clusters import get_initial_matrix, show_dendrogram
from utils.initial_data_handle import clean_and_filter_df, handle_non_numerical_data, reverse_genre_mapping, \
    standardize_dataframe


def main():
    df = pd.read_csv("data/movie_data.csv", delimiter=';', encoding='utf-8')
    df = clean_and_filter_df(df)
    df, genre_mapping = handle_non_numerical_data(df)
    df = standardize_dataframe(df)
    # show_dendrogram(df)


if __name__ == '__main__':
    main()
