import pandas as pd
from utils.initial_data_handle import clean_and_filter_df, handle_non_numerical_data, standardize_dataframe


def main():
    df = pd.read_csv("data/movie_data.csv", delimiter=';', encoding='utf-8')
    df = clean_and_filter_df(df)

    print(df.iloc[[1339, 2024]])  # 4686
    print("se les suma")
    print(df.iloc[[3182]])  # 7507: array([3182., 4686.]), 8197: array([7507., 7714.])
    print("se une con")  # 7714: array([6849., 7362.])
    print(df.iloc[[1786, 2764]])  # 6849: array([1786., 2764.]), 7714: array([6849., 7362.]),
    print("tambien")  # 7362: array([5122., 5983.])
    print(df.iloc[[3779, 4356]])  # 5122: array([3779., 4356.])
    print(df.iloc[[3303, 3605]])  # 5983: array([3303., 3605.])
    df, genre_mapping = handle_non_numerical_data(df)
    df = standardize_dataframe(df)
    # show_dendrogram(df)


if __name__ == '__main__':
    main()
