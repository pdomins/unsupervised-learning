import pandas as pd


def clean_and_filter_df(df: pd.DataFrame):
    df = df[df['genres'] != 'TV Movie']
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df[df['release_date'].dt.year != 2017]
    df = df[df['imdb_id'].isnull() | ~df['imdb_id'].duplicated(keep='first')]  # keep only the non-repeated values
    df = df.dropna()  # drop null values
    return df


def handle_non_numerical_data(df: pd.DataFrame):
    columns_to_remove = ['imdb_id', 'original_title', 'overview']
    df = df.drop(columns=columns_to_remove, errors='ignore')

    unique_genres = df['genres'].unique()
    genre_mapping = {genre: idx + 1 for idx, genre in enumerate(unique_genres)}

    df['genres'] = df['genres'].map(genre_mapping)
    return df, genre_mapping


def reverse_genre_mapping(df: pd.DataFrame, genre_mapping: dict):
    reverse_mapping = {v: k for k, v in genre_mapping.items()}
    df['genres'] = df['genres'].map(reverse_mapping)
    return df
