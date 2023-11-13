import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_and_filter_df(df: pd.DataFrame):
    df = df[df['genres'] != 'TV Movie']
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df[df['release_date'].dt.year != 2017]
    df['year'] = df['release_date'].dt.year
    df = df[df['imdb_id'].isnull() | ~df['imdb_id'].duplicated(keep='first')]  # keep only the non-repeated values
    df = df.dropna()  # drop null values
    return df


def handle_non_numerical_data(df: pd.DataFrame):
    columns_to_remove = ['imdb_id', 'original_title', 'overview', 'release_date']
    df = df.drop(columns=columns_to_remove, errors='ignore')

    unique_genres = df['genres'].unique()
    genre_mapping = {genre: idx + 1 for idx, genre in enumerate(unique_genres)}

    df['genres'] = df['genres'].map(genre_mapping)
    return df, genre_mapping


def reverse_genre_mapping(df: pd.DataFrame, genre_mapping: dict):
    reverse_mapping = {v: k for k, v in genre_mapping.items()}
    df['genres'] = df['genres'].map(reverse_mapping)
    return df


def standardize_dataframe(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df
