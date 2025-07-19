import pandas as pd

def load_data():
    reviews = pd.read_csv('data/zomato_reviews.csv')
    metadata = pd.read_csv('data/zomato_metadata.csv')
    return reviews, metadata

def clean_data(df):
    # remove nulls, standardize columns
    df = df.dropna()
    return df

def merge_data(reviews, metadata):
    merged = pd.merge(reviews, metadata, on='restaurant_id', how='inner')
    return merged
