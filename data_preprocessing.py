import pandas as pd

def preprocess_data():
    ratings = pd.read_csv('data/ratings.csv')
    movie_data = pd.read_csv('data/movie_data.csv')
    
    num_users = ratings['userId'].nunique()
    num_items = movie_data['movie_id'].nunique()
    
    user_movie_ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Create mappings
    user_id_mapping = {id: i for i, id in enumerate(ratings['userId'].unique())}
    movie_id_mapping = {id: i for i, id in enumerate(movie_data['movie_id'].unique())}
    
    # Reverse mappings
    user_id_reverse_mapping = {i: id for id, i in user_id_mapping.items()}
    movie_id_reverse_mapping = {i: id for id, i in movie_id_mapping.items()}
    
    return user_movie_ratings_matrix, num_users, num_items, movie_data, ratings, user_id_mapping, movie_id_mapping, user_id_reverse_mapping, movie_id_reverse_mapping
