# train.py
import pandas as pd
import torch
from data_preprocessing import preprocess_data
from model import MatrixFactorization, train_model, save_model

def run_training():
    # Preprocess the data
    user_movie_ratings_matrix, num_users, num_items, movie_data, ratings, user_id_mapping, movie_id_mapping, user_id_reverse_mapping, movie_id_reverse_mapping = preprocess_data()

    # Initialize the model
    model = MatrixFactorization(num_users, num_items)

    # Train the model
    train_model(model, user_movie_ratings_matrix)

    # Save the model
    save_model(model, 'model.pth')

    return user_id_mapping, movie_id_mapping, user_id_reverse_mapping, movie_id_reverse_mapping

if __name__ == '__main__':
    run_training()
