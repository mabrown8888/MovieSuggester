from flask import Flask, request, render_template, session, redirect, url_for
import pandas as pd
import torch
import os
from train import run_training
from data_collection import fetch_movies_from_tmdb, fetch_movie_details, fetch_movie_credits, save_movie_data_to_csv, load_existing_movie_ids
from data_preprocessing import preprocess_data
from model import MatrixFactorization, load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

MODEL_FILE = 'model.pth'
CSV_FILE_PATH = 'data/movie_data.csv'
RATINGS_FILE_PATH = 'data/ratings.csv'
USER_MAPPING_FILE = 'data/user_mapping.csv'

# Load the model
user_movie_ratings_matrix, num_users, num_items, movie_data, ratings, user_id_mapping, movie_id_mapping, user_id_reverse_mapping, movie_id_reverse_mapping = preprocess_data()
model = load_model(MODEL_FILE, num_users, num_items)

def get_user_id(user_name):
    if not os.path.exists(USER_MAPPING_FILE):
        df = pd.DataFrame(columns=['userName', 'userId'])
        df.to_csv(USER_MAPPING_FILE, index=False)
    
    user_mapping = pd.read_csv(USER_MAPPING_FILE)
    if user_name in user_mapping['userName'].values:
        return user_mapping[user_mapping['userName'] == user_name]['userId'].values[0]
    else:
        new_user_id = user_mapping['userId'].max() + 1 if not user_mapping.empty else 0
        new_user = pd.DataFrame([[user_name, new_user_id]], columns=['userName', 'userId'])
        user_mapping = pd.concat([user_mapping, new_user], ignore_index=True)
        user_mapping.to_csv(USER_MAPPING_FILE, index=False)
        return new_user_id

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_movie_list', methods=['POST'])
def update_movie_list():
    existing_movie_ids = load_existing_movie_ids()
    movies = fetch_movies_from_tmdb()
    
    for movie in movies:
        movie_id = movie['id']
        if movie_id in existing_movie_ids:
            continue

        movie_details = fetch_movie_details(movie_id)
        movie_credits = fetch_movie_credits(movie_id)
        
        if 'status_code' in movie_details and movie_details['status_code'] == 34:
            continue

        movie_data = {
            'movie_id': movie_details['id'],
            'movie_title': movie_details['title'],
            'release_date': movie_details['release_date'],
            'overview': movie_details['overview'],
            'genres': ", ".join([genre['name'] for genre in movie_details['genres']]),
            'vote_average': movie_details['vote_average'],
            'vote_count': movie_details['vote_count'],
            'runtime': movie_details['runtime'],
            'budget': movie_details['budget'],
            'revenue': movie_details['revenue'],
            'production_companies': ", ".join([company['name'] for company in movie_details['production_companies']]),
            'cast': ", ".join([cast_member['name'] + " as " + cast_member['character'] for cast_member in movie_credits['cast'][:10]]),
            'crew': ", ".join([crew_member['name'] + " (" + crew_member['job'] + ")" for crew_member in movie_credits['crew'][:10]])
        }

        save_movie_data_to_csv(movie_data)
        existing_movie_ids.add(movie_id)

    return render_template('index.html', message="Movies data fetched and saved successfully.")

@app.route('/login', methods=['POST'])
def login():
    user_name = request.form['user_name']
    session['user_name'] = user_name
    return render_template('loggedIn.html', message="Successfully logged in as " + user_name)

@app.route('/add_rating', methods=['POST'])
def add_rating():
    if 'user_name' not in session:
        return redirect(url_for('home'))

    user_name = session['user_name']
    movie_name = request.form['movie_name']
    rating = float(request.form['rating'])

    user_id = get_user_id(user_name)

    movie_data = pd.read_csv(CSV_FILE_PATH)
    movie_row = movie_data[movie_data['movie_title'] == movie_name]
    if movie_row.empty:
        return render_template('index.html', message="Movie not found.")

    movie_id = movie_row['movie_id'].values[0]

    ratings_df = pd.read_csv(RATINGS_FILE_PATH)

    # Check if the user has already rated the movie
    if ((ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)).any():
        ratings_df.loc[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id), 'rating'] = rating
        message = "Movie Already Added."
    else:
        # Add new rating
        new_rating = pd.DataFrame([[user_id, movie_id, rating]], columns=['userId', 'movieId', 'rating'])
        ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
        message = "Rating added successfully."

    ratings_df.to_csv(RATINGS_FILE_PATH, index=False)

    return render_template('loggedIn.html', message=message)

@app.route('/view_ratings')
def view_ratings():
    if 'user_name' not in session:
        return redirect(url_for('home'))

    user_name = session['user_name']
    user_id = get_user_id(user_name)

    ratings_df = pd.read_csv(RATINGS_FILE_PATH)
    user_ratings = ratings_df[ratings_df['userId'] == user_id]

    movie_data = pd.read_csv(CSV_FILE_PATH)
    user_ratings = user_ratings.merge(movie_data, left_on='movieId', right_on='movie_id')

    return render_template('view_ratings.html', ratings=user_ratings.to_dict(orient='records'))

@app.route('/edit_rating', methods=['POST'])
def edit_rating():
    if 'user_name' not in session:
        return redirect(url_for('home'))

    user_name = session['user_name']
    movie_id = int(request.form['movie_id'])
    new_rating = float(request.form['new_rating'])

    user_id = get_user_id(user_name)

    ratings_df = pd.read_csv(RATINGS_FILE_PATH)
    if ((ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)).any():
        ratings_df.loc[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id), 'rating'] = new_rating
        ratings_df.to_csv(RATINGS_FILE_PATH, index=False)
        message = "Rating updated successfully."
    else:
        message = "Rating not found."

    return redirect(url_for('view_ratings', message=message))

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_name' not in session:
        return redirect(url_for('home'))

    user_name = session['user_name']
    user_id = get_user_id(user_name)

    # Run the training process
    user_id_mapping, movie_id_mapping, user_id_reverse_mapping, movie_id_reverse_mapping = run_training()

    # Reload the model
    model = load_model(MODEL_FILE, len(user_id_mapping), len(movie_id_mapping))

    # Debug: print user ID
    print(f"User ID for {user_name}: {user_id}")

    if user_id >= model.user_factors.weight.shape[0]:
        return render_template('index.html', message="User ID exceeds model capacity.")

    user_vector = torch.tensor([user_id], dtype=torch.long)
    num_items = model.item_factors.weight.shape[0]

    if (num_items < 10): 
        message = "Not enough movies inputted."
        return render_template('loggedIn.html', message=message)

    scores = model(user_vector, torch.arange(num_items))
    top_scores, top_items = torch.topk(scores, 10)

    # Debug: print top items
    print(f"Top recommended movie IDs: {top_items.tolist()}")

    # Use the movie_id_mapping to get actual movie IDs
    actual_movie_ids = [movie_id_reverse_mapping[i] for i in top_items.tolist()]

    # Debug: print actual movie IDs
    print(f"Actual movie IDs: {actual_movie_ids}")

    movie_data = pd.read_csv(CSV_FILE_PATH)
    
    recommended_movies = movie_data[movie_data['movie_id'].isin(actual_movie_ids)]

    # Debug: print recommended movies
    print(f"Recommended movies: {recommended_movies}")

    return render_template('results.html', recommendations=recommended_movies.to_dict(orient='records'))

if __name__ == '__main__':
    if not os.path.exists(RATINGS_FILE_PATH):
        pd.DataFrame(columns=['userId', 'movieId', 'rating']).to_csv(RATINGS_FILE_PATH, index=False)
    
    if not os.path.exists(USER_MAPPING_FILE):
        pd.DataFrame(columns=['userName', 'userId']).to_csv(USER_MAPPING_FILE, index=False)
    
    app.run(debug=True)
