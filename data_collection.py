import requests
import pandas as pd
import os

TMDB_API_KEY = '151da84efafb0a56a0d088d9f8c7cc51'
CSV_FILE_PATH = 'data/movie_data.csv'

def fetch_movies_from_tmdb():
    all_movies = []
    page = 1
    while True:
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}"
        response = requests.get(url)
        data = response.json()
        
        if 'results' not in data or not data['results']:
            break
        
        all_movies.extend(data['results'])
        print(f"Fetched {len(data['results'])} movies from page {page}")
        page += 1
    
    print(f"Total movies fetched: {len(all_movies)}")
    return all_movies

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    return response.json()

def fetch_movie_credits(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    return response.json()

def save_movie_data_to_csv(movie_data):
    df = pd.DataFrame([movie_data])
    df.to_csv(CSV_FILE_PATH, mode='a', header=not os.path.exists(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) == 0, index=False)
    print(f"Saved movie data: {movie_data['movie_id']}")

def load_existing_movie_ids():
    if not os.path.exists(CSV_FILE_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        return set(df['movie_id'].values)
    except pd.errors.EmptyDataError:
        return set()
