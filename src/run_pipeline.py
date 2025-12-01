"""
MovieLens Movie Recommender - Item-KNN Pipeline
================================================
Run this script to execute the full recommendation pipeline:
  1. Load and clean data
  2. Build user-item matrix
  3. Train Item-KNN model
  4. Get recommendations

Usage:
    python run_pipeline.py

Item-KNN was chosen because it achieved the best performance:
  - Hit Rate@10: 54.6%
  - Recall@10: 11.2%
  - Precision@10: 10.3%
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import sqlite3
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_RAW = "../data/raw/"
DATA_PROCESSED = "../data/processed/"
DB_PATH = DATA_PROCESSED + "movielens.db"

os.makedirs(DATA_PROCESSED, exist_ok=True)


# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================

def load_and_clean_data():
    """Load MovieLens data and perform basic cleaning."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Cleaning Data")
    print("=" * 60)
    
    # load ratings
    ratings_df = pd.read_csv(
        DATA_RAW + "u.data",
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        header=None
    )
    ratings_df = ratings_df.drop(columns=['timestamp'])
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # load movies
    movies_df = pd.read_csv(
        DATA_RAW + "u.item",
        sep='|',
        encoding='latin-1',
        header=None,
        names=['movie_id', 'title', 'release_date', 'video_release_date',
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    movies_df = movies_df.drop(columns=['video_release_date'])
    print(f"Loaded {len(movies_df):,} movies")
    
    # handle duplicate movie titles
    movies_df['primary_movie_id'] = (
        movies_df.groupby('title')['movie_id'].transform('min')
    )
    id_map = dict(zip(movies_df['movie_id'], movies_df['primary_movie_id']))
    ratings_df['movie_id'] = ratings_df['movie_id'].map(id_map)
    
    # average duplicate ratings
    ratings_df = (
        ratings_df
        .groupby(['user_id', 'movie_id'], as_index=False)['rating']
        .mean()
    )
    
    # keep one row per movie
    movies_df = (
        movies_df
        .drop(columns='primary_movie_id')
        .drop_duplicates(subset='title', keep='first')
    )
    
    print(f"After cleaning: {len(ratings_df):,} ratings, {len(movies_df):,} movies")
    
    # save to SQLite
    conn = sqlite3.connect(DB_PATH)
    ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)
    movies_df.to_sql('movies', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved to {DB_PATH}")
    
    return ratings_df, movies_df


# ============================================================================
# STEP 2: BUILD USER-ITEM MATRIX
# ============================================================================

def build_user_item_matrix(ratings_df):
    """Create the user-item matrix for collaborative filtering."""
    print("\n" + "=" * 60)
    print("STEP 2: Building User-Item Matrix")
    print("=" * 60)
    
    # create pivot table: rows = movies, columns = users
    user_item_df = ratings_df.pivot_table(
        index='movie_id',
        columns='user_id',
        values='rating',
        fill_value=0
    )
    
    print(f"Matrix shape: {user_item_df.shape} (movies × users)")
    
    # calculate sparsity
    total = user_item_df.shape[0] * user_item_df.shape[1]
    filled = (user_item_df != 0).sum().sum()
    sparsity = 1 - (filled / total)
    print(f"Sparsity: {sparsity:.1%}")
    
    return user_item_df


# ============================================================================
# STEP 3: TRAIN ITEM-KNN MODEL
# ============================================================================

def train_model(user_item_df, movies_df):
    """Train the Item-KNN model."""
    print("\n" + "=" * 60)
    print("STEP 3: Training Item-KNN Model")
    print("=" * 60)
    
    # convert to sparse matrix
    movie_ids = user_item_df.index.tolist()
    item_user_matrix = csr_matrix(user_item_df.values)
    
    # create lookup dictionaries
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}
    movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))
    title_to_movie_id = {title.lower(): mid for mid, title in movie_id_to_title.items()}
    
    # train Item-KNN
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model.fit(item_user_matrix)
    
    print("Item-KNN model trained!")
    print(f"  - {len(movie_ids)} movies in model")
    
    return {
        'model': model,
        'matrix': item_user_matrix,
        'movie_ids': movie_ids,
        'movie_id_to_idx': movie_id_to_idx,
        'idx_to_movie_id': idx_to_movie_id,
        'movie_id_to_title': movie_id_to_title,
        'title_to_movie_id': title_to_movie_id
    }


# ============================================================================
# STEP 4: RECOMMENDATION FUNCTIONS
# ============================================================================

def get_similar_movies(movie_title, data, n=10):
    """Find movies similar to the given movie."""
    title_lower = movie_title.lower()
    
    if title_lower not in data['title_to_movie_id']:
        print(f"Movie '{movie_title}' not found!")
        return []
    
    movie_id = data['title_to_movie_id'][title_lower]
    
    if movie_id not in data['movie_id_to_idx']:
        print(f"Movie not in matrix!")
        return []
    
    movie_idx = data['movie_id_to_idx'][movie_id]
    
    # find neighbors
    distances, indices = data['model'].kneighbors(
        data['matrix'][movie_idx],
        n_neighbors=n + 1
    )
    
    # build results (skip first - it's the movie itself)
    results = []
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        rec_movie_id = data['idx_to_movie_id'][idx]
        rec_title = data['movie_id_to_title'].get(rec_movie_id, "Unknown")
        similarity = round(1 - distances[0][i], 3)
        results.append((rec_title, similarity))
    
    return results


def recommend_for_user(user_id, ratings_df, data, n=10):
    """Recommend movies for a user based on their liked movies."""
    # get user's highly rated movies
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if user_ratings.empty:
        print(f"User {user_id} not found!")
        return []
    
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
    seen_movies = set(user_ratings['movie_id'])
    
    if not liked_movies:
        print(f"User {user_id} has no highly rated movies!")
        return []
    
    # find similar movies to what user liked
    candidates = {}
    for movie_id in liked_movies:
        if movie_id not in data['movie_id_to_idx']:
            continue
        
        movie_idx = data['movie_id_to_idx'][movie_id]
        distances, indices = data['model'].kneighbors(
            data['matrix'][movie_idx],
            n_neighbors=20
        )
        
        for i, idx in enumerate(indices[0]):
            rec_id = data['idx_to_movie_id'][idx]
            if rec_id not in seen_movies:
                sim = 1 - distances[0][i]
                candidates[rec_id] = candidates.get(rec_id, 0) + sim
    
    # sort by score
    sorted_recs = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_recs:
        return []
    
    # normalize scores to 0-1 range (divide by max score)
    max_score = sorted_recs[0][1]
    
    # build results
    results = []
    for movie_id, score in sorted_recs[:n]:
        title = data['movie_id_to_title'].get(movie_id, "Unknown")
        normalized_score = score / max_score if max_score > 0 else 0
        results.append((title, round(normalized_score, 3)))
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full pipeline."""
    print("\n" + "=" * 60)
    print("MovieLens Item-KNN Recommender Pipeline")
    print("=" * 60)
    
    # Step 1: Load and clean data
    ratings_df, movies_df = load_and_clean_data()
    
    # Step 2: Build user-item matrix
    user_item_df = build_user_item_matrix(ratings_df)
    
    # Step 3: Train model
    data = train_model(user_item_df, movies_df)
    
    # Step 4: Interactive mode
    interactive_mode(ratings_df, data)


def interactive_mode(ratings_df, data):
    """Interactive command line interface."""
    print("\n" + "=" * 60)
    print("ITEM-KNN RECOMMENDER")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("  1 - Find similar movies (enter a movie title)")
        print("  2 - Get recommendations for a user (enter user ID)")
        print("  3 - Exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            movie_title = input("Enter movie title (e.g., 'Toy Story (1995)'): ").strip()
            if not movie_title:
                continue
            
            n = input("How many recommendations? (default 10): ").strip()
            n = int(n) if n.isdigit() else 10
            
            recs = get_similar_movies(movie_title, data, n=n)
            
            if recs:
                print(f"\nMovies similar to '{movie_title}':")
                print("-" * 55)
                for i, (title, score) in enumerate(recs, 1):
                    print(f"  {i:2}. {title:<40} (similarity: {score:.3f})")
        
        elif choice == "2":
            user_input = input("Enter user ID (1-943): ").strip()
            if not user_input.isdigit():
                print("Please enter a valid user ID.")
                continue
            
            user_id = int(user_input)
            n = input("How many recommendations? (default 10): ").strip()
            n = int(n) if n.isdigit() else 10
            
            recs = recommend_for_user(user_id, ratings_df, data, n=n)
            
            if recs:
                print(f"\nRecommendations for User {user_id}:")
                print("-" * 55)
                for i, (title, score) in enumerate(recs, 1):
                    print(f"  {i:2}. {title:<40} (score: {score:.3f})")
        
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
