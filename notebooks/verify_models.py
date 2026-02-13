import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from collections import defaultdict
import os

# Set paths
DATA_DIR = 'data/ml-latest-small'
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')

def load_data():
    print("Loading data...")
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(f"Ratings file not found at {RATINGS_PATH}")
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(f"Movies file not found at {MOVIES_PATH}")
        
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    print(f"Ratings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    return ratings, movies

def temporal_split(df, test_size=0.2):
    print("Performing temporal split...")
    df_sorted = df.sort_values(by=['userId', 'timestamp'])
    train_data = []
    test_data = []
    
    for _, group in df_sorted.groupby('userId'):
        n_test = int(len(group) * test_size)
        if n_test == 0:
            train_data.append(group)
        else:
            train_data.append(group.iloc[:-n_test])
            test_data.append(group.iloc[-n_test:])
        
    return pd.concat(train_data), pd.concat(test_data)

def calculate_metrics(predictions, k=10, threshold=3.5):
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    mean_precision = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    mean_recall = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    
    return rmse, mae, mean_precision, mean_recall

def main():
    ratings, movies = load_data()
    
    # Analyze sparsity
    n_users = ratings['userId'].nunique()
    n_items = ratings['movieId'].nunique()
    n_ratings = ratings.shape[0]
    sparsity = 1 - (n_ratings / (n_users * n_items))
    print(f"Sparsity: {sparsity:.4f}")

    # Split data
    train_df, test_df = temporal_split(ratings)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    full_trainset = train_data.build_full_trainset()
    test_set = list(test_df[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))
    
    results = []
    
    # User-Based CF
    print("\nTraining User-Based CF...")
    sim_options_user = {'name': 'cosine', 'user_based': True}
    user_based = KNNBasic(sim_options=sim_options_user, verbose=False)
    user_based.fit(full_trainset)
    predictions = user_based.test(test_set)
    metrics = calculate_metrics(predictions)
    results.append(('User-Based CF', *metrics))
    print(f"RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, Precision@10: {metrics[2]:.4f}, Recall@10: {metrics[3]:.4f}")
    
    # Item-Based CF
    print("\nTraining Item-Based CF...")
    sim_options_item = {'name': 'cosine', 'user_based': False}
    item_based = KNNBasic(sim_options=sim_options_item, verbose=False)
    item_based.fit(full_trainset)
    predictions = item_based.test(test_set)
    metrics = calculate_metrics(predictions)
    results.append(('Item-Based CF', *metrics))
    print(f"RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, Precision@10: {metrics[2]:.4f}, Recall@10: {metrics[3]:.4f}")
    
    # SVD
    print("\nTraining SVD...")
    svd = SVD(n_factors=100, random_state=42)
    svd.fit(full_trainset)
    predictions = svd.test(test_set)
    metrics = calculate_metrics(predictions)
    results.append(('SVD', *metrics))
    print(f"RMSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, Precision@10: {metrics[2]:.4f}, Recall@10: {metrics[3]:.4f}")
    
    # Summary
    print("\n--- Summary ---")
    results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'MAE', 'Precision@10', 'Recall@10'])
    print(results_df)

    # Recommendation Check
    user_id = 1
    print(f"\nGenerating recommendations for User {user_id} using SVD...")
    all_movie_ids = set(movies['movieId'].unique())
    user_rated_movies = set(train_df[train_df['userId'] == user_id]['movieId'].unique())
    movies_to_predict = list(all_movie_ids - user_rated_movies)
    
    preds = []
    for mid in movies_to_predict:
        preds.append((mid, svd.predict(user_id, mid).est))
    
    preds.sort(key=lambda x: x[1], reverse=True)
    top_10 = preds[:10]
    top_ids = [x[0] for x in top_10]
    
    recs = movies[movies['movieId'].isin(top_ids)][['title', 'genres']]
    print(recs)

if __name__ == "__main__":
    main()
