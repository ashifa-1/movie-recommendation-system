import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
import os

# Set paths
DATA_DIR = 'data/ml-latest-small'
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.csv')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')
OUTPUT_IMAGE = 'svd_latent_factors.png'

def main():
    if not os.path.exists(RATINGS_PATH):
        print(f"File not found: {RATINGS_PATH}")
        return

    # Load data
    print("Loading data for visualization...")
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    
    # Prepare for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    full_trainset = data.build_full_trainset()
    
    # Train SVD
    print("Training SVD for latent factors...")
    svd = SVD(n_factors=100, random_state=42)
    svd.fit(full_trainset)
    
    # Extract item factors (matrix Q)
    item_factors = svd.qi
    print(f"Item factors shape: {item_factors.shape}")
    
    # Visualize first 2 dimensions
    plt.figure(figsize=(10, 8))
    plt.scatter(item_factors[:, 0], item_factors[:, 1], alpha=0.5)
    plt.title('SVD Item Latent Factors (First 2 Dimensions)')
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.grid(True)
    
    print(f"Saving plot to {OUTPUT_IMAGE}...")
    plt.savefig(OUTPUT_IMAGE)
    print("Done.")

if __name__ == "__main__":
    main()
