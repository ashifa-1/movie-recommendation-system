# Movie Recommendation System

A comprehensive movie recommendation system built with Python, using the MovieLens dataset. This project implements and compares three collaborative filtering techniques:
1.  **User-Based Collaborative Filtering**
2.  **Item-Based Collaborative Filtering**
3.  **Matrix Factorization (SVD)**

## Features

-   **Exploratory Data Analysis (EDA)**: Visualizations of rating distributions and data sparsity.
-   **Data Preprocessing**: Temporal train-test split to simulate real-world scenarios.
-   **Model Implementation**: Comparison of memory-based (KNN) and model-based (SVD) approaches.
-   **Evaluation**: Metrics include RMSE, MAE, Precision@k, and Recall@k.
-   **Recommendation Engine**: Generates personalized top-N movie recommendations.
-   **Cold-Start Strategy**: Handles new users by recommending popular, high-rated movies. [Details](docs/cold_start_strategy.md).

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashifa-1/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data:**
    The project uses the MovieLens `ml-latest-small` dataset. The script automatically checks for data in `data/ml-latest-small/`.
    Ensure `ratings.csv` and `movies.csv` are present.

## Running the Project

### Jupyter Notebook
Launch the Jupyter Notebook to explore the code, analysis, and visualizations interactively:
```bash
jupyter notebook notebooks/movie_recommender.ipynb
```

### Verification Script
To verify the model implementations and see the evaluation metrics in the terminal:
```bash
python notebooks/verify_models.py
```

## Methodology & Results

### Data Split
We used a **temporal split**, sorting ratings by timestamp and using the last 20% of each user's ratings for testing. This evaluates the model's ability to predict future preferences.

### Model Comparison
The models were evaluated on RMSE (lower is better) and Ranking Metrics (Precision@10, Recall@10).

| Model | RMSE | MAE | Precision@10 | Recall@10 |
| :--- | :--- | :--- | :--- | :--- |
| **User-Based CF** | 1.0242 | 0.7834 | 0.478 | 0.474 |
| **Item-Based CF** | 1.0504 | 0.8094 | 0.477 | 0.473 |
| **SVD** | **0.8941** | **0.6882** | **0.518** | **0.483** |

**Key Findings:**
-   **SVD** significantly outperforms memory-based methods (User-Based and Item-Based CF) across all metrics.
-   It achieves the lowest Error (RMSE ~0.89) and highest Recommendation Quality (Precision@10 ~0.52).
-   User-Based and Item-Based CF have similar performance, with User-Based being slightly better in this dataset.

### Visualizations
A visualization of the first 2 latent factors from the SVD model can be found in `docs/svd_latent_factors.png`.

## Trade-offs Between Models

Example:

- User-based CF sensitive to sparse data
- Item-based more stable
- SVD handles sparsity via latent features


## Project Structure
-   `data/`: Contains the dataset.
-   `notebooks/`:
    -   `movie_recommender.ipynb`: Main project notebook.
    -   `verify_models.py`: Script to verify and evaluate models.
-   `requirements.txt`: Python dependencies.
-   `README.md`: Project documentation.
