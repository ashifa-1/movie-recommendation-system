# Cold-Start Strategy

## Problem
New users or items with no interaction history present a challenge for collaborative filtering models, which rely on past behavior to make predictions. This is known as the **Cold-Start Problem**.

## Proposed Strategy

### 1. New Users
For users with no rating history, we cannot compute similarity or latent factors.
**Strategy**: Recommend **Popular Items**.
-   Calculate the average rating and number of ratings for all movies.
-   Filter for movies with a minimum number of ratings (e.g., > 50) to ensure reliability.
-   Rank by average rating in descending order.
-   Present the top-N movies to the new user.
-   *Extension*: If demographic data were available (age, gender), we could recommend popular items within their demographic group.

### 2. New Items
For items with no ratings, they will never be recommended by standard CF.
**Strategy**: **Content-Based Filtering** (Future Enhancement).
-   Use item metadata (genres, actors, directors) to find similar items to those a user has liked.
-   For this implementation, new items are excluded from recommendations until they receive a minimum number of ratings.

## Implementation
The `get_recommendations` function handles this check:
```python
if user_id not in train_df['userId'].unique():
    return get_cold_start_recommendations(n)
```
The `get_cold_start_recommendations` function aggregates the global ratings to find the highest-rated popular movies.
