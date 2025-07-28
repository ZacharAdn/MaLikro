#!/usr/bin/env python3
"""
Working Item-Based Collaborative Filtering - simplified and robust implementation.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

class WorkingItemBasedCF:
    def __init__(self, min_ratings_per_book=50, min_ratings_per_user=20, top_k_similar=50):
        self.min_ratings_per_book = min_ratings_per_book
        self.min_ratings_per_user = min_ratings_per_user
        self.top_k_similar = top_k_similar
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        self.user_id_to_idx = {}
        self.book_means = {}
        self.all_books = []  # All books for recommendation
        
    def train(self, ratings_df):
        """Train the model."""
        print("Training Working Item-Based CF...")
        start_time = time.time()
        
        # Filter data
        book_counts = ratings_df['book_id'].value_counts()
        user_counts = ratings_df['user_id'].value_counts()
        
        valid_books = book_counts[book_counts >= self.min_ratings_per_book].index
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        
        filtered_df = ratings_df[
            (ratings_df['book_id'].isin(valid_books)) & 
            (ratings_df['user_id'].isin(valid_users))
        ].copy()
        
        print(f"Filtered: {len(filtered_df):,} ratings, {filtered_df['user_id'].nunique():,} users, {filtered_df['book_id'].nunique():,} books")
        
        # Create mappings
        unique_books = sorted(filtered_df['book_id'].unique())
        unique_users = sorted(filtered_df['user_id'].unique())
        
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.all_books = unique_books
        
        # Create user-item matrix
        user_indices = filtered_df['user_id'].map(self.user_id_to_idx)
        book_indices = filtered_df['book_id'].map(self.book_id_to_idx)
        
        self.user_item_matrix = csr_matrix(
            (filtered_df['rating'], (user_indices, book_indices)),
            shape=(len(unique_users), len(unique_books))
        )
        
        # Calculate book means
        for book_id in unique_books:
            book_ratings = filtered_df[filtered_df['book_id'] == book_id]['rating']
            self.book_means[book_id] = book_ratings.mean()
        
        # Compute similarity matrix
        print("Computing item similarity...")
        item_user_matrix = self.user_item_matrix.T
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f}s")
        
        self.filtered_data = filtered_df
        
    def predict_rating(self, user_id, book_id):
        """Predict rating for user-book pair."""
        if user_id not in self.user_id_to_idx or book_id not in self.book_id_to_idx:
            return self.book_means.get(book_id, 3.5)
            
        user_idx = self.user_id_to_idx[user_id]
        book_idx = self.book_id_to_idx[book_id]
        
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_book_indices = np.nonzero(user_ratings)[0]
        
        if len(rated_book_indices) == 0:
            return self.book_means[book_id]
        
        # Get similarities
        similarities = self.item_similarity_matrix[book_idx][rated_book_indices]
        
        # Filter positive similarities
        positive_mask = similarities > 0.1
        if not np.any(positive_mask):
            return self.book_means[book_id]
            
        similarities = similarities[positive_mask]
        rated_indices = rated_book_indices[positive_mask]
        
        # Get top-k most similar
        if len(similarities) > self.top_k_similar:
            top_k_idx = np.argsort(similarities)[-self.top_k_similar:]
            similarities = similarities[top_k_idx]
            rated_indices = rated_indices[top_k_idx]
        
        # Weighted prediction
        user_ratings_for_similar = user_ratings[rated_indices]
        
        if len(similarities) == 0 or np.sum(similarities) == 0:
            return self.book_means[book_id]
            
        predicted_rating = np.sum(similarities * user_ratings_for_similar) / np.sum(similarities)
        return np.clip(predicted_rating, 1, 5)
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations."""
        if user_id not in self.user_id_to_idx:
            # Return most popular books by average rating
            popular_books = sorted(self.book_means.items(), key=lambda x: x[1], reverse=True)
            return [book_id for book_id, _ in popular_books[:n_recommendations]]
        
        # Get user's rated books
        user_ratings = self.filtered_data[self.filtered_data['user_id'] == user_id]
        rated_books = set(user_ratings['book_id'])
        
        # Get candidate books (unrated)
        candidate_books = [book_id for book_id in self.all_books if book_id not in rated_books]
        
        # Predict ratings for all candidates
        predictions = []
        for book_id in candidate_books:
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in predictions[:n_recommendations]]

class WorkingBaselineRecommender:
    """Working popularity-based recommender."""
    
    def __init__(self):
        self.popular_books = []
        
    def train(self, ratings_df):
        """Train on ratings data."""
        # Calculate popularity score for all books
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        })
        book_stats.columns = ['count', 'mean']
        book_stats = book_stats.reset_index()
        
        # Filter books with reasonable number of ratings
        book_stats = book_stats[book_stats['count'] >= 10]
        
        # Popularity score
        book_stats['popularity_score'] = (
            book_stats['mean'] * 0.7 + 
            np.log1p(book_stats['count']) * 0.3
        )
        
        # Sort by popularity
        self.popular_books = book_stats.sort_values(
            'popularity_score', ascending=False
        )['book_id'].tolist()
        
    def recommend(self, user_id, user_rated_books, n_recommendations=10):
        """Generate recommendations."""
        # Get books user hasn't rated
        rated_books = set(user_rated_books)
        
        recommendations = []
        for book_id in self.popular_books:
            if book_id not in rated_books:
                recommendations.append(book_id)
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

def create_proper_split(ratings_df, test_ratio=0.2):
    """Create proper train/test split."""
    print("Creating train/test split...")
    
    train_data = []
    test_data = []
    
    for user_id, user_ratings in ratings_df.groupby('user_id'):
        user_ratings = user_ratings.sort_values('book_id')  # Stable sort
        n_ratings = len(user_ratings)
        
        if n_ratings < 10:  # Skip users with too few ratings
            train_data.append(user_ratings)
            continue
            
        # Split - last 20% for test, rest for train
        n_test = max(2, int(n_ratings * test_ratio))  # At least 2 for test
        n_test = min(n_test, n_ratings - 5)  # At least 5 for train
        
        train_data.append(user_ratings.iloc[:-n_test])
        test_data.append(user_ratings.iloc[-n_test:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"Train: {len(train_df):,} ratings from {train_df['user_id'].nunique():,} users")
    print(f"Test: {len(test_df):,} ratings from {test_df['user_id'].nunique():,} users")
    
    return train_df, test_df

def evaluate_models(item_cf, baseline, train_df, test_df, n_users=500):
    """Evaluate both models."""
    print(f"Evaluating models on {n_users} users...")
    
    # Get users who exist in both train and test, and are in the model
    test_users = test_df['user_id'].unique()
    train_users = train_df['user_id'].unique()
    model_users = set(item_cf.user_id_to_idx.keys())
    
    valid_users = list(set(test_users) & set(train_users) & model_users)
    
    if len(valid_users) > n_users:
        np.random.seed(42)
        valid_users = np.random.choice(valid_users, n_users, replace=False)
    
    print(f"Evaluating {len(valid_users)} valid users...")
    
    baseline_precisions = []
    item_cf_precisions = []
    
    for i, user_id in enumerate(valid_users):
        # Get user's test data
        user_test = test_df[test_df['user_id'] == user_id]
        user_train = train_df[train_df['user_id'] == user_id]
        
        # Define relevant books (rating >= 4)
        relevant_books = set(user_test[user_test['rating'] >= 4]['book_id'])
        
        if len(relevant_books) == 0:
            continue  # Skip if no relevant books
        
        # Get recommendations from both models
        item_cf_recs = item_cf.recommend(user_id, 10)
        baseline_recs = baseline.recommend(user_id, user_train['book_id'].tolist(), 10)
        
        # Calculate precision
        item_cf_hits = len(set(item_cf_recs) & relevant_books)
        baseline_hits = len(set(baseline_recs) & relevant_books)
        
        item_cf_precision = item_cf_hits / 10
        baseline_precision = baseline_hits / 10
        
        item_cf_precisions.append(item_cf_precision)
        baseline_precisions.append(baseline_precision)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(valid_users)}")
    
    baseline_avg = np.mean(baseline_precisions) if baseline_precisions else 0
    item_cf_avg = np.mean(item_cf_precisions) if item_cf_precisions else 0
    
    print(f"✓ Baseline precision@10: {baseline_avg:.4f}")
    print(f"✓ Item-CF precision@10: {item_cf_avg:.4f}")
    print(f"✓ Improvement: {item_cf_avg - baseline_avg:.4f} ({((item_cf_avg/baseline_avg-1)*100) if baseline_avg > 0 else 0:.1f}%)")
    
    return baseline_avg, item_cf_avg

def main():
    """Main execution."""
    print("=== Working Item-Based Collaborative Filtering ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # Create train/test split
    train_df, test_df = create_proper_split(ratings_df)
    
    # Train models
    print("\nTraining baseline...")
    baseline = WorkingBaselineRecommender()
    baseline.train(train_df)
    
    print("\nTraining Item-Based CF...")
    item_cf = WorkingItemBasedCF()
    item_cf.train(train_df)
    
    # Evaluate
    print("\nEvaluating models...")
    baseline_precision, item_cf_precision = evaluate_models(item_cf, baseline, train_df, test_df)
    
    print(f"\n=== Final Results ===")
    print(f"Baseline precision@10: {baseline_precision:.4f}")
    print(f"Item-CF precision@10: {item_cf_precision:.4f}")
    
    if item_cf_precision > baseline_precision:
        improvement = (item_cf_precision / baseline_precision - 1) * 100
        print(f"✅ Item-CF is {improvement:.1f}% better than baseline!")
    else:
        print(f"❌ Item-CF needs improvement")
    
    return item_cf, baseline

if __name__ == "__main__":
    model, baseline = main()
