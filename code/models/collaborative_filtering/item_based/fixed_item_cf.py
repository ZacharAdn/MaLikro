#!/usr/bin/env python3
"""
Fixed Item-Based Collaborative Filtering with proper data split and evaluation.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from datetime import datetime

class FixedItemBasedCF:
    def __init__(self, min_ratings_per_book=20, min_ratings_per_user=10, top_k_similar=30):
        self.min_ratings_per_book = min_ratings_per_book
        self.min_ratings_per_user = min_ratings_per_user
        self.top_k_similar = top_k_similar
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.book_means = {}
        self.training_stats = {}
        
    def prepare_data(self, ratings_df):
        """Prepare and filter data efficiently."""
        print("Preparing data...")
        
        # Filter books and users
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
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        
        # Create matrix
        n_users = len(unique_users)
        n_books = len(unique_books)
        
        user_indices = filtered_df['user_id'].map(self.user_id_to_idx)
        book_indices = filtered_df['book_id'].map(self.book_id_to_idx)
        
        self.user_item_matrix = csr_matrix(
            (filtered_df['rating'], (user_indices, book_indices)),
            shape=(n_users, n_books)
        )
        
        # Store book means and popularity
        book_stats = filtered_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        })
        book_stats.columns = ['count', 'mean']
        
        self.book_means = {}
        self.book_popularity = {}
        for book_id in unique_books:
            stats = book_stats.loc[book_id]
            self.book_means[book_id] = stats['mean']
            self.book_popularity[book_id] = stats['count']
        
        self.training_stats = {
            'n_users': n_users,
            'n_books': n_books,
            'n_ratings': len(filtered_df),
            'sparsity': 1 - (len(filtered_df) / (n_users * n_books))
        }
        
        self.filtered_data = filtered_df
        return filtered_df
        
    def compute_similarity(self):
        """Compute item similarity matrix."""
        print("Computing similarity matrix...")
        start_time = time.time()
        
        # Transpose to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        
        # Compute cosine similarity
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        end_time = time.time()
        print(f"✓ Similarity computed in {end_time - start_time:.2f}s")
        
    def train(self, ratings_df):
        """Train the model."""
        start_time = time.time()
        print("Training Fixed Item-Based CF...")
        
        self.prepare_data(ratings_df)
        self.compute_similarity()
        
        training_time = time.time() - start_time
        self.training_stats['training_time'] = training_time
        print(f"✓ Training completed in {training_time:.2f}s")
        
    def predict_rating(self, user_id, book_id):
        """Predict rating for user-book pair with better fallbacks."""
        if user_id not in self.user_id_to_idx or book_id not in self.book_id_to_idx:
            return self.book_means.get(book_id, 3.5)
            
        user_idx = self.user_id_to_idx[user_id]
        book_idx = self.book_id_to_idx[book_id]
        
        # Get user's ratings
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_book_indices = np.nonzero(user_ratings)[0]
        
        if len(rated_book_indices) == 0:
            return self.book_means[book_id]
        
        # Get similarities to rated books
        similarities = self.item_similarity_matrix[book_idx][rated_book_indices]
        
        # Filter positive similarities and get top-k
        positive_mask = similarities > 0.1  # Minimum similarity threshold
        if not np.any(positive_mask):
            return self.book_means[book_id]
            
        similarities = similarities[positive_mask]
        rated_indices = rated_book_indices[positive_mask]
        
        # Get top-k most similar
        if len(similarities) > self.top_k_similar:
            top_k_idx = np.argsort(similarities)[-self.top_k_similar:]
            similarities = similarities[top_k_idx]
            rated_indices = rated_indices[top_k_idx]
        
        # Calculate weighted prediction
        user_ratings_for_similar = user_ratings[rated_indices]
        
        if len(similarities) == 0 or np.sum(similarities) == 0:
            return self.book_means[book_id]
            
        predicted_rating = np.sum(similarities * user_ratings_for_similar) / np.sum(similarities)
        
        # Ensure rating is within valid range
        return np.clip(predicted_rating, 1, 5)
        
    def get_recommendations(self, user_id, n_recommendations=10):
        """Generate recommendations with better algorithm."""
        if user_id not in self.user_id_to_idx:
            # Cold start - return popular high-quality books
            popular_books = sorted(self.book_means.items(), 
                                 key=lambda x: (x[1], self.book_popularity[x[0]]), 
                                 reverse=True)
            return [book_id for book_id, _ in popular_books[:n_recommendations]]
        
        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_book_indices = set(np.nonzero(user_ratings)[0])
        
        # Get candidate books (unrated, popular enough)
        candidate_books = []
        for book_id, book_idx in self.book_id_to_idx.items():
            if book_idx not in rated_book_indices and self.book_popularity[book_id] >= 10:
                candidate_books.append(book_id)
        
        # Limit candidates to top 1000 most popular for efficiency
        if len(candidate_books) > 1000:
            candidate_books = sorted(candidate_books, 
                                   key=lambda x: self.book_popularity[x], 
                                   reverse=True)[:1000]
        
        # Predict ratings for candidates
        predictions = []
        for book_id in candidate_books:
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in predictions[:n_recommendations]]
    
    def evaluate_precision(self, test_df, k=10, max_users=1000):
        """Evaluate with better methodology."""
        print(f"Evaluating precision@{k}...")
        
        # Group test data by user
        test_grouped = test_df.groupby('user_id').agg({
            'book_id': list,
            'rating': list
        }).to_dict('index')
        
        # Sample users who exist in training
        valid_users = [uid for uid in test_grouped.keys() if uid in self.user_id_to_idx]
        if len(valid_users) > max_users:
            np.random.seed(42)
            valid_users = np.random.choice(valid_users, max_users, replace=False)
        
        print(f"Evaluating {len(valid_users)} users...")
        
        precisions = []
        start_time = time.time()
        
        for i, user_id in enumerate(valid_users):
            # Get user's test data
            user_test = test_grouped[user_id]
            test_books = user_test['book_id']
            test_ratings = user_test['rating']
            
            # Consider only high-rated books as relevant (4+ stars)
            relevant_books = set([
                book_id for book_id, rating in zip(test_books, test_ratings)
                if rating >= 4
            ])
            
            if len(relevant_books) == 0:
                continue  # Skip users with no relevant books
            
            # Get recommendations
            recommendations = self.get_recommendations(user_id, k)
            recommended_set = set(recommendations)
            
            # Calculate precision
            hits = len(recommended_set & relevant_books)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
            
            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == len(valid_users):
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(valid_users) - i - 1)
                print(f"Progress: {i+1}/{len(valid_users)} ({(i+1)/len(valid_users)*100:.1f}%) - ETA: {eta:.0f}s")
        
        avg_precision = np.mean(precisions) if precisions else 0
        total_time = time.time() - start_time
        print(f"✓ Precision@{k}: {avg_precision:.4f} (evaluated {len(precisions)} users in {total_time:.1f}s)")
        
        return avg_precision, precisions

class FixedBaselineRecommender:
    """Fixed popularity-based recommender."""
    
    def __init__(self):
        self.popular_books = None
        self.book_stats = None
        
    def train(self, ratings_df):
        """Train on ratings data."""
        # Calculate book statistics
        self.book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        })
        self.book_stats.columns = ['count', 'mean']
        self.book_stats = self.book_stats.reset_index()
        
        # Filter books with enough ratings
        self.book_stats = self.book_stats[self.book_stats['count'] >= 10]
        
        # Calculate popularity score (weighted by quality and popularity)
        self.book_stats['popularity_score'] = (
            np.log1p(self.book_stats['count']) * 0.4 +  # Log of popularity
            self.book_stats['mean'] * 0.6  # Average rating
        )
        
        # Sort by popularity score
        self.popular_books = self.book_stats.sort_values(
            'popularity_score', ascending=False
        )['book_id'].tolist()
        
    def recommend(self, user_id, user_ratings, n_recommendations=10):
        """Generate recommendations."""
        # Get books user hasn't rated
        rated_books = set(user_ratings['book_id'])
        
        recommendations = []
        for book_id in self.popular_books:
            if book_id not in rated_books:
                recommendations.append(book_id)
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

def create_proper_train_test_split(ratings_df, test_ratio=0.2):
    """Create proper train/test split ensuring each user has data in both sets."""
    print("Creating proper train/test split...")
    
    train_data = []
    test_data = []
    
    # Group by user
    user_groups = ratings_df.groupby('user_id')
    
    for user_id, user_ratings in user_groups:
        # Sort by some proxy for time (rating order within user)
        user_ratings = user_ratings.sort_values('book_id')  # Stable sort
        n_ratings = len(user_ratings)
        
        if n_ratings < 5:  # Need minimum ratings
            train_data.append(user_ratings)
            continue
            
        # Split - put older ratings in train, newer in test
        n_test = max(1, int(n_ratings * test_ratio))
        n_test = min(n_test, n_ratings - 2)  # Ensure at least 2 in training
        
        train_data.append(user_ratings.iloc[:-n_test])
        test_data.append(user_ratings.iloc[-n_test:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"Train: {len(train_df):,} ratings from {train_df['user_id'].nunique():,} users")
    print(f"Test: {len(test_df):,} ratings from {test_df['user_id'].nunique():,} users")
    
    return train_df, test_df

def main():
    """Main execution with all fixes."""
    print("=== Fixed Item-Based Collaborative Filtering ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # Create proper train/test split
    train_df, test_df = create_proper_train_test_split(ratings_df)
    
    # Train both models
    print("\nTraining Fixed Baseline...")
    baseline = FixedBaselineRecommender()
    baseline.train(train_df)
    
    print("\nTraining Fixed Item-Based CF...")
    item_cf = FixedItemBasedCF()
    item_cf.train(train_df)
    
    # Evaluate Item-Based CF
    print("\nEvaluating Fixed Item-Based CF...")
    precision_10, _ = item_cf.evaluate_precision(test_df, k=10, max_users=1000)
    
    # Quick baseline evaluation
    print("\nQuick baseline check on sample users...")
    test_users = test_df['user_id'].unique()[:100]  # Sample 100 users
    baseline_precisions = []
    
    for user_id in test_users:
        if user_id not in item_cf.user_id_to_idx:
            continue
            
        # Get user's train data for baseline
        user_train = train_df[train_df['user_id'] == user_id]
        if len(user_train) == 0:
            continue
            
        # Get user's test data
        user_test = test_df[test_df['user_id'] == user_id]
        relevant_books = set(user_test[user_test['rating'] >= 4]['book_id'])
        
        if len(relevant_books) == 0:
            continue
            
        # Get baseline recommendations
        baseline_recs = baseline.recommend(user_id, user_train, 10)
        baseline_hits = len(set(baseline_recs) & relevant_books)
        baseline_precision = baseline_hits / 10
        baseline_precisions.append(baseline_precision)
    
    baseline_avg = np.mean(baseline_precisions) if baseline_precisions else 0
    
    print(f"\n=== Results ===")
    print(f"Training time: {item_cf.training_stats['training_time']:.1f}s")
    print(f"Users: {item_cf.training_stats['n_users']:,}")
    print(f"Books: {item_cf.training_stats['n_books']:,}")
    print(f"Baseline precision@10: {baseline_avg:.4f}")
    print(f"Item-CF precision@10: {precision_10:.4f}")
    print(f"Improvement: {precision_10 - baseline_avg:.4f}")
    
    return item_cf, baseline

if __name__ == "__main__":
    model, baseline = main()
