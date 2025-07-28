#!/usr/bin/env python3
"""
Optimized Item-Based Collaborative Filtering with working evaluation.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from datetime import datetime

class OptimizedItemBasedCF:
    def __init__(self, min_ratings_per_book=30, min_ratings_per_user=15, top_k_similar=50):
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
        
        # Store book means
        for book_id in unique_books:
            book_ratings = filtered_df[filtered_df['book_id'] == book_id]['rating']
            self.book_means[book_id] = book_ratings.mean()
        
        self.training_stats = {
            'n_users': n_users,
            'n_books': n_books,
            'n_ratings': len(filtered_df),
            'sparsity': 1 - (len(filtered_df) / (n_users * n_books))
        }
        
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
        print("Training Item-Based CF...")
        
        self.filtered_data = self.prepare_data(ratings_df)
        self.compute_similarity()
        
        training_time = time.time() - start_time
        self.training_stats['training_time'] = training_time
        print(f"✓ Training completed in {training_time:.2f}s")
        
    def get_recommendations_fast(self, user_id, n_recommendations=10):
        """Fast recommendation generation using vectorized operations."""
        if user_id not in self.user_id_to_idx:
            # Cold start - return popular books
            popular_books = self.filtered_data.groupby('book_id')['rating'].agg(['count', 'mean'])
            popular_books = popular_books[popular_books['count'] >= 20]
            popular_books = popular_books.sort_values(['mean', 'count'], ascending=[False, False])
            return list(popular_books.head(n_recommendations).index)
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get user's ratings as dense array
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_book_indices = np.nonzero(user_ratings)[0]
        
        if len(rated_book_indices) == 0:
            # No ratings - return popular books
            popular_books = self.filtered_data.groupby('book_id')['rating'].agg(['count', 'mean'])
            popular_books = popular_books.sort_values(['mean', 'count'], ascending=[False, False])
            return list(popular_books.head(n_recommendations).index)
        
        # Get similarities for all books vs user's rated books
        similarities = self.item_similarity_matrix[:, rated_book_indices]
        user_ratings_for_rated = user_ratings[rated_book_indices]
        
        # Calculate predictions using vectorized operations
        # For each book, get weighted average of similar books the user rated
        predictions = np.zeros(len(self.book_id_to_idx))
        
        for book_idx in range(len(self.book_id_to_idx)):
            if book_idx in rated_book_indices:
                continue  # Skip already rated books
                
            # Get similarities between this book and user's rated books
            book_similarities = similarities[book_idx]
            
            # Only use positive similarities
            positive_mask = book_similarities > 0
            if not np.any(positive_mask):
                predictions[book_idx] = self.book_means.get(self.idx_to_book_id[book_idx], 3.5)
                continue
            
            # Get top-k most similar
            if np.sum(positive_mask) > self.top_k_similar:
                top_k_indices = np.argsort(book_similarities)[-self.top_k_similar:]
                book_similarities = book_similarities[top_k_indices]
                user_ratings_subset = user_ratings_for_rated[top_k_indices]
            else:
                book_similarities = book_similarities[positive_mask]
                user_ratings_subset = user_ratings_for_rated[positive_mask]
            
            # Weighted prediction
            if len(book_similarities) > 0 and np.sum(book_similarities) > 0:
                predictions[book_idx] = np.sum(book_similarities * user_ratings_subset) / np.sum(book_similarities)
            else:
                predictions[book_idx] = self.book_means.get(self.idx_to_book_id[book_idx], 3.5)
        
        # Get top recommendations
        top_indices = np.argsort(predictions)[-n_recommendations:][::-1]
        return [self.idx_to_book_id[idx] for idx in top_indices]
    
    def evaluate_precision_fast(self, test_df, k=10, max_users=1000):
        """Fast evaluation using optimized recommendation generation."""
        print(f"Evaluating precision@{k} on {max_users} users...")
        
        # Group test data
        test_grouped = test_df.groupby('user_id')['book_id'].apply(list).to_dict()
        
        # Sample users
        valid_users = [uid for uid in test_grouped.keys() if uid in self.user_id_to_idx]
        if len(valid_users) > max_users:
            np.random.seed(42)
            valid_users = np.random.choice(valid_users, max_users, replace=False)
        
        print(f"Evaluating {len(valid_users)} users...")
        
        precisions = []
        start_time = time.time()
        
        for i, user_id in enumerate(valid_users):
            # Get recommendations
            recommended_books = self.get_recommendations_fast(user_id, k)
            
            # Calculate precision
            actual_books = set(test_grouped[user_id])
            recommended_set = set(recommended_books)
            hits = len(actual_books & recommended_set)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
            
            # Progress update
            if (i + 1) % 100 == 0 or (i + 1) == len(valid_users):
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(valid_users) - i - 1)
                print(f"Progress: {i+1}/{len(valid_users)} ({(i+1)/len(valid_users)*100:.1f}%) - ETA: {eta:.0f}s")
        
        avg_precision = np.mean(precisions)
        total_time = time.time() - start_time
        print(f"✓ Precision@{k}: {avg_precision:.4f} (completed in {total_time:.1f}s)")
        
        return avg_precision, precisions

def main():
    """Main execution."""
    print("=== Optimized Item-Based Collaborative Filtering ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    # Simple train/test split
    print("Creating train/test split...")
    shuffled = ratings_df.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Train model
    model = OptimizedItemBasedCF()
    model.train(train_df)
    
    # Evaluate
    precision_10, _ = model.evaluate_precision_fast(test_df, k=10, max_users=1000)
    
    print(f"\n=== Results ===")
    print(f"Training time: {model.training_stats['training_time']:.1f}s")
    print(f"Users: {model.training_stats['n_users']:,}")
    print(f"Books: {model.training_stats['n_books']:,}")
    print(f"Precision@10: {precision_10:.4f}")
    
    return model

if __name__ == "__main__":
    model = main()
