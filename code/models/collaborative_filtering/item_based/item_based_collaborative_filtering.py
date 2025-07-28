#!/usr/bin/env python3
"""
Item-Based Collaborative Filtering Implementation
Finds similar books based on user rating patterns and recommends accordingly.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ItemBasedCollaborativeFiltering:
    def __init__(self, min_ratings_per_book=50, min_ratings_per_user=10, top_k_similar=50):
        """
        Initialize Item-Based CF model.
        
        Args:
            min_ratings_per_book: Minimum ratings required for a book to be considered
            min_ratings_per_user: Minimum ratings required for a user to be considered
            top_k_similar: Number of similar items to consider for recommendations
        """
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
        
    def prepare_data(self, ratings_df, books_df=None):
        """Prepare and filter data for training."""
        print("Preparing data...")
        
        # Filter books and users with sufficient ratings
        book_counts = ratings_df['book_id'].value_counts()
        user_counts = ratings_df['user_id'].value_counts()
        
        valid_books = book_counts[book_counts >= self.min_ratings_per_book].index
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        
        filtered_df = ratings_df[
            (ratings_df['book_id'].isin(valid_books)) & 
            (ratings_df['user_id'].isin(valid_users))
        ].copy()
        
        print(f"Original data: {len(ratings_df)} ratings, {ratings_df['user_id'].nunique()} users, {ratings_df['book_id'].nunique()} books")
        print(f"Filtered data: {len(filtered_df)} ratings, {filtered_df['user_id'].nunique()} users, {filtered_df['book_id'].nunique()} books")
        
        # Create mappings
        unique_books = sorted(filtered_df['book_id'].unique())
        unique_users = sorted(filtered_df['user_id'].unique())
        
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        
        # Create user-item matrix
        n_users = len(unique_users)
        n_books = len(unique_books)
        
        user_indices = filtered_df['user_id'].map(self.user_id_to_idx)
        book_indices = filtered_df['book_id'].map(self.book_id_to_idx)
        
        self.user_item_matrix = csr_matrix(
            (filtered_df['rating'], (user_indices, book_indices)),
            shape=(n_users, n_books)
        )
        
        # Calculate book means for prediction
        for book_id in unique_books:
            book_ratings = filtered_df[filtered_df['book_id'] == book_id]['rating']
            self.book_means[book_id] = book_ratings.mean()
        
        self.training_stats['n_users'] = n_users
        self.training_stats['n_books'] = n_books
        self.training_stats['n_ratings'] = len(filtered_df)
        self.training_stats['sparsity'] = 1 - (len(filtered_df) / (n_users * n_books))
        
        return filtered_df
        
    def compute_item_similarity(self):
        """Compute item-item similarity matrix using cosine similarity."""
        print("Computing item-item similarity matrix...")
        
        # Transpose to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        n_items = item_user_matrix.shape[0]
        
        print(f"Computing similarities for {n_items} items...")
        print("This should take about 10-15 seconds for ~10k items...")
        
        # Compute cosine similarity between items
        # Using sklearn's cosine_similarity for efficiency
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        
        # Set diagonal to 0 (book is not similar to itself for recommendation purposes)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print(f"✓ Item similarity matrix computed: {self.item_similarity_matrix.shape}")
        print(f"✓ Memory usage: ~{(self.item_similarity_matrix.nbytes / 1024**2):.1f} MB")
        
    def train(self, ratings_df, books_df=None):
        """Train the Item-Based CF model."""
        start_time = datetime.now()
        print("Training Item-Based Collaborative Filtering model...")
        
        # Prepare data
        self.filtered_data = self.prepare_data(ratings_df, books_df)
        
        # Compute similarity matrix
        self.compute_item_similarity()
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        self.training_stats['training_time_seconds'] = training_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        return self
        
    def predict_rating(self, user_id, book_id):
        """Predict rating for a user-book pair."""
        if user_id not in self.user_id_to_idx or book_id not in self.book_id_to_idx:
            return self.book_means.get(book_id, 3.5)  # Global fallback
            
        user_idx = self.user_id_to_idx[user_id]
        book_idx = self.book_id_to_idx[book_id]
        
        # Get books rated by this user
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_book_indices = np.nonzero(user_ratings)[0]
        
        if len(rated_book_indices) == 0:
            return self.book_means.get(book_id, 3.5)
        
        # Get similarities between target book and user's rated books
        similarities = self.item_similarity_matrix[book_idx][rated_book_indices]
        
        # Get top-k most similar books
        if len(similarities) > self.top_k_similar:
            top_k_indices = np.argsort(similarities)[-self.top_k_similar:]
            similarities = similarities[top_k_indices]
            rated_book_indices = rated_book_indices[top_k_indices]
        
        # Filter out zero similarities
        positive_sim_mask = similarities > 0
        if not np.any(positive_sim_mask):
            return self.book_means.get(book_id, 3.5)
            
        similarities = similarities[positive_sim_mask]
        rated_book_indices = rated_book_indices[positive_sim_mask]
        
        # Calculate weighted prediction
        user_ratings_for_similar = user_ratings[rated_book_indices]
        
        if len(similarities) == 0 or np.sum(similarities) == 0:
            return self.book_means.get(book_id, 3.5)
            
        predicted_rating = np.sum(similarities * user_ratings_for_similar) / np.sum(similarities)
        
        # Ensure rating is within valid range
        return np.clip(predicted_rating, 1, 5)
        
    def recommend_books(self, user_id, n_recommendations=10, exclude_rated=True, max_candidates=2000):
        """Generate book recommendations for a user."""
        if user_id not in self.user_id_to_idx:
            # For cold start, recommend most popular books
            popular_books = self.filtered_data.groupby('book_id')['rating'].agg(['count', 'mean']).reset_index()
            popular_books = popular_books[popular_books['count'] >= 20]
            popular_books = popular_books.sort_values(['mean', 'count'], ascending=[False, False])
            return list(popular_books.head(n_recommendations)['book_id'])
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get books already rated by user
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_books = set(np.nonzero(user_ratings)[0])
        
        # Get candidate books (unrated books)
        all_book_indices = set(range(len(self.idx_to_book_id)))
        candidate_books = all_book_indices - rated_books if exclude_rated else all_book_indices
        
        # If too many candidates, sample the most popular ones
        if len(candidate_books) > max_candidates:
            # Get book popularity scores
            book_counts = {}
            for book_idx in candidate_books:
                book_id = self.idx_to_book_id[book_idx]
                book_counts[book_idx] = len(self.filtered_data[self.filtered_data['book_id'] == book_id])
            
            # Take top popular books
            sorted_candidates = sorted(book_counts.items(), key=lambda x: x[1], reverse=True)
            candidate_books = set([book_idx for book_idx, _ in sorted_candidates[:max_candidates]])
        
        # Generate predictions for candidate books only
        predictions = []
        
        for book_idx in candidate_books:
            book_id = self.idx_to_book_id[book_idx]
            predicted_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in predictions[:n_recommendations]]
        
    def evaluate_precision_at_k(self, test_data, k=10, max_users=5000):
        """Evaluate precision@k on test data."""
        print(f"Evaluating precision@{k}...", flush=True)
        
        # Group test data by user
        print("Grouping test data by user...", flush=True)
        test_grouped = test_data.groupby('user_id')['book_id'].apply(list).to_dict()
        
        # Sample users if too many
        valid_users = [user_id for user_id in test_grouped.keys() if user_id in self.user_id_to_idx]
        if len(valid_users) > max_users:
            print(f"Sampling {max_users} users from {len(valid_users)} valid users for evaluation...", flush=True)
            np.random.seed(42)  # For reproducibility
            valid_users = np.random.choice(valid_users, max_users, replace=False)
        
        precisions = []
        total_users = len(valid_users)
        
        print(f"Starting evaluation of {total_users} users...", flush=True)
        start_time = time.time()
        
        for i, user_id in enumerate(valid_users):
            actual_books = test_grouped[user_id]
                
            # Get recommendations
            recommended_books = self.recommend_books(user_id, n_recommendations=k, max_candidates=1000)
            
            # Calculate precision
            relevant_recommendations = set(recommended_books) & set(actual_books)
            precision = len(relevant_recommendations) / k if k > 0 else 0
            precisions.append(precision)
            
            # Progress indicator with time estimation
            if (i + 1) % 100 == 0 or (i + 1) == total_users:
                elapsed = time.time() - start_time
                progress = ((i + 1) / total_users) * 100
                avg_time_per_user = elapsed / (i + 1)
                remaining_time = avg_time_per_user * (total_users - i - 1)
                
                print(f"Progress: {i+1}/{total_users} users ({progress:.1f}%) - "
                      f"ETA: {remaining_time:.0f}s", flush=True)
        
        avg_precision = np.mean(precisions) if precisions else 0
        total_time = time.time() - start_time
        print(f"✓ Precision@{k}: {avg_precision:.4f} (evaluated {len(precisions)} users in {total_time:.1f}s)", flush=True)
        
        return avg_precision, precisions

def load_data():
    """Load the ratings and books data."""
    print("Loading data...")
    
    # Load ratings
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    # Load books if available
    books_df = None
    if os.path.exists('data_goodbooks_10k/books.csv'):
        books_df = pd.read_csv('data_goodbooks_10k/books.csv')
    
    return ratings_df, books_df

def create_train_test_split(ratings_df, test_ratio=0.2, min_test_ratings=5):
    """Create train/test split ensuring each user has enough data."""
    print("Creating train/test split...")
    
    train_data = []
    test_data = []
    
    # Group by user
    user_groups = ratings_df.groupby('user_id')
    
    for user_id, user_ratings in user_groups:
        user_ratings = user_ratings.sample(frac=1, random_state=42)  # Shuffle
        n_ratings = len(user_ratings)
        
        if n_ratings < min_test_ratings * 2:  # Need enough for both train and test
            train_data.append(user_ratings)  # Put all in training
            continue
            
        n_test = max(min_test_ratings, int(n_ratings * test_ratio))
        
        test_data.append(user_ratings.iloc[:n_test])
        train_data.append(user_ratings.iloc[n_test:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")
    return train_df, test_df

def main():
    """Main execution function."""
    print("=== Item-Based Collaborative Filtering ===")
    
    # Load data
    ratings_df, books_df = load_data()
    
    # Create train/test split
    train_df, test_df = create_train_test_split(ratings_df)
    
    # Initialize and train model
    model = ItemBasedCollaborativeFiltering(
        min_ratings_per_book=50,
        min_ratings_per_user=20,
        top_k_similar=50
    )
    
    model.train(train_df, books_df)
    
    # Evaluate model
    precision_10, _ = model.evaluate_precision_at_k(test_df, k=10)
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Training time: {model.training_stats['training_time_seconds']:.2f} seconds")
    print(f"Users: {model.training_stats['n_users']:,}")
    print(f"Books: {model.training_stats['n_books']:,}")
    print(f"Ratings: {model.training_stats['n_ratings']:,}")
    print(f"Sparsity: {model.training_stats['sparsity']:.3f}")
    print(f"Precision@10: {precision_10:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/item_based_cf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved to models/item_based_cf_model.pkl")
    return model

if __name__ == "__main__":
    model = main() 