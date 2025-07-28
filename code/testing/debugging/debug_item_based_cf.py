#!/usr/bin/env python3
"""
Debug script for Item-Based Collaborative Filtering
Tests each component step by step to identify bottlenecks.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
import sys

def load_and_sample_data():
    """Load and sample data for debugging."""
    print("Loading data...")
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    print(f"Original data: {len(ratings_df):,} ratings")
    print(f"Users: {ratings_df['user_id'].nunique():,}")
    print(f"Books: {ratings_df['book_id'].nunique():,}")
    
    # Sample smaller dataset for debugging
    print("\nSampling data for debugging...")
    
    # Take top 1000 books by number of ratings
    book_counts = ratings_df['book_id'].value_counts()
    top_books = book_counts.head(1000).index
    
    # Take users who rated at least 10 of these books
    filtered_ratings = ratings_df[ratings_df['book_id'].isin(top_books)]
    user_book_counts = filtered_ratings['user_id'].value_counts()
    active_users = user_book_counts[user_book_counts >= 10].index
    
    # Final sample
    sample_data = filtered_ratings[filtered_ratings['user_id'].isin(active_users)]
    
    print(f"Sample data: {len(sample_data):,} ratings")
    print(f"Sample users: {sample_data['user_id'].nunique():,}")
    print(f"Sample books: {sample_data['book_id'].nunique():,}")
    
    return sample_data

def test_similarity_computation(ratings_df):
    """Test the similarity computation step by step."""
    print("\n=== Testing Similarity Computation ===")
    
    # Create mappings
    unique_books = sorted(ratings_df['book_id'].unique())
    unique_users = sorted(ratings_df['user_id'].unique())
    
    book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    print(f"Creating {len(unique_users)} × {len(unique_books)} matrix...")
    
    # Create user-item matrix
    user_indices = ratings_df['user_id'].map(user_id_to_idx)
    book_indices = ratings_df['book_id'].map(book_id_to_idx)
    
    user_item_matrix = csr_matrix(
        (ratings_df['rating'], (user_indices, book_indices)),
        shape=(len(unique_users), len(unique_books))
    )
    
    print(f"Matrix created: {user_item_matrix.shape}")
    print(f"Sparsity: {1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.3f}")
    
    # Test similarity computation
    print("\nComputing item-item similarity...")
    start_time = time.time()
    
    item_user_matrix = user_item_matrix.T
    print(f"Transposed matrix: {item_user_matrix.shape}")
    
    # Time the similarity computation
    similarity_matrix = cosine_similarity(item_user_matrix)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    print(f"✓ Similarity computation completed!")
    print(f"Time taken: {computation_time:.2f} seconds")
    print(f"Result shape: {similarity_matrix.shape}")
    print(f"Memory usage: {similarity_matrix.nbytes / 1024**2:.1f} MB")
    
    # Analyze similarity matrix
    np.fill_diagonal(similarity_matrix, 0)
    print(f"Max similarity: {np.max(similarity_matrix):.3f}")
    print(f"Mean similarity: {np.mean(similarity_matrix[similarity_matrix > 0]):.3f}")
    print(f"Non-zero similarities: {np.sum(similarity_matrix > 0):,}")
    
    return user_item_matrix, similarity_matrix, book_id_to_idx, user_id_to_idx

def test_recommendation_speed(user_item_matrix, similarity_matrix, book_id_to_idx, user_id_to_idx, ratings_df):
    """Test recommendation generation speed."""
    print("\n=== Testing Recommendation Speed ===")
    
    # Sample a few users
    sample_users = list(user_id_to_idx.keys())[:10]
    
    idx_to_book_id = {idx: book_id for book_id, idx in book_id_to_idx.items()}
    
    total_time = 0
    
    for i, user_id in enumerate(sample_users):
        print(f"Testing user {i+1}/10...", end='', flush=True)
        start_time = time.time()
        
        user_idx = user_id_to_idx[user_id]
        user_ratings = user_item_matrix[user_idx].toarray().flatten()
        rated_books = set(np.nonzero(user_ratings)[0])
        
        # Generate predictions for unrated books (sample)
        unrated_books = [idx for idx in range(len(book_id_to_idx)) if idx not in rated_books]
        sample_unrated = unrated_books[:100]  # Sample 100 books
        
        predictions = []
        for book_idx in sample_unrated:
            # Simplified prediction - just get similarity to rated books
            similarities = similarity_matrix[book_idx][list(rated_books)]
            if len(similarities) > 0 and np.sum(similarities > 0) > 0:
                pred = np.mean(similarities[similarities > 0])
            else:
                pred = 3.0
            predictions.append((idx_to_book_id[book_idx], pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations = [book_id for book_id, _ in predictions[:10]]
        
        end_time = time.time()
        user_time = end_time - start_time
        total_time += user_time
        
        print(f" {user_time:.3f}s")
    
    avg_time = total_time / len(sample_users)
    print(f"\nAverage time per user: {avg_time:.3f} seconds")
    print(f"Estimated time for 3000 users: {avg_time * 3000:.1f} seconds ({avg_time * 3000 / 60:.1f} minutes)")
    
    return avg_time

def main():
    """Main debug function."""
    print("=== Item-Based CF Debug Script ===")
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Load and sample data
    sample_data = load_and_sample_data()
    
    # Step 2: Test similarity computation
    user_item_matrix, similarity_matrix, book_id_to_idx, user_id_to_idx = test_similarity_computation(sample_data)
    
    # Step 3: Test recommendation speed
    avg_time = test_recommendation_speed(user_item_matrix, similarity_matrix, book_id_to_idx, user_id_to_idx, sample_data)
    
    print(f"\n=== Debug Complete ===")
    print(f"The bottleneck is likely in the recommendation generation phase.")
    print(f"With current speed, evaluation would take {avg_time * 3000 / 60:.1f} minutes for 3000 users.")

if __name__ == "__main__":
    main() 