#!/usr/bin/env python3
"""
Fast Item-Based CF with optimized evaluation
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

def load_and_prepare_data():
    """Load and prepare data efficiently."""
    print("Loading data...")
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    # Simple train/test split (80/20)
    shuffled = ratings_df.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df):,} ratings")
    print(f"Test: {len(test_df):,} ratings")
    
    return train_df, test_df

def build_item_similarity_matrix(train_df, min_ratings_per_book=50):
    """Build item similarity matrix efficiently."""
    print("Building similarity matrix...")
    
    # Filter books with enough ratings
    book_counts = train_df['book_id'].value_counts()
    valid_books = book_counts[book_counts >= min_ratings_per_book].index
    
    # Filter data
    filtered_df = train_df[train_df['book_id'].isin(valid_books)].copy()
    print(f"Using {len(valid_books)} books with {len(filtered_df):,} ratings")
    
    # Create mappings
    unique_books = sorted(filtered_df['book_id'].unique())
    unique_users = sorted(filtered_df['user_id'].unique())
    
    book_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # Create user-item matrix
    user_indices = filtered_df['user_id'].map(user_to_idx)
    book_indices = filtered_df['book_id'].map(book_to_idx)
    
    user_item_matrix = csr_matrix(
        (filtered_df['rating'], (user_indices, book_indices)),
        shape=(len(unique_users), len(unique_books))
    )
    
    # Compute similarity
    print(f"Computing {len(unique_books)}x{len(unique_books)} similarity matrix...")
    start_time = time.time()
    
    item_user_matrix = user_item_matrix.T
    similarity_matrix = cosine_similarity(item_user_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity
    
    comp_time = time.time() - start_time
    print(f"✓ Similarity computed in {comp_time:.1f} seconds")
    
    return similarity_matrix, book_to_idx, unique_books, filtered_df

def fast_recommend(user_id, user_item_matrix, similarity_matrix, book_to_idx, 
                   idx_to_book, n_recommendations=10):
    """Fast recommendation generation."""
    if user_id not in user_item_matrix.index:
        return []
    
    # Get user ratings
    user_ratings = user_item_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings > 0]
    
    if len(rated_books) == 0:
        return []
    
    # Get similarities for rated books
    rated_indices = [book_to_idx.get(book_id) for book_id in rated_books.index 
                     if book_id in book_to_idx]
    
    if len(rated_indices) == 0:
        return []
    
    # Calculate scores for all books
    scores = np.zeros(similarity_matrix.shape[0])
    
    for book_idx in rated_indices:
        rating = rated_books[idx_to_book[book_idx]]
        similarities = similarity_matrix[book_idx]
        scores += similarities * rating
    
    # Remove already rated books
    for book_idx in rated_indices:
        scores[book_idx] = -1
    
    # Get top recommendations
    top_indices = np.argsort(scores)[-n_recommendations:][::-1]
    top_indices = top_indices[scores[top_indices] > 0]  # Only positive scores
    
    return [idx_to_book[idx] for idx in top_indices]

def fast_evaluation(test_df, similarity_matrix, book_to_idx, unique_books, 
                    filtered_df, max_users=1000):
    """Fast evaluation using vectorized operations."""
    print(f"Starting fast evaluation with {max_users} users...")
    
    # Create book index mapping
    idx_to_book = {idx: book_id for book_id, idx in book_to_idx.items()}
    
    # Create user-item matrix from training data for recommendations
    user_item_pivot = filtered_df.pivot_table(
        index='user_id', columns='book_id', values='rating', fill_value=0
    )
    
    # Sample test users
    test_users = test_df['user_id'].unique()
    if len(test_users) > max_users:
        test_users = np.random.choice(test_users, max_users, replace=False)
    
    print(f"Evaluating {len(test_users)} users...")
    
    precisions = []
    start_time = time.time()
    
    for i, user_id in enumerate(test_users):
        # Get user's test books
        user_test_books = set(test_df[test_df['user_id'] == user_id]['book_id'])
        
        # Get recommendations
        recommendations = fast_recommend(
            user_id, user_item_pivot, similarity_matrix, 
            book_to_idx, idx_to_book, n_recommendations=10
        )
        
        # Calculate precision
        if len(recommendations) > 0:
            hits = len(set(recommendations) & user_test_books)
            precision = hits / len(recommendations)
        else:
            precision = 0
        
        precisions.append(precision)
        
        # Progress tracking
        if (i + 1) % 100 == 0 or (i + 1) == len(test_users):
            elapsed = time.time() - start_time
            progress = (i + 1) / len(test_users) * 100
            print(f"Progress: {i+1}/{len(test_users)} ({progress:.1f}%) - "
                  f"Avg precision so far: {np.mean(precisions):.3f}")
    
    total_time = time.time() - start_time
    avg_precision = np.mean(precisions)
    
    print(f"\n✓ Evaluation completed in {total_time:.1f} seconds")
    print(f"✓ Precision@10: {avg_precision:.4f}")
    print(f"✓ Users with recommendations: {sum(1 for p in precisions if p > 0)}/{len(precisions)}")
    
    return avg_precision, precisions

def main():
    """Main execution."""
    print("=== Fast Item-Based Collaborative Filtering ===")
    
    # Load data
    train_df, test_df = load_and_prepare_data()
    
    # Build similarity matrix
    similarity_matrix, book_to_idx, unique_books, filtered_df = build_item_similarity_matrix(
        train_df, min_ratings_per_book=30
    )
    
    # Fast evaluation
    precision, precisions = fast_evaluation(
        test_df, similarity_matrix, book_to_idx, unique_books, filtered_df
    )
    
    # Create results folder
    os.makedirs('results/item_cf_fast', exist_ok=True)
    
    # Save results
    results = {
        'precision_at_10': precision,
        'num_books': len(unique_books),
        'num_users_evaluated': len(precisions),
        'users_with_recommendations': sum(1 for p in precisions if p > 0)
    }
    
    # Save summary
    with open('results/item_cf_fast/results_summary.txt', 'w') as f:
        f.write("Item-Based Collaborative Filtering Results\n")
        f.write("==========================================\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✓ Results saved to results/item_cf_fast/")
    print(f"✓ Final Precision@10: {precision:.4f}")
    
    return precision

if __name__ == "__main__":
    main()
