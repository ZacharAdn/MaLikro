#!/usr/bin/env python3
"""
Debug precision calculation to understand why we're getting 0.0000
"""

import pandas as pd
import numpy as np

def debug_precision_calculation():
    """Debug step by step what's happening in precision calculation."""
    print("=== Debugging Precision Calculation ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # Check rating distribution
    print("\nRating distribution:")
    print(ratings_df['rating'].value_counts().sort_index())
    
    # Check user rating counts
    user_counts = ratings_df['user_id'].value_counts()
    print(f"\nUser rating stats:")
    print(f"Users with 1+ ratings: {len(user_counts)}")
    print(f"Users with 10+ ratings: {sum(user_counts >= 10)}")
    print(f"Users with 50+ ratings: {sum(user_counts >= 50)}")
    
    # Check book rating counts
    book_counts = ratings_df['book_id'].value_counts()
    print(f"\nBook rating stats:")
    print(f"Books with 1+ ratings: {len(book_counts)}")
    print(f"Books with 10+ ratings: {sum(book_counts >= 10)}")
    print(f"Books with 50+ ratings: {sum(book_counts >= 50)}")
    
    # Simple train/test split for one user
    print("\n=== Testing with one specific user ===")
    
    # Get a user with many ratings
    heavy_user = user_counts[user_counts >= 50].index[0]
    user_data = ratings_df[ratings_df['user_id'] == heavy_user].copy()
    user_data = user_data.sort_values('book_id')  # Stable sort
    
    print(f"User {heavy_user} has {len(user_data)} ratings")
    print(f"Rating distribution: {user_data['rating'].value_counts().sort_index().to_dict()}")
    
    # Split user data
    n_test = max(1, int(len(user_data) * 0.2))
    train_data = user_data.iloc[:-n_test]
    test_data = user_data.iloc[-n_test:]
    
    print(f"Train: {len(train_data)} ratings")
    print(f"Test: {len(test_data)} ratings")
    print(f"Test ratings 4+: {sum(test_data['rating'] >= 4)}")
    
    # Check if we have relevant books
    relevant_books = set(test_data[test_data['rating'] >= 4]['book_id'])
    print(f"Relevant books in test: {len(relevant_books)}")
    print(f"Relevant book IDs: {list(relevant_books)[:5]}")
    
    # Get popular books from training
    popular_books = train_data.groupby('book_id')['rating'].agg(['count', 'mean'])
    popular_books = popular_books[popular_books['count'] >= 1]  # Any book
    popular_books['score'] = popular_books['mean'] * 0.7 + np.log1p(popular_books['count']) * 0.3
    popular_books = popular_books.sort_values('score', ascending=False)
    
    print(f"\nTop 10 popular books from training:")
    for i, (book_id, stats) in enumerate(popular_books.head(10).iterrows()):
        print(f"  {book_id}: {stats['count']} ratings, avg {stats['mean']:.2f}")
    
    # Check overlap
    train_books = set(train_data['book_id'])
    test_books = set(test_data['book_id'])
    popular_book_ids = set(popular_books.head(20).index)
    
    print(f"\nOverlap analysis:")
    print(f"Books in train: {len(train_books)}")
    print(f"Books in test: {len(test_books)}")
    print(f"Train ∩ Test: {len(train_books & test_books)}")
    print(f"Popular books: {len(popular_book_ids)}")
    print(f"Popular ∩ Relevant: {len(popular_book_ids & relevant_books)}")
    
    # Manual precision calculation
    rated_books = set(train_data['book_id'])
    recommendations = [book_id for book_id in popular_books.index[:10] if book_id not in rated_books][:10]
    
    print(f"\nManual calculation:")
    print(f"Recommendations: {recommendations[:5]}")
    print(f"Relevant books: {list(relevant_books)[:5]}")
    
    hits = len(set(recommendations) & relevant_books)
    precision = hits / len(recommendations) if len(recommendations) > 0 else 0
    
    print(f"Hits: {hits}")
    print(f"Precision: {precision:.4f}")
    
    # Check if the issue is no recommendations or no overlap
    if len(recommendations) == 0:
        print("❌ ISSUE: No recommendations generated")
    elif len(relevant_books) == 0:
        print("❌ ISSUE: No relevant books in test set")
    elif hits == 0:
        print("❌ ISSUE: No overlap between recommendations and relevant books")
    else:
        print("✅ Manual calculation shows some precision")
        
    return {
        'user_id': heavy_user,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'relevant_books': len(relevant_books),
        'recommendations': len(recommendations),
        'hits': hits,
        'precision': precision
    }

if __name__ == "__main__":
    result = debug_precision_calculation()
