#!/usr/bin/env python3
"""
Trace recommendations for specific users to understand the 0.0000 precision issue.
"""

import pandas as pd
import numpy as np

def trace_user_recommendations():
    """Trace step by step for a specific user."""
    print("=== Tracing User Recommendations ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    # Simple split for one user
    user_id = 1  # Use user 1
    user_data = ratings_df[ratings_df['user_id'] == user_id].copy()
    user_data = user_data.sort_values('book_id')
    
    print(f"User {user_id} data:")
    print(f"  Total ratings: {len(user_data)}")
    print(f"  Rating distribution: {user_data['rating'].value_counts().sort_index().to_dict()}")
    
    # Split
    n_test = max(2, int(len(user_data) * 0.2))
    train_data = user_data.iloc[:-n_test]
    test_data = user_data.iloc[-n_test:]
    
    print(f"\nAfter split:")
    print(f"  Train: {len(train_data)} ratings")
    print(f"  Test: {len(test_data)} ratings")
    
    # Show actual data
    print(f"\nTrain books: {train_data['book_id'].tolist()}")
    print(f"Train ratings: {train_data['rating'].tolist()}")
    print(f"Test books: {test_data['book_id'].tolist()}")
    print(f"Test ratings: {test_data['rating'].tolist()}")
    
    # Find relevant books (rating >= 4)
    relevant_books = set(test_data[test_data['rating'] >= 4]['book_id'])
    print(f"\nRelevant books (rating >= 4): {relevant_books}")
    
    # Get popular books from ALL training data (not just this user)
    all_train = ratings_df[~ratings_df.index.isin(test_data.index)]  # Exclude this user's test data
    
    book_stats = all_train.groupby('book_id').agg({
        'rating': ['count', 'mean']
    })
    book_stats.columns = ['count', 'mean']
    book_stats = book_stats[book_stats['count'] >= 10]  # At least 10 ratings
    book_stats['score'] = book_stats['mean'] * 0.7 + np.log1p(book_stats['count']) * 0.3
    popular_books = book_stats.sort_values('score', ascending=False)
    
    print(f"\nTop 15 popular books globally:")
    for i, (book_id, stats) in enumerate(popular_books.head(15).iterrows()):
        in_relevant = "✓" if book_id in relevant_books else " "
        print(f"  {in_relevant} {book_id}: {stats['count']} ratings, avg {stats['mean']:.2f}, score {stats['score']:.3f}")
    
    # Generate recommendations (exclude already rated)
    rated_books = set(train_data['book_id'])
    recommendations = []
    
    for book_id in popular_books.index:
        if book_id not in rated_books:
            recommendations.append(book_id)
        if len(recommendations) >= 10:
            break
    
    print(f"\nRecommendations (top 10 unrated popular books): {recommendations}")
    
    # Calculate precision manually
    hits = len(set(recommendations) & relevant_books)
    precision = hits / 10
    
    print(f"\nPrecision calculation:")
    print(f"  Recommendations: {recommendations}")
    print(f"  Relevant books: {list(relevant_books)}")
    print(f"  Hits: {hits}")
    print(f"  Precision@10: {precision:.4f}")
    
    # Check why we might have 0 precision
    if len(relevant_books) == 0:
        print(f"❌ ISSUE: No relevant books (no ratings >= 4 in test)")
    elif len(recommendations) == 0:
        print(f"❌ ISSUE: No recommendations generated")
    elif hits == 0:
        print(f"❌ ISSUE: No overlap between popular books and user's future preferences")
        print(f"   This could be normal - user might have niche taste!")
    else:
        print(f"✅ Everything looks normal, we have some precision")
    
    # Try with different users
    print(f"\n=== Testing 5 different users ===")
    user_results = []
    
    for test_user in [1, 10, 100, 1000, 5000]:
        user_data = ratings_df[ratings_df['user_id'] == test_user]
        if len(user_data) < 10:
            continue
            
        user_data = user_data.sort_values('book_id')
        n_test = max(2, int(len(user_data) * 0.2))
        train_data = user_data.iloc[:-n_test]
        test_data = user_data.iloc[-n_test:]
        
        relevant_books = set(test_data[test_data['rating'] >= 4]['book_id'])
        rated_books = set(train_data['book_id'])
        
        recommendations = []
        for book_id in popular_books.index:
            if book_id not in rated_books:
                recommendations.append(book_id)
            if len(recommendations) >= 10:
                break
        
        hits = len(set(recommendations) & relevant_books)
        precision = hits / 10
        
        user_results.append({
            'user_id': test_user,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'relevant_books': len(relevant_books),
            'precision': precision
        })
        
        print(f"User {test_user}: {len(train_data)} train, {len(test_data)} test, {len(relevant_books)} relevant, precision {precision:.4f}")
    
    avg_precision = np.mean([r['precision'] for r in user_results])
    print(f"\nAverage precision across {len(user_results)} users: {avg_precision:.4f}")
    
    return user_results

if __name__ == "__main__":
    results = trace_user_recommendations()
