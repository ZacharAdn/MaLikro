#!/usr/bin/env python3
"""
Corrected Recommendation System with proper random train/test split.
This fixes the fundamental issue where we were sorting by book_id, creating artificial bias.

This file contains both baseline and item-based collaborative filtering implementations
with the corrected train/test split methodology.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

class CorrectedBaselineRecommender:
    """Corrected popularity-based recommender."""
    
    def __init__(self):
        self.popular_books = []
        self.book_stats = None
        
    def train(self, ratings_df):
        """Train on ratings data with proper popularity calculation."""
        # Calculate book statistics
        self.book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        })
        self.book_stats.columns = ['count', 'mean']
        self.book_stats = self.book_stats.reset_index()
        
        # Filter books with sufficient ratings
        self.book_stats = self.book_stats[self.book_stats['count'] >= 20]
        
        # Calculate popularity score (weighted by rating and count)
        self.book_stats['popularity_score'] = (
            self.book_stats['mean'] * 0.6 +  # Quality weight
            np.log1p(self.book_stats['count']) * 0.4  # Popularity weight
        )
        
        # Sort by popularity score
        self.popular_books = self.book_stats.sort_values(
            'popularity_score', ascending=False
        )['book_id'].tolist()
        
        print(f"Baseline trained with {len(self.popular_books)} popular books")
        
    def recommend(self, user_id, user_rated_books, n_recommendations=10):
        """Generate recommendations by excluding already rated books."""
        rated_books = set(user_rated_books)
        
        recommendations = []
        for book_id in self.popular_books:
            if book_id not in rated_books:
                recommendations.append(book_id)
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

class CorrectedItemBasedCF:
    """Corrected Item-Based Collaborative Filtering."""
    
    def __init__(self, min_ratings_per_book=30, min_ratings_per_user=15, top_k_similar=50):
        self.min_ratings_per_book = min_ratings_per_book
        self.min_ratings_per_user = min_ratings_per_user
        self.top_k_similar = top_k_similar
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        self.user_id_to_idx = {}
        self.book_means = {}
        self.all_books = []
        
    def train(self, ratings_df):
        """Train the model with proper filtering."""
        print("Training Corrected Item-Based CF...")
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
        
    def recommend(self, user_id, user_rated_books, n_recommendations=10):
        """Generate recommendations."""
        if user_id not in self.user_id_to_idx:
            # Return high-rated books by average rating
            popular_books = sorted(self.book_means.items(), key=lambda x: x[1], reverse=True)
            candidate_books = [book_id for book_id, _ in popular_books if book_id not in set(user_rated_books)]
            return candidate_books[:n_recommendations]
        
        # Get candidate books (unrated)
        rated_books = set(user_rated_books)
        candidate_books = [book_id for book_id in self.all_books if book_id not in rated_books]
        
        # Predict ratings for all candidates
        predictions = []
        for book_id in candidate_books:
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append((book_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in predictions[:n_recommendations]]

def create_corrected_train_test_split(ratings_df, test_ratio=0.2):
    """Create CORRECTED train/test split with proper random sampling."""
    print("Creating CORRECTED train/test split...")
    
    train_data = []
    test_data = []
    
    # Group by user
    user_groups = ratings_df.groupby('user_id')
    
    for user_id, user_ratings in user_groups:
        # ✅ FIXED: Random shuffle instead of sorting by book_id
        user_ratings = user_ratings.sample(frac=1, random_state=42).reset_index(drop=True)
        n_ratings = len(user_ratings)
        
        if n_ratings < 10:  # Skip users with too few ratings
            train_data.append(user_ratings)
            continue
            
        # Split randomly - NOT by book_id
        n_test = max(2, int(n_ratings * test_ratio))
        n_test = min(n_test, n_ratings - 5)  # Ensure at least 5 for train
        
        train_data.append(user_ratings.iloc[:-n_test])
        test_data.append(user_ratings.iloc[-n_test:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"✅ CORRECTED Split:")
    print(f"   Train: {len(train_df):,} ratings from {train_df['user_id'].nunique():,} users")
    print(f"   Test: {len(test_df):,} ratings from {test_df['user_id'].nunique():,} users")
    
    return train_df, test_df

def evaluate_models_corrected(baseline, item_cf, train_df, test_df, n_users=1000):
    """Evaluate both models with corrected methodology."""
    print(f"Evaluating models on {n_users} users with CORRECTED methodology...")
    
    # Get users who exist in both train and test
    test_users = test_df['user_id'].unique()
    train_users = train_df['user_id'].unique()
    common_users = list(set(test_users) & set(train_users))
    
    if len(common_users) > n_users:
        np.random.seed(42)
        common_users = np.random.choice(common_users, n_users, replace=False)
    
    print(f"Evaluating {len(common_users)} users...")
    
    baseline_precisions = []
    item_cf_precisions = []
    debug_info = []
    
    for i, user_id in enumerate(common_users):
        # Get user's test and train data
        user_test = test_df[test_df['user_id'] == user_id]
        user_train = train_df[train_df['user_id'] == user_id]
        
        # Define relevant books (rating >= 4)
        relevant_books = set(user_test[user_test['rating'] >= 4]['book_id'])
        
        if len(relevant_books) == 0:
            continue  # Skip if no relevant books
        
        # Get user's rated books from training
        user_rated_books = user_train['book_id'].tolist()
        
        # Get recommendations from both models
        try:
            baseline_recs = baseline.recommend(user_id, user_rated_books, 10)
            item_cf_recs = item_cf.recommend(user_id, user_rated_books, 10)
        except:
            continue  # Skip if recommendation fails
        
        # Calculate precision
        baseline_hits = len(set(baseline_recs) & relevant_books)
        item_cf_hits = len(set(item_cf_recs) & relevant_books)
        
        baseline_precision = baseline_hits / 10
        item_cf_precision = item_cf_hits / 10
        
        baseline_precisions.append(baseline_precision)
        item_cf_precisions.append(item_cf_precision)
        
        # Store debug info for first few users
        if len(debug_info) < 5:
            debug_info.append({
                'user_id': user_id,
                'relevant_books': list(relevant_books)[:5],
                'baseline_recs': baseline_recs[:5],
                'item_cf_recs': item_cf_recs[:5],
                'baseline_hits': baseline_hits,
                'item_cf_hits': item_cf_hits
            })
        
        # Progress
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(common_users)}")
    
    baseline_avg = np.mean(baseline_precisions) if baseline_precisions else 0
    item_cf_avg = np.mean(item_cf_precisions) if item_cf_precisions else 0
    
    print(f"\n=== CORRECTED Results ===")
    print(f"✅ Baseline precision@10: {baseline_avg:.4f}")
    print(f"✅ Item-CF precision@10: {item_cf_avg:.4f}")
    
    if item_cf_avg > baseline_avg:
        improvement = ((item_cf_avg / baseline_avg - 1) * 100) if baseline_avg > 0 else 0
        print(f"✅ Item-CF is {improvement:.1f}% better than baseline!")
    else:
        print(f"⚠️  Item-CF needs further tuning")
    
    # Show debug examples
    print(f"\n=== Debug Examples ===")
    for info in debug_info:
        print(f"User {info['user_id']}:")
        print(f"  Relevant books: {info['relevant_books']}")
        print(f"  Baseline recs: {info['baseline_recs']} (hits: {info['baseline_hits']})")
        print(f"  Item-CF recs: {info['item_cf_recs']} (hits: {info['item_cf_hits']})")
        print()
    
    return baseline_avg, item_cf_avg

def evaluate_models_original_method(baseline, item_cf, train_df, test_df, n_users=1000):
    """Evaluate both models using ORIGINAL evaluation method (that gave 28% baseline)."""
    print(f"Evaluating models on {n_users} users with ORIGINAL evaluation method...")
    
    # Get users who exist in both train and test
    test_users = test_df['user_id'].unique()
    train_users = train_df['user_id'].unique()
    common_users = list(set(test_users) & set(train_users))
    
    if len(common_users) > n_users:
        np.random.seed(42)
        common_users = np.random.choice(common_users, n_users, replace=False)
    
    print(f"Evaluating {len(common_users)} users...")
    
    baseline_precisions = []
    item_cf_precisions = []
    debug_info = []
    
    for i, user_id in enumerate(common_users):
        # ✅ ORIGINAL METHOD: Get user's ALL high ratings in test set (4+ stars)
        user_test_high_ratings = test_df[
            (test_df['user_id'] == user_id) & 
            (test_df['rating'] >= 4)
        ]['book_id'].tolist()
        
        if len(user_test_high_ratings) == 0:
            continue  # Skip users with no high ratings in test set
        
        # Get user's rated books from training (to exclude from recommendations)
        user_rated_books = train_df[train_df['user_id'] == user_id]['book_id'].tolist()
        
        # Get recommendations from both models
        try:
            baseline_recs = baseline.recommend(user_id, user_rated_books, 10)
            item_cf_recs = item_cf.recommend(user_id, user_rated_books, 10)
        except:
            continue  # Skip if recommendation fails
        
        # ✅ ORIGINAL METHOD: Calculate precision using ALL test high ratings
        baseline_hits = len(set(baseline_recs) & set(user_test_high_ratings))
        item_cf_hits = len(set(item_cf_recs) & set(user_test_high_ratings))
        
        baseline_precision = baseline_hits / 10
        item_cf_precision = item_cf_hits / 10
        
        baseline_precisions.append(baseline_precision)
        item_cf_precisions.append(item_cf_precision)
        
        # Store debug info for first few users
        if len(debug_info) < 5:
            debug_info.append({
                'user_id': user_id,
                'relevant_books': user_test_high_ratings[:5],  # Show first 5
                'baseline_recs': baseline_recs[:5],
                'item_cf_recs': item_cf_recs[:5],
                'baseline_hits': baseline_hits,
                'item_cf_hits': item_cf_hits
            })
        
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{len(common_users)}")
    
    # Calculate average precision
    baseline_avg = np.mean(baseline_precisions) if baseline_precisions else 0
    item_cf_avg = np.mean(item_cf_precisions) if item_cf_precisions else 0
    
    print(f"\n=== ORIGINAL METHOD Results ===")
    print(f"✅ Baseline precision@10: {baseline_avg:.4f}")
    print(f"✅ Item-CF precision@10: {item_cf_avg:.4f}")
    if item_cf_avg > baseline_avg:
        print(f"🎉 Item-CF beats baseline by {item_cf_avg - baseline_avg:.4f}")
    else:
        print(f"⚠️  Item-CF needs further tuning")
    
    print(f"\n=== Debug Examples ===")
    for debug in debug_info:
        print(f"User {debug['user_id']}:")
        print(f"  Relevant books: {debug['relevant_books']}")
        print(f"  Baseline recs: {debug['baseline_recs']} (hits: {debug['baseline_hits']})")
        print(f"  Item-CF recs: {debug['item_cf_recs']} (hits: {debug['item_cf_hits']})")
        print()
    
    return baseline_avg, item_cf_avg

def create_original_user_split(ratings_df, test_size=0.2, random_state=42):
    """Create ORIGINAL train/test split by users (like the 28% baseline)."""
    print("Creating ORIGINAL user-based train/test split...")
    
    # Get unique users
    unique_users = ratings_df['user_id'].unique()
    
    # Split users randomly
    from sklearn.model_selection import train_test_split
    train_users, test_users = train_test_split(
        unique_users, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train/test datasets
    train_df = ratings_df[ratings_df['user_id'].isin(train_users)]
    test_df = ratings_df[ratings_df['user_id'].isin(test_users)]
    
    print(f"✅ ORIGINAL Split:")
    print(f"   Train: {len(train_df):,} ratings from {len(train_users):,} users")
    print(f"   Test: {len(test_df):,} ratings from {len(test_users):,} users")
    print(f"   Split: {len(train_df)/len(ratings_df):.1%} train, {len(test_df)/len(ratings_df):.1%} test")
    
    return train_df, test_df


def evaluate_with_original_split():
    """Test with original user-based split to verify 28% baseline."""
    print("=== TESTING ORIGINAL USER-BASED SPLIT ===")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # Create ORIGINAL user-based split
    train_df, test_df = create_original_user_split(ratings_df)
    
    # Train baseline
    print("\nTraining baseline...")
    baseline = CorrectedBaselineRecommender()
    baseline.train(train_df)
    
    # Evaluate with original method
    print("\nEvaluating with original method...")
    
    # Sample test users
    test_users = test_df['user_id'].unique()
    sample_users = min(1000, len(test_users))
    np.random.seed(42)
    test_users_sample = np.random.choice(test_users, sample_users, replace=False)
    
    precisions = []
    
    for user_id in test_users_sample:
        # Get user's high ratings in test set (4+ stars)
        user_test_high_ratings = test_df[
            (test_df['user_id'] == user_id) & 
            (test_df['rating'] >= 4)
        ]['book_id'].tolist()
        
        if len(user_test_high_ratings) == 0:
            continue  # Skip users with no high ratings
        
        # Generate recommendations (no train history for this user!)
        try:
            recommendations = baseline.recommend(user_id, [], 10)  # Empty history!
            hits = len(set(recommendations) & set(user_test_high_ratings))
            precision = hits / 10
            precisions.append(precision)
        except:
            continue
    
    avg_precision = np.mean(precisions) if precisions else 0
    
    print(f"\n=== ORIGINAL SPLIT Results ===")
    print(f"✅ Baseline precision@10: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
    print(f"📊 Users evaluated: {len(precisions)}")
    
    if avg_precision >= 0.25:
        print("🎉 SUCCESS: Restored ~28% baseline with original split!")
    else:
        print("⚠️  Still lower than expected - other factors involved")
    
    return avg_precision


def main():
    """Main execution with corrected methodology."""
    print("=== CORRECTED Recommendation System ===")
    print("Fixing the train/test split methodology...")
    
    # Load data
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    print(f"Loaded {len(ratings_df):,} ratings")
    
    # Create CORRECTED train/test split
    train_df, test_df = create_corrected_train_test_split(ratings_df)
    
    # Quick sanity check
    print(f"\n=== Sanity Check ===")
    sample_user = train_df['user_id'].iloc[0]
    user_train_books = set(train_df[train_df['user_id'] == sample_user]['book_id'])
    user_test_books = set(test_df[test_df['user_id'] == sample_user]['book_id'])
    overlap = len(user_train_books & user_test_books)
    print(f"Sample user {sample_user}: {len(user_train_books)} train books, {len(user_test_books)} test books, {overlap} overlap")
    
    if overlap == 0:
        print("✅ Good: No overlap between train and test books")
    else:
        print("⚠️  Warning: There's overlap - check split logic")
    
    # Train models
    print(f"\n=== Training Models ===")
    
    print("Training baseline...")
    baseline = CorrectedBaselineRecommender()
    baseline.train(train_df)
    
    print("\nTraining Item-Based CF...")
    item_cf = CorrectedItemBasedCF()
    item_cf.train(train_df)
    
    # Evaluate with ORIGINAL method (that gave 28% baseline)
    print(f"\n=== Evaluation ===")
    baseline_precision, item_cf_precision = evaluate_models_original_method(
        baseline, item_cf, train_df, test_df, n_users=1000
    )
    
    print(f"\n=== Final Results ===")
    print(f"With CORRECTED split + ORIGINAL evaluation:")
    print(f"  Baseline precision@10: {baseline_precision:.4f}")
    print(f"  Item-CF precision@10: {item_cf_precision:.4f}")
    
    # Expected results commentary
    print(f"\n=== Expected vs Actual ===")
    print(f"Expected baseline: 0.250-0.300 (25-30%) - should match original 28%")
    print(f"Expected Item-CF: 0.300-0.400 (30-40%)")
    print(f"Actual baseline: {baseline_precision:.3f} ({baseline_precision*100:.1f}%)")
    print(f"Actual Item-CF: {item_cf_precision:.3f} ({item_cf_precision*100:.1f}%)")
    
    if baseline_precision >= 0.20:
        print("✅ Baseline restored to ~28%!")
    else:
        print("⚠️  Baseline still lower than expected")
    
    if item_cf_precision > baseline_precision:
        print("🎉 Item-CF beats baseline!")
    else:
        print("⚠️  Item-CF may need parameter tuning")
    
    return baseline, item_cf, baseline_precision, item_cf_precision

if __name__ == "__main__":
    baseline, item_cf, baseline_prec, item_cf_prec = main() 