"""
Phase 2: Optimized User-Based Collaborative Filtering Recommender
Book Recommendation System Project

Optimized implementation with sampling and efficient similarity calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path('data_goodbooks_10k')
RESULTS_PATH = Path('results')

class OptimizedCollaborativeFiltering:
    """Optimized user-based collaborative filtering recommendation system"""
    
    def __init__(self, n_neighbors=30, min_common_ratings=3, max_users_for_similarity=2000):
        """
        Initialize collaborative filtering recommender
        
        Args:
            n_neighbors: Number of similar users to consider
            min_common_ratings: Minimum common ratings required for similarity calculation
            max_users_for_similarity: Maximum number of users to consider for similarity (for speed)
        """
        self.n_neighbors = n_neighbors
        self.min_common_ratings = min_common_ratings
        self.max_users_for_similarity = max_users_for_similarity
        self.user_ratings = {}  # Dictionary of user_id -> {book_id: rating}
        self.user_means = {}
        self.books_data = None
        self.power_users = []  # Users with many ratings
        
    def fit(self, ratings_df, books_df):
        """
        Train the model by organizing data for efficient similarity calculation
        
        Args:
            ratings_df: DataFrame with columns [user_id, book_id, rating]
            books_df: DataFrame with book metadata
        """
        print("Training Optimized Collaborative Filtering...")
        
        # Store books data
        self.books_data = books_df
        
        # Convert to dictionary format for faster lookup
        print("Converting to dictionary format...")
        for _, row in ratings_df.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            rating = row['rating']
            
            if user_id not in self.user_ratings:
                self.user_ratings[user_id] = {}
            self.user_ratings[user_id][book_id] = rating
        
        # Calculate user means
        print("Calculating user means...")
        for user_id, ratings in self.user_ratings.items():
            self.user_means[user_id] = np.mean(list(ratings.values()))
        
        # Identify power users (users with many ratings)
        user_rating_counts = {user_id: len(ratings) for user_id, ratings in self.user_ratings.items()}
        sorted_users = sorted(user_rating_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top users for similarity calculation
        self.power_users = [user_id for user_id, count in sorted_users[:self.max_users_for_similarity] if count >= 20]
        
        print(f"Total users: {len(self.user_ratings):,}")
        print(f"Power users for similarity: {len(self.power_users):,}")
        print(f"Average ratings per user: {np.mean(list(user_rating_counts.values())):.1f}")
        
    def _calculate_similarity(self, user1_ratings, user2_ratings):
        """Calculate Pearson correlation between two users"""
        # Find common books
        common_books = set(user1_ratings.keys()) & set(user2_ratings.keys())
        
        if len(common_books) < self.min_common_ratings:
            return 0.0
        
        # Get ratings for common books
        user1_common = [user1_ratings[book] for book in common_books]
        user2_common = [user2_ratings[book] for book in common_books]
        
        # Calculate Pearson correlation
        try:
            correlation, _ = pearsonr(user1_common, user2_common)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _find_similar_users(self, target_user_id):
        """Find similar users for a given target user"""
        if target_user_id not in self.user_ratings:
            return []
        
        target_ratings = self.user_ratings[target_user_id]
        
        if len(target_ratings) < self.min_common_ratings:
            return []
        
        similarities = []
        
        # Only calculate similarity with power users (for speed)
        for other_user_id in self.power_users:
            if other_user_id == target_user_id:
                continue
            
            other_ratings = self.user_ratings[other_user_id]
            similarity = self._calculate_similarity(target_ratings, other_ratings)
            
            if similarity > 0.1:  # Only keep meaningful similarities
                similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.n_neighbors]
    
    def _predict_rating(self, user_id, book_id, similar_users):
        """Predict rating for a user-book pair based on similar users"""
        if user_id not in self.user_means:
            return 3.0  # Default rating
        
        user_mean = self.user_means[user_id]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_id, similarity in similar_users:
            if similar_user_id in self.user_ratings:
                similar_user_ratings = self.user_ratings[similar_user_id]
                
                if book_id in similar_user_ratings:
                    similar_user_rating = similar_user_ratings[book_id]
                    similar_user_mean = self.user_means[similar_user_id]
                    
                    # Weight by similarity
                    weighted_sum += similarity * (similar_user_rating - similar_user_mean)
                    similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return user_mean
        
        predicted_rating = user_mean + (weighted_sum / similarity_sum)
        return max(1, min(5, predicted_rating))  # Clamp to 1-5 scale
    
    def recommend(self, user_id, ratings_df, n_recommendations=10):
        """
        Generate recommendations for a user
        
        Args:
            user_id: Target user ID
            ratings_df: Training ratings data (not used in optimized version)
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books and predicted ratings
        """
        # Find similar users
        similar_users = self._find_similar_users(user_id)
        
        if not similar_users:
            # Fallback to popularity-based recommendations
            return self._popularity_fallback(user_id, n_recommendations)
        
        # Get books already rated by user
        user_rated_books = set()
        if user_id in self.user_ratings:
            user_rated_books = set(self.user_ratings[user_id].keys())
        
        # Get candidate books from similar users
        candidate_books = set()
        for similar_user_id, similarity in similar_users:
            if similar_user_id in self.user_ratings:
                similar_user_ratings = self.user_ratings[similar_user_id]
                # Only consider books they rated highly
                liked_books = [book_id for book_id, rating in similar_user_ratings.items() if rating >= 4]
                candidate_books.update(liked_books)
        
        # Remove already rated books
        candidate_books = candidate_books - user_rated_books
        
        # Predict ratings for candidate books
        predictions = []
        for book_id in candidate_books:
            predicted_rating = self._predict_rating(user_id, book_id, similar_users)
            predictions.append((book_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n_recommendations]
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame(top_predictions, columns=['book_id', 'predicted_rating'])
        
        # Add book metadata
        if self.books_data is not None:
            recommendations = recommendations.merge(
                self.books_data[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']], 
                on='book_id', 
                how='left'
            )
        
        return recommendations
    
    def _popularity_fallback(self, user_id, n_recommendations):
        """Fallback to popularity-based recommendations"""
        # Calculate book popularity from all users
        book_ratings = {}
        book_counts = {}
        
        for user_ratings in self.user_ratings.values():
            for book_id, rating in user_ratings.items():
                if book_id not in book_ratings:
                    book_ratings[book_id] = []
                book_ratings[book_id].append(rating)
        
        # Calculate popularity scores
        book_scores = []
        for book_id, ratings in book_ratings.items():
            if len(ratings) >= 10:  # Minimum ratings threshold
                avg_rating = np.mean(ratings)
                rating_count = len(ratings)
                popularity_score = (rating_count * 0.3) + (avg_rating * 0.7)
                book_scores.append((book_id, popularity_score, avg_rating))
        
        # Sort by popularity and get top books
        book_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out already rated books
        user_rated_books = set()
        if user_id in self.user_ratings:
            user_rated_books = set(self.user_ratings[user_id].keys())
        
        recommendations = []
        for book_id, _, avg_rating in book_scores:
            if book_id not in user_rated_books:
                recommendations.append((book_id, avg_rating))
                if len(recommendations) >= n_recommendations:
                    break
        
        # Create recommendations DataFrame
        rec_df = pd.DataFrame(recommendations, columns=['book_id', 'predicted_rating'])
        
        # Add book metadata
        if self.books_data is not None:
            rec_df = rec_df.merge(
                self.books_data[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']], 
                on='book_id', 
                how='left'
            )
        
        return rec_df

def load_data():
    """Load the Goodbooks-10k dataset"""
    print("Loading Goodbooks-10k dataset...")
    
    # Load data files
    books_df = pd.read_csv(DATA_PATH / 'books.csv')
    ratings_df = pd.read_csv(DATA_PATH / 'ratings.csv')
    
    print(f"Books: {len(books_df):,}")
    print(f"Ratings: {len(ratings_df):,}")
    print(f"Users: {ratings_df['user_id'].nunique():,}")
    
    return ratings_df, books_df

def create_train_test_split(ratings_df, test_size=0.2, random_state=42):
    """Create train/test split for evaluation"""
    print(f"\nCreating train/test split ({test_size:.0%} test)...")
    
    # Random split by users
    users = ratings_df['user_id'].unique()
    train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)
    
    # Get ratings for train/test users
    train_ratings = ratings_df[ratings_df['user_id'].isin(train_users)]
    test_ratings = ratings_df[ratings_df['user_id'].isin(test_users)]
    
    print(f"Train ratings: {len(train_ratings):,}")
    print(f"Test ratings: {len(test_ratings):,}")
    print(f"Train users: {train_ratings['user_id'].nunique():,}")
    print(f"Test users: {test_ratings['user_id'].nunique():,}")
    
    return train_ratings, test_ratings

def evaluate_recommendations(recommender, test_ratings, train_ratings, books_df, n_recommendations=10, sample_users=200):
    """Evaluate recommendation quality"""
    print("\n3. EVALUATING RECOMMENDATIONS")
    print("-" * 40)
    
    # Sample test users for evaluation
    test_users = test_ratings['user_id'].unique()
    if len(test_users) > sample_users:
        test_users_sample = np.random.choice(test_users, sample_users, replace=False)
    else:
        test_users_sample = test_users
    
    print(f"Evaluating on {len(test_users_sample)} test users...")
    
    precision_scores = []
    coverage_books = set()
    successful_recommendations = 0
    fallback_count = 0
    
    for i, user_id in enumerate(test_users_sample):
        if i % 50 == 0:
            print(f"Processing user {i+1}/{len(test_users_sample)}")
        
        # Get user's actual high ratings in test set (4+ stars)
        user_test_high_ratings = test_ratings[
            (test_ratings['user_id'] == user_id) & 
            (test_ratings['rating'] >= 4)
        ]['book_id'].tolist()
        
        if len(user_test_high_ratings) == 0:
            continue  # Skip users with no high ratings in test set
        
        # Generate recommendations
        try:
            recommendations = recommender.recommend(user_id, train_ratings, n_recommendations)
            
            if len(recommendations) == 0:
                continue
            
            recommended_books = recommendations['book_id'].tolist()
            successful_recommendations += 1
            
            # Check if we used fallback (popularity-based)
            if user_id not in recommender.user_ratings or not recommender._find_similar_users(user_id):
                fallback_count += 1
            
            # Calculate precision@K
            hits = len(set(recommended_books) & set(user_test_high_ratings))
            precision = hits / min(len(recommended_books), n_recommendations)
            precision_scores.append(precision)
            
            # Track coverage
            coverage_books.update(recommended_books)
            
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue
    
    # Calculate metrics
    metrics = {
        'precision_at_k': np.mean(precision_scores) if precision_scores else 0,
        'coverage': len(coverage_books) / len(books_df),
        'successful_recommendations': successful_recommendations,
        'total_test_users': len(test_users_sample),
        'fallback_count': fallback_count,
        'collaborative_filtering_rate': (successful_recommendations - fallback_count) / successful_recommendations if successful_recommendations > 0 else 0,
        'precision_scores': precision_scores
    }
    
    return metrics

def create_visual_examples(recommender, train_ratings, test_ratings, books_df, n_examples=5):
    """Create visual examples of recommendations"""
    print("\n4. CREATING VISUAL EXAMPLES")
    print("-" * 40)
    
    # Get users with different rating counts
    user_rating_counts = train_ratings.groupby('user_id').size()
    
    # Power users (100+ ratings)
    power_users = user_rating_counts[user_rating_counts >= 100].index.tolist()
    
    # Casual users (20-99 ratings)
    casual_users = user_rating_counts[
        (user_rating_counts >= 20) & (user_rating_counts < 100)
    ].index.tolist()
    
    examples = []
    
    # Get examples from different user types
    sample_users = []
    if len(power_users) > 0:
        sample_users.extend(np.random.choice(power_users, min(3, len(power_users)), replace=False))
    if len(casual_users) > 0:
        sample_users.extend(np.random.choice(casual_users, min(2, len(casual_users)), replace=False))
    
    for user_id in sample_users[:n_examples]:
        # Get user's rating history
        user_history = train_ratings[train_ratings['user_id'] == user_id].merge(
            books_df[['book_id', 'title', 'authors']], on='book_id', how='left'
        ).sort_values('rating', ascending=False)
        
        # Generate recommendations
        try:
            recommendations = recommender.recommend(user_id, train_ratings, 10)
            
            # Check if collaborative filtering was used
            similar_users = recommender._find_similar_users(user_id)
            method_used = "Collaborative Filtering" if similar_users else "Popularity Fallback"
            
            example = {
                'user_id': user_id,
                'rating_count': len(user_history),
                'average_rating': user_history['rating'].mean(),
                'method_used': method_used,
                'similarity_count': len(similar_users),
                'top_rated_books': user_history.head(5)[['title', 'authors', 'rating']].to_dict('records'),
                'recommendations': recommendations[['title', 'authors', 'predicted_rating']].to_dict('records')
            }
            
            examples.append(example)
            
        except Exception as e:
            print(f"Error creating example for user {user_id}: {e}")
            continue
    
    return examples

def print_results(metrics, examples):
    """Print evaluation results and examples"""
    print("\n" + "="*60)
    print("OPTIMIZED COLLABORATIVE FILTERING RESULTS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Precision@10: {metrics['precision_at_k']:.3f} ({metrics['precision_at_k']*100:.1f}%)")
    print(f"Coverage: {metrics['coverage']:.3f} ({metrics['coverage']*100:.1f}%)")
    print(f"Successful Recommendations: {metrics['successful_recommendations']}/{metrics['total_test_users']}")
    print(f"Collaborative Filtering Rate: {metrics['collaborative_filtering_rate']:.1%}")
    print(f"Fallback to Popularity: {metrics['fallback_count']}/{metrics['successful_recommendations']}")
    
    # Compare with baseline (28% precision)
    baseline_precision = 0.28
    improvement = (metrics['precision_at_k'] - baseline_precision) / baseline_precision * 100
    print(f"\n📈 IMPROVEMENT OVER BASELINE:")
    print(f"Baseline Precision@10: {baseline_precision:.3f} ({baseline_precision*100:.1f}%)")
    print(f"Improvement: {improvement:+.1f}%")
    
    print(f"\n👥 EXAMPLE RECOMMENDATIONS:")
    print("-" * 40)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. User {example['user_id']} ({example['rating_count']} ratings, avg: {example['average_rating']:.1f}★)")
        print(f"   Method: {example['method_used']} ({example['similarity_count']} similar users)")
        
        print("   Top-rated books:")
        for book in example['top_rated_books'][:3]:
            print(f"   • {book['title']} by {book['authors']} ({book['rating']}★)")
        
        print("   Recommendations:")
        for book in example['recommendations'][:5]:
            print(f"   → {book['title']} by {book['authors']} (predicted: {book['predicted_rating']:.1f}★)")

def main():
    """Main execution function"""
    print("OPTIMIZED USER-BASED COLLABORATIVE FILTERING")
    print("="*50)
    
    # Load data
    ratings_df, books_df = load_data()
    
    # Create train/test split
    train_ratings, test_ratings = create_train_test_split(ratings_df, test_size=0.2)
    
    # Initialize and train model
    print("\n2. TRAINING MODEL")
    print("-" * 40)
    
    recommender = OptimizedCollaborativeFiltering(
        n_neighbors=30,
        min_common_ratings=3,
        max_users_for_similarity=2000  # Limit for speed
    )
    
    recommender.fit(train_ratings, books_df)
    
    # Evaluate model
    metrics = evaluate_recommendations(
        recommender, test_ratings, train_ratings, books_df, 
        n_recommendations=10, sample_users=200
    )
    
    # Create visual examples
    examples = create_visual_examples(
        recommender, train_ratings, test_ratings, books_df, n_examples=5
    )
    
    # Print results
    print_results(metrics, examples)
    
    # Save results
    results_file = RESULTS_PATH / 'collaborative_filtering_optimized_results.txt'
    with open(results_file, 'w') as f:
        f.write("OPTIMIZED COLLABORATIVE FILTERING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Precision@10: {metrics['precision_at_k']:.3f}\n")
        f.write(f"Coverage: {metrics['coverage']:.3f}\n")
        f.write(f"Collaborative Filtering Rate: {metrics['collaborative_filtering_rate']:.1%}\n")
        f.write(f"Successful Recommendations: {metrics['successful_recommendations']}/{metrics['total_test_users']}\n")
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main() 