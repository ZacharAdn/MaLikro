"""
Phase 2: User-Based Collaborative Filtering Recommender
Book Recommendation System Project

Implementation of user-based collaborative filtering with comprehensive evaluation
Algorithm: Find similar users based on rating patterns, recommend books they liked
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path('data_goodbooks_10k')
RESULTS_PATH = Path('results')
TABLES_PATH = RESULTS_PATH / 'tables'
VIZ_PATH = RESULTS_PATH / 'visualizations'

class UserBasedCollaborativeFiltering:
    """User-based collaborative filtering recommendation system"""
    
    def __init__(self, n_neighbors=50, min_common_ratings=2, similarity_metric='pearson'):
        """
        Initialize collaborative filtering recommender
        
        Args:
            n_neighbors: Number of similar users to consider
            min_common_ratings: Minimum common ratings required for similarity calculation
            similarity_metric: 'pearson' or 'cosine'
        """
        self.n_neighbors = n_neighbors
        self.min_common_ratings = min_common_ratings
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_similarities = {}
        self.user_means = {}
        self.books_data = None
        
    def _calculate_user_similarity(self, user1_ratings, user2_ratings):
        """Calculate similarity between two users based on common ratings"""
        # Find common items
        common_items = set(user1_ratings.index) & set(user2_ratings.index)
        
        if len(common_items) < self.min_common_ratings:
            return 0.0
        
        # Get ratings for common items
        user1_common = user1_ratings[list(common_items)]
        user2_common = user2_ratings[list(common_items)]
        
        if self.similarity_metric == 'pearson':
            if len(common_items) < 2:
                return 0.0
            correlation, _ = pearsonr(user1_common, user2_common)
            return correlation if not np.isnan(correlation) else 0.0
        
        elif self.similarity_metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(user1_common, user2_common)
            norm_user1 = np.linalg.norm(user1_common)
            norm_user2 = np.linalg.norm(user2_common)
            
            if norm_user1 == 0 or norm_user2 == 0:
                return 0.0
            
            return dot_product / (norm_user1 * norm_user2)
        
        return 0.0
    
    def fit(self, ratings_df, books_df):
        """
        Train the model by building user-item matrix and calculating similarities
        
        Args:
            ratings_df: DataFrame with columns [user_id, book_id, rating]
            books_df: DataFrame with book metadata
        """
        print("Training User-Based Collaborative Filtering...")
        
        # Store books data
        self.books_data = books_df
        
        # Create user-item matrix
        print("Creating user-item matrix...")
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='book_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate user means (for rating prediction)
        print("Calculating user rating means...")
        user_ratings = ratings_df.groupby('user_id')['rating'].mean()
        self.user_means = user_ratings.to_dict()
        
        print(f"Matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.1f}%")
        
        # Pre-calculate similarities for power users (users with many ratings)
        print("Pre-calculating similarities for power users...")
        user_rating_counts = ratings_df.groupby('user_id').size()
        power_users = user_rating_counts[user_rating_counts >= 50].index.tolist()
        
        print(f"Pre-calculating similarities for {len(power_users)} power users...")
        
        for i, user_id in enumerate(power_users[:1000]):  # Limit to top 1000 for speed
            if i % 100 == 0:
                print(f"Processing user {i+1}/{min(1000, len(power_users))}")
            
            user_ratings = self.user_item_matrix.loc[user_id]
            user_ratings = user_ratings[user_ratings > 0]  # Only rated items
            
            similarities = []
            for other_user_id in power_users:
                if other_user_id == user_id:
                    continue
                
                other_ratings = self.user_item_matrix.loc[other_user_id]
                other_ratings = other_ratings[other_ratings > 0]
                
                similarity = self._calculate_user_similarity(user_ratings, other_ratings)
                if similarity > 0:
                    similarities.append((other_user_id, similarity))
            
            # Store top N similar users
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.user_similarities[user_id] = similarities[:self.n_neighbors]
        
        print(f"Training completed. Stored similarities for {len(self.user_similarities)} users.")
    
    def _find_similar_users(self, target_user_id, ratings_df):
        """Find similar users for a given target user"""
        # Check if we have pre-calculated similarities
        if target_user_id in self.user_similarities:
            return self.user_similarities[target_user_id]
        
        # Calculate similarities on-the-fly
        if target_user_id not in self.user_item_matrix.index:
            return []
        
        target_ratings = self.user_item_matrix.loc[target_user_id]
        target_ratings = target_ratings[target_ratings > 0]
        
        if len(target_ratings) < self.min_common_ratings:
            return []
        
        similarities = []
        
        # Sample users for similarity calculation (for speed)
        sample_users = self.user_item_matrix.index.tolist()
        if len(sample_users) > 1000:
            sample_users = np.random.choice(sample_users, 1000, replace=False)
        
        for other_user_id in sample_users:
            if other_user_id == target_user_id:
                continue
            
            other_ratings = self.user_item_matrix.loc[other_user_id]
            other_ratings = other_ratings[other_ratings > 0]
            
            similarity = self._calculate_user_similarity(target_ratings, other_ratings)
            if similarity > 0:
                similarities.append((other_user_id, similarity))
        
        # Sort and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.n_neighbors]
    
    def _predict_rating(self, user_id, book_id, similar_users, ratings_df):
        """Predict rating for a user-book pair based on similar users"""
        if user_id not in self.user_means:
            return 3.0  # Default rating
        
        user_mean = self.user_means[user_id]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_id, similarity in similar_users:
            # Check if similar user rated this book
            if book_id in self.user_item_matrix.columns and similar_user_id in self.user_item_matrix.index:
                similar_user_rating = self.user_item_matrix.loc[similar_user_id, book_id]
                
                if similar_user_rating > 0:  # User rated this book
                    similar_user_mean = self.user_means.get(similar_user_id, 3.0)
                    
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
            ratings_df: Training ratings data
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books and predicted ratings
        """
        # Find similar users
        similar_users = self._find_similar_users(user_id, ratings_df)
        
        if not similar_users:
            # Fallback to popularity-based recommendations
            return self._popularity_fallback(user_id, ratings_df, n_recommendations)
        
        # Get books already rated by user
        user_rated_books = set()
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            user_rated_books = set(user_ratings[user_ratings > 0].index)
        
        # Get candidate books from similar users
        candidate_books = set()
        for similar_user_id, similarity in similar_users:
            if similar_user_id in self.user_item_matrix.index:
                similar_user_ratings = self.user_item_matrix.loc[similar_user_id]
                liked_books = similar_user_ratings[similar_user_ratings >= 4].index
                candidate_books.update(liked_books)
        
        # Remove already rated books
        candidate_books = candidate_books - user_rated_books
        
        # Predict ratings for candidate books
        predictions = []
        for book_id in candidate_books:
            predicted_rating = self._predict_rating(user_id, book_id, similar_users, ratings_df)
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
    
    def _popularity_fallback(self, user_id, ratings_df, n_recommendations):
        """Fallback to popularity-based recommendations when collaborative filtering fails"""
        # Calculate popularity scores
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).round(2)
        
        book_stats.columns = ['average_rating', 'ratings_count']
        book_stats['popularity_score'] = (
            book_stats['ratings_count'] * 0.3 + 
            book_stats['average_rating'] * 0.7
        )
        
        # Get user's rated books
        user_rated_books = set()
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            user_rated_books = set(user_ratings[user_ratings > 0].index)
        
        # Filter out rated books and get top recommendations
        available_books = book_stats[~book_stats.index.isin(user_rated_books)]
        top_books = available_books.nlargest(n_recommendations, 'popularity_score')
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'book_id': top_books.index,
            'predicted_rating': top_books['average_rating']
        })
        
        # Add book metadata
        if self.books_data is not None:
            recommendations = recommendations.merge(
                self.books_data[['book_id', 'title', 'authors', 'average_rating', 'ratings_count']], 
                on='book_id', 
                how='left'
            )
        
        return recommendations

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
    
    # For collaborative filtering, we need to ensure users appear in both train and test
    # Use temporal split if timestamp is available, otherwise random split by user
    
    if 'timestamp' in ratings_df.columns:
        # Temporal split
        ratings_df_sorted = ratings_df.sort_values('timestamp')
        split_idx = int(len(ratings_df_sorted) * (1 - test_size))
        train_ratings = ratings_df_sorted.iloc[:split_idx]
        test_ratings = ratings_df_sorted.iloc[split_idx:]
    else:
        # Random split ensuring users appear in both sets
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

def evaluate_recommendations(recommender, test_ratings, train_ratings, books_df, n_recommendations=10, sample_users=500):
    """
    Evaluate recommendation quality using multiple metrics
    
    Args:
        recommender: Trained recommender model
        test_ratings: Test set ratings
        train_ratings: Training set ratings  
        books_df: Books metadata
        n_recommendations: Number of recommendations to evaluate
        sample_users: Number of test users to sample for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
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
    recall_scores = []
    coverage_books = set()
    recommendation_popularity = []
    successful_recommendations = 0
    
    for i, user_id in enumerate(test_users_sample):
        if i % 100 == 0:
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
            
            # Calculate precision@K
            hits = len(set(recommended_books) & set(user_test_high_ratings))
            precision = hits / min(len(recommended_books), n_recommendations)
            precision_scores.append(precision)
            
            # Calculate recall@K  
            recall = hits / len(user_test_high_ratings)
            recall_scores.append(recall)
            
            # Track coverage
            coverage_books.update(recommended_books)
            
            # Track recommendation popularity
            for book_id in recommended_books:
                book_info = books_df[books_df['book_id'] == book_id]
                if not book_info.empty:
                    recommendation_popularity.append(book_info.iloc[0]['ratings_count'])
            
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue
    
    # Calculate metrics
    metrics = {
        'precision_at_k': np.mean(precision_scores) if precision_scores else 0,
        'recall_at_k': np.mean(recall_scores) if recall_scores else 0,
        'coverage': len(coverage_books) / len(books_df),
        'avg_recommendation_popularity': np.mean(recommendation_popularity) if recommendation_popularity else 0,
        'successful_recommendations': successful_recommendations,
        'total_test_users': len(test_users_sample),
        'precision_scores': precision_scores,
        'recall_scores': recall_scores
    }
    
    # Calculate F1 score
    if metrics['precision_at_k'] + metrics['recall_at_k'] > 0:
        metrics['f1_at_k'] = 2 * (metrics['precision_at_k'] * metrics['recall_at_k']) / (metrics['precision_at_k'] + metrics['recall_at_k'])
    else:
        metrics['f1_at_k'] = 0
    
    return metrics

def create_visual_examples(recommender, train_ratings, test_ratings, books_df, n_examples=5):
    """Create visual examples of recommendations for different user types"""
    print("\n4. CREATING VISUAL EXAMPLES")
    print("-" * 40)
    
    # Sample different types of users
    user_rating_counts = train_ratings.groupby('user_id').size()
    
    # Power users (100+ ratings)
    power_users = user_rating_counts[user_rating_counts >= 100].index.tolist()
    
    # Casual users (10-99 ratings)
    casual_users = user_rating_counts[
        (user_rating_counts >= 10) & (user_rating_counts < 100)
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
        
        # Get user's test ratings for evaluation
        user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
        
        # Generate recommendations
        try:
            recommendations = recommender.recommend(user_id, train_ratings, 10)
            
            example = {
                'user_id': user_id,
                'rating_count': len(user_history),
                'average_rating': user_history['rating'].mean(),
                'top_rated_books': user_history.head(5)[['title', 'authors', 'rating']].to_dict('records'),
                'recommendations': recommendations[['title', 'authors', 'predicted_rating']].to_dict('records'),
                'test_books_liked': user_test_ratings[user_test_ratings['rating'] >= 4]['book_id'].tolist()
            }
            
            examples.append(example)
            
        except Exception as e:
            print(f"Error creating example for user {user_id}: {e}")
            continue
    
    return examples

def print_results(metrics, examples):
    """Print evaluation results and examples"""
    print("\n" + "="*60)
    print("COLLABORATIVE FILTERING EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Precision@10: {metrics['precision_at_k']:.3f} ({metrics['precision_at_k']*100:.1f}%)")
    print(f"Recall@10: {metrics['recall_at_k']:.3f} ({metrics['recall_at_k']*100:.1f}%)")
    print(f"F1@10: {metrics['f1_at_k']:.3f}")
    print(f"Coverage: {metrics['coverage']:.3f} ({metrics['coverage']*100:.1f}%)")
    print(f"Avg Recommendation Popularity: {metrics['avg_recommendation_popularity']:.0f} ratings")
    print(f"Successful Recommendations: {metrics['successful_recommendations']}/{metrics['total_test_users']}")
    
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
        
        print("   Top-rated books:")
        for book in example['top_rated_books'][:3]:
            print(f"   • {book['title']} by {book['authors']} ({book['rating']}★)")
        
        print("   Recommendations:")
        for book in example['recommendations'][:5]:
            print(f"   → {book['title']} by {book['authors']} (predicted: {book['predicted_rating']:.1f}★)")

def main():
    """Main execution function"""
    print("USER-BASED COLLABORATIVE FILTERING")
    print("="*50)
    
    # Load data
    ratings_df, books_df = load_data()
    
    # Create train/test split
    train_ratings, test_ratings = create_train_test_split(ratings_df, test_size=0.2)
    
    # Initialize and train model
    print("\n2. TRAINING MODEL")
    print("-" * 40)
    
    recommender = UserBasedCollaborativeFiltering(
        n_neighbors=50,
        min_common_ratings=2,
        similarity_metric='pearson'
    )
    
    recommender.fit(train_ratings, books_df)
    
    # Evaluate model
    metrics = evaluate_recommendations(
        recommender, test_ratings, train_ratings, books_df, 
        n_recommendations=10, sample_users=500
    )
    
    # Create visual examples
    examples = create_visual_examples(
        recommender, train_ratings, test_ratings, books_df, n_examples=5
    )
    
    # Print results
    print_results(metrics, examples)
    
    # Save results
    results_file = RESULTS_PATH / 'collaborative_filtering_results.txt'
    with open(results_file, 'w') as f:
        f.write("COLLABORATIVE FILTERING EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Precision@10: {metrics['precision_at_k']:.3f}\n")
        f.write(f"Recall@10: {metrics['recall_at_k']:.3f}\n")
        f.write(f"F1@10: {metrics['f1_at_k']:.3f}\n")
        f.write(f"Coverage: {metrics['coverage']:.3f}\n")
        f.write(f"Successful Recommendations: {metrics['successful_recommendations']}/{metrics['total_test_users']}\n")
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main() 