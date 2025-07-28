"""
Phase 1: Baseline Popularity-Based Recommender
Book Recommendation System Project

Implementation of simple popularity baseline to establish performance threshold
Algorithm: popularity_score = (rating_count * 0.3) + (average_rating * 0.7)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path('data')
RESULTS_PATH = Path('results')
TABLES_PATH = RESULTS_PATH / 'tables'
VIZ_PATH = RESULTS_PATH / 'visualizations'

class PopularityBasedRecommender:
    """Simple popularity-based recommendation system"""
    
    def __init__(self, rating_weight=0.7, count_weight=0.3):
        """
        Initialize recommender with weighting parameters
        
        Args:
            rating_weight: Weight for average rating (default 0.7)
            count_weight: Weight for rating count (default 0.3)
        """
        self.rating_weight = rating_weight
        self.count_weight = count_weight
        self.book_scores = None
        self.books_data = None
        
    def fit(self, ratings_df, books_df):
        """
        Train the model by calculating popularity scores for all books
        
        Args:
            ratings_df: DataFrame with columns [user_id, book_id, rating]
            books_df: DataFrame with book metadata
        """
        print("Training Popularity-Based Recommender...")
        
        # Calculate book statistics
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        }).round(3)
        
        # Flatten column names
        book_stats.columns = ['rating_count', 'average_rating']
        book_stats = book_stats.reset_index()
        
        # Normalize rating count to 0-5 scale to match rating scale
        max_count = book_stats['rating_count'].max()
        book_stats['normalized_count'] = (book_stats['rating_count'] / max_count) * 5
        
        # Calculate popularity score
        book_stats['popularity_score'] = (
            book_stats['normalized_count'] * self.count_weight + 
            book_stats['average_rating'] * self.rating_weight
        )
        
        # Merge with book metadata
        self.book_scores = book_stats.merge(
            books_df[['book_id', 'title', 'authors']], 
            on='book_id', 
            how='left'
        ).sort_values('popularity_score', ascending=False)
        
        self.books_data = books_df
        
        print(f"✓ Trained on {len(book_stats)} books")
        print(f"✓ Score range: {book_stats['popularity_score'].min():.2f} - {book_stats['popularity_score'].max():.2f}")
        
    def recommend(self, user_id, ratings_df, n_recommendations=10):
        """
        Generate recommendations for a specific user
        
        Args:
            user_id: Target user ID
            ratings_df: Full ratings dataset
            n_recommendations: Number of books to recommend
            
        Returns:
            DataFrame with recommended books
        """
        if self.book_scores is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get books already rated by user
        user_rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])
        
        # Filter out already rated books
        available_books = self.book_scores[~self.book_scores['book_id'].isin(user_rated_books)]
        
        # Return top N recommendations
        recommendations = available_books.head(n_recommendations).copy()
        recommendations['user_id'] = user_id
        
        return recommendations[['user_id', 'book_id', 'title', 'authors', 'popularity_score', 'average_rating', 'rating_count']]

def load_data():
    """Load ratings and books data"""
    print("="*50)
    print("BASELINE MODEL: POPULARITY RECOMMENDER")
    print("="*50)
    
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    # Load ratings
    ratings_path = DATA_PATH / 'ratings.csv'
    if not ratings_path.exists():
        raise FileNotFoundError("ratings.csv not found!")
    
    ratings_df = pd.read_csv(ratings_path)
    print(f"✓ Ratings loaded: {len(ratings_df):,} ratings")
    
    # Load books
    books_path = DATA_PATH / 'books.csv'
    if not books_path.exists():
        raise FileNotFoundError("books.csv not found!")
    
    books_df = pd.read_csv(books_path)
    print(f"✓ Books loaded: {len(books_df):,} books")
    
    return ratings_df, books_df

def create_train_test_split(ratings_df, test_size=0.2, random_state=42):
    """
    Create train/test split by users (80% users for training, 20% for testing)
    
    Args:
        ratings_df: Full ratings dataset
        test_size: Fraction of users for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_ratings, test_ratings DataFrames
    """
    print("\n2. CREATING TRAIN/TEST SPLIT")
    print("-" * 40)
    
    # Get unique users
    unique_users = ratings_df['user_id'].unique()
    
    # Split users
    train_users, test_users = train_test_split(
        unique_users, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train/test datasets
    train_ratings = ratings_df[ratings_df['user_id'].isin(train_users)]
    test_ratings = ratings_df[ratings_df['user_id'].isin(test_users)]
    
    print(f"✓ Train users: {len(train_users):,} ({len(train_ratings):,} ratings)")
    print(f"✓ Test users: {len(test_users):,} ({len(test_ratings):,} ratings)")
    print(f"✓ Split ratio: {len(train_ratings)/len(ratings_df):.1%} train, {len(test_ratings)/len(ratings_df):.1%} test")
    
    return train_ratings, test_ratings, train_users, test_users

def evaluate_recommendations(recommender, test_ratings, train_ratings, books_df, n_recommendations=10, sample_users=1000):
    """
    Evaluate recommendation quality using Precision@K and other metrics
    
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
    
    # Sample test users for evaluation (for speed)
    test_users = test_ratings['user_id'].unique()
    if len(test_users) > sample_users:
        test_users_sample = np.random.choice(test_users, sample_users, replace=False)
    else:
        test_users_sample = test_users
    
    print(f"Evaluating on {len(test_users_sample)} test users...")
    
    precision_scores = []
    coverage_books = set()
    recommendation_popularity = []
    
    for user_id in test_users_sample:
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
            recommended_books = recommendations['book_id'].tolist()
            
            # Calculate precision@K
            hits = len(set(recommended_books) & set(user_test_high_ratings))
            precision = hits / min(len(recommended_books), n_recommendations)
            precision_scores.append(precision)
            
            # Track coverage
            coverage_books.update(recommended_books)
            
            # Track recommendation popularity
            rec_popularity = recommendations['popularity_score'].mean()
            recommendation_popularity.append(rec_popularity)
            
        except Exception as e:
            continue  # Skip problematic users
    
    # Calculate metrics
    metrics = {
        'precision_at_k': np.mean(precision_scores) if precision_scores else 0,
        'coverage': len(coverage_books) / len(books_df),
        'avg_recommendation_popularity': np.mean(recommendation_popularity) if recommendation_popularity else 0,
        'users_evaluated': len(precision_scores),
        'total_unique_recommendations': len(coverage_books)
    }
    
    print(f"✓ Precision@{n_recommendations}: {metrics['precision_at_k']:.3f}")
    print(f"✓ Coverage: {metrics['coverage']:.1%} ({len(coverage_books)} unique books recommended)")
    print(f"✓ Avg recommendation popularity: {metrics['avg_recommendation_popularity']:.2f}")
    print(f"✓ Users evaluated: {metrics['users_evaluated']}")
    
    return metrics

def analyze_model_performance(recommender, metrics, books_df):
    """Analyze and visualize model performance"""
    print("\n4. ANALYZING MODEL PERFORMANCE")
    print("-" * 40)
    
    # Get top books by popularity score
    top_books = recommender.book_scores.head(20).copy()
    
    print("Top 10 most recommended books:")
    for i, (_, book) in enumerate(top_books.head(10).iterrows(), 1):
        print(f"  {i}. {book['title']} by {book['authors']}")
        print(f"     Score: {book['popularity_score']:.2f} (avg: {book['average_rating']:.1f}, count: {book['rating_count']})")
    
    # Save results
    results_summary = pd.DataFrame([{
        'Metric': 'Precision@10',
        'Value': f"{metrics['precision_at_k']:.3f}",
        'Description': 'Fraction of top-10 recommendations that user would rate 4+ stars'
    }, {
        'Metric': 'Coverage', 
        'Value': f"{metrics['coverage']:.1%}",
        'Description': 'Percentage of catalog that gets recommended'
    }, {
        'Metric': 'Avg Popularity',
        'Value': f"{metrics['avg_recommendation_popularity']:.2f}",
        'Description': 'Average popularity score of recommendations'
    }, {
        'Metric': 'Users Evaluated',
        'Value': f"{metrics['users_evaluated']}",
        'Description': 'Number of test users used for evaluation'
    }])
    
    results_summary.to_csv(TABLES_PATH / 'baseline_model_results.csv', index=False)
    top_books.to_csv(TABLES_PATH / 'top_recommended_books.csv', index=False)
    
    print(f"\n📊 Results saved to: {TABLES_PATH}/")

def create_model_visualizations(recommender, metrics):
    """Create visualizations for model analysis"""
    print("\n5. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top 15 books by popularity score
    top_books = recommender.book_scores.head(15)
    ax1.barh(range(len(top_books)), top_books['popularity_score'])
    ax1.set_yticks(range(len(top_books)))
    ax1.set_yticklabels([f"{title[:30]}..." if len(title) > 30 else title 
                        for title in top_books['title']], fontsize=8)
    ax1.set_xlabel('Popularity Score')
    ax1.set_title('Top 15 Books by Popularity Score', fontweight='bold')
    ax1.invert_yaxis()
    
    # 2. Rating count vs Average rating scatter
    sample_books = recommender.book_scores.sample(min(1000, len(recommender.book_scores)))
    scatter = ax2.scatter(sample_books['rating_count'], sample_books['average_rating'], 
                         c=sample_books['popularity_score'], cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Number of Ratings')
    ax2.set_ylabel('Average Rating')
    ax2.set_title('Rating Count vs Average Rating', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Popularity Score')
    
    # 3. Popularity score distribution
    ax3.hist(recommender.book_scores['popularity_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(recommender.book_scores['popularity_score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {recommender.book_scores["popularity_score"].mean():.2f}')
    ax3.set_xlabel('Popularity Score')
    ax3.set_ylabel('Number of Books')
    ax3.set_title('Distribution of Popularity Scores', fontweight='bold')
    ax3.legend()
    
    # 4. Model performance metrics
    metric_names = ['Precision@10', 'Coverage', 'Avg Popularity']
    metric_values = [metrics['precision_at_k'], metrics['coverage'], 
                    metrics['avg_recommendation_popularity']/5]  # Normalize popularity to 0-1
    
    bars = ax4.bar(metric_names, metric_values, color=['lightcoral', 'lightgreen', 'lightblue'])
    ax4.set_ylabel('Score')
    ax4.set_title('Model Performance Metrics', fontweight='bold')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, [metrics['precision_at_k'], metrics['coverage'], 
                                metrics['avg_recommendation_popularity']]):
        height = bar.get_height()
        if bar == bars[2]:  # Popularity score
            label = f'{value:.2f}'
        else:
            label = f'{value:.3f}'
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'baseline_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualizations saved to: {VIZ_PATH}/")

def demonstrate_recommendations(recommender, train_ratings, books_df, n_users=3):
    """Show example recommendations for sample users"""
    print("\n6. EXAMPLE RECOMMENDATIONS")
    print("-" * 40)
    
    sample_users = train_ratings['user_id'].unique()[:n_users]
    
    for user_id in sample_users:
        print(f"\nRecommendations for User {user_id}:")
        
        # Show user's reading history (sample)
        user_history = train_ratings[train_ratings['user_id'] == user_id].merge(
            books_df[['book_id', 'title', 'authors']], on='book_id'
        ).sort_values('rating', ascending=False).head(3)
        
        print("  Recent high-rated books:")
        for _, book in user_history.iterrows():
            print(f"    • {book['title']} ({book['rating']} stars)")
        
        # Generate recommendations
        recommendations = recommender.recommend(user_id, train_ratings, 5)
        print("  Recommended books:")
        for i, (_, book) in enumerate(recommendations.iterrows(), 1):
            print(f"    {i}. {book['title']} by {book['authors']}")
            print(f"       Score: {book['popularity_score']:.2f}")

def main():
    """Main execution function for baseline model"""
    try:
        # Load data
        ratings_df, books_df = load_data()
        
        # Create train/test split
        train_ratings, test_ratings, train_users, test_users = create_train_test_split(ratings_df)
        
        # Initialize and train model
        recommender = PopularityBasedRecommender(rating_weight=0.7, count_weight=0.3)
        recommender.fit(train_ratings, books_df)
        
        # Evaluate model
        metrics = evaluate_recommendations(recommender, test_ratings, train_ratings, books_df)
        
        # Analyze performance
        analyze_model_performance(recommender, metrics, books_df)
        
        # Create visualizations
        create_model_visualizations(recommender, metrics)
        
        # Show example recommendations
        demonstrate_recommendations(recommender, train_ratings, books_df)
        
        # Final summary
        print("\n" + "="*50)
        print("BASELINE MODEL COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"🎯 Precision@10: {metrics['precision_at_k']:.3f}")
        print(f"📊 Coverage: {metrics['coverage']:.1%}")
        print(f"⭐ Avg Popularity: {metrics['avg_recommendation_popularity']:.2f}")
        print(f"👥 Users Evaluated: {metrics['users_evaluated']}")
        print(f"📁 Results saved in: {RESULTS_PATH}")
        
        return recommender, metrics
        
    except Exception as e:
        print(f"\n❌ Error in baseline model: {str(e)}")
        return None, None

if __name__ == "__main__":
    recommender, metrics = main() 