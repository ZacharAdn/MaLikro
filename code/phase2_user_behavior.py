"""
Phase 2: User Behavior Analysis
Book Recommendation System Project

Tasks:
- Rating distribution per user
- User activity levels (power users vs casual)
- Reading preferences patterns

Key Visualizations:
- User rating distribution histogram
- User activity level pie chart
- Top 10 most active users bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = Path('data')
RESULTS_PATH = Path('results')
TABLES_PATH = RESULTS_PATH / 'tables'
VIZ_PATH = RESULTS_PATH / 'visualizations'

def load_required_data():
    """Load the datasets needed for user behavior analysis"""
    print("="*50)
    print("PHASE 2: USER BEHAVIOR ANALYSIS")
    print("="*50)
    
    print("\n1. LOADING REQUIRED DATASETS")
    print("-" * 40)
    
    datasets = {}
    
    # Load ratings dataset (main for user analysis)
    ratings_path = DATA_PATH / 'ratings.csv'
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        datasets['ratings'] = ratings
        print(f"✓ Ratings loaded: {len(ratings):,} ratings")
    else:
        print("❌ ratings.csv not found!")
        return None
    
    # Load books for additional context
    books_path = DATA_PATH / 'books.csv'
    if books_path.exists():
        books = pd.read_csv(books_path)
        datasets['books'] = books
        print(f"✓ Books loaded: {len(books):,} books")
    
    # Load to_read for user engagement analysis
    to_read_path = DATA_PATH / 'to_read.csv'
    if to_read_path.exists():
        to_read = pd.read_csv(to_read_path)
        datasets['to_read'] = to_read
        print(f"✓ To-read loaded: {len(to_read):,} entries")
    
    return datasets

def analyze_rating_distribution(ratings_df):
    """Analyze how ratings are distributed per user"""
    print("\n2. RATING DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # User rating statistics
    user_stats = ratings_df.groupby('user_id')['rating'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).rename(columns={'count': 'total_ratings'})
    
    print(f"Total unique users: {len(user_stats):,}")
    print(f"Average ratings per user: {user_stats['total_ratings'].mean():.1f}")
    print(f"Median ratings per user: {user_stats['total_ratings'].median():.1f}")
    
    # Rating distribution summary
    rating_dist = ratings_df['rating'].value_counts().sort_index()
    print(f"\nOverall rating distribution:")
    for rating, count in rating_dist.items():
        pct = (count / len(ratings_df)) * 100
        print(f"  {rating} stars: {count:,} ({pct:.1f}%)")
    
    # User rating patterns
    user_rating_counts = user_stats['total_ratings']
    print(f"\nUser activity statistics:")
    print(f"  Min ratings per user: {user_rating_counts.min()}")
    print(f"  Max ratings per user: {user_rating_counts.max()}")
    print(f"  Users with 1 rating: {(user_rating_counts == 1).sum():,}")
    print(f"  Users with 100+ ratings: {(user_rating_counts >= 100).sum():,}")
    
    # Save user statistics
    user_stats_summary = pd.DataFrame({
        'Metric': ['Total Users', 'Avg Ratings per User', 'Median Ratings per User',
                  'Users with 1 rating', 'Users with 10+ ratings', 'Users with 100+ ratings'],
        'Value': [
            len(user_stats),
            round(user_stats['total_ratings'].mean(), 1),
            user_stats['total_ratings'].median(),
            (user_rating_counts == 1).sum(),
            (user_rating_counts >= 10).sum(),
            (user_rating_counts >= 100).sum()
        ]
    })
    
    user_stats_summary.to_csv(TABLES_PATH / 'user_statistics_summary.csv', index=False)
    
    return user_stats, rating_dist

def analyze_user_activity_levels(user_stats):
    """Categorize users by activity levels"""
    print("\n3. USER ACTIVITY LEVEL ANALYSIS")
    print("-" * 40)
    
    # Define activity levels
    def categorize_user_activity(rating_count):
        if rating_count == 1:
            return 'One-time (1 rating)'
        elif rating_count <= 5:
            return 'Casual (2-5 ratings)'
        elif rating_count <= 20:
            return 'Regular (6-20 ratings)'
        elif rating_count <= 100:
            return 'Active (21-100 ratings)'
        else:
            return 'Power User (100+ ratings)'
    
    user_stats['activity_level'] = user_stats['total_ratings'].apply(categorize_user_activity)
    
    # Activity level distribution
    activity_dist = user_stats['activity_level'].value_counts()
    activity_pct = (activity_dist / len(user_stats) * 100).round(1)
    
    print("User activity level distribution:")
    for level, count in activity_dist.items():
        pct = activity_pct[level]
        print(f"  {level}: {count:,} users ({pct}%)")
    
    # Top 10 most active users
    top_users = user_stats.nlargest(10, 'total_ratings')[['total_ratings', 'mean']]
    top_users.index.name = 'user_id'
    top_users.columns = ['Total_Ratings', 'Average_Rating']
    
    print(f"\nTop 10 most active users:")
    for i, (user_id, row) in enumerate(top_users.iterrows(), 1):
        print(f"  {i}. User {user_id}: {row['Total_Ratings']} ratings (avg: {row['Average_Rating']:.2f})")
    
    # Save results
    activity_summary = pd.DataFrame({
        'Activity_Level': activity_dist.index,
        'User_Count': activity_dist.values,
        'Percentage': activity_pct.values
    })
    activity_summary.to_csv(TABLES_PATH / 'user_activity_levels.csv', index=False)
    
    top_users.to_csv(TABLES_PATH / 'top_10_users.csv')
    
    return activity_dist, top_users

def analyze_reading_preferences(ratings_df, books_df=None):
    """Analyze user reading preferences"""
    print("\n4. READING PREFERENCES ANALYSIS")
    print("-" * 40)
    
    # Average rating per user
    user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean()
    
    print(f"User rating behavior:")
    print(f"  Users who rate strictly (avg < 3.0): {(user_avg_ratings < 3.0).sum():,}")
    print(f"  Users who rate moderately (3.0-4.0): {((user_avg_ratings >= 3.0) & (user_avg_ratings <= 4.0)).sum():,}")
    print(f"  Users who rate generously (avg > 4.0): {(user_avg_ratings > 4.0).sum():,}")
    
    # Rating variance per user (consistency)
    user_rating_variance = ratings_df.groupby('user_id')['rating'].std().fillna(0)
    
    print(f"\nUser rating consistency:")
    print(f"  Very consistent users (std < 0.5): {(user_rating_variance < 0.5).sum():,}")
    print(f"  Moderate variance (0.5-1.5): {((user_rating_variance >= 0.5) & (user_rating_variance <= 1.5)).sum():,}")
    print(f"  High variance users (std > 1.5): {(user_rating_variance > 1.5).sum():,}")
    
    # Save preferences analysis
    preferences_summary = pd.DataFrame({
        'Metric': ['Strict Raters (avg < 3.0)', 'Moderate Raters (3.0-4.0)', 'Generous Raters (avg > 4.0)',
                  'Consistent Raters (std < 0.5)', 'Moderate Variance (0.5-1.5)', 'High Variance (std > 1.5)'],
        'Count': [
            (user_avg_ratings < 3.0).sum(),
            ((user_avg_ratings >= 3.0) & (user_avg_ratings <= 4.0)).sum(),
            (user_avg_ratings > 4.0).sum(),
            (user_rating_variance < 0.5).sum(),
            ((user_rating_variance >= 0.5) & (user_rating_variance <= 1.5)).sum(),
            (user_rating_variance > 1.5).sum()
        ]
    })
    
    preferences_summary.to_csv(TABLES_PATH / 'user_preferences_summary.csv', index=False)
    
    return user_avg_ratings, user_rating_variance

def create_user_behavior_visualizations(ratings_df, user_stats, activity_dist, top_users, rating_dist):
    """Create visualizations for user behavior analysis"""
    print("\n5. CREATING USER BEHAVIOR VISUALIZATIONS")
    print("-" * 40)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. User rating distribution histogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    user_stats['total_ratings'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Ratings per User', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.yscale('log')  # Log scale due to long tail
    
    # 2. Overall rating distribution
    plt.subplot(2, 2, 2)
    rating_dist.plot(kind='bar', color='lightcoral', alpha=0.8)
    plt.title('Overall Rating Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=0)
    
    # 3. User activity level pie chart
    plt.subplot(2, 2, 3)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    activity_dist.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('User Activity Levels', fontsize=12, fontweight='bold')
    plt.ylabel('')
    
    # 4. Top 10 users bar chart
    plt.subplot(2, 2, 4)
    top_10_ratings = top_users['Total_Ratings'].head(10)
    bars = plt.bar(range(len(top_10_ratings)), top_10_ratings.values, color='gold', alpha=0.8)
    plt.title('Top 10 Most Active Users', fontsize=12, fontweight='bold')
    plt.xlabel('User Rank')
    plt.ylabel('Number of Ratings')
    plt.xticks(range(len(top_10_ratings)), [f'#{i+1}' for i in range(len(top_10_ratings))])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'user_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed user activity distribution
    plt.figure(figsize=(12, 6))
    
    # User ratings histogram with better binning
    plt.subplot(1, 2, 1)
    # Create custom bins for better visualization
    max_rating = user_stats['total_ratings'].max()
    bins = [0, 1, 5, 10, 20, 50, 100, 200, max_rating + 1]  # Add 1 to avoid duplicate edge
    labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '101-200', '200+']
    
    hist_data = pd.cut(user_stats['total_ratings'], bins=bins, labels=labels, include_lowest=True)
    hist_counts = hist_data.value_counts().sort_index()
    
    bars = plt.bar(range(len(hist_counts)), hist_counts.values, color='steelblue', alpha=0.7)
    plt.title('User Activity Distribution (Detailed)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.xticks(range(len(labels)), labels, rotation=45)
    
    # Add percentage labels
    total_users = len(user_stats)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / total_users) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # User average rating distribution
    plt.subplot(1, 2, 2)
    user_stats['mean'].hist(bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribution of User Average Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Average Rating per User')
    plt.ylabel('Number of Users')
    plt.axvline(user_stats['mean'].mean(), color='red', linestyle='--', 
                label=f'Overall Mean: {user_stats["mean"].mean():.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'detailed_user_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 User behavior visualizations saved to: {VIZ_PATH}/")

def main():
    """Main execution function for Phase 2"""
    try:
        # Load required data
        datasets = load_required_data()
        if not datasets:
            return None
        
        ratings_df = datasets['ratings']
        books_df = datasets.get('books')
        
        # Analyze rating distribution
        user_stats, rating_dist = analyze_rating_distribution(ratings_df)
        
        # Analyze user activity levels
        activity_dist, top_users = analyze_user_activity_levels(user_stats)
        
        # Analyze reading preferences
        user_avg_ratings, user_rating_variance = analyze_reading_preferences(ratings_df, books_df)
        
        # Create visualizations
        create_user_behavior_visualizations(ratings_df, user_stats, activity_dist, top_users, rating_dist)
        
        # Summary
        print("\n" + "="*50)
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Users analyzed: {len(user_stats):,}")
        print(f"📊 Ratings analyzed: {len(ratings_df):,}")
        print(f"📈 Visualizations created: 5")
        print(f"📋 Tables generated: 4")
        print(f"📁 Results saved in: {RESULTS_PATH}")
        
        return {
            'user_stats': user_stats,
            'activity_dist': activity_dist,
            'top_users': top_users,
            'user_avg_ratings': user_avg_ratings,
            'user_rating_variance': user_rating_variance
        }
        
    except Exception as e:
        print(f"\n❌ Error in Phase 2: {str(e)}")
        return None

if __name__ == "__main__":
    results = main() 