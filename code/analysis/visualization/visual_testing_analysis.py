#!/usr/bin/env python3
"""
Visual Testing Analysis - Compare algorithms with specific user examples
to understand why Item-Based CF has low precision.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from item_cf_optimized import OptimizedItemBasedCF
import os
from datetime import datetime
import random

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BaselinePopularityRecommender:
    """Simple popularity-based recommender for comparison."""
    
    def __init__(self):
        self.popular_books = None
        
    def train(self, ratings_df):
        """Train on ratings data."""
        # Calculate popularity score
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['count', 'mean']
        }).round(3)
        book_stats.columns = ['rating_count', 'avg_rating']
        book_stats = book_stats.reset_index()
        
        # Popularity score: weighted combination
        book_stats['popularity_score'] = (
            book_stats['rating_count'] * 0.3 + 
            book_stats['avg_rating'] * 0.7
        )
        
        # Store sorted by popularity
        self.popular_books = book_stats.sort_values(
            'popularity_score', ascending=False
        )['book_id'].tolist()
        
    def recommend(self, user_id, user_ratings, n_recommendations=10):
        """Generate recommendations by popularity."""
        # Get books user hasn't rated
        rated_books = set(user_ratings['book_id'])
        
        recommendations = []
        for book_id in self.popular_books:
            if book_id not in rated_books:
                recommendations.append(book_id)
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations

def create_results_folder():
    """Create dedicated results folder for visual testing."""
    folder = 'results/visual_testing_analysis'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/visualizations', exist_ok=True)
    os.makedirs(f'{folder}/tables', exist_ok=True)
    return folder

def get_user_profile(user_id, ratings_df, books_df=None):
    """Get detailed user profile information."""
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
    
    if len(user_ratings) == 0:
        return None
        
    # Basic stats
    profile = {
        'user_id': user_id,
        'total_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'rating_std': user_ratings['rating'].std(),
        'min_rating': user_ratings['rating'].min(),
        'max_rating': user_ratings['rating'].max()
    }
    
    # Get top rated books
    top_books = user_ratings[user_ratings['rating'] >= 4].sort_values(
        'rating', ascending=False
    ).head(10)
    
    profile['top_rated_books'] = []
    for _, row in top_books.iterrows():
        book_info = f"Book {row['book_id']} ({row['rating']}★)"
        if books_df is not None:
            book_data = books_df[books_df['book_id'] == row['book_id']]
            if len(book_data) > 0:
                title = book_data.iloc[0].get('title', f'Book {row["book_id"]}')
                book_info = f"{title} ({row['rating']}★)"
        profile['top_rated_books'].append(book_info)
    
    # Rating distribution
    rating_dist = user_ratings['rating'].value_counts().sort_index()
    profile['rating_distribution'] = rating_dist.to_dict()
    
    return profile

def analyze_recommendations_quality(user_id, user_ratings, recommendations, test_ratings):
    """Analyze the quality of recommendations for a specific user."""
    # Get actual books user rated highly in test set
    actual_high_rated = set(
        test_ratings[
            (test_ratings['user_id'] == user_id) & 
            (test_ratings['rating'] >= 4)
        ]['book_id']
    )
    
    # Calculate hits
    recommended_set = set(recommendations)
    hits = recommended_set & actual_high_rated
    
    # Analysis
    analysis = {
        'total_recommendations': len(recommendations),
        'actual_high_rated_books': len(actual_high_rated),
        'hits': len(hits),
        'precision': len(hits) / len(recommendations) if len(recommendations) > 0 else 0,
        'recall': len(hits) / len(actual_high_rated) if len(actual_high_rated) > 0 else 0,
        'hit_books': list(hits),
        'missed_opportunities': list(actual_high_rated - recommended_set),
        'irrelevant_recommendations': list(recommended_set - actual_high_rated)
    }
    
    return analysis

def select_diverse_users(ratings_df, n_users=20):
    """Select diverse users for analysis."""
    # Calculate user statistics
    user_stats = ratings_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std']
    }).round(3)
    user_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
    user_stats = user_stats.reset_index()
    user_stats = user_stats.dropna()
    
    # Define user segments
    segments = {
        'power_users': user_stats[user_stats['rating_count'] >= 100],
        'casual_users': user_stats[
            (user_stats['rating_count'] >= 20) & 
            (user_stats['rating_count'] < 100)
        ],
        'light_users': user_stats[
            (user_stats['rating_count'] >= 10) & 
            (user_stats['rating_count'] < 20)
        ],
        'generous_raters': user_stats[user_stats['avg_rating'] >= 4.0],
        'critical_raters': user_stats[user_stats['avg_rating'] <= 3.5],
        'diverse_raters': user_stats[user_stats['rating_std'] >= 1.0]
    }
    
    # Sample from each segment
    selected_users = []
    users_per_segment = max(1, n_users // len(segments))
    
    for segment_name, segment_data in segments.items():
        if len(segment_data) > 0:
            sample_size = min(users_per_segment, len(segment_data))
            sampled = segment_data.sample(sample_size, random_state=42)
            for _, user in sampled.iterrows():
                selected_users.append({
                    'user_id': user['user_id'],
                    'segment': segment_name,
                    'rating_count': user['rating_count'],
                    'avg_rating': user['avg_rating'],
                    'rating_std': user['rating_std']
                })
    
    # Remove duplicates and limit to n_users
    seen_users = set()
    unique_users = []
    for user in selected_users:
        if user['user_id'] not in seen_users:
            seen_users.add(user['user_id'])
            unique_users.append(user)
        if len(unique_users) >= n_users:
            break
    
    return unique_users[:n_users]

def run_visual_testing(train_df, test_df, books_df=None, n_examples=20):
    """Run comprehensive visual testing analysis."""
    print("=== Visual Testing Analysis ===")
    
    # Train both models
    print("Training models...")
    
    # Baseline model
    baseline = BaselinePopularityRecommender()
    baseline.train(train_df)
    
    # Item-based CF model
    item_cf = OptimizedItemBasedCF()
    item_cf.train(train_df)
    
    print("Models trained successfully!")
    
    # Select diverse users
    print(f"Selecting {n_examples} diverse users for analysis...")
    selected_users = select_diverse_users(train_df, n_examples)
    print(f"Selected {len(selected_users)} users from different segments")
    
    # Analyze each user
    results = []
    
    print("Analyzing user examples...")
    for i, user_info in enumerate(selected_users):
        user_id = user_info['user_id']
        
        print(f"Analyzing user {i+1}/{len(selected_users)}: {user_id}")
        
        # Get user profile
        profile = get_user_profile(user_id, train_df, books_df)
        if profile is None:
            continue
            
        # Get user's training ratings
        user_train_ratings = train_df[train_df['user_id'] == user_id]
        user_test_ratings = test_df[test_df['user_id'] == user_id]
        
        # Generate recommendations
        baseline_recs = baseline.recommend(user_id, user_train_ratings, 10)
        
        # Item-based CF recommendations
        if user_id in item_cf.user_id_to_idx:
            item_cf_recs = item_cf.get_recommendations_fast(user_id, 10)
        else:
            item_cf_recs = baseline_recs  # Fallback
        
        # Analyze recommendation quality
        baseline_analysis = analyze_recommendations_quality(
            user_id, user_train_ratings, baseline_recs, test_df
        )
        item_cf_analysis = analyze_recommendations_quality(
            user_id, user_train_ratings, item_cf_recs, test_df
        )
        
        # Store results
        result = {
            'user_info': user_info,
            'profile': profile,
            'baseline_recommendations': baseline_recs,
            'item_cf_recommendations': item_cf_recs,
            'baseline_analysis': baseline_analysis,
            'item_cf_analysis': item_cf_analysis,
            'test_books_count': len(user_test_ratings)
        }
        
        results.append(result)
    
    return results

def create_visualizations(results, results_folder):
    """Create comprehensive visualizations of the analysis."""
    print("Creating visualizations...")
    
    # Extract data for visualization
    user_segments = [r['user_info']['segment'] for r in results]
    baseline_precisions = [r['baseline_analysis']['precision'] for r in results]
    item_cf_precisions = [r['item_cf_analysis']['precision'] for r in results]
    user_rating_counts = [r['profile']['total_ratings'] for r in results]
    user_avg_ratings = [r['profile']['avg_rating'] for r in results]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Precision comparison by algorithm
    algorithms = ['Baseline\n(Popularity)', 'Item-Based\nCollaborative Filtering']
    avg_precisions = [np.mean(baseline_precisions), np.mean(item_cf_precisions)]
    
    bars = axes[0,0].bar(algorithms, avg_precisions, color=['lightcoral', 'skyblue'], alpha=0.8)
    axes[0,0].set_ylabel('Average Precision@10')
    axes[0,0].set_title('Algorithm Performance Comparison')
    axes[0,0].set_ylim(0, max(avg_precisions) * 1.2)
    
    for bar, precision in zip(bars, avg_precisions):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{precision:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. User-by-user precision comparison
    x = range(len(results))
    axes[0,1].scatter(x, baseline_precisions, alpha=0.7, label='Baseline', color='lightcoral')
    axes[0,1].scatter(x, item_cf_precisions, alpha=0.7, label='Item-Based CF', color='skyblue')
    axes[0,1].set_xlabel('User Index')
    axes[0,1].set_ylabel('Precision@10')
    axes[0,1].set_title('Per-User Precision Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precision by user segment
    segments = list(set(user_segments))
    segment_baseline = [np.mean([baseline_precisions[i] for i, seg in enumerate(user_segments) if seg == s]) for s in segments]
    segment_item_cf = [np.mean([item_cf_precisions[i] for i, seg in enumerate(user_segments) if seg == s]) for s in segments]
    
    x_seg = np.arange(len(segments))
    width = 0.35
    
    axes[0,2].bar(x_seg - width/2, segment_baseline, width, label='Baseline', alpha=0.8, color='lightcoral')
    axes[0,2].bar(x_seg + width/2, segment_item_cf, width, label='Item-Based CF', alpha=0.8, color='skyblue')
    axes[0,2].set_ylabel('Average Precision@10')
    axes[0,2].set_title('Performance by User Segment')
    axes[0,2].set_xticks(x_seg)
    axes[0,2].set_xticklabels(segments, rotation=45)
    axes[0,2].legend()
    
    # 4. Precision vs user activity
    axes[1,0].scatter(user_rating_counts, baseline_precisions, alpha=0.7, label='Baseline', color='lightcoral')
    axes[1,0].scatter(user_rating_counts, item_cf_precisions, alpha=0.7, label='Item-Based CF', color='skyblue')
    axes[1,0].set_xlabel('Number of Training Ratings')
    axes[1,0].set_ylabel('Precision@10')
    axes[1,0].set_title('Precision vs User Activity')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Precision vs rating generosity
    axes[1,1].scatter(user_avg_ratings, baseline_precisions, alpha=0.7, label='Baseline', color='lightcoral')
    axes[1,1].scatter(user_avg_ratings, item_cf_precisions, alpha=0.7, label='Item-Based CF', color='skyblue')
    axes[1,1].set_xlabel('Average Rating Given')
    axes[1,1].set_ylabel('Precision@10')
    axes[1,1].set_title('Precision vs Rating Generosity')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Success/failure distribution
    baseline_wins = sum(1 for i in range(len(results)) if baseline_precisions[i] > item_cf_precisions[i])
    item_cf_wins = sum(1 for i in range(len(results)) if item_cf_precisions[i] > baseline_precisions[i])
    ties = len(results) - baseline_wins - item_cf_wins
    
    categories = ['Baseline\nWins', 'Item-CF\nWins', 'Ties']
    counts = [baseline_wins, item_cf_wins, ties]
    colors = ['lightcoral', 'skyblue', 'lightgray']
    
    axes[1,2].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1,2].set_title('Algorithm Wins Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{results_folder}/visualizations/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_examples_table(results, results_folder):
    """Create detailed table with user examples."""
    print("Creating detailed examples table...")
    
    # Create summary table
    summary_data = []
    
    for result in results:
        user_info = result['user_info']
        profile = result['profile']
        baseline_analysis = result['baseline_analysis']
        item_cf_analysis = result['item_cf_analysis']
        
        summary_data.append({
            'user_id': user_info['user_id'],
            'segment': user_info['segment'],
            'train_ratings': profile['total_ratings'],
            'avg_rating': profile['avg_rating'],
            'test_books': result['test_books_count'],
            'baseline_precision': baseline_analysis['precision'],
            'baseline_hits': baseline_analysis['hits'],
            'item_cf_precision': item_cf_analysis['precision'],
            'item_cf_hits': item_cf_analysis['hits'],
            'improvement': item_cf_analysis['precision'] - baseline_analysis['precision'],
            'top_rated_books': '; '.join(profile['top_rated_books'][:3])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{results_folder}/tables/user_examples_summary.csv', index=False)
    
    # Create detailed examples for top/bottom performers
    detailed_examples = []
    
    # Sort by improvement
    sorted_results = sorted(results, key=lambda x: x['item_cf_analysis']['precision'] - x['baseline_analysis']['precision'], reverse=True)
    
    # Top 5 improvements and bottom 5
    interesting_results = sorted_results[:5] + sorted_results[-5:]
    
    for result in interesting_results:
        user_info = result['user_info']
        profile = result['profile']
        baseline_analysis = result['baseline_analysis']
        item_cf_analysis = result['item_cf_analysis']
        
        example = {
            'user_id': user_info['user_id'],
            'segment': user_info['segment'],
            'profile_summary': f"{profile['total_ratings']} ratings, avg {profile['avg_rating']:.2f}",
            'baseline_recommendations': result['baseline_recommendations'][:5],
            'item_cf_recommendations': result['item_cf_recommendations'][:5],
            'baseline_precision': baseline_analysis['precision'],
            'item_cf_precision': item_cf_analysis['precision'],
            'improvement': item_cf_analysis['precision'] - baseline_analysis['precision'],
            'analysis': 'Improvement' if item_cf_analysis['precision'] > baseline_analysis['precision'] else 'Degradation'
        }
        
        detailed_examples.append(example)
    
    detailed_df = pd.DataFrame(detailed_examples)
    detailed_df.to_csv(f'{results_folder}/tables/detailed_examples.csv', index=False)
    
    return summary_df, detailed_df

def create_diagnosis_report(results, results_folder):
    """Create diagnostic report to understand low precision."""
    print("Creating diagnostic report...")
    
    # Overall statistics
    total_users = len(results)
    baseline_precisions = [r['baseline_analysis']['precision'] for r in results]
    item_cf_precisions = [r['item_cf_analysis']['precision'] for r in results]
    
    avg_baseline = np.mean(baseline_precisions)
    avg_item_cf = np.mean(item_cf_precisions)
    
    # Analyze failure cases
    low_precision_users = [r for r in results if r['item_cf_analysis']['precision'] < 0.1]
    zero_precision_users = [r for r in results if r['item_cf_analysis']['precision'] == 0.0]
    
    # Test data analysis
    test_book_counts = [r['test_books_count'] for r in results]
    users_with_test_data = [r for r in results if r['test_books_count'] > 0]
    
    report_text = f"""
# Visual Testing Analysis - Diagnostic Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance Summary
- **Total users analyzed**: {total_users}
- **Baseline average precision@10**: {avg_baseline:.4f}
- **Item-CF average precision@10**: {avg_item_cf:.4f}
- **Improvement**: {avg_item_cf - avg_baseline:.4f} ({((avg_item_cf/avg_baseline-1)*100):.1f}%)

## Key Findings

### Test Data Issues
- **Users with test data**: {len(users_with_test_data)}/{total_users}
- **Average test books per user**: {np.mean(test_book_counts):.1f}
- **Users with zero test books**: {total_users - len(users_with_test_data)}

### Performance Issues
- **Users with zero precision**: {len(zero_precision_users)}/{total_users} ({len(zero_precision_users)/total_users*100:.1f}%)
- **Users with precision < 0.1**: {len(low_precision_users)}/{total_users} ({len(low_precision_users)/total_users*100:.1f}%)

### Potential Causes of Low Precision
1. **Data Split Problem**: Simple random split may not preserve temporal order
2. **Cold Start Issues**: Many users may not be in training set properly
3. **Evaluation Method**: Precision@10 may be too strict for sparse test data
4. **Algorithm Issues**: Item-based CF may need parameter tuning

## Segment Analysis
"""
    
    # Add segment analysis
    segments = {}
    for result in results:
        segment = result['user_info']['segment']
        if segment not in segments:
            segments[segment] = []
        segments[segment].append(result)
    
    for segment, segment_results in segments.items():
        baseline_avg = np.mean([r['baseline_analysis']['precision'] for r in segment_results])
        item_cf_avg = np.mean([r['item_cf_analysis']['precision'] for r in segment_results])
        
        report_text += f"""
### {segment.replace('_', ' ').title()}
- **Count**: {len(segment_results)} users
- **Baseline precision**: {baseline_avg:.4f}
- **Item-CF precision**: {item_cf_avg:.4f}
- **Improvement**: {item_cf_avg - baseline_avg:.4f}
"""
    
    report_text += f"""
## Recommendations for Next Steps

1. **Fix Data Split**:
   - Use temporal split instead of random split
   - Ensure users have sufficient data in both train/test

2. **Improve Evaluation**:
   - Use recall@10 and F1@10 in addition to precision
   - Consider implicit feedback evaluation

3. **Algorithm Tuning**:
   - Adjust similarity thresholds
   - Experiment with different similarity metrics
   - Tune top-k similar items parameter

4. **Baseline Comparison**:
   - Current baseline might be artificially high
   - Need to verify baseline implementation

## Files Generated
- `comprehensive_analysis.png`: Visual comparison charts
- `user_examples_summary.csv`: Summary of all {total_users} users
- `detailed_examples.csv`: Top/bottom 10 performers
- `diagnosis_report.txt`: This diagnostic report
"""
    
    # Save report
    with open(f'{results_folder}/diagnosis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"Diagnostic report saved to {results_folder}/diagnosis_report.txt")

def main():
    """Main execution function."""
    print("=== Visual Testing Analysis - Algorithm Comparison ===")
    
    # Create results folder
    results_folder = create_results_folder()
    print(f"Results will be saved to: {results_folder}")
    
    # Load data
    print("Loading data...")
    ratings_df = pd.read_csv('data_goodbooks_10k/ratings.csv')
    
    books_df = None
    if os.path.exists('data_goodbooks_10k/books.csv'):
        books_df = pd.read_csv('data_goodbooks_10k/books.csv')
    
    # Create train/test split (same as before for consistency)
    print("Creating train/test split...")
    shuffled = ratings_df.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]
    
    print(f"Train: {len(train_df):,} ratings")
    print(f"Test: {len(test_df):,} ratings")
    
    # Run visual testing
    results = run_visual_testing(train_df, test_df, books_df, n_examples=20)
    
    # Create visualizations and analysis
    create_visualizations(results, results_folder)
    summary_df, detailed_df = create_detailed_examples_table(results, results_folder)
    create_diagnosis_report(results, results_folder)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {results_folder}")
    print(f"Analyzed {len(results)} users across different segments")
    
    # Print quick summary
    baseline_avg = np.mean([r['baseline_analysis']['precision'] for r in results])
    item_cf_avg = np.mean([r['item_cf_analysis']['precision'] for r in results])
    
    print(f"\nQuick Summary:")
    print(f"Baseline precision@10: {baseline_avg:.4f}")
    print(f"Item-CF precision@10: {item_cf_avg:.4f}")
    print(f"Improvement: {item_cf_avg - baseline_avg:.4f}")

if __name__ == "__main__":
    main()
