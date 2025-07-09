"""
Phase 1: Data Quality & Structure Analysis
Book Recommendation System Project

Tasks:
- Load and validate all CSV files
- Check for missing values, duplicates, and inconsistencies
- Verify data types and relationships between files
- Document basic statistics (rows, columns, memory usage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up paths
DATA_PATH = Path('data')
RESULTS_PATH = Path('results')
TABLES_PATH = RESULTS_PATH / 'tables'
VIZ_PATH = RESULTS_PATH / 'visualizations'

# Create directories if they don't exist
TABLES_PATH.mkdir(parents=True, exist_ok=True)
VIZ_PATH.mkdir(parents=True, exist_ok=True)

def load_and_validate_data():
    """Load all CSV files and perform basic validation"""
    print("="*50)
    print("PHASE 1: DATA QUALITY & STRUCTURE ANALYSIS")
    print("="*50)
    
    # Define expected files
    files_info = {
        'ratings': 'ratings.csv',
        'books': 'books.csv', 
        'book_tags': 'book_tags.csv',
        'tags': 'tags.csv',
        'to_read': 'to_read.csv'
    }
    
    datasets = {}
    basic_stats = []
    
    print("\n1. LOADING AND VALIDATING DATA FILES")
    print("-" * 40)
    
    for name, filename in files_info.items():
        filepath = DATA_PATH / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)
            datasets[name] = df
            
            # Basic statistics
            stats = {
                'Dataset': name,
                'Filename': filename,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory_MB': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                'File_Size_MB': round(filepath.stat().st_size / 1024**2, 2)
            }
            basic_stats.append(stats)
            
            print(f"  ✓ {name}: {len(df):,} rows × {len(df.columns)} columns")
        else:
            print(f"  ✗ {filename} not found!")
    
    # Save basic statistics
    stats_df = pd.DataFrame(basic_stats)
    stats_df.to_csv(TABLES_PATH / 'basic_statistics.csv', index=False)
    print(f"\n📊 Basic statistics saved to: {TABLES_PATH / 'basic_statistics.csv'}")
    
    return datasets, stats_df

def analyze_data_quality(datasets):
    """Analyze data quality issues across all datasets"""
    print("\n2. DATA QUALITY ANALYSIS")
    print("-" * 40)
    
    quality_report = []
    
    for name, df in datasets.items():
        print(f"\nAnalyzing {name} dataset:")
        
        # Missing values
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        print(f"  Missing values:")
        for col in df.columns:
            if missing_count[col] > 0:
                print(f"    {col}: {missing_count[col]:,} ({missing_pct[col]}%)")
        
        # Duplicates
        duplicate_count = df.duplicated().sum()
        print(f"  Duplicate rows: {duplicate_count:,}")
        
        # Data types
        print(f"  Data types:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Add to quality report
        for col in df.columns:
            quality_report.append({
                'Dataset': name,
                'Column': col,
                'Data_Type': str(df[col].dtype),
                'Missing_Count': missing_count[col],
                'Missing_Percentage': missing_pct[col],
                'Unique_Values': df[col].nunique(),
                'Non_Null_Count': df[col].count()
            })
    
    # Save quality report
    quality_df = pd.DataFrame(quality_report)
    quality_df.to_csv(TABLES_PATH / 'data_quality_report.csv', index=False)
    print(f"\n📊 Data quality report saved to: {TABLES_PATH / 'data_quality_report.csv'}")
    
    return quality_df

def analyze_relationships(datasets):
    """Analyze relationships between datasets"""
    print("\n3. DATASET RELATIONSHIPS ANALYSIS")
    print("-" * 40)
    
    relationships = []
    
    if 'ratings' in datasets and 'books' in datasets:
        # Check book_id relationships
        ratings_books = set(datasets['ratings']['book_id'].unique())
        books_ids = set(datasets['books']['book_id'].unique())
        
        common_books = len(ratings_books.intersection(books_ids))
        ratings_only = len(ratings_books - books_ids)
        books_only = len(books_ids - ratings_books)
        
        print(f"Books relationship (ratings ↔ books):")
        print(f"  Books in both: {common_books:,}")
        print(f"  Only in ratings: {ratings_only:,}")
        print(f"  Only in books: {books_only:,}")
        
        relationships.append({
            'Relationship': 'ratings_books',
            'Common': common_books,
            'Left_Only': ratings_only,
            'Right_Only': books_only
        })
    
    if 'book_tags' in datasets and 'tags' in datasets:
        # Check tag relationships
        book_tags_tags = set(datasets['book_tags']['tag_id'].unique())
        tags_ids = set(datasets['tags']['tag_id'].unique())
        
        common_tags = len(book_tags_tags.intersection(tags_ids))
        book_tags_only = len(book_tags_tags - tags_ids)
        tags_only = len(tags_ids - book_tags_tags)
        
        print(f"\nTags relationship (book_tags ↔ tags):")
        print(f"  Tags in both: {common_tags:,}")
        print(f"  Only in book_tags: {book_tags_only:,}")
        print(f"  Only in tags: {tags_only:,}")
        
        relationships.append({
            'Relationship': 'book_tags_tags',
            'Common': common_tags,
            'Left_Only': book_tags_only,
            'Right_Only': tags_only
        })
    
    # Save relationships analysis
    if relationships:
        rel_df = pd.DataFrame(relationships)
        rel_df.to_csv(TABLES_PATH / 'dataset_relationships.csv', index=False)
        print(f"\n📊 Relationships analysis saved to: {TABLES_PATH / 'dataset_relationships.csv'}")
    
    return relationships

def create_quality_visualizations(datasets, stats_df):
    """Create visualizations for data quality analysis"""
    print("\n4. CREATING DATA QUALITY VISUALIZATIONS")
    print("-" * 40)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Dataset sizes comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Rows comparison
    ax1.bar(stats_df['Dataset'], stats_df['Rows'])
    ax1.set_title('Dataset Sizes (Number of Rows)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Rows')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(stats_df['Rows']):
        ax1.text(i, v + max(stats_df['Rows'])*0.01, f'{v:,}', ha='center', fontweight='bold')
    
    # Memory usage comparison
    ax2.bar(stats_df['Dataset'], stats_df['Memory_MB'], color='orange')
    ax2.set_title('Memory Usage by Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(stats_df['Memory_MB']):
        ax2.text(i, v + max(stats_df['Memory_MB'])*0.01, f'{v}MB', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Missing values heatmap
    if 'ratings' in datasets:
        df = datasets['ratings']
        missing_matrix = df.isnull()
        
        if missing_matrix.any().any():
            plt.figure(figsize=(12, 8))
            sns.heatmap(missing_matrix, cbar=True, cmap='viridis')
            plt.title('Missing Values Pattern - Ratings Dataset', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(VIZ_PATH / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"📈 Visualizations saved to: {VIZ_PATH}/")

def main():
    """Main execution function"""
    try:
        # Load and validate data
        datasets, stats_df = load_and_validate_data()
        
        # Analyze data quality
        quality_df = analyze_data_quality(datasets)
        
        # Analyze relationships
        relationships = analyze_relationships(datasets)
        
        # Create visualizations
        create_quality_visualizations(datasets, stats_df)
        
        # Summary
        print("\n" + "="*50)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Files analyzed: {len(datasets)}")
        print(f"📈 Visualizations created: 2")
        print(f"📋 Reports generated: 3")
        print(f"📁 Results saved in: {RESULTS_PATH}")
        
        return datasets, stats_df, quality_df
        
    except Exception as e:
        print(f"\n❌ Error in Phase 1: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    datasets, stats_df, quality_df = main() 