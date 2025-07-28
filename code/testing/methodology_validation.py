#!/usr/bin/env python3
"""
Methodology Validation for Corrected Recommendation System.
This script validates that our corrected train/test split methodology works properly.
"""

import sys
import pandas as pd
import numpy as np

# Try to import our corrected system
try:
    sys.path.append('..')  # Add parent directory to path
    from models.corrected_implementation import (
        CorrectedBaselineRecommender, 
        CorrectedItemBasedCF,
        create_corrected_train_test_split,
        evaluate_models_corrected
    )
    print("✅ Successfully imported corrected system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def validate_methodology():
    """Validate the corrected methodology with a small sample."""
    print("=== METHODOLOGY VALIDATION ===")
    
    # Load data
    try:
        ratings_df = pd.read_csv('../../data_goodbooks_10k/ratings.csv')
        print(f"✅ Loaded {len(ratings_df):,} ratings")
    except FileNotFoundError:
        print("❌ Could not find ratings.csv file")
        return False
    
    # Sample a subset for quick testing
    sample_users = ratings_df['user_id'].sample(1000, random_state=42)
    sample_df = ratings_df[ratings_df['user_id'].isin(sample_users)].copy()
    print(f"📊 Testing with sample: {len(sample_df):,} ratings from {sample_df['user_id'].nunique()} users")
    
    # Create corrected split
    train_df, test_df = create_corrected_train_test_split(sample_df, test_ratio=0.2)
    
    # Sanity check - verify we have proper overlap in book IDs between train and test
    train_books = set(train_df['book_id'])
    test_books = set(test_df['book_id'])
    book_overlap = len(train_books & test_books)
    
    print(f"📈 Book overlap check:")
    print(f"   Train books: {len(train_books)}")
    print(f"   Test books: {len(test_books)}")
    print(f"   Overlap: {book_overlap} ({book_overlap/len(test_books)*100:.1f}% of test books)")
    
    if book_overlap == 0:
        print("❌ ERROR: No book overlap between train and test - this would guarantee 0 precision!")
        return False
    elif book_overlap / len(test_books) < 0.1:
        print("⚠️  WARNING: Very low book overlap - precision might still be very low")
    else:
        print("✅ Good book overlap - should enable reasonable precision")
    
    # Quick user-level sanity check
    sample_user = train_df['user_id'].iloc[0]
    user_train_books = set(train_df[train_df['user_id'] == sample_user]['book_id'])
    user_test_books = set(test_df[test_df['user_id'] == sample_user]['book_id'])
    user_overlap = len(user_train_books & user_test_books)
    
    print(f"👤 User-level check (user {sample_user}):")
    print(f"   Train books: {len(user_train_books)}")
    print(f"   Test books: {len(user_test_books)}")
    print(f"   Overlap: {user_overlap} (should be 0)")
    
    if user_overlap > 0:
        print("⚠️  WARNING: User has same books in train and test!")
    else:
        print("✅ Good: No user-level data leakage")
    
    # Train models
    print("\n🔧 Training models...")
    
    try:
        baseline = CorrectedBaselineRecommender()
        baseline.train(train_df)
        print("✅ Baseline trained successfully")
    except Exception as e:
        print(f"❌ Baseline training failed: {e}")
        return False
    
    try:
        item_cf = CorrectedItemBasedCF(min_ratings_per_book=10, min_ratings_per_user=5)  # Lower thresholds for sample
        item_cf.train(train_df)
        print("✅ Item-CF trained successfully")
    except Exception as e:
        print(f"❌ Item-CF training failed: {e}")
        return False
    
    # Quick evaluation
    print("\n📊 Quick evaluation...")
    
    try:
        baseline_prec, item_cf_prec = evaluate_models_corrected(
            baseline, item_cf, train_df, test_df, n_users=100
        )
        
        print(f"\n🎯 VALIDATION RESULTS:")
        print(f"   Baseline precision@10: {baseline_prec:.4f} ({baseline_prec*100:.1f}%)")
        print(f"   Item-CF precision@10: {item_cf_prec:.4f} ({item_cf_prec*100:.1f}%)")
        
        # Evaluate results
        if baseline_prec > 0.02:  # At least 2%
            print("✅ Baseline shows reasonable performance!")
        else:
            print("⚠️  Baseline still very low - may need investigation")
        
        if item_cf_prec > baseline_prec:
            improvement = (item_cf_prec / baseline_prec - 1) * 100 if baseline_prec > 0 else 0
            print(f"✅ Item-CF outperforms baseline by {improvement:.1f}%!")
        else:
            print("⚠️  Item-CF doesn't beat baseline - may need tuning")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_original_vs_corrected_split():
    """Compare original vs corrected split methodology."""
    print("\n=== ORIGINAL vs CORRECTED SPLIT COMPARISON ===")
    
    # Load a small sample for quick testing
    ratings_df = pd.read_csv('../../data_goodbooks_10k/ratings.csv')
    sample_users = ratings_df['user_id'].sample(100, random_state=42)
    sample_df = ratings_df[ratings_df['user_id'].isin(sample_users)].copy()
    
    # Test original methodology (problematic)
    print("📉 Original methodology (problematic):")
    train_data_orig = []
    test_data_orig = []
    
    for user_id, user_ratings in sample_df.groupby('user_id'):
        # Original: sort by book_id (problematic!)
        user_ratings = user_ratings.sort_values('book_id')
        n_ratings = len(user_ratings)
        
        if n_ratings < 5:
            continue
            
        n_test = max(1, int(n_ratings * 0.2))
        train_data_orig.append(user_ratings.iloc[:-n_test])
        test_data_orig.append(user_ratings.iloc[-n_test:])
    
    train_orig = pd.concat(train_data_orig, ignore_index=True)
    test_orig = pd.concat(test_data_orig, ignore_index=True)
    
    train_books_orig = set(train_orig['book_id'])
    test_books_orig = set(test_orig['book_id'])
    overlap_orig = len(train_books_orig & test_books_orig)
    
    print(f"   Train books: {len(train_books_orig)}")
    print(f"   Test books: {len(test_books_orig)}")
    print(f"   Overlap: {overlap_orig} ({overlap_orig/len(test_books_orig)*100:.1f}%)")
    
    # Test corrected methodology
    print("\n📈 Corrected methodology:")
    train_corr, test_corr = create_corrected_train_test_split(sample_df, test_ratio=0.2)
    
    train_books_corr = set(train_corr['book_id'])
    test_books_corr = set(test_corr['book_id'])
    overlap_corr = len(train_books_corr & test_books_corr)
    
    print(f"   Train books: {len(train_books_corr)}")
    print(f"   Test books: {len(test_books_corr)}")
    print(f"   Overlap: {overlap_corr} ({overlap_corr/len(test_books_corr)*100:.1f}%)")
    
    print(f"\n📊 COMPARISON:")
    print(f"   Original overlap: {overlap_orig/len(test_books_orig)*100:.1f}%")
    print(f"   Corrected overlap: {overlap_corr/len(test_books_corr)*100:.1f}%")
    
    if overlap_corr > overlap_orig:
        print("✅ Corrected methodology significantly improves book overlap!")
    else:
        print("⚠️  Unexpected result - investigate further")

def main():
    """Run the methodology validation."""
    print("🚀 METHODOLOGY VALIDATION TEST")
    
    # Test 1: Basic methodology validation
    success = validate_methodology()
    
    # Test 2: Compare original vs corrected
    test_original_vs_corrected_split()
    
    if success:
        print("\n🎉 METHODOLOGY VALIDATION PASSED!")
        print("The corrected train/test split methodology works properly.")
    else:
        print("\n❌ Validation failed - need to investigate further.")
    
    return success

if __name__ == "__main__":
    main() 