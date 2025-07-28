#!/usr/bin/env python3
"""
Quick test of Item-Based CF with progress tracking.
"""

from item_based_collaborative_filtering import ItemBasedCollaborativeFiltering, load_data, create_train_test_split
import pandas as pd

def main():
    print("=== Quick Item-Based CF Test ===")
    
    # Load data
    ratings_df, books_df = load_data()
    
    # Take smaller sample for quick test
    print("Sampling data for quick test...")
    sample_size = 50000  # Much smaller sample
    sample_df = ratings_df.sample(n=sample_size, random_state=42)
    
    train_df, test_df = create_train_test_split(sample_df)
    
    print(f"Training with {len(train_df):,} ratings")
    print(f"Testing with {len(test_df):,} ratings")
    
    # Initialize and train model (smaller parameters)
    model = ItemBasedCollaborativeFiltering(
        min_ratings_per_book=10,  # Much lower threshold
        min_ratings_per_user=5,   # Much lower threshold
        top_k_similar=20
    )
    
    print("\nTraining model...")
    model.train(train_df, books_df)
    
    # Test evaluation with very small sample
    print("\nTesting evaluation with 100 users...")
    precision_10, _ = model.evaluate_precision_at_k(test_df, k=10, max_users=100)
    
    print(f"\n✓ Test completed successfully!")
    print(f"Precision@10: {precision_10:.4f}")

if __name__ == "__main__":
    main() 