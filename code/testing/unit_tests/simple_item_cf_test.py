#!/usr/bin/env python3
"""
Simple test of Item-Based CF with fixed progress tracking.
"""

import pandas as pd
import numpy as np
from item_based_collaborative_filtering import ItemBasedCollaborativeFiltering, load_data
import time

def create_simple_train_test_split(ratings_df, test_ratio=0.2):
    """Create a simple train/test split."""
    print("Creating simple train/test split...")
    
    # Shuffle and split
    shuffled = ratings_df.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df):,} ratings")
    print(f"Test: {len(test_df):,} ratings")
    
    return train_df, test_df

def main():
    print("=== Simple Item-Based CF Test ===")
    
    # Load and sample data
    ratings_df, books_df = load_data()
    
    print("Sampling 100k ratings for quick test...")
    sample_df = ratings_df.sample(n=100000, random_state=42)
    
    train_df, test_df = create_simple_train_test_split(sample_df)
    
    # Initialize and train model
    model = ItemBasedCollaborativeFiltering(
        min_ratings_per_book=20,
        min_ratings_per_user=10,
        top_k_similar=30
    )
    
    print("\nTraining model...")
    start_time = time.time()
    model.train(train_df, books_df)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Test evaluation with small sample
    print("\nTesting evaluation with 500 users...")
    eval_start = time.time()
    precision_10, _ = model.evaluate_precision_at_k(test_df, k=10, max_users=500)
    eval_time = time.time() - eval_start
    
    print(f"\n✓ Evaluation completed in {eval_time:.1f} seconds")
    print(f"✓ Precision@10: {precision_10:.4f}")
    
    # Show what a full evaluation would take
    print(f"\nEstimated time for 3000 users: {eval_time * 6:.1f} seconds ({eval_time * 6 / 60:.1f} minutes)")

if __name__ == "__main__":
    main() 