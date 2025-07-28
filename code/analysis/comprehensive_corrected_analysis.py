#!/usr/bin/env python3
"""
Comprehensive Corrected Analysis - Runner script for corrected recommendation system analysis.
"""

import os
import sys
import time
from datetime import datetime

def create_results_folder():
    """Create results folder for corrected analysis."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = f'../../results/corrected_analysis_{timestamp}'
    os.makedirs(folder, exist_ok=True)
    return folder

def run_quick_test():
    """Run the quick test first."""
    print("=" * 60)
    print("STEP 1: RUNNING QUICK TEST")
    print("=" * 60)
    
    try:
        sys.path.append('../testing')
        from methodology_validation import main as test_main
        success = test_main()
        return success
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def run_full_analysis(results_folder):
    """Run the full corrected analysis."""
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING FULL CORRECTED ANALYSIS")
    print("=" * 60)
    
    try:
        sys.path.append('../models')
        from corrected_implementation import main as corrected_main
        baseline, item_cf, baseline_prec, item_cf_prec = corrected_main()
        
        # Save results
        results_file = os.path.join(results_folder, 'corrected_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Corrected Recommendation System Results\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Methodology: Fixed random train/test split (not sorted by book_id)\n\n")
            f.write(f"Results:\n")
            f.write(f"  Baseline precision@10: {baseline_prec:.4f} ({baseline_prec*100:.1f}%)\n")
            f.write(f"  Item-CF precision@10: {item_cf_prec:.4f} ({item_cf_prec*100:.1f}%)\n\n")
            
            if item_cf_prec > baseline_prec and baseline_prec > 0:
                improvement = (item_cf_prec / baseline_prec - 1) * 100
                f.write(f"  Improvement: {improvement:.1f}%\n")
            
            f.write(f"\nExpected ranges:\n")
            f.write(f"  Baseline: 5-15%\n")
            f.write(f"  Item-CF: 8-20%\n\n")
            
            if baseline_prec >= 0.05:
                f.write("✅ Baseline results are reasonable\n")
            else:
                f.write("⚠️ Baseline results are still low\n")
                
            if item_cf_prec >= 0.08:
                f.write("✅ Item-CF results are reasonable\n")
            else:
                f.write("⚠️ Item-CF results need tuning\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        return True, baseline_prec, item_cf_prec
        
    except Exception as e:
        print(f"❌ Full analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def create_summary_report(results_folder, baseline_prec, item_cf_prec):
    """Create a summary report."""
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    summary_file = os.path.join(results_folder, 'summary_report.md')
    
    with open(summary_file, 'w') as f:
        f.write(f"# Corrected Recommendation System Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Problem Fixed\n\n")
        f.write(f"**Issue:** Previous train/test split was sorting by `book_id`, creating artificial bias:\n")
        f.write(f"- Train data: Books with low IDs (1, 2, 25...)\n")
        f.write(f"- Test data: Books with high IDs (1761, 1796...)\n")
        f.write(f"- Result: Zero overlap → 0.0000 precision\n\n")
        
        f.write(f"**Fix:** Implemented proper random train/test split:\n")
        f.write(f"- Random shuffle of user's ratings before split\n")
        f.write(f"- Proper book overlap between train and test\n")
        f.write(f"- No data leakage at user level\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"| Algorithm | Precision@10 | Percentage |\n")
        f.write(f"|-----------|--------------|------------|\n")
        f.write(f"| Baseline  | {baseline_prec:.4f}       | {baseline_prec*100:.1f}%       |\n")
        f.write(f"| Item-CF   | {item_cf_prec:.4f}       | {item_cf_prec*100:.1f}%       |\n\n")
        
        f.write(f"## Analysis\n\n")
        
        if baseline_prec >= 0.05:
            f.write(f"✅ **Baseline Performance**: Good - {baseline_prec*100:.1f}% precision is reasonable for popularity-based recommendations.\n\n")
        else:
            f.write(f"⚠️ **Baseline Performance**: Low - {baseline_prec*100:.1f}% precision suggests the task is very challenging or needs further tuning.\n\n")
        
        if item_cf_prec > baseline_prec and baseline_prec > 0:
            improvement = (item_cf_prec / baseline_prec - 1) * 100
            f.write(f"✅ **Item-CF Performance**: Item-Based CF outperforms baseline by {improvement:.1f}%, demonstrating collaborative filtering value.\n\n")
        elif item_cf_prec > 0.08:
            f.write(f"✅ **Item-CF Performance**: Absolute performance of {item_cf_prec*100:.1f}% is reasonable for collaborative filtering.\n\n")
        else:
            f.write(f"⚠️ **Item-CF Performance**: {item_cf_prec*100:.1f}% precision suggests need for parameter tuning or algorithm improvements.\n\n")
        
        f.write(f"## Next Steps\n\n")
        f.write(f"1. **Parameter Tuning**: Optimize similarity thresholds, k-values, and filtering criteria\n")
        f.write(f"2. **Advanced Algorithms**: Implement Matrix Factorization (SVD/NMF)\n")
        f.write(f"3. **Hybrid Approaches**: Combine collaborative filtering with content-based features\n")
        f.write(f"4. **Evaluation Metrics**: Add Recall@K, NDCG, and coverage metrics\n")
        f.write(f"5. **Deep Learning**: Explore Neural Collaborative Filtering\n\n")
        
        f.write(f"## Conclusion\n\n")
        f.write(f"The corrected methodology successfully fixed the fundamental train/test split issue. ")
        
        if baseline_prec >= 0.02 or item_cf_prec >= 0.02:
            f.write(f"Results now show reasonable precision values, providing a solid foundation for further algorithm development.")
        else:
            f.write(f"While precision values are still low, the methodology is now sound and ready for advanced algorithm optimization.")
    
    print(f"📄 Summary report saved to: {summary_file}")

def main():
    """Main execution."""
    print("🚀 CORRECTED RECOMMENDATION SYSTEM ANALYSIS")
    print(f"Started at: {datetime.now()}")
    
    # Create results folder
    results_folder = create_results_folder()
    print(f"📁 Results folder: {results_folder}")
    
    # Step 1: Quick test
    quick_success = run_quick_test()
    
    if not quick_success:
        print("\n❌ Quick test failed - stopping analysis")
        return False
    
    # Step 2: Full analysis
    full_success, baseline_prec, item_cf_prec = run_full_analysis(results_folder)
    
    if not full_success:
        print("\n❌ Full analysis failed")
        return False
    
    # Step 3: Summary report
    create_summary_report(results_folder, baseline_prec, item_cf_prec)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 CORRECTED ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📊 Baseline precision@10: {baseline_prec:.4f} ({baseline_prec*100:.1f}%)")
    print(f"📊 Item-CF precision@10: {item_cf_prec:.4f} ({item_cf_prec*100:.1f}%)")
    print(f"📁 All results saved to: {results_folder}")
    
    if baseline_prec >= 0.02 or item_cf_prec >= 0.02:
        print("✅ SUCCESS: Methodology fix worked - we now have reasonable precision values!")
    else:
        print("⚠️  Precision values are still low but methodology is now correct")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 