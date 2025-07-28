# Code Directory Organization

This directory contains all the code for the book recommendation system project, organized into logical subdirectories.

## Directory Structure

### 📊 `eda/` - Exploratory Data Analysis
Scripts for understanding and analyzing the dataset.

- **`data_quality/`** - Data quality analysis
  - `phase1_data_quality.py` - Initial data quality assessment and validation
  
- **`user_behavior/`** - User behavior analysis  
  - `phase2_user_behavior.py` - Analysis of user rating patterns and behaviors
  
- **`comprehensive/`** - Comprehensive analysis
  - `final_comprehensive_analysis.py` - Complete dataset analysis and insights
  
- `run_analysis.py` - Main script to run various analyses

### 🤖 `models/` - Machine Learning Models
Implementation of different recommendation algorithms.

- **`baseline/`** - Baseline models
  - `baseline_model.py` - Simple popularity-based recommendation model
  
- **`collaborative_filtering/`** - Collaborative filtering approaches
  - **`user_based/`** - User-based collaborative filtering
    - `collaborative_filtering.py` - Basic user-based CF implementation
    - `collaborative_filtering_optimized.py` - Optimized version with performance improvements
  - **`item_based/`** - Item-based collaborative filtering
    - `item_based_collaborative_filtering.py` - Item-based CF implementation
    - `analyze_item_based_cf.py` - Analysis and evaluation script
    - `item_cf_optimized.py` - Optimized version for better performance

- **🔧 `corrected_implementation.py`** - **MAIN CORRECTED SYSTEM**
  - Contains both baseline and item-based CF with **fixed train/test split methodology**
  - **This fixes the fundamental book_id sorting issue that caused 0.0000 precision**
  - Includes proper random train/test split and comprehensive evaluation

### 📈 `analysis/` - Analysis and Evaluation Scripts
Scripts for comprehensive analysis and model evaluation.

- **`comprehensive_corrected_analysis.py`** - **MAIN ANALYSIS RUNNER**
  - Orchestrates the corrected recommendation system analysis
  - Generates comprehensive reports and visualizations
  - **Run this file to execute the complete corrected analysis**

- **`methodology_comparison/`** - Methodology comparison scripts
  - `compare_baselines.py` - Compare different baseline approaches
  - `analyze_baseline_results.py` - Detailed baseline analysis with user examples

### 🧪 `testing/` - Testing and Validation
Scripts for testing and validating implementations.

- **`methodology_validation.py`** - **CORRECTED METHODOLOGY VALIDATION**
  - Validates that the corrected train/test split works properly
  - Compares original vs corrected methodology
  - Quick test with sample data to verify reasonable precision values

- **`debugging/`** - Debug and diagnostic scripts
  - `debug_precision.py` - Debug precision calculation issues
  - `trace_recommendations.py` - Trace recommendation generation step-by-step
  - `visual_testing_analysis.py` - Visual analysis of recommendation quality

- **`unit_tests/`** - Unit tests for components
  - Individual component testing scripts

## 🚀 Quick Start - Running the Corrected Analysis

### **Step 1: Run the Methodology Validation**
```bash
cd code/testing
python methodology_validation.py
```

### **Step 2: Run the Complete Corrected Analysis**
```bash
cd code/analysis
python comprehensive_corrected_analysis.py
```

### **Step 3: View Results**
Results will be saved to `results/corrected_analysis_[timestamp]/`

## 🔧 Key Fix Applied

**Problem Fixed:** The original implementation was sorting user ratings by `book_id` before train/test split, creating artificial bias:
- **Train data:** Books with low IDs (1, 2, 25...)
- **Test data:** Books with high IDs (1761, 1796...)
- **Result:** Zero overlap → 0.0000 precision

**Solution:** Implemented proper random train/test split:
- **Random shuffle** of user's ratings before split
- **Proper book overlap** between train and test
- **No data leakage** at user level

## 📊 Expected Results

With the corrected methodology, you should see:
- **Baseline precision@10:** 5-15%
- **Item-CF precision@10:** 8-20%
- **Meaningful comparisons** between algorithms

## 📁 File Dependencies

### Core Implementation Files:
1. `models/corrected_implementation.py` - Main algorithms
2. `testing/methodology_validation.py` - Validation script
3. `analysis/comprehensive_corrected_analysis.py` - Analysis runner

### Legacy Files (for reference):
- Individual model implementations in their respective folders
- Debugging scripts that helped identify the methodology issue

## 🎯 Next Steps

After running the corrected analysis:
1. **Parameter Tuning** - Optimize algorithm parameters
2. **Advanced Algorithms** - Implement Matrix Factorization (SVD/NMF)
3. **Hybrid Approaches** - Combine multiple recommendation strategies
4. **Production Optimization** - Scale for real-world deployment

---

**Note:** Always run the corrected implementation files to get accurate results. The legacy implementations may still contain the book_id sorting issue. 