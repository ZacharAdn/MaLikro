# Book Recommendation System - Data Analysis Project

## Project Overview
A comprehensive data science project for building a book recommendation system based on user reading patterns and preferences using the goodbooks-10k dataset.

## Project Structure
```
books/
├── data/                          # Raw datasets
│   ├── ratings.csv               # User-book ratings (5.9M ratings)
│   ├── books.csv                 # Book metadata (10K books)
│   ├── book_tags.csv             # User-generated book tags
│   ├── tags.csv                  # Tag definitions
│   └── to_read.csv              # Books marked "to read"
├── code/                          # Analysis scripts
│   ├── models/                   # Model implementations
│   │   └── corrected_implementation.py  # Recommendation models
│   ├── testing/                  # Testing & validation
│   │   └── methodology_validation.py    # Evaluation framework
│   ├── analysis/                 # Analysis scripts
│   │   └── comprehensive_corrected_analysis.py  # Main analysis
│   └── README.md                 # Code documentation
├── results/                       # Analysis outputs
│   ├── tables/                   # Generated CSV reports
│   └── visualizations/           # Generated charts and plots
├── specs/                         # Project specifications
│   └── data_understanding_specification.md
└── book_recommendation_venv/      # Virtual environment
```

## Dataset Summary
- **Users**: 53,424 unique users
- **Books**: 10,000 books with full metadata
- **Ratings**: 5,976,479 ratings (scale 1-5)
- **Tags**: 34,252 unique tags for content categorization
- **To-Read**: 912,705 "want to read" entries

## Latest Progress

### ✅ Dual Evaluation Framework
Our system evaluates recommendations in two critical scenarios:

#### 1. Cold Start (New Users)
- **Baseline**: 28.2% precision@10
- **Use Case**: New platform users
- **Method**: User-based train/test split
- **Advantage**: Perfect for user onboarding

#### 2. Warm Start (Existing Users)
- **Baseline**: 6.6% precision@10
- **Use Case**: Personalized recommendations
- **Method**: Per-user rating split
- **Advantage**: Realistic prediction scenario

### 🔄 Item-Based Collaborative Filtering
**Current status:**
- Initial implementation complete
- Training time: 25.88s
- Memory usage: 762MB
- Warm start precision@10: 0.4% (needs tuning)
- Cold start evaluation pending

## Running the Analysis

### Prerequisites
```bash
# Create and activate virtual environment
python -m venv book_recommendation_venv
source book_recommendation_venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn plotly scipy
```

### Execute Analysis
```bash
# Run methodology validation (both cold/warm start)
python code/testing/methodology_validation.py

# Run comprehensive analysis
python code/analysis/comprehensive_corrected_analysis.py
```

## Technical Notes
- Dual evaluation framework for cold/warm start
- Memory-efficient processing of 6M ratings
- Modular code organization by functionality
- Comprehensive testing and validation
- Clear performance benchmarks

## Next Steps
1. **Item-CF Optimization**
   - Complete cold start evaluation
   - Parameter tuning for both scenarios
   - Improve recommendation diversity

2. **Matrix Factorization**
   - SVD implementation
   - Performance comparison
   - Hybrid approach planning

3. **Production Readiness**
   - Code optimization
   - Documentation updates
   - Performance monitoring 