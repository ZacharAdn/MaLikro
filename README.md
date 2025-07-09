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
│   └── to_read.csv               # Books marked "to read"
├── code/                          # Analysis scripts
│   ├── phase1_data_quality.py    # Data quality & structure analysis
│   ├── phase2_user_behavior.py   # User behavior analysis
│   └── run_analysis.py           # Main runner script
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

## Completed Analysis Phases

### ✅ Phase 1: Data Quality & Structure
**Objective**: Understand data structure and quality issues
**Key Findings**:
- Perfect relationship integrity between ratings and books datasets
- Missing values primarily in book metadata (ISBN, language, publication year)
- 6 duplicate entries in book_tags dataset
- No missing values in core ratings data

**Generated Outputs**:
- `basic_statistics.csv` - Dataset size and memory usage stats
- `data_quality_report.csv` - Detailed quality analysis per column
- `dataset_relationships.csv` - Relationship integrity analysis
- `dataset_overview.png` - Visual overview of dataset sizes

### ✅ Phase 2: User Behavior Analysis
**Objective**: Analyze user rating patterns and activity levels
**Key Findings**:
- **Highly Active Users**: 67.8% are power users (100+ ratings)
- **Rating Bias**: Positive skew (35.8% give 4 stars, 33.2% give 5 stars)
- **User Preferences**: 42% rate generously (avg >4.0), 56% rate moderately
- **Consistency**: 95.9% of users have moderate rating variance

**Generated Outputs**:
- `user_statistics_summary.csv` - User activity metrics
- `user_activity_levels.csv` - User segmentation by activity
- `top_10_users.csv` - Most active users ranking
- `user_preferences_summary.csv` - Rating behavior patterns
- `user_behavior_analysis.png` - Comprehensive user analysis dashboard
- `detailed_user_analysis.png` - Deep dive into user distributions

## Key Insights for Recommendation System

### User Segmentation
1. **Power Users (67.8%)**: 100+ ratings, high engagement
2. **Active Users (32.2%)**: 21-100 ratings, moderate engagement  
3. **Regular Users (0.0%)**: 6-20 ratings, minimal presence

### Rating Patterns
- **Positive Bias**: 69% of ratings are 4-5 stars
- **Low Negative Ratings**: Only 8.1% are 1-2 stars
- **Balanced User Base**: Mix of strict, moderate, and generous raters

### Data Quality Status
- **Excellent Core Data**: No missing values in ratings
- **Good Metadata**: 90%+ completeness for essential book info
- **Perfect Relationships**: All datasets properly linked

## Running the Analysis

### Prerequisites
```bash
# Activate virtual environment
source book_recommendation_venv/bin/activate

# Install required packages (already installed)
pip install pandas numpy matplotlib seaborn plotly scipy
```

### Execute Analysis
```bash
# Run both Phase 1 and Phase 2
python code/run_analysis.py

# Or run phases individually
python code/phase1_data_quality.py
python code/phase2_user_behavior.py
```

## Next Steps
1. **Phase 3**: Book & Content Analysis
2. **Phase 4**: Recommendation System Specific Analysis  
3. **Phase 5**: Advanced Analysis (correlations, feature documentation)
4. **Model Development**: Baseline and advanced recommendation models

## Technical Notes
- All analysis performed in Python with pandas, matplotlib, seaborn
- Results saved as CSV tables and PNG visualizations
- Modular script design for easy expansion and modification
- Comprehensive logging and error handling implemented 