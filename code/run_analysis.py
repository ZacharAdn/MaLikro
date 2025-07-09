"""
Main Runner Script for Book Recommendation Data Analysis
Executes Phase 1 and Phase 2 according to the specification
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import phase modules
from phase1_data_quality import main as phase1_main
from phase2_user_behavior import main as phase2_main

def run_complete_analysis():
    """Run the complete data understanding analysis"""
    print("🚀 STARTING BOOK RECOMMENDATION DATA ANALYSIS")
    print("=" * 60)
    
    # Check if data directory exists
    data_path = Path('data')
    if not data_path.exists():
        print("❌ Data directory not found! Please ensure data files are in 'data/' folder")
        return
    
    # List available data files
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        print("❌ No CSV files found in data directory!")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    print("\n" + "=" * 60)
    
    # Execute Phase 1
    print("\n🔍 EXECUTING PHASE 1: DATA QUALITY & STRUCTURE")
    phase1_results = phase1_main()
    
    if phase1_results[0] is None:
        print("❌ Phase 1 failed! Stopping analysis.")
        return
    
    print("\n✅ Phase 1 completed successfully!")
    
    # Execute Phase 2
    print("\n👥 EXECUTING PHASE 2: USER BEHAVIOR ANALYSIS")
    phase2_results = phase2_main()
    
    if phase2_results is None:
        print("❌ Phase 2 failed!")
        return
    
    print("\n✅ Phase 2 completed successfully!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    results_path = Path('results')
    tables_count = len(list((results_path / 'tables').glob('*.csv')))
    viz_count = len(list((results_path / 'visualizations').glob('*.png')))
    
    print(f"📊 Total tables generated: {tables_count}")
    print(f"📈 Total visualizations created: {viz_count}")
    print(f"📁 All results saved in: {results_path}")
    
    print("\nGenerated files:")
    print("Tables:")
    for table in (results_path / 'tables').glob('*.csv'):
        print(f"  - {table.name}")
    
    print("Visualizations:")
    for viz in (results_path / 'visualizations').glob('*.png'):
        print(f"  - {viz.name}")
    
    print("\n🔍 Next steps:")
    print("1. Review the generated reports and visualizations")
    print("2. Check data quality issues in the reports")
    print("3. Proceed with Phase 3: Book & Content Analysis")
    print("4. Continue with Phase 4: Recommendation System Analysis")

if __name__ == "__main__":
    run_complete_analysis() 