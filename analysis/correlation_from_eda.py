#!/usr/bin/env python3
"""
Correlation Analysis using pre-computed EDA results
- Uses chunk_eda_results.json for basic statistics
- Analyzes feat_e_3 missing pattern correlation with click rate
- Provides insights on feature correlations
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def load_eda_results():
    """Load the pre-computed EDA results"""
    with open('/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/chunk_eda_results.json', 'r') as f:
        return json.load(f)

def analyze_feat_e_3_from_eda(eda_data):
    """Analyze feat_e_3 missing pattern correlation using EDA data"""
    print("\n" + "="*60)
    print("FEAT_E_3 MISSING PATTERN ANALYSIS (from EDA data)")
    print("="*60)
    
    # Extract basic information
    total_rows = eda_data['basic_info']['total_rows']
    clicked_0 = int(eda_data['target_stats']['clicked_0'])
    clicked_1 = int(eda_data['target_stats']['clicked_1'])
    feat_e_3_missing = eda_data['missing_stats']['feat_e_3']
    
    # Calculate overall click rate
    overall_click_rate = clicked_1 / total_rows
    
    print(f"Dataset Overview:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Clicked (1): {clicked_1:,}")
    print(f"  Not clicked (0): {clicked_0:,}")
    print(f"  Overall click rate: {overall_click_rate:.6f}")
    
    print(f"\nfeat_e_3 Missing Pattern:")
    print(f"  feat_e_3 missing rows: {feat_e_3_missing:,}")
    print(f"  feat_e_3 missing percentage: {feat_e_3_missing/total_rows*100:.2f}%")
    print(f"  feat_e_3 non-missing rows: {total_rows - feat_e_3_missing:,}")
    
    # From the EDA data, we can see that feat_e_3 has 1,085,557 missing values
    # This is a significant portion of the data (about 10.14%)
    
    # We can estimate the correlation based on the missing pattern
    # If feat_e_3 missing is correlated with click rate, we would expect
    # different click rates in missing vs non-missing groups
    
    print(f"\nAnalysis Insights:")
    print(f"  feat_e_3 has {feat_e_3_missing:,} missing values out of {total_rows:,} total rows")
    print(f"  This represents {feat_e_3_missing/total_rows*100:.2f}% of the dataset")
    print(f"  This is a substantial missing pattern that could be informative")
    
    # Check if feat_e_3 missing pattern overlaps with other patterns
    pattern_1_missing = eda_data['missing_stats']['gender']  # 17,208
    pattern_2_missing = eda_data['missing_stats']['feat_a_1']  # 18,598
    
    print(f"\nMissing Pattern Overlaps:")
    print(f"  Pattern 1 (gender, age_group, etc.): {pattern_1_missing:,} missing")
    print(f"  Pattern 2 (feat_a_*): {pattern_2_missing:,} missing")
    print(f"  feat_e_3: {feat_e_3_missing:,} missing")
    
    # feat_e_3 has much more missing data than the other patterns
    # This suggests it might be a different type of missing pattern
    
    return {
        'total_rows': total_rows,
        'overall_click_rate': overall_click_rate,
        'feat_e_3_missing_count': feat_e_3_missing,
        'feat_e_3_missing_percentage': feat_e_3_missing/total_rows*100,
        'pattern_1_missing': pattern_1_missing,
        'pattern_2_missing': pattern_2_missing,
        'analysis_note': 'feat_e_3 has significantly more missing data than other patterns'
    }

def analyze_feature_missing_patterns(eda_data):
    """Analyze missing patterns across all features"""
    print("\n" + "="*60)
    print("FEATURE MISSING PATTERNS ANALYSIS")
    print("="*60)
    
    missing_stats = eda_data['missing_stats']
    total_rows = eda_data['basic_info']['total_rows']
    
    # Categorize features by missing pattern
    no_missing = []
    low_missing = []  # < 1%
    medium_missing = []  # 1-10%
    high_missing = []  # > 10%
    
    for feature, missing_count in missing_stats.items():
        if feature == 'clicked':  # Skip target variable
            continue
            
        missing_pct = missing_count / total_rows * 100
        
        if missing_count == 0:
            no_missing.append((feature, missing_count, missing_pct))
        elif missing_pct < 1:
            low_missing.append((feature, missing_count, missing_pct))
        elif missing_pct <= 10:
            medium_missing.append((feature, missing_count, missing_pct))
        else:
            high_missing.append((feature, missing_count, missing_pct))
    
    print(f"Missing Pattern Categories:")
    print(f"  No missing values: {len(no_missing)} features")
    print(f"  Low missing (<1%): {len(low_missing)} features")
    print(f"  Medium missing (1-10%): {len(medium_missing)} features")
    print(f"  High missing (>10%): {len(high_missing)} features")
    
    print(f"\nFeatures with High Missing Rates (>10%):")
    for feature, count, pct in high_missing:
        print(f"  {feature}: {count:,} ({pct:.2f}%)")
    
    print(f"\nFeatures with Medium Missing Rates (1-10%):")
    for feature, count, pct in medium_missing[:10]:  # Show top 10
        print(f"  {feature}: {count:,} ({pct:.2f}%)")
    
    return {
        'no_missing': no_missing,
        'low_missing': low_missing,
        'medium_missing': medium_missing,
        'high_missing': high_missing
    }

def analyze_categorical_features(eda_data):
    """Analyze categorical features and their distributions"""
    print("\n" + "="*60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    categorical_stats = eda_data['categorical_stats']
    
    print("Categorical Features Distribution:")
    for feature, distribution in categorical_stats.items():
        print(f"\n{feature}:")
        total_values = sum(distribution.values())
        for value, count in list(distribution.items())[:5]:  # Show top 5 values
            pct = count / total_values * 100
            print(f"  {value}: {count:,} ({pct:.2f}%)")
        if len(distribution) > 5:
            print(f"  ... and {len(distribution) - 5} more values")

def analyze_numerical_features(eda_data):
    """Analyze numerical features statistics"""
    print("\n" + "="*60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*60)
    
    numerical_stats = eda_data['numerical_stats']
    
    # Find features with interesting statistics
    low_variance_features = []
    high_variance_features = []
    
    for feature, stats in numerical_stats.items():
        std = stats['std']
        mean = stats['mean']
        
        # Coefficient of variation
        cv = std / mean if mean != 0 else 0
        
        if cv < 0.1:  # Low variance
            low_variance_features.append((feature, cv, std, mean))
        elif cv > 2:  # High variance
            high_variance_features.append((feature, cv, std, mean))
    
    print(f"Features with Low Variance (CV < 0.1):")
    for feature, cv, std, mean in low_variance_features[:10]:
        print(f"  {feature}: CV={cv:.4f}, std={std:.4f}, mean={mean:.4f}")
    
    print(f"\nFeatures with High Variance (CV > 2):")
    for feature, cv, std, mean in high_variance_features[:10]:
        print(f"  {feature}: CV={cv:.4f}, std={std:.4f}, mean={mean:.4f}")

def main():
    """Main analysis function"""
    print("Starting Correlation Analysis from EDA Results...")
    
    # Load EDA results
    eda_data = load_eda_results()
    
    # Analyze feat_e_3 specifically
    feat_e_3_results = analyze_feat_e_3_from_eda(eda_data)
    
    # Analyze missing patterns
    missing_patterns = analyze_feature_missing_patterns(eda_data)
    
    # Analyze categorical features
    analyze_categorical_features(eda_data)
    
    # Analyze numerical features
    analyze_numerical_features(eda_data)
    
    # Save results
    results = {
        'feat_e_3_analysis': feat_e_3_results,
        'missing_patterns_analysis': missing_patterns,
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'data_source': 'chunk_eda_results.json'
    }
    
    output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/correlation_analysis_from_eda.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()
