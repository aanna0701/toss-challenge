#!/usr/bin/env python3
"""
Summary Correlation Analysis
- Uses existing EDA results to provide correlation insights
- Focuses on feat_e_3 missing pattern analysis
- Provides actionable insights for feature engineering
"""

import json
import numpy as np
import pandas as pd

def load_eda_results():
    """Load the pre-computed EDA results"""
    with open('/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/chunk_eda_results.json', 'r') as f:
        return json.load(f)

def analyze_feat_e_3_insights(eda_data):
    """Analyze feat_e_3 missing pattern and its implications"""
    print("\n" + "="*60)
    print("FEAT_E_3 MISSING PATTERN - DETAILED ANALYSIS")
    print("="*60)
    
    # Extract key statistics
    total_rows = eda_data['basic_info']['total_rows']
    clicked_1 = int(eda_data['target_stats']['clicked_1'])
    clicked_0 = int(eda_data['target_stats']['clicked_0'])
    feat_e_3_missing = eda_data['missing_stats']['feat_e_3']
    
    overall_click_rate = clicked_1 / total_rows
    
    print(f"Dataset Overview:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Clicked (1): {clicked_1:,}")
    print(f"  Not clicked (0): {clicked_0:,}")
    print(f"  Overall click rate: {overall_click_rate:.6f} ({overall_click_rate*100:.4f}%)")
    
    print(f"\nfeat_e_3 Missing Pattern:")
    print(f"  feat_e_3 missing rows: {feat_e_3_missing:,}")
    print(f"  feat_e_3 missing percentage: {feat_e_3_missing/total_rows*100:.2f}%")
    print(f"  feat_e_3 non-missing rows: {total_rows - feat_e_3_missing:,}")
    
    # Compare with other missing patterns
    pattern_1_missing = eda_data['missing_stats']['gender']  # 17,208
    pattern_2_missing = eda_data['missing_stats']['feat_a_1']  # 18,598
    
    print(f"\nMissing Pattern Comparison:")
    print(f"  Pattern 1 (gender, age_group, etc.): {pattern_1_missing:,} ({pattern_1_missing/total_rows*100:.2f}%)")
    print(f"  Pattern 2 (feat_a_*): {pattern_2_missing:,} ({pattern_2_missing/total_rows*100:.2f}%)")
    print(f"  feat_e_3: {feat_e_3_missing:,} ({feat_e_3_missing/total_rows*100:.2f}%)")
    
    # Key insights
    print(f"\nKey Insights:")
    print(f"  1. feat_e_3 has {feat_e_3_missing:,} missing values ({feat_e_3_missing/total_rows*100:.2f}% of data)")
    print(f"  2. This is {feat_e_3_missing/pattern_1_missing:.1f}x more missing than Pattern 1")
    print(f"  3. This is {feat_e_3_missing/pattern_2_missing:.1f}x more missing than Pattern 2")
    print(f"  4. feat_e_3 missing pattern is unique and substantial")
    
    # Statistical significance estimation
    # If feat_e_3 missing is informative, we would expect different click rates
    # Let's estimate the potential impact
    
    print(f"\nStatistical Significance Estimation:")
    print(f"  Sample size: {total_rows:,} (very large)")
    print(f"  Missing group size: {feat_e_3_missing:,}")
    print(f"  Non-missing group size: {total_rows - feat_e_3_missing:,}")
    print(f"  Both groups are large enough for reliable statistical tests")
    
    # Effect size estimation
    # If there's a meaningful difference in click rates between missing/non-missing
    # groups, it would be detectable with this sample size
    
    print(f"\nEffect Size Considerations:")
    print(f"  With {total_rows:,} total observations, even small differences")
    print(f"  in click rates between missing/non-missing groups would be")
    print(f"  statistically significant (p < 0.05)")
    print(f"  The key question is practical significance, not statistical significance")
    
    return {
        'total_rows': total_rows,
        'overall_click_rate': overall_click_rate,
        'feat_e_3_missing_count': feat_e_3_missing,
        'feat_e_3_missing_percentage': feat_e_3_missing/total_rows*100,
        'pattern_1_missing': pattern_1_missing,
        'pattern_2_missing': pattern_2_missing,
        'feat_e_3_ratio_to_pattern_1': feat_e_3_missing/pattern_1_missing,
        'feat_e_3_ratio_to_pattern_2': feat_e_3_missing/pattern_2_missing
    }

def analyze_feature_importance_indicators(eda_data):
    """Analyze features that might be important based on EDA statistics"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE INDICATORS")
    print("="*60)
    
    numerical_stats = eda_data['numerical_stats']
    categorical_stats = eda_data['categorical_stats']
    
    # Find features with interesting characteristics
    print("Features with Low Variance (might be less informative):")
    low_variance_features = []
    
    for feature, stats in numerical_stats.items():
        std = stats['std']
        mean = stats['mean']
        cv = std / mean if mean != 0 else 0
        
        if cv < 0.05:  # Very low variance
            low_variance_features.append((feature, cv, std, mean))
    
    for feature, cv, std, mean in low_variance_features[:10]:
        print(f"  {feature}: CV={cv:.4f}, std={std:.4f}, mean={mean:.4f}")
    
    print(f"\nFeatures with High Variance (might be more informative):")
    high_variance_features = []
    
    for feature, stats in numerical_stats.items():
        std = stats['std']
        mean = stats['mean']
        cv = std / mean if mean != 0 else 0
        
        if cv > 1.0:  # High variance
            high_variance_features.append((feature, cv, std, mean))
    
    # Sort by coefficient of variation
    high_variance_features.sort(key=lambda x: x[1], reverse=True)
    
    for feature, cv, std, mean in high_variance_features[:10]:
        print(f"  {feature}: CV={cv:.4f}, std={std:.4f}, mean={mean:.4f}")
    
    print(f"\nCategorical Features with Balanced Distribution:")
    balanced_categorical = []
    
    for feature, distribution in categorical_stats.items():
        values = list(distribution.values())
        if len(values) > 1:
            # Calculate Gini coefficient (measure of inequality)
            n = len(values)
            total = sum(values)
            gini = 1 - sum((count/total)**2 for count in values)
            
            # Lower Gini means more balanced distribution
            if gini < 0.5:  # Relatively balanced
                balanced_categorical.append((feature, gini, len(values)))
    
    # Sort by Gini coefficient (ascending = more balanced)
    balanced_categorical.sort(key=lambda x: x[1])
    
    for feature, gini, n_categories in balanced_categorical[:10]:
        print(f"  {feature}: Gini={gini:.4f}, categories={n_categories}")
    
    return {
        'low_variance_features': low_variance_features,
        'high_variance_features': high_variance_features,
        'balanced_categorical_features': balanced_categorical
    }

def provide_correlation_recommendations():
    """Provide recommendations for correlation analysis"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS RECOMMENDATIONS")
    print("="*60)
    
    print("Based on the EDA results, here are the key findings:")
    
    print("\n1. feat_e_3 Missing Pattern:")
    print("   - Has 1,085,557 missing values (10.14% of data)")
    print("   - This is a substantial missing pattern")
    print("   - Should be investigated for correlation with click rate")
    print("   - Recommendation: Create missing indicator feature")
    
    print("\n2. Missing Pattern Categories:")
    print("   - No missing: 22 features (clean data)")
    print("   - Low missing (<1%): 95 features (minor missing)")
    print("   - High missing (>10%): 1 feature (feat_e_3)")
    print("   - Recommendation: Focus on feat_e_3 missing pattern")
    
    print("\n3. Feature Types for Correlation Analysis:")
    print("   - Continuous features: Use Spearman correlation")
    print("   - Categorical features: Use Chi-square test or ANOVA")
    print("   - Binary features: Use Chi-square test")
    
    print("\n4. Statistical Considerations:")
    print("   - Sample size is very large (10.7M rows)")
    print("   - Statistical significance is almost guaranteed")
    print("   - Focus on practical significance (effect size)")
    print("   - Use Cramer's V for categorical variables")
    print("   - Use correlation coefficient magnitude for continuous")
    
    print("\n5. Feature Engineering Recommendations:")
    print("   - Create missing indicators for all features with missing values")
    print("   - Pay special attention to feat_e_3 missing pattern")
    print("   - Consider interaction terms between missing patterns")
    print("   - Normalize continuous features before correlation analysis")
    
    print("\n6. Next Steps:")
    print("   - Implement missing indicator features")
    print("   - Run correlation analysis on sample data")
    print("   - Focus on features with high variance")
    print("   - Investigate feat_e_3 missing pattern in detail")

def main():
    """Main analysis function"""
    print("Starting Summary Correlation Analysis...")
    
    # Load EDA results
    eda_data = load_eda_results()
    
    # Analyze feat_e_3 specifically
    feat_e_3_results = analyze_feat_e_3_insights(eda_data)
    
    # Analyze feature importance indicators
    feature_indicators = analyze_feature_importance_indicators(eda_data)
    
    # Provide recommendations
    provide_correlation_recommendations()
    
    # Save results
    results = {
        'feat_e_3_analysis': feat_e_3_results,
        'feature_importance_indicators': feature_indicators,
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'data_source': 'chunk_eda_results.json',
        'recommendations': {
            'focus_on_feat_e_3': True,
            'create_missing_indicators': True,
            'use_spearman_for_continuous': True,
            'use_chi2_for_categorical': True,
            'sample_size_sufficient': True
        }
    }
    
    output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/summary_correlation_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("Summary analysis completed!")

if __name__ == "__main__":
    main()
