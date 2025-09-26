#!/usr/bin/env python3
"""
Lightweight Correlation Analysis between Features and Click Rate
- Processes data in smaller chunks to manage memory
- feat_e_3 missing pattern analysis
- All features correlation with click rate
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def load_data_sample():
    """Load a smaller sample of the training data"""
    print("Loading data sample...")
    
    parquet_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/train.parquet'
    
    # Load only a subset of columns first to check structure
    df_sample = pd.read_parquet(parquet_file, engine='pyarrow')
    print(f"Full dataset shape: {df_sample.shape}")
    
    # Take a smaller sample for analysis
    sample_size = min(100000, len(df_sample))  # 100K rows max
    print(f"Taking sample of {sample_size:,} rows for analysis...")
    
    df_sample = df_sample.sample(n=sample_size, random_state=42)
    print(f"Sample loaded: {df_sample.shape}")
    
    return df_sample

def analyze_feat_e_3_correlation(df):
    """Analyze correlation between feat_e_3 missing pattern and click rate"""
    print("\n" + "="*60)
    print("FEAT_E_3 MISSING PATTERN ANALYSIS")
    print("="*60)
    
    # Create missing indicator for feat_e_3
    df['feat_e_3_missing'] = df['feat_e_3'].isna().astype(int)
    
    # Calculate click rates
    total_rows = len(df)
    missing_rows = df['feat_e_3_missing'].sum()
    non_missing_rows = total_rows - missing_rows
    
    # Click rates by missing status
    click_rate_missing = df[df['feat_e_3_missing'] == 1]['clicked'].mean()
    click_rate_non_missing = df[df['feat_e_3_missing'] == 0]['clicked'].mean()
    
    # Overall click rate
    overall_click_rate = df['clicked'].mean()
    
    print(f"Sample size: {total_rows:,}")
    print(f"feat_e_3 missing rows: {missing_rows:,} ({missing_rows/total_rows*100:.2f}%)")
    print(f"feat_e_3 non-missing rows: {non_missing_rows:,} ({non_missing_rows/total_rows*100:.2f}%)")
    print(f"\nClick rates:")
    print(f"  Overall: {overall_click_rate:.6f}")
    print(f"  When feat_e_3 missing: {click_rate_missing:.6f}")
    print(f"  When feat_e_3 non-missing: {click_rate_non_missing:.6f}")
    print(f"  Difference: {click_rate_non_missing - click_rate_missing:.6f}")
    
    # Statistical significance test (Chi-square test)
    contingency_table = pd.crosstab(df['feat_e_3_missing'], df['clicked'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nStatistical significance (Chi-square test):")
    print(f"  Chi-square statistic: {chi2:.6f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Effect size (Cramer's V)
    n = total_rows
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    print(f"  Cramer's V (effect size): {cramers_v:.6f}")
    
    return {
        'sample_size': total_rows,
        'missing_rows': missing_rows,
        'missing_percentage': missing_rows/total_rows*100,
        'click_rate_overall': overall_click_rate,
        'click_rate_missing': click_rate_missing,
        'click_rate_non_missing': click_rate_non_missing,
        'click_rate_difference': click_rate_non_missing - click_rate_missing,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cramers_v': cramers_v
    }

def determine_variable_type(series):
    """Determine if a variable is continuous or categorical"""
    # Remove missing values for analysis
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return 'categorical'
    
    # If all values are integers and range is small, consider categorical
    if clean_series.dtype in ['int64', 'int32', 'int16', 'int8']:
        unique_count = clean_series.nunique()
        total_count = len(clean_series)
        if unique_count <= 20 or unique_count / total_count < 0.1:
            return 'categorical'
    
    # If float but has many unique values, consider continuous
    if clean_series.dtype in ['float64', 'float32', 'float16']:
        unique_count = clean_series.nunique()
        total_count = len(clean_series)
        if unique_count > 20 and unique_count / total_count > 0.1:
            return 'continuous'
    
    # Default to categorical for safety
    return 'categorical'

def analyze_feature_correlation(df, feature_name):
    """Analyze correlation between a single feature and click rate"""
    if feature_name not in df.columns:
        return None
    
    # Skip if feature is the target variable
    if feature_name == 'clicked':
        return None
    
    # Determine variable type
    var_type = determine_variable_type(df[feature_name])
    
    result = {
        'feature': feature_name,
        'variable_type': var_type,
        'missing_count': df[feature_name].isna().sum(),
        'missing_percentage': df[feature_name].isna().sum() / len(df) * 100,
        'unique_values': df[feature_name].nunique()
    }
    
    try:
        if var_type == 'continuous':
            # Spearman correlation for continuous variables
            # Use non-missing values only
            non_missing_mask = df[feature_name].notna()
            if non_missing_mask.sum() > 10:  # Need at least 10 observations
                corr_coef, p_value = spearmanr(
                    df.loc[non_missing_mask, feature_name], 
                    df.loc[non_missing_mask, 'clicked']
                )
                result['correlation_coefficient'] = corr_coef
                result['p_value'] = p_value
                result['significant'] = p_value < 0.05
            else:
                result['correlation_coefficient'] = np.nan
                result['p_value'] = np.nan
                result['significant'] = False
        
        else:  # categorical
            # ANOVA for categorical variables
            # Group by feature values and calculate click rates
            click_rates = df.groupby(feature_name)['clicked'].agg(['mean', 'count']).reset_index()
            click_rates = click_rates[click_rates['count'] > 0]  # Remove groups with no data
            
            if len(click_rates) > 1:
                # Perform ANOVA
                groups = []
                for value in click_rates[feature_name].values:
                    group_data = df[df[feature_name] == value]['clicked'].values
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    result['f_statistic'] = f_stat
                    result['p_value'] = p_value
                    result['significant'] = p_value < 0.05
                    
                    # Calculate click rate by category (top 5 only)
                    result['click_rates_by_category'] = click_rates.head(5).to_dict('records')
                else:
                    result['f_statistic'] = np.nan
                    result['p_value'] = np.nan
                    result['significant'] = False
                    result['click_rates_by_category'] = []
            else:
                result['f_statistic'] = np.nan
                result['p_value'] = np.nan
                result['significant'] = False
                result['click_rates_by_category'] = []
    
    except Exception as e:
        print(f"Error analyzing {feature_name}: {str(e)}")
        result['error'] = str(e)
        result['correlation_coefficient'] = np.nan
        result['p_value'] = np.nan
        result['significant'] = False
    
    return result

def analyze_all_features_correlation(df):
    """Analyze correlation between all features and click rate"""
    print("\n" + "="*60)
    print("ALL FEATURES CORRELATION ANALYSIS")
    print("="*60)
    
    # Get all feature columns (exclude target and sequence)
    feature_columns = [col for col in df.columns if col not in ['clicked', 'seq']]
    
    results = []
    
    print(f"Analyzing {len(feature_columns)} features...")
    
    for i, feature in enumerate(feature_columns):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(feature_columns)}")
        
        result = analyze_feature_correlation(df, feature)
        if result:
            results.append(result)
    
    # Sort by significance and correlation strength
    continuous_features = [r for r in results if r['variable_type'] == 'continuous']
    categorical_features = [r for r in results if r['variable_type'] == 'categorical']
    
    # Sort continuous features by absolute correlation coefficient
    continuous_features.sort(key=lambda x: abs(x.get('correlation_coefficient', 0)), reverse=True)
    
    # Sort categorical features by F-statistic
    categorical_features.sort(key=lambda x: x.get('f_statistic', 0), reverse=True)
    
    print(f"\nTop 10 Continuous Features by Correlation Strength:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Correlation':<12} {'P-value':<12} {'Significant':<12} {'Missing%':<10}")
    print("-" * 80)
    
    for feature in continuous_features[:10]:
        corr = feature.get('correlation_coefficient', np.nan)
        p_val = feature.get('p_value', np.nan)
        sig = feature.get('significant', False)
        missing_pct = feature.get('missing_percentage', 0)
        
        print(f"{feature['feature']:<20} {corr:<12.6f} {p_val:<12.2e} {str(sig):<12} {missing_pct:<10.2f}")
    
    print(f"\nTop 10 Categorical Features by F-statistic:")
    print("-" * 80)
    print(f"{'Feature':<20} {'F-statistic':<12} {'P-value':<12} {'Significant':<12} {'Missing%':<10}")
    print("-" * 80)
    
    for feature in categorical_features[:10]:
        f_stat = feature.get('f_statistic', np.nan)
        p_val = feature.get('p_value', np.nan)
        sig = feature.get('significant', False)
        missing_pct = feature.get('missing_percentage', 0)
        
        print(f"{feature['feature']:<20} {f_stat:<12.6f} {p_val:<12.2e} {str(sig):<12} {missing_pct:<10.2f}")
    
    # Summary statistics
    significant_continuous = sum(1 for f in continuous_features if f.get('significant', False))
    significant_categorical = sum(1 for f in categorical_features if f.get('significant', False))
    
    print(f"\nSummary:")
    print(f"Total features analyzed: {len(results)}")
    print(f"Continuous features: {len(continuous_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Significant continuous features: {significant_continuous}")
    print(f"Significant categorical features: {significant_categorical}")
    print(f"Total significant features: {significant_continuous + significant_categorical}")
    
    return {
        'continuous_features': continuous_features,
        'categorical_features': categorical_features,
        'summary': {
            'total_features': len(results),
            'continuous_count': len(continuous_features),
            'categorical_count': len(categorical_features),
            'significant_continuous': significant_continuous,
            'significant_categorical': significant_categorical,
            'total_significant': significant_continuous + significant_categorical
        }
    }

def main():
    """Main analysis function"""
    print("Starting Lightweight Correlation Analysis...")
    
    # Load data
    df = load_data_sample()
    
    # Analyze feat_e_3 specifically
    feat_e_3_results = analyze_feat_e_3_correlation(df)
    
    # Analyze all features
    all_features_results = analyze_all_features_correlation(df)
    
    # Save results
    results = {
        'feat_e_3_analysis': feat_e_3_results,
        'all_features_analysis': all_features_results,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/correlation_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()
