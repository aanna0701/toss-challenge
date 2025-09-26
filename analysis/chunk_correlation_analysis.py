#!/usr/bin/env python3
"""
Chunk-based Correlation Analysis
- Processes data in small chunks to manage memory
- Computes actual correlations between features and click rate
- Uses Spearman correlation for continuous variables
- Uses Chi-square test for categorical variables
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def process_data_in_chunks(file_path, chunk_size=50000):
    """Process data in chunks and compute correlations"""
    print("Processing data in chunks for correlation analysis...")
    
    # Initialize correlation accumulators
    correlations = {}
    chi2_stats = {}
    sample_counts = {}
    
    chunk_count = 0
    total_rows_processed = 0
    
    # Read data in chunks
    for chunk in pd.read_parquet(file_path, engine='pyarrow', chunksize=chunk_size):
        chunk_count += 1
        total_rows_processed += len(chunk)
        
        if chunk_count % 10 == 0:
            print(f"Processed {chunk_count} chunks, {total_rows_processed:,} rows")
        
        # Analyze each feature in this chunk
        for column in chunk.columns:
            if column in ['clicked', 'seq']:
                continue
                
            if column not in correlations:
                correlations[column] = []
                chi2_stats[column] = []
                sample_counts[column] = 0
            
            # Skip if all values are missing
            if chunk[column].isna().all():
                continue
            
            # Determine if continuous or categorical
            clean_series = chunk[column].dropna()
            if len(clean_series) < 10:  # Need minimum samples
                continue
                
            unique_count = clean_series.nunique()
            total_count = len(clean_series)
            
            if unique_count > 20 and unique_count / total_count > 0.1:
                # Continuous variable - use Spearman correlation
                non_missing_mask = chunk[column].notna()
                if non_missing_mask.sum() > 10:
                    try:
                        corr_coef, p_value = spearmanr(
                            chunk.loc[non_missing_mask, column], 
                            chunk.loc[non_missing_mask, 'clicked']
                        )
                        if not np.isnan(corr_coef):
                            correlations[column].append(corr_coef)
                            sample_counts[column] += non_missing_mask.sum()
                    except:
                        pass
            else:
                # Categorical variable - use Chi-square test
                try:
                    contingency_table = pd.crosstab(chunk[column], chunk['clicked'])
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        if not np.isnan(chi2):
                            chi2_stats[column].append(chi2)
                            sample_counts[column] += len(chunk)
                except:
                    pass
    
    print(f"Completed processing {chunk_count} chunks, {total_rows_processed:,} total rows")
    return correlations, chi2_stats, sample_counts

def compute_final_correlations(correlations, chi2_stats, sample_counts):
    """Compute final correlation statistics from accumulated data"""
    print("\nComputing final correlation statistics...")
    
    results = []
    
    # Process continuous variables (Spearman correlations)
    for feature, corr_list in correlations.items():
        if len(corr_list) > 0:
            # Average correlation across chunks
            avg_corr = np.mean(corr_list)
            std_corr = np.std(corr_list)
            
            # Approximate p-value using t-test
            n = sample_counts[feature]
            if n > 2:
                t_stat = avg_corr * np.sqrt((n - 2) / (1 - avg_corr**2))
                # Approximate p-value (two-tailed)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            else:
                p_value = 1.0
            
            results.append({
                'feature': feature,
                'type': 'continuous',
                'correlation_coefficient': avg_corr,
                'correlation_std': std_corr,
                'p_value': p_value,
                'sample_size': n,
                'significant': p_value < 0.05,
                'chunks_analyzed': len(corr_list)
            })
    
    # Process categorical variables (Chi-square statistics)
    for feature, chi2_list in chi2_stats.items():
        if len(chi2_list) > 0:
            # Average chi-square statistic across chunks
            avg_chi2 = np.mean(chi2_list)
            std_chi2 = np.std(chi2_list)
            
            # Approximate p-value
            n = sample_counts[feature]
            if n > 0:
                # For chi-square, we need degrees of freedom
                # Approximate as 1 (since we're testing independence)
                p_value = 1 - stats.chi2.cdf(avg_chi2, 1)
            else:
                p_value = 1.0
            
            results.append({
                'feature': feature,
                'type': 'categorical',
                'chi2_statistic': avg_chi2,
                'chi2_std': std_chi2,
                'p_value': p_value,
                'sample_size': n,
                'significant': p_value < 0.05,
                'chunks_analyzed': len(chi2_list)
            })
    
    return results

def analyze_feat_e_3_detailed(file_path, sample_size=100000):
    """Detailed analysis of feat_e_3 missing pattern"""
    print("\n" + "="*60)
    print("DETAILED FEAT_E_3 ANALYSIS")
    print("="*60)
    
    # Load a sample for detailed analysis
    df_sample = pd.read_parquet(file_path, engine='pyarrow')
    df_sample = df_sample.sample(n=min(sample_size, len(df_sample)), random_state=42)
    
    print(f"Sample size: {len(df_sample):,}")
    
    # Create missing indicator
    df_sample['feat_e_3_missing'] = df_sample['feat_e_3'].isna().astype(int)
    
    # Calculate click rates
    total_rows = len(df_sample)
    missing_rows = df_sample['feat_e_3_missing'].sum()
    non_missing_rows = total_rows - missing_rows
    
    click_rate_missing = df_sample[df_sample['feat_e_3_missing'] == 1]['clicked'].mean()
    click_rate_non_missing = df_sample[df_sample['feat_e_3_missing'] == 0]['clicked'].mean()
    overall_click_rate = df_sample['clicked'].mean()
    
    print(f"feat_e_3 missing rows: {missing_rows:,} ({missing_rows/total_rows*100:.2f}%)")
    print(f"feat_e_3 non-missing rows: {non_missing_rows:,} ({non_missing_rows/total_rows*100:.2f}%)")
    print(f"\nClick rates:")
    print(f"  Overall: {overall_click_rate:.6f}")
    print(f"  When feat_e_3 missing: {click_rate_missing:.6f}")
    print(f"  When feat_e_3 non-missing: {click_rate_non_missing:.6f}")
    print(f"  Difference: {click_rate_non_missing - click_rate_missing:.6f}")
    
    # Statistical significance test
    contingency_table = pd.crosstab(df_sample['feat_e_3_missing'], df_sample['clicked'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nStatistical significance (Chi-square test):")
    print(f"  Chi-square statistic: {chi2:.6f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Effect size (Cramer's V)
    cramers_v = np.sqrt(chi2 / (total_rows * (min(contingency_table.shape) - 1)))
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

def main():
    """Main analysis function"""
    print("Starting Chunk-based Correlation Analysis...")
    
    file_path = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/train.parquet'
    
    # Detailed feat_e_3 analysis
    feat_e_3_results = analyze_feat_e_3_detailed(file_path)
    
    # Process all features in chunks
    correlations, chi2_stats, sample_counts = process_data_in_chunks(file_path)
    
    # Compute final correlations
    all_features_results = compute_final_correlations(correlations, chi2_stats, sample_counts)
    
    # Separate continuous and categorical results
    continuous_results = [r for r in all_features_results if r['type'] == 'continuous']
    categorical_results = [r for r in all_features_results if r['type'] == 'categorical']
    
    # Sort by significance
    continuous_results.sort(key=lambda x: abs(x['correlation_coefficient']), reverse=True)
    categorical_results.sort(key=lambda x: x['chi2_statistic'], reverse=True)
    
    print(f"\nTop 10 Continuous Features by Correlation Strength:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Correlation':<12} {'P-value':<12} {'Significant':<12} {'Sample Size':<12}")
    print("-" * 80)
    
    for feature in continuous_results[:10]:
        corr = feature['correlation_coefficient']
        p_val = feature['p_value']
        sig = feature['significant']
        sample_size = feature['sample_size']
        
        print(f"{feature['feature']:<20} {corr:<12.6f} {p_val:<12.2e} {str(sig):<12} {sample_size:<12,}")
    
    print(f"\nTop 10 Categorical Features by Chi-square Statistic:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Chi-square':<12} {'P-value':<12} {'Significant':<12} {'Sample Size':<12}")
    print("-" * 80)
    
    for feature in categorical_results[:10]:
        chi2 = feature['chi2_statistic']
        p_val = feature['p_value']
        sig = feature['significant']
        sample_size = feature['sample_size']
        
        print(f"{feature['feature']:<20} {chi2:<12.6f} {p_val:<12.2e} {str(sig):<12} {sample_size:<12,}")
    
    # Summary
    significant_continuous = sum(1 for f in continuous_results if f['significant'])
    significant_categorical = sum(1 for f in categorical_results if f['significant'])
    
    print(f"\nSummary:")
    print(f"Total features analyzed: {len(all_features_results)}")
    print(f"Continuous features: {len(continuous_results)}")
    print(f"Categorical features: {len(categorical_results)}")
    print(f"Significant continuous features: {significant_continuous}")
    print(f"Significant categorical features: {significant_categorical}")
    print(f"Total significant features: {significant_continuous + significant_categorical}")
    
    # Save results
    results = {
        'feat_e_3_analysis': feat_e_3_results,
        'continuous_features': continuous_results,
        'categorical_features': categorical_results,
        'summary': {
            'total_features': len(all_features_results),
            'continuous_count': len(continuous_results),
            'categorical_count': len(categorical_results),
            'significant_continuous': significant_continuous,
            'significant_categorical': significant_categorical,
            'total_significant': significant_continuous + significant_categorical
        },
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/chunk_correlation_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("Analysis completed!")

if __name__ == "__main__":
    main()
