"""
Ensemble predictions by averaging multiple model results.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def find_submission_file(result_folder):
    """
    Find submission CSV file in the given folder.
    
    Args:
        result_folder: Path to result folder
        
    Returns:
        Path to submission CSV file
    """
    folder_path = Path(result_folder)
    
    # Look for submission_*.csv files
    csv_files = list(folder_path.glob("submission_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No submission CSV file found in {result_folder}")
    
    if len(csv_files) > 1:
        print(f"Warning: Multiple submission files found in {result_folder}. Using the first one: {csv_files[0].name}")
    
    return csv_files[0]


def load_predictions(csv_path):
    """
    Load predictions from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with ID and predictions
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path.name}")
    return df


def ensemble_average(result_folders, output_dir=None):
    """
    Average predictions from multiple result folders.
    
    Args:
        result_folders: List of paths to result folders
        output_dir: Directory to save the averaged result (default: ./ensemble_results)
        
    Returns:
        DataFrame with averaged predictions
    """
    print(f"\n{'='*80}")
    print(f"Ensembling {len(result_folders)} model results by averaging...")
    print(f"{'='*80}\n")
    
    # Load all predictions
    all_dfs = []
    for i, folder in enumerate(result_folders, 1):
        print(f"[{i}/{len(result_folders)}] Processing: {folder}")
        csv_path = find_submission_file(folder)
        df = load_predictions(csv_path)
        all_dfs.append(df)
    
    # Check that all DataFrames have the same IDs
    print("\nVerifying ID consistency across all files...")
    first_ids = all_dfs[0]['ID'].values
    for i, df in enumerate(all_dfs[1:], 2):
        if not np.array_equal(first_ids, df['ID'].values):
            raise ValueError(f"ID mismatch between file 1 and file {i}")
    print("✓ All IDs match!")
    
    # Average predictions
    print("\nAveraging predictions...")
    averaged_df = all_dfs[0][['ID']].copy()
    
    # Stack all prediction columns and compute mean
    prediction_arrays = [df['clicked'].values for df in all_dfs]
    averaged_predictions = np.mean(prediction_arrays, axis=0)
    averaged_df['clicked'] = averaged_predictions
    
    # Print statistics
    print(f"\nEnsemble Statistics:")
    print(f"  Number of models: {len(result_folders)}")
    print(f"  Number of predictions: {len(averaged_df)}")
    print(f"  Mean prediction: {averaged_predictions.mean():.6f}")
    print(f"  Std prediction: {averaged_predictions.std():.6f}")
    print(f"  Min prediction: {averaged_predictions.min():.6f}")
    print(f"  Max prediction: {averaged_predictions.max():.6f}")
    
    # Print individual model statistics
    print(f"\nIndividual Model Statistics:")
    for i, (folder, df) in enumerate(zip(result_folders, all_dfs), 1):
        folder_name = Path(folder).parent.name + "/" + Path(folder).name
        mean_pred = df['clicked'].mean()
        print(f"  Model {i} ({folder_name}): mean={mean_pred:.6f}")
    
    # Save result
    if output_dir is None:
        output_dir = Path("./ensemble_results")
    else:
        output_dir = Path(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = output_dir / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = output_folder / f"submission_ensemble_avg_{len(result_folders)}models.csv"
    averaged_df.to_csv(output_path, index=False)
    print(f"\n✓ Ensemble result saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'num_models': len(result_folders),
        'result_folders': [str(Path(f).resolve()) for f in result_folders],
        'ensemble_method': 'average',
        'num_predictions': len(averaged_df),
        'statistics': {
            'mean': float(averaged_predictions.mean()),
            'std': float(averaged_predictions.std()),
            'min': float(averaged_predictions.min()),
            'max': float(averaged_predictions.max())
        }
    }
    
    import json
    metadata_path = output_folder / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"\n{'='*80}")
    print(f"Ensemble complete!")
    print(f"{'='*80}\n")
    
    return averaged_df


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble predictions by averaging multiple model results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Average 2 models
  python ensemble_average.py \\
    result_GBDT_xgboost/20251010_225523 \\
    result_GBDT_catboost/20251010_231501
  
  # Average 3 models
  python ensemble_average.py \\
    result_GBDT_xgboost/20251010_225523 \\
    result_GBDT_catboost/20251010_231501 \\
    result_model3/20251010_120000
  
  # Specify custom output directory
  python ensemble_average.py \\
    result_GBDT_xgboost/20251010_225523 \\
    result_GBDT_catboost/20251010_231501 \\
    --output-dir custom_ensemble_results
        """
    )
    
    parser.add_argument(
        'result_folders',
        nargs='+',
        help='Paths to result folders containing submission CSV files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for ensemble results (default: ./ensemble_results)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.result_folders) < 2:
        parser.error("At least 2 result folders are required for ensemble")
    
    # Check that all folders exist
    for folder in args.result_folders:
        if not os.path.exists(folder):
            parser.error(f"Folder does not exist: {folder}")
    
    # Run ensemble
    ensemble_average(args.result_folders, args.output_dir)


if __name__ == "__main__":
    main()

