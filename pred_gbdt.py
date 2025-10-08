# Environment setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured")

# Core imports
import gc
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# GPU libraries
import cupy as cp

# ==== RMM / cuDF allocator 초기화 ====
try:
    import rmm, cudf
    initial_pool_size_bytes = 10 * 1024 * 1024 * 1024
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=initial_pool_size_bytes,
        managed_memory=True,
    )
    print("✅ RMM initialized (pool=10GB, managed_memory=True)")
except (ImportError, RuntimeError) as e:
    print(f"⚠️ RMM init skipped: {e}")

# Set GPU device
cp.cuda.Device(0).use()

# NVTabular
import nvtabular as nvt
from nvtabular import ops
from merlin.io import Dataset

# ML libraries
import xgboost as xgb
import catboost as cb

# Configuration
import argparse

# Import common functions
from utils import clear_gpu_memory, print_memory, calculate_competition_score

# ============================================================================
# Calibration Classes
# ============================================================================

class TemperatureScaling:
    """Temperature scaling for calibration"""
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Find optimal temperature using validation set"""
        from scipy.optimize import minimize
        
        def nll_loss(temp):
            scaled_logits = logits / temp
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
            return loss
        
        result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        print(f"      Optimal temperature: {self.temperature:.4f}")
    
    def predict_proba(self, logits):
        """Apply temperature scaling"""
        scaled_logits = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))
        return probs

# ============================================================================
# Helper Functions
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GBDT Prediction Script')
    
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model (e.g., result_GBDT_xgboost/20231201_120000)')
    parser.add_argument('--test-path', type=str, default='data/test.parquet',
                        help='Path to test data')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save submission file (default: {model_dir}/submission.csv)')
    parser.add_argument('--use-calibration', action='store_true', default=True,
                        help='Use calibration model for predictions (default: True)')
    parser.add_argument('--no-calibration', dest='use_calibration', action='store_false',
                        help='Disable calibration')
    parser.add_argument('--temp-dir', type=str, default='tmp',
                        help='Temporary directory for intermediate files')
    
    return parser.parse_args()

def load_model_and_metadata(model_dir):
    """Load model, calibrator, and metadata"""
    print(f"\n📦 Loading model from {model_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"   Model: {metadata['model_name']}")
    print(f"   Validation Score: {metadata['val_score']:.6f}")
    print(f"   Calibration method: {metadata['calibration_method']}")
    
    # Find model file
    model_name = metadata['model_name']
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name) and (f.endswith('.json') or f.endswith('.cbm'))]
    
    if not model_files:
        raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    model_path = os.path.join(model_dir, model_files[0])
    print(f"   Loading model from {model_path}")
    
    # Load model
    if model_name == 'xgboost':
        model = xgb.Booster()
        model.load_model(model_path)
    else:  # catboost
        model = cb.CatBoostClassifier()
        model.load_model(model_path)
    
    print("   ✅ Model loaded")
    
    return model, metadata

def load_and_preprocess_test(test_path, workflow_dir, temp_dir):
    """Load and preprocess test data"""
    print(f"\n📦 Loading and preprocessing test data from {test_path}...")
    
    # Create temp directory if needed
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load workflow
    workflow_path = os.path.join(workflow_dir, 'workflow')
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")
    
    workflow = nvt.Workflow.load(workflow_path)
    print(f"   ✅ Workflow loaded from {workflow_path}")
    
    # Create temp test file without excluded columns
    temp_test_path = os.path.join(temp_dir, 'test_no_seq.parquet')
    if not os.path.exists(temp_test_path):
        print("   Creating temp test file without excluded columns...")
        pf = pq.ParquetFile(test_path)
        cols = [c for c in pf.schema.names if c not in ['seq', 'l_feat_20', 'l_feat_23', '']]
        print(f"   Total columns: {len(pf.schema.names)}, Using: {len(cols)} (excluded 'seq', 'l_feat_20', 'l_feat_23')")
        
        df_test = pd.read_parquet(test_path, columns=cols)
        df_test.to_parquet(temp_test_path, index=False)
        del df_test
        gc.collect()
        print("   ✅ Temp test file created")
    else:
        print(f"   ✅ Using existing temp test file: {temp_test_path}")
    
    # Apply workflow to test data
    print("   Applying workflow to test data...")
    test_dataset = Dataset(temp_test_path, engine='parquet', part_size='64MB')
    
    # Recreate workflow components without 'clicked'
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] +
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]
    )
    
    cat_features = true_categorical >> ops.Categorify(freq_threshold=0, max_size=50000)
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)
    
    # Create test workflow without 'clicked'
    test_workflow = nvt.Workflow(cat_features + cont_features)
    test_workflow.fit(test_dataset)
    gdf_test = test_workflow.transform(test_dataset).to_ddf().compute()
    
    # Get test IDs from original file
    test_df_original = pd.read_parquet(test_path, columns=['ID'])
    test_ids = test_df_original['ID'].values
    del test_df_original
    
    # Convert to float32
    feature_cols = [col for col in gdf_test.columns if col != 'ID']
    X_test = gdf_test[feature_cols].astype('float32', copy=False)
    X_test_np = X_test.to_numpy()
    del X_test, gdf_test
    gc.collect()
    
    print(f"   ✅ Test data loaded and preprocessed: {X_test_np.shape}")
    
    return X_test_np, test_ids, temp_test_path

def find_best_calibration(model, workflow_dir, model_name, temp_dir, test_ratio=0.5):
    """
    Find best calibration method by comparing all methods on calibration test set
    
    Returns:
        best_method: Best calibration method name ('none', 'isotonic', 'sigmoid', or 'temperature')
        best_calibrator: Fitted calibrator (or None if 'none' is best)
        calibration_results: Dict of all results
    """
    print("\n" + "="*70)
    print("🎯 Finding Best Calibration Method")
    print("="*70)
    
    # Load and preprocess train_c
    print(f"\n📦 Loading and preprocessing calibration data (train_c)...")
    cal_path = "data/train_c.parquet"
    X_cal, y_cal, temp_cal_path = load_and_preprocess_calibration(cal_path, workflow_dir, temp_dir)
    
    # Get raw predictions on train_c
    print("   Collecting predictions on train_c...")
    if model_name == 'xgboost':
        dcal = xgb.DMatrix(X_cal)
        cal_preds = model.predict(dcal)
        del dcal
    else:
        cal_preds = model.predict_proba(X_cal)[:, 1]
    
    # Split train_c: balanced set for fitting, rest for testing
    pos_idx = np.where(y_cal == 1)[0]
    neg_idx = np.where(y_cal == 0)[0]
    
    print(f"   train_c total: {len(y_cal):,} samples (pos: {len(pos_idx):,}, neg: {len(neg_idx):,})")
    
    # Sample balanced training set
    np.random.seed(42)
    n_pos_train = int(len(pos_idx) * test_ratio)
    pos_train_idx = np.random.choice(pos_idx, size=n_pos_train, replace=False)
    pos_test_idx = np.setdiff1d(pos_idx, pos_train_idx)
    
    neg_train_idx = np.random.choice(neg_idx, size=n_pos_train, replace=False)
    neg_test_idx = np.setdiff1d(neg_idx, neg_train_idx)
    
    cal_train_idx = np.concatenate([pos_train_idx, neg_train_idx])
    cal_test_idx = np.concatenate([pos_test_idx, neg_test_idx])
    
    np.random.shuffle(cal_train_idx)
    np.random.shuffle(cal_test_idx)
    
    print(f"   → Calibration fit set: {len(cal_train_idx):,} samples (balanced, pos={len(pos_train_idx):,}, neg={len(neg_train_idx):,})")
    print(f"   → Calibration test set: {len(cal_test_idx):,} samples (imbalanced, pos={len(pos_test_idx):,}, neg={len(neg_test_idx):,})")
    
    # Test all calibration methods
    methods = ['none', 'isotonic', 'sigmoid', 'temperature']
    results = {}
    calibrators = {}
    
    print(f"\n🔬 Testing calibration methods...")
    
    for method in methods:
        print(f"\n   [{method.upper()}]")
        
        if method == 'none':
            # No calibration - use raw predictions
            cal_test_calibrated = cal_preds[cal_test_idx]
            calibrators[method] = None
        else:
            # Get training data for calibration
            cal_train_preds = cal_preds[cal_train_idx]
            cal_train_targets = y_cal[cal_train_idx]
            
            # Fit calibrator
            if method == 'temperature':
                # Convert probabilities to logits
                cal_train_preds_clipped = np.clip(cal_train_preds, 1e-7, 1 - 1e-7)
                train_logits = np.log(cal_train_preds_clipped / (1 - cal_train_preds_clipped))
                
                calibrator = TemperatureScaling()
                calibrator.fit(train_logits, cal_train_targets)
                
                # Apply to test set
                cal_test_preds_clipped = np.clip(cal_preds[cal_test_idx], 1e-7, 1 - 1e-7)
                test_logits = np.log(cal_test_preds_clipped / (1 - cal_test_preds_clipped))
                cal_test_calibrated = calibrator.predict_proba(test_logits)
                
            elif method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(cal_train_preds, cal_train_targets)
                
                cal_test_calibrated = calibrator.predict(cal_preds[cal_test_idx])
                
            else:  # sigmoid
                calibrator = LogisticRegression()
                calibrator.fit(cal_train_preds.reshape(-1, 1), cal_train_targets)
                
                cal_test_calibrated = calibrator.predict_proba(cal_preds[cal_test_idx].reshape(-1, 1))[:, 1]
            
            calibrators[method] = calibrator
        
        # Evaluate on calibration test set
        cal_test_targets = y_cal[cal_test_idx]
        score, ap, wll = calculate_competition_score(cal_test_targets, cal_test_calibrated)
        
        # Store results
        results[method] = {
            'score': score,
            'ap': ap,
            'wll': wll
        }
        
        print(f"      Score: {score:.6f}, AP: {ap:.6f}, WLL: {wll:.6f}")
    
    # Find best method based on calibration test score
    best_method = max(results.keys(), key=lambda m: results[m]['score'])
    best_score = results[best_method]['score']
    
    print(f"\n🏆 Best Method: {best_method.upper()}")
    print(f"   Score: {best_score:.6f}")
    print(f"   AP: {results[best_method]['ap']:.6f}")
    print(f"   WLL: {results[best_method]['wll']:.6f}")
    
    # Compare with no calibration
    if best_method != 'none':
        improvement = best_score - results['none']['score']
        print(f"   Improvement over raw: {improvement:+.6f}")
    
    # Cleanup temp files
    if os.path.exists(temp_cal_path):
        os.remove(temp_cal_path)
    
    print("="*70)
    
    return best_method, calibrators[best_method], results

def load_and_preprocess_calibration(data_path, workflow_dir, temp_dir):
    """Load and preprocess calibration/validation data using the same workflow as test"""
    import shutil
    
    # Create temp directory if needed
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load workflow
    workflow_path = os.path.join(workflow_dir, 'workflow')
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow not found: {workflow_path}")
    
    workflow = nvt.Workflow.load(workflow_path)
    
    # Create temp file without excluded columns
    temp_path = os.path.join(temp_dir, f'{os.path.basename(data_path).replace(".parquet", "_no_seq.parquet")}')
    
    if not os.path.exists(temp_path):
        pf = pq.ParquetFile(data_path)
        cols = [c for c in pf.schema.names if c not in ['seq', 'l_feat_20', 'l_feat_23', '']]
        
        df = pd.read_parquet(data_path, columns=cols)
        df.to_parquet(temp_path, index=False)
        del df
        gc.collect()
    
    # Apply workflow
    dataset = Dataset(temp_path, engine='parquet', part_size='64MB')
    
    # Recreate workflow components
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +
        [f'feat_b_{i}' for i in range(1, 7)] +
        [f'feat_c_{i}' for i in range(1, 9)] +
        [f'feat_d_{i}' for i in range(1, 7)] +
        [f'feat_e_{i}' for i in range(1, 11)] +
        [f'history_a_{i}' for i in range(1, 8)] +
        [f'history_b_{i}' for i in range(1, 31)] +
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]
    )
    
    cat_features = true_categorical >> ops.Categorify(freq_threshold=0, max_size=50000)
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)
    
    # Add 'clicked' if exists
    from nvtabular import ColumnSelector
    if 'clicked' in dataset.to_ddf().columns:
        cal_workflow = nvt.Workflow(cat_features + cont_features + ['clicked'])
    else:
        cal_workflow = nvt.Workflow(cat_features + cont_features)
    
    cal_workflow.fit(dataset)
    gdf = cal_workflow.transform(dataset).to_ddf().compute()
    
    # Extract features and target
    if 'clicked' in gdf.columns:
        y = gdf['clicked'].to_numpy()
        X = gdf.drop('clicked', axis=1)
    else:
        y = None
        X = gdf
    
    feature_cols = [col for col in X.columns if col != 'ID']
    X = X[feature_cols].astype('float32', copy=False)
    X_np = X.to_numpy()
    del X, gdf
    gc.collect()
    
    return X_np, y, temp_path

def predict_and_save(model, best_method, best_calibrator, X_test, test_ids, model_name, output_path):
    """Generate predictions and save submission file"""
    print(f"\n🔮 Predicting on test data...")
    print(f"   Using calibration method: {best_method.upper()}")
    
    # Get raw predictions
    if model_name == 'xgboost':
        dtest = xgb.DMatrix(X_test)
        test_predictions = model.predict(dtest)
        del dtest
    else:  # catboost
        test_predictions = model.predict_proba(X_test)[:, 1]
    
    print(f"   Raw prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    
    # Apply calibration if needed
    if best_method != 'none' and best_calibrator is not None:
        print(f"   Applying {best_method} calibration...")
        
        if best_method == 'temperature':
            # Convert probabilities to logits
            test_preds_clipped = np.clip(test_predictions, 1e-7, 1 - 1e-7)
            logits = np.log(test_preds_clipped / (1 - test_preds_clipped))
            test_predictions = best_calibrator.predict_proba(logits)
        elif best_method == 'isotonic':
            test_predictions = best_calibrator.predict(test_predictions)
        else:  # sigmoid
            test_predictions = best_calibrator.predict_proba(test_predictions.reshape(-1, 1))[:, 1]
        
        print(f"   Calibrated prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    else:
        print("   ✅ Using raw predictions (no calibration)")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'clicked': test_predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to {output_path}")
    print(f"   Samples: {len(submission_df):,}")
    print(f"   Prediction stats:")
    print(f"      Mean: {test_predictions.mean():.6f}")
    print(f"      Std: {test_predictions.std():.6f}")
    print(f"      Min: {test_predictions.min():.6f}")
    print(f"      Max: {test_predictions.max():.6f}")

def main():
    """Main prediction pipeline"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🔮 GBDT Prediction Pipeline")
    print("="*70)
    print(f"   Model directory: {args.model_dir}")
    print(f"   Test data: {args.test_path}")
    print(f"   Use calibration: {args.use_calibration}")
    
    # Verify paths
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test data not found: {args.test_path}")
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_dir, 'submission.csv')
    
    # Load model and metadata
    model, metadata = load_model_and_metadata(args.model_dir)
    model_name = metadata['model_name']
    
    # Determine workflow directory (parent of model_dir)
    workflow_dir = os.path.dirname(args.model_dir)
    if not os.path.exists(os.path.join(workflow_dir, 'workflow')):
        # If workflow not found in parent, try the model_dir itself
        workflow_dir = args.model_dir
        if not os.path.exists(os.path.join(workflow_dir, 'workflow')):
            raise FileNotFoundError(f"Workflow not found in {workflow_dir}")
    
    # Find best calibration method if enabled
    best_method = 'none'
    best_calibrator = None
    
    if args.use_calibration:
        best_method, best_calibrator, calibration_results = find_best_calibration(
            model, workflow_dir, model_name, args.temp_dir
        )
    else:
        print("\n⚠️  Calibration disabled by user (--no-calibration)")
    
    # Load and preprocess test data
    X_test, test_ids, temp_test_path = load_and_preprocess_test(
        args.test_path, 
        workflow_dir, 
        args.temp_dir
    )
    
    # Predict and save
    predict_and_save(
        model, 
        best_method,
        best_calibrator,
        X_test, 
        test_ids, 
        model_name,
        args.output_path
    )
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    if os.path.exists(temp_test_path):
        os.remove(temp_test_path)
        print(f"   ✅ Removed {temp_test_path}")
    
    if os.path.exists(args.temp_dir) and not os.listdir(args.temp_dir):
        os.rmdir(args.temp_dir)
        print(f"   ✅ Removed empty directory {args.temp_dir}")
    
    clear_gpu_memory()
    print_memory()
    
    print("\n" + "🎉"*35)
    print("PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"✅ Submission saved to: {args.output_path}")
    print("="*70)

if __name__ == "__main__":
    main()

