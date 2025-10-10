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
import numpy as np
import pandas as pd

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

# ML libraries
import xgboost as xgb
import catboost as cb

# Configuration
import argparse

# Import common functions
from utils import clear_gpu_memory, print_memory
from data_loader import load_processed_data_gbdt

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
    parser.add_argument('--test-path', type=str, default='data/proc_test',
                        help='Path to preprocessed test data (default: data/proc_test)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save submission file (default: {model_dir}/submission.csv)')
    
    return parser.parse_args()

def load_model_and_metadata(model_dir):
    """Load model, calibrator, and metadata"""
    print(f"\n📦 Loading model from {model_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"   Model: {metadata['model_name']}")
    print(f"   Validation Score: {metadata['val_score']:.6f}")
    print(f"   Calibration method: {metadata.get('calibration_method', 'none')}")
    
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
    
    # Load calibrator if exists
    calibrator = None
    calibrator_path = os.path.join(model_dir, 'calibrator.pkl')
    if os.path.exists(calibrator_path):
        import pickle
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"   ✅ Calibrator loaded from {calibrator_path}")
    else:
        print("   ℹ️  No calibrator found (will use raw predictions)")
    
    return model, calibrator, metadata

def load_test_data(test_path):
    """Load preprocessed test data"""
    print(f"\n📦 Loading preprocessed test data from {test_path}...")
    
    # Load test data using the same loader as HPO
    X_test, _ = load_processed_data_gbdt(test_path)
    
    # Look for original parquet file to get IDs
    original_test_path = test_path.replace('proc_test', 'test.parquet')
    if os.path.exists(original_test_path):
        test_df_original = pd.read_parquet(original_test_path, columns=['ID'])
        test_ids = test_df_original['ID'].values
        del test_df_original
    else:
        # If original file not found, generate sequential IDs
        print(f"   ⚠️ Original test file not found at {original_test_path}, generating sequential IDs")
        test_ids = np.arange(len(X_test))
    
    gc.collect()
    
    print(f"   ✅ Test data loaded: {X_test.shape}")
    print(f"   ✅ Test IDs: {len(test_ids):,}")
    
    return X_test, test_ids

def predict_and_save(model, calibrator, calibration_method, X_test, test_ids, model_name, output_path):
    """Generate predictions and save submission file"""
    print(f"\n🔮 Predicting on test data...")
    
    # Get raw predictions
    if model_name == 'xgboost':
        dtest = xgb.DMatrix(X_test)
        test_predictions = model.predict(dtest)
        del dtest
    else:  # catboost
        test_predictions = model.predict_proba(X_test)[:, 1]
    
    print(f"   Raw prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    
    # Apply calibration if calibrator exists
    if calibrator is not None:
        print(f"   Applying {calibration_method} calibration...")
        
        if calibration_method == 'temperature':
            # Convert probabilities to logits
            test_preds_clipped = np.clip(test_predictions, 1e-7, 1 - 1e-7)
            logits = np.log(test_preds_clipped / (1 - test_preds_clipped))
            test_predictions = calibrator.predict_proba(logits)
        elif calibration_method == 'isotonic':
            test_predictions = calibrator.predict(test_predictions)
        elif calibration_method == 'sigmoid':
            test_predictions = calibrator.predict_proba(test_predictions.reshape(-1, 1))[:, 1]
        else:
            print(f"   ⚠️  Unknown calibration method: {calibration_method}")
            print("   Using raw predictions")
        
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
    
    # Verify paths
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test data not found: {args.test_path}")
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_dir, 'submission.csv')
    
    # Load model, calibrator, and metadata
    model, calibrator, metadata = load_model_and_metadata(args.model_dir)
    model_name = metadata['model_name']
    calibration_method = metadata.get('calibration_method', 'none')
    
    # Load preprocessed test data
    X_test, test_ids = load_test_data(args.test_path)
    
    # Predict and save
    predict_and_save(
        model, 
        calibrator,
        calibration_method,
        X_test, 
        test_ids, 
        model_name,
        args.output_path
    )
    
    clear_gpu_memory()
    print_memory()
    
    print("\n" + "🎉"*35)
    print("PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"✅ Submission saved to: {args.output_path}")
    print("="*70)

if __name__ == "__main__":
    main()

