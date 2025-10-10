# Environment setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# NOTE: GBDT는 단일 GPU 사용. 원하는 GPU 인덱스를 설정하거나 환경변수로 전달
# Example: export CUDA_VISIBLE_DEVICES=4 python train_gbdt.py
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured")

# Required libraries and versions
required_libs = {
    'nvtabular': '23.08.00',
    'cudf': '23.10',
    'cupy': '13.6',
    'xgboost': '3.0',
    'catboost': '1.2',
    'dask': '2023.9',
    'pandas': '1.5',
    'numpy': '1.24',
    'scikit-learn': '1.7',
    'psutil': '5.9',
    'pyarrow': '12.0'
}

# Check installed versions
import importlib

# Suppress deprecation warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        import pkg_resources
    except ImportError:
        pkg_resources = None

missing_libs = []
all_good = True

for lib, required_version in required_libs.items():
    try:
        # Map library names for import
        import_name = lib
        if lib == 'scikit-learn':
            import_name = 'sklearn'

        # Check if library is installed
        module = importlib.import_module(import_name)

        # Get installed version
        try:
            if hasattr(module, '__version__'):
                installed_version = module.__version__
            elif pkg_resources:
                installed_version = pkg_resources.get_distribution(lib).version
            else:
                installed_version = 'unknown'
        except (AttributeError, ImportError):
            installed_version = 'unknown'

        # Simple version check (lenient)
        if installed_version == 'unknown':
            status = "⚠️ "
        else:
            # Extract major version for comparison
            try:
                inst_major = int(installed_version.split('.')[0])
                req_major = int(required_version.split('.')[0])
                status = "✅" if inst_major >= req_major else "⚠️ "
            except (ValueError, IndexError):
                status = "✅"  # If can't parse, assume OK
        
        print(f"{status} {lib:15} {installed_version:15} (required: ≥{required_version})")

    except ImportError:
        missing_libs.append(lib)
        print(f"❌ {lib:15} NOT INSTALLED (required: ≥{required_version})")
        all_good = False

# Report
if missing_libs:
    print(f"\n❌ Missing libraries: {', '.join(missing_libs)}")
    print("Please install them using conda or pip")
elif all_good:
    print("\n✅ All required libraries are installed and compatible!")

# Standard library
import argparse
import json
import pickle
import shutil
import time
import yaml
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

# Third-party libraries
import cupy as cp
import numpy as np

# Initialize RMM for GPU memory management
try:
    import rmm
    
    initial_pool_size_bytes = 10 * 1024 * 1024 * 1024  # 10GB
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
import catboost as cb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Custom modules
from mixup import apply_mixup_to_dataset
from utils import calculate_competition_score, clear_gpu_memory, print_memory

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

@dataclass
class GBDTConfig:
    """Main GBDT configuration"""
    train_t_path: str
    train_v_path: str
    train_c_path: str
    output_dir: str
    temp_dir: str
    model_name: str
    model_params: Dict[str, Any]
    use_mixup: bool = False
    mixup_alpha: float = 0.3
    mixup_ratio: float = 0.5
    calibration_method: str = 'none'

def load_yaml_config(config_path: str = 'config_GBDT.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_config_from_yaml(config_path: str = 'config_GBDT.yaml', preset: str = None) -> GBDTConfig:
    """Create GBDTConfig from YAML file"""
    yaml_config = load_yaml_config(config_path)

    # Use preset if specified
    if preset and 'presets' in yaml_config and preset in yaml_config['presets']:
        preset_config = yaml_config['presets'][preset]
        # Merge preset with base config
        _deep_update(yaml_config, preset_config)

    # Extract data config
    data_config = yaml_config.get('data', {})
    train_t_path = data_config.get('train_t_path', 'data/train_t.parquet')
    train_v_path = data_config.get('train_v_path', 'data/train_v.parquet')
    train_c_path = data_config.get('train_c_path', 'data/train_c.parquet')
    temp_dir = data_config.get('temp_dir', 'tmp')
    
    # Convert temp_dir to absolute path relative to script location
    if not os.path.isabs(temp_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, temp_dir)

    # Extract model config
    model_config = yaml_config.get('model', {})
    final_model_name = model_config.get('name', 'xgboost')

    # Automatically set output directory based on model name
    output_dir = f"result_GBDT_{final_model_name}"
    # Convert to absolute path relative to script location
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get model parameters and extract MixUp settings from model config
    if final_model_name == 'xgboost':
        model_config_full = yaml_config.get('xgboost', {}).copy()
    else:  # catboost
        model_config_full = yaml_config.get('catboost', {}).copy()
    
    # Extract MixUp settings from model config
    use_mixup = model_config_full.pop('use_mixup', False)
    mixup_alpha = model_config_full.pop('mixup_alpha', 0.3)
    mixup_ratio = model_config_full.pop('mixup_ratio', 0.5)
    
    # Extract calibration method from model config
    calibration_method = model_config_full.pop('calibration_method', 'none')
    
    # Prepare model parameters (MixUp settings and calibration method removed)
    model_params = model_config_full
    
    if final_model_name == 'xgboost':
        # Add XGBoost-specific parameters
        model_params['objective'] = 'binary:logistic'
        model_params['predictor'] = 'gpu_predictor'
    else:  # catboost
        # GPU에서는 colsample_bylevel (rsm) 지원하지 않음
        if model_params.get('task_type') == 'GPU' and 'colsample_bylevel' in model_params:
            print("   ⚠️ Removing colsample_bylevel for GPU training (not supported)")
            model_params.pop('colsample_bylevel', None)

    return GBDTConfig(
        train_t_path=train_t_path,
        train_v_path=train_v_path,
        train_c_path=train_c_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        model_name=final_model_name,
        model_params=model_params,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        mixup_ratio=mixup_ratio,
        calibration_method=calibration_method
    )

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update dictionary"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def print_config(config: GBDTConfig):
    """Print configuration details"""
    print(f"\n📋 GBDT Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Train data: {config.train_t_path}")
    print(f"   Val data: {config.train_v_path}")
    print(f"   Calibration data: {config.train_c_path}")
    print(f"   Output: {config.output_dir}")
    print(f"   MixUp enabled: {config.use_mixup}")
    if config.use_mixup:
        print(f"   MixUp alpha: {config.mixup_alpha}")
        print(f"   MixUp ratio: {config.mixup_ratio}")
    print(f"   Calibration method: {config.calibration_method}")

    print("\n🔧 Model Parameters:")
    for key, value in config.model_params.items():
        print(f"   {key}: {value}")

print("✅ All libraries imported successfully")
print(f"   XGBoost: {xgb.__version__}")
print(f"   CatBoost: {cb.__version__}")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Argument parser
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GBDT Training Script')

    parser.add_argument('--config', type=str, default='config_GBDT.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        help='Preset configuration (e.g., xgboost_fast, catboost_deep)')

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration - use command line arguments
print("\n🔧 Command line arguments:")
print(f"   Config file: {args.config}")
print(f"   Preset: {args.preset}")

CONFIG = create_config_from_yaml(args.config, preset=args.preset)

print_config(CONFIG)

# Skip process_data() - use HPO-style independent processing instead
# Each split will be processed independently in run_train() using load_processed_data_gbdt()
print("\n📋 Data processing will be done independently for each split (HPO-style)")
print("   This ensures identical behavior between HPO and training")

def run_train():
    """Run training using pre-processed data"""
    print("\n" + "="*70)
    print("🔄 Loading Pre-Processed Data")
    print("="*70)
    print(f"   Train T: {CONFIG.train_t_path}")
    print(f"   Train V: {CONFIG.train_v_path}")
    print(f"   Train C: {CONFIG.train_c_path}")
    print(f"   ⚡ Using pre-processed data (fast loading)")

    start_load = time.time()

    # Import the same function HPO uses
    from data_loader import load_processed_data_gbdt
    
    # Load TRAIN_T data
    print("\n📦 Loading train_t data...")
    X_train_t, y_train_t = load_processed_data_gbdt(CONFIG.train_t_path)
    print(f"   Train_t shape: {X_train_t.shape}")
    print(f"   Train_t positive ratio: {y_train_t.mean():.6f}")
    
    clear_gpu_memory()

    # Load TRAIN_V data
    print("\n📦 Loading train_v data...")
    X_train_v, y_train_v = load_processed_data_gbdt(CONFIG.train_v_path)
    print(f"   Train_v shape: {X_train_v.shape}")
    print(f"   Train_v positive ratio: {y_train_v.mean():.6f}")
    
    clear_gpu_memory()

    # Load TRAIN_C data
    print("\n📦 Loading train_c data...")
    X_train_c, y_train_c = load_processed_data_gbdt(CONFIG.train_c_path)
    print(f"   Train_c shape: {X_train_c.shape}")
    print(f"   Train_c positive ratio: {y_train_c.mean():.6f}")
    
    clear_gpu_memory()

    print(f"\n⏱️  Loading time: {time.time() - start_load:.1f}s")
    print_memory()
    
    # ========================================================================
    # Split train_c into calibration set and additional training set
    # ========================================================================
    print("\n" + "="*70)
    print("🔀 Splitting train_c for Calibration")
    print("="*70)
    
    pos_idx = np.where(y_train_c == 1)[0]
    neg_idx = np.where(y_train_c == 0)[0]
    
    print(f"   train_c total: {len(y_train_c):,} samples")
    print(f"   Positive: {len(pos_idx):,}, Negative: {len(neg_idx):,}")
    
    # Sample half of positive samples for calibration
    np.random.seed(42)
    n_pos_cal = len(pos_idx) // 2
    pos_cal_idx = np.random.choice(pos_idx, size=n_pos_cal, replace=False)
    pos_train_idx = np.setdiff1d(pos_idx, pos_cal_idx)
    
    # Sample same number of negative samples for calibration
    neg_cal_idx = np.random.choice(neg_idx, size=n_pos_cal, replace=False)
    neg_train_idx = np.setdiff1d(neg_idx, neg_cal_idx)
    
    cal_idx = np.concatenate([pos_cal_idx, neg_cal_idx])
    train_c_remaining_idx = np.concatenate([pos_train_idx, neg_train_idx])
    
    np.random.shuffle(cal_idx)
    np.random.shuffle(train_c_remaining_idx)
    
    # Create calibration set
    X_cal = X_train_c[cal_idx]
    y_cal = y_train_c[cal_idx]
    
    # Create remaining train_c for training
    X_train_c_remaining = X_train_c[train_c_remaining_idx]
    y_train_c_remaining = y_train_c[train_c_remaining_idx]
    
    print(f"\n   → Calibration set: {len(cal_idx):,} samples")
    print(f"      Positive: {len(pos_cal_idx):,}, Negative: {len(neg_cal_idx):,}")
    print(f"      Positive ratio: {y_cal.mean():.6f}")
    
    print(f"\n   → Remaining train_c: {len(train_c_remaining_idx):,} samples")
    print(f"      Positive: {len(pos_train_idx):,}, Negative: {len(neg_train_idx):,}")
    print(f"      Positive ratio: {y_train_c_remaining.mean():.6f}")
    
    # Cleanup original train_c
    del X_train_c, y_train_c
    clear_gpu_memory()
    
    # ========================================================================
    # Combine train_t + train_v + remaining train_c for full training
    # ========================================================================
    print("\n" + "="*70)
    print("🔗 Combining Training Data")
    print("="*70)
    
    X_train = np.vstack([X_train_t, X_train_v, X_train_c_remaining])
    y_train = np.concatenate([y_train_t, y_train_v, y_train_c_remaining])
    
    print(f"   Combined training set: {X_train.shape[0]:,} samples")
    print(f"      From train_t: {len(X_train_t):,}")
    print(f"      From train_v: {len(X_train_v):,}")
    print(f"      From train_c: {len(X_train_c_remaining):,}")
    print(f"   Combined positive ratio: {y_train.mean():.6f}")
    
    # Cleanup individual datasets
    del X_train_t, y_train_t, X_train_v, y_train_v, X_train_c_remaining, y_train_c_remaining
    clear_gpu_memory()
    
    # Use train_v as validation set (we're using all data for training, but need eval set for early stopping)
    # Actually, let's use calibration set as validation for early stopping
    X_val = X_cal
    y_val = y_cal
    
    print(f"   Using calibration set as validation for early stopping")
    print(f"   Val shape: {X_val.shape}")
    print_memory()

    # Class distribution
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print("\n📊 Train data class distribution:")
    print(f"   Positive ratio: {pos_ratio:.6f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    # Get model parameters and add scale_pos_weight
    params = CONFIG.model_params.copy()
    if scale_pos_weight:
        if CONFIG.model_name == 'xgboost':
            params['scale_pos_weight'] = scale_pos_weight
        else:  # catboost
            params['class_weights'] = [1.0, scale_pos_weight]

    print(f"\n🔧 Using {CONFIG.model_name} with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")

    # Store original train size before MixUp
    original_train_size = len(X_train)
    sample_weight_train = None
    
    if CONFIG.use_mixup:
        print(f"\n🎨 Applying MixUp augmentation...")
        print(f"   Alpha: {CONFIG.mixup_alpha}")
        print(f"   Ratio: {CONFIG.mixup_ratio}")
        
        # Calculate class weight for base_weight
        class_weight = (1.0, scale_pos_weight)
        
        X_train, y_train, sample_weight_train = apply_mixup_to_dataset(
            X_train, y_train, 
            class_weight=class_weight,
            alpha=CONFIG.mixup_alpha,
            ratio=CONFIG.mixup_ratio,
            rng=np.random.default_rng(42)
        )
        
        print(f"   Original train samples: {original_train_size:,}")
        print(f"   Augmented train samples: {len(X_train):,}")
        print(f"   Added {len(X_train) - original_train_size:,} MixUp samples")
        print(f"   Train positive ratio (soft): {y_train.mean():.4f}")
    else:
        print("\n⚠️  MixUp disabled")

    # Training
    print("\n🔄 Training model...")
    train_start = time.time()

    # Initialize variables
    y_pred = None
    best_iteration = None
    dtrain = None
    dval = None
    
    if CONFIG.model_name == 'xgboost':
        # Create DMatrix with sample weights if MixUp is enabled
        if CONFIG.use_mixup and sample_weight_train is not None:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight_train)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        print("   Training XGBoost...")
        if CONFIG.use_mixup and sample_weight_train is not None:
            print(f"   Using sample weights (range: [{sample_weight_train.min():.2f}, {sample_weight_train.max():.2f}])")
        model = xgb.train(
            params, dtrain,
            num_boost_round=CONFIG.model_params['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
            verbose_eval=False
        )
        y_pred = model.predict(dval)
        best_iteration = model.best_iteration

    elif CONFIG.model_name == 'catboost':
        print("   Training CatBoost...")
        model = cb.CatBoostClassifier(**params)
        
        # Use sample weights if MixUp is enabled
        if CONFIG.use_mixup and sample_weight_train is not None:
            print(f"   Using sample weights (range: [{sample_weight_train.min():.2f}, {sample_weight_train.max():.2f}])")
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
                verbose=False
            )
        y_pred = model.predict_proba(X_val)[:, 1]
        best_iteration = model.get_best_iteration()
    
    else:
        raise ValueError(f"Unknown model: {CONFIG.model_name}")

    # Validation results
    score, ap, wll = calculate_competition_score(y_val, y_pred)

    print("\n📊 Validation Results (on calibration set):")
    print(f"   Score: {score:.6f}")
    print(f"   AP: {ap:.6f}")
    print(f"   WLL: {wll:.6f}")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Training time: {time.time() - train_start:.1f}s")
    
    # ========================================================================
    # Train Calibrator
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 Training Calibrator")
    print("="*70)
    print(f"   Method: {CONFIG.calibration_method}")
    
    calibrator = None
    calibrated_preds = y_pred  # Default to raw predictions
    
    if CONFIG.calibration_method != 'none':
        print(f"   Fitting {CONFIG.calibration_method} calibrator on calibration set...")
        
        if CONFIG.calibration_method == 'temperature':
            # Convert probabilities to logits
            cal_preds_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            logits = np.log(cal_preds_clipped / (1 - cal_preds_clipped))
            
            calibrator = TemperatureScaling()
            calibrator.fit(logits, y_val)
            
            # Apply calibration
            calibrated_preds = calibrator.predict_proba(logits)
            
        elif CONFIG.calibration_method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred, y_val)
            
            calibrated_preds = calibrator.predict(y_pred)
            
        elif CONFIG.calibration_method == 'sigmoid':
            calibrator = LogisticRegression()
            calibrator.fit(y_pred.reshape(-1, 1), y_val)
            
            calibrated_preds = calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        
        else:
            print(f"   ⚠️  Unknown calibration method: {CONFIG.calibration_method}")
            print("   Using raw predictions (no calibration)")
        
        # Evaluate calibrated predictions
        if calibrator is not None:
            cal_score, cal_ap, cal_wll = calculate_competition_score(y_val, calibrated_preds)
            
            print(f"\n   📊 Calibrated Results:")
            print(f"      Score: {cal_score:.6f} (raw: {score:.6f}, Δ: {cal_score - score:+.6f})")
            print(f"      AP: {cal_ap:.6f} (raw: {ap:.6f}, Δ: {cal_ap - ap:+.6f})")
            print(f"      WLL: {cal_wll:.6f} (raw: {wll:.6f}, Δ: {cal_wll - wll:+.6f})")
    else:
        print("   No calibration method specified (using raw predictions)")
    
    print("="*70)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(CONFIG.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n💾 Saving models to {save_dir}...")
    
    # Save model with score in filename
    model_ext = 'json' if CONFIG.model_name == 'xgboost' else 'cbm'
    model_filename = f"{CONFIG.model_name}_{score:.6f}.{model_ext}"
    model_path = os.path.join(save_dir, model_filename)
    
    if CONFIG.model_name == 'xgboost':
        model.save_model(model_path)
    else:
        model.save_model(model_path)
    print(f"   ✅ Model saved to {model_path}")
    
    # Save calibrator if exists
    if calibrator is not None:
        calibrator_path = os.path.join(save_dir, 'calibrator.pkl')
        with open(calibrator_path, 'wb') as f:
            pickle.dump(calibrator, f)
        print(f"   ✅ Calibrator saved to {calibrator_path}")
    
    # Save metadata
    metadata = {
        'model_name': CONFIG.model_name,
        'val_score': float(score),
        'val_ap': float(ap),
        'val_wll': float(wll),
        'best_iteration': int(best_iteration) if best_iteration else None,
        'train_samples': len(y_train) if not CONFIG.use_mixup else original_train_size,
        'val_samples': len(y_val),
        'cal_samples': len(y_cal),
        'use_mixup': CONFIG.use_mixup,
        'mixup_alpha': CONFIG.mixup_alpha if CONFIG.use_mixup else None,
        'mixup_ratio': CONFIG.mixup_ratio if CONFIG.use_mixup else None,
        'calibration_method': CONFIG.calibration_method,
        'timestamp': timestamp,
        'model_params': CONFIG.model_params,
        'processing_method': 'full_train_with_calibration',  # train_t + train_v + remaining train_c
        'data_source': 'combined_all_data_with_calibration_split'  # All data used with calibration
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to {metadata_path}")

    # Cleanup
    print("\n🧹 Cleaning up memory...")
    del X_train, y_train, X_val, y_val, X_cal, y_cal, model, y_pred
    if sample_weight_train is not None:
        del sample_weight_train
    if dtrain is not None:
        del dtrain
    if dval is not None:
        del dval
    if calibrator is not None:
        del calibrator
    clear_gpu_memory()
    print("   ✅ Memory cleaned")

    return score, ap, wll, save_dir

# Run training
val_score, val_ap, val_wll, save_dir = run_train()

# Final summary
if val_score:
    print("\n" + "🎉"*35)
    print("TRAINING COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Validation Score: {val_score:.6f}")
    print(f"✅ Validation AP: {val_ap:.6f}")
    print(f"✅ Validation WLL: {val_wll:.6f}")
    print(f"✅ Model used: {CONFIG.model_name}")
    print(f"✅ Calibration method: {CONFIG.calibration_method}")
    print(f"✅ Models saved to: {save_dir}")
    print("✅ Training data: train_t + train_v + train_c (remaining)")
    print("✅ Calibration data: train_c (balanced subset)")
    print("✅ GBDT-optimized preprocessing (no normalization)")
    print(f"✅ Data source: {CONFIG.train_t_path}, {CONFIG.train_v_path}, {CONFIG.train_c_path}")
    print("="*70)
else:
    print("\n⚠️ Training did not complete. Please check for errors above.")

# Final cleanup
clear_gpu_memory()

# Clean up temporary files created by load_processed_data_gbdt
print("\n🧹 Final cleanup...")
print("   Temporary files are managed by data_loader.load_processed_data_gbdt()")

# Clean up any temp directories if they exist
for data_file in ['train_t', 'train_v', 'train_c']:
    temp_subdir = f'{CONFIG.temp_dir}_{data_file}'
    if os.path.exists(temp_subdir):
        try:
            shutil.rmtree(temp_subdir)
            print(f"   ✅ Removed {temp_subdir}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {temp_subdir}: {e}")

print("🧹 Final cleanup complete")
print_memory()

