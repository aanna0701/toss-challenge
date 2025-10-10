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

# Custom modules
from mixup import apply_mixup_to_dataset
from utils import calculate_competition_score, clear_gpu_memory, print_memory

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
    
    # Prepare model parameters (MixUp settings removed)
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
        mixup_ratio=mixup_ratio
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
    print(f"   Output: {config.output_dir}")
    print(f"   MixUp enabled: {config.use_mixup}")
    if config.use_mixup:
        print(f"   MixUp alpha: {config.mixup_alpha}")
        print(f"   MixUp ratio: {config.mixup_ratio}")
    print(f"   Note: Calibration will be performed during prediction")

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
    print(f"   Train: {CONFIG.train_t_path}")
    print(f"   Val: {CONFIG.train_v_path}")
    print(f"   ⚡ Using pre-processed data (fast loading)")

    start_load = time.time()

    # Import the same function HPO uses
    from data_loader import load_processed_data_gbdt
    
    # Load TRAIN data (pre-processed, seq column automatically excluded)
    print("\n📦 Loading train data...")
    X_train, y_train = load_processed_data_gbdt(CONFIG.train_t_path)
    print(f"   Train shape: {X_train.shape}")
    print(f"   Train positive ratio: {y_train.mean():.6f}")
    
    clear_gpu_memory()

    # Load VAL data (pre-processed, seq column automatically excluded)
    print("\n📦 Loading val data...")
    X_val, y_val = load_processed_data_gbdt(CONFIG.train_v_path)
    print(f"   Val shape: {X_val.shape}")
    print(f"   Val positive ratio: {y_val.mean():.6f}")
    
    clear_gpu_memory()

    print(f"\n⏱️  Loading time: {time.time() - start_load:.1f}s")
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

    print("\n📊 Validation Results:")
    print(f"   Score: {score:.6f}")
    print(f"   AP: {ap:.6f}")
    print(f"   WLL: {wll:.6f}")
    print(f"   Best iteration: {best_iteration}")
    print(f"   Training time: {time.time() - train_start:.1f}s")
    
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
    
    # Save metadata
    metadata = {
        'model_name': CONFIG.model_name,
        'val_score': float(score),
        'val_ap': float(ap),
        'val_wll': float(wll),
        'best_iteration': int(best_iteration) if best_iteration else None,
        'train_samples': len(y_train) if not CONFIG.use_mixup else original_train_size,
        'val_samples': len(y_val),
        'use_mixup': CONFIG.use_mixup,
        'mixup_alpha': CONFIG.mixup_alpha if CONFIG.use_mixup else None,
        'mixup_ratio': CONFIG.mixup_ratio if CONFIG.use_mixup else None,
        'timestamp': timestamp,
        'model_params': CONFIG.model_params,
        'processing_method': 'hpo_compatible_independent',  # HPO-compatible independent processing
        'data_source': 'raw_parquet_with_independent_workflow'  # Each split processed independently
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to {metadata_path}")

    # Cleanup
    print("\n🧹 Cleaning up memory...")
    del X_train, y_train, X_val, y_val, model, y_pred
    if sample_weight_train is not None:
        del sample_weight_train
    if dtrain is not None:
        del dtrain
    if dval is not None:
        del dval
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
    print(f"✅ Models saved to: {save_dir}")
    print("✅ HPO-Compatible Processing: Each split processed independently")
    print("✅ GBDT-optimized preprocessing (no normalization)")
    print(f"✅ Data source: {CONFIG.train_t_path}, {CONFIG.train_v_path}")
    print(f"✅ Identical behavior to hpo_xgboost.py for consistency")
    print("="*70)
else:
    print("\n⚠️ Training did not complete. Please check for errors above.")

# Final cleanup
clear_gpu_memory()

# Clean up temporary files created by load_processed_data_gbdt
print("\n🧹 Final cleanup...")
print("   Temporary files are managed by data_loader.load_processed_data_gbdt()")

# Clean up any temp directories if they exist
for data_file in ['train_t', 'train_v']:
    temp_subdir = f'{CONFIG.temp_dir}_{data_file}'
    if os.path.exists(temp_subdir):
        try:
            shutil.rmtree(temp_subdir)
            print(f"   ✅ Removed {temp_subdir}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {temp_subdir}: {e}")

print("🧹 Final cleanup complete")
print_memory()

