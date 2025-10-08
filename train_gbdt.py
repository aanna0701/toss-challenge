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
    'cudf': '23.10',      # Prefix match
    'cupy': '13.6',       # Prefix match
    'xgboost': '3.0',     # Minimum version
    'catboost': '1.2',    # Minimum version
    'dask': '2023.9',
    'pandas': '1.5',
    'numpy': '1.24',
    'scikit-learn': '1.7',
    'psutil': '5.9',      # 5.9.1 works fine (used in working code)
    'pyarrow': '12.0'     # 12.0.1 works fine (used in working code)
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

        # Check version compatibility (lenient)
        req_major = required_version.split('.')[0]
        inst_version_parts = installed_version.split('.')
        inst_major = inst_version_parts[0] if installed_version != 'unknown' else ''

        if installed_version == 'unknown':
            print(f"⚠️  {lib:15} {installed_version:15} (required: ≥{required_version})")
        elif float(inst_major) >= float(req_major) if inst_major.isdigit() and req_major.isdigit() else installed_version.startswith(required_version[:3]):
            print(f"✅ {lib:15} {installed_version:15} (required: ≥{required_version})")
        else:
            print(f"⚠️  {lib:15} {installed_version:15} (required: ≥{required_version}) - but should work")

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

# Core imports
import gc
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

# GPU libraries
import cupy as cp

# ==== RMM / cuDF allocator 초기화 (NVTabular/ cuDF 사용 전 1회) ====
try:
    import rmm, cudf
    # RTX 3090 24GB 기준: 10~14GB 선할당 권장 (필요 시 조정)
    # 10GB를 바이트로 변환: 10 * 1024^3
    initial_pool_size_bytes = 10 * 1024 * 1024 * 1024
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=initial_pool_size_bytes,    # 8~14GB 사이에서 조정 가능
        managed_memory=True,         # 부족분 UVM 사용
    )
    print("✅ RMM initialized (pool=10GB, managed_memory=True)")
except (ImportError, RuntimeError) as e:
    print(f"⚠️ RMM init skipped: {e}")

# Set GPU device (CUDA_VISIBLE_DEVICES 고려: 가시 목록 내 0번째)
cp.cuda.Device(0).use()

# NVTabular
import nvtabular as nvt
from nvtabular import ops
from merlin.io import Dataset

# ML libraries
import xgboost as xgb
import catboost as cb

# Configuration
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, Any

# Import common functions
from utils import (
    calculate_competition_score, 
    clear_gpu_memory, 
    print_memory
)
from data_loader import create_workflow_gbdt
from mixup import apply_mixup_to_dataset

@dataclass
class GBDTConfig:
    """Main GBDT configuration"""
    train_t_path: str
    train_v_path: str
    train_c_path: str
    output_dir: str
    temp_dir: str
    force_reprocess: bool
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

    # Extract training config
    training_config = yaml_config.get('training', {})
    force_reprocess = training_config.get('force_reprocess', False)
    use_mixup = training_config.get('use_mixup', False)
    mixup_alpha = training_config.get('mixup_alpha', 0.3)
    mixup_ratio = training_config.get('mixup_ratio', 0.5)

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

    # Get model parameters
    if final_model_name == 'xgboost':
        xgb_config = yaml_config.get('xgboost', {})
        model_params = {
            'n_estimators': xgb_config.get('n_estimators', 200),
            'learning_rate': xgb_config.get('learning_rate', 0.1),
            'max_depth': xgb_config.get('max_depth', 8),
            'subsample': xgb_config.get('subsample', 0.8),
            'colsample_bytree': xgb_config.get('colsample_bytree', 0.8),
            'tree_method': xgb_config.get('tree_method', 'gpu_hist'),
            'gpu_id': xgb_config.get('gpu_id', 0),
            'verbosity': xgb_config.get('verbosity', 0),
            'early_stopping_rounds': xgb_config.get('early_stopping_rounds', 20),
            'random_state': xgb_config.get('random_state', 42)
        }
    else:  # catboost
        cb_config = yaml_config.get('catboost', {})
        model_params = {
            'n_estimators': cb_config.get('n_estimators', 200),
            'learning_rate': cb_config.get('learning_rate', 0.1),
            'max_depth': cb_config.get('max_depth', 8),
            'colsample_bylevel': cb_config.get('colsample_bylevel', 0.8),
            'task_type': cb_config.get('task_type', 'GPU'),
            'devices': cb_config.get('devices', '0'),
            'verbose': cb_config.get('verbose', False),
            'early_stopping_rounds': cb_config.get('early_stopping_rounds', 20),
            'thread_count': cb_config.get('thread_count', -1),
            'random_state': cb_config.get('random_state', 42),
            'bootstrap_type': cb_config.get('bootstrap_type', 'Bayesian')
        }

    return GBDTConfig(
        train_t_path=train_t_path,
        train_v_path=train_v_path,
        train_c_path=train_c_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        force_reprocess=force_reprocess,
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

def get_model_params_dict(config: GBDTConfig, scale_pos_weight: float = None) -> Dict[str, Any]:
    """Get model parameters as dictionary for training"""
    if config.model_name == 'xgboost':
        params = {
            'objective': 'binary:logistic',
            'tree_method': config.model_params['tree_method'],
            'max_depth': config.model_params['max_depth'],
            'learning_rate': config.model_params['learning_rate'],
            'subsample': config.model_params['subsample'],
            'colsample_bytree': config.model_params['colsample_bytree'],
            'gpu_id': config.model_params['gpu_id'],
            'verbosity': config.model_params['verbosity'],
            'seed': config.model_params['random_state'],

            # 메모리 사용 감소
            'max_bin': 128,
            'predictor': 'gpu_predictor',
        }

        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
        return params

    elif config.model_name == 'catboost':
        params = {
            'task_type': config.model_params['task_type'],
            'devices': config.model_params['devices'],
            'iterations': config.model_params['n_estimators'],
            'learning_rate': config.model_params['learning_rate'],
            'depth': config.model_params['max_depth'],
            'verbose': config.model_params['verbose'],
            'random_seed': config.model_params['random_state'],
            'thread_count': config.model_params['thread_count'],
            'bootstrap_type': config.model_params['bootstrap_type']
        }
        
        # GPU에서는 colsample_bylevel (rsm) 지원하지 않음 - CPU에서만 지원
        if config.model_params['task_type'] == 'GPU':
            print("   ⚠️ Skipping colsample_bylevel for GPU training (not supported)")
        else:
            params['colsample_bylevel'] = config.model_params['colsample_bylevel']
            
        if scale_pos_weight:
            params['class_weights'] = [1.0, scale_pos_weight]
        return params

    else:
        raise ValueError(f"Unknown model: {config.model_name}")

def print_config(config: GBDTConfig):
    """Print configuration details"""
    print(f"\n📋 GBDT Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Train data: {config.train_t_path}")
    print(f"   Val data: {config.train_v_path}")
    print(f"   Cal data: {config.train_c_path}")
    print(f"   Output: {config.output_dir}")
    print(f"   Force reprocess: {config.force_reprocess}")
    print(f"   MixUp enabled: {config.use_mixup}")
    if config.use_mixup:
        print(f"   MixUp alpha: {config.mixup_alpha}")
        print(f"   MixUp ratio: {config.mixup_ratio}")
    print(f"   Note: Calibration will be performed during prediction")

    print("\n🔧 Model Parameters:")
    for key, value in config.model_params.items():
        print(f"   {key}: {value}")

print("✅ All libraries imported successfully")
print(f"NVTabular version: {nvt.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"CatBoost version: {cb.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Test imported functions
print("Testing memory functions:")
print_memory()
clear_gpu_memory()

# Argument parser
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GBDT Training Script')

    parser.add_argument('--config', type=str, default='config_GBDT.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        help='Preset configuration (e.g., xgboost_fast, catboost_deep)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data even if processed data exists')

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration - use command line arguments
print("\n🔧 Command line arguments:")
print(f"   Config file: {args.config}")
print(f"   Preset: {args.preset}")
print(f"   Force reprocess: {args.force_reprocess}")

CONFIG = create_config_from_yaml(args.config, preset=args.preset)

# Override config with command line arguments if provided
if args.force_reprocess:
    CONFIG.force_reprocess = True

print_config(CONFIG)

# Test workflow creation
test_workflow = create_workflow_gbdt()
print("✅ Workflow creation tested successfully")

def process_data():
    """Process data with NVTabular - process each split separately"""
    import shutil

    print("\n" + "="*70)
    print("🚀 NVTabular Data Processing")
    print("="*70)

    # Check if already processed
    if os.path.exists(CONFIG.output_dir) and not CONFIG.force_reprocess:
        try:
            test_dataset = Dataset(CONFIG.output_dir, engine='parquet')
            print(f"✅ Using existing processed data from {CONFIG.output_dir}")
            return CONFIG.output_dir
        except:
            print(f"⚠️ Existing data corrupted, reprocessing...")
            shutil.rmtree(CONFIG.output_dir)

    # Clear existing if needed
    if os.path.exists(CONFIG.output_dir):
        print(f"🗑️ Removing existing directory {CONFIG.output_dir}")
        shutil.rmtree(CONFIG.output_dir)

    start_time = time.time()
    initial_mem = print_memory()

    # Create temp directory if it doesn't exist
    os.makedirs(CONFIG.temp_dir, exist_ok=True)

    # Process all three splits: train_t, train_v, train_c
    split_paths = {
        'train': CONFIG.train_t_path,
        'val': CONFIG.train_v_path,
        'cal': CONFIG.train_c_path
    }
    
    temp_paths = {}
    
    print("\n📋 Creating temp files without 'seq' column...")
    for split_name, split_path in split_paths.items():
        temp_path = f'{CONFIG.temp_dir}/{split_name}_no_seq.parquet'
        temp_paths[split_name] = temp_path
        
        if not os.path.exists(temp_path):
            print(f"   Processing {split_name} split: {split_path}")
            pf = pq.ParquetFile(split_path)
            cols = [c for c in pf.schema.names if c not in ['seq', '']]
            
            df = pd.read_parquet(split_path, columns=cols)
            print(f"      Loaded {len(df):,} rows")
            df.to_parquet(temp_path, index=False)
            del df
            gc.collect()
            print(f"      ✅ Temp file created: {temp_path}")
        else:
            print(f"   ✅ Using existing temp file: {temp_path}")

    # Combine all temp files into one dataset for workflow fitting
    # ⚠️ IMPORTANT: 순서를 유지해야 함! (train → val → cal)
    # 나중에 run_train()에서 인덱스로 다시 분할하므로 순서가 중요
    print("\n📦 Combining data for workflow fitting...")
    combined_temp_path = f'{CONFIG.temp_dir}/combined_no_seq.parquet'
    
    if not os.path.exists(combined_temp_path):
        dfs = []
        for split_name in ['train', 'val', 'cal']:  # ✅ 순서 보존!
            df = pd.read_parquet(temp_paths[split_name])
            dfs.append(df)
            print(f"   Loaded {split_name}: {len(df):,} rows")
        
        combined_df = pd.concat(dfs, ignore_index=True)  # ✅ 순서대로 concat
        print(f"   Total combined: {len(combined_df):,} rows")
        print(f"   ⚠️  Order preserved: train → val → cal")
        combined_df.to_parquet(combined_temp_path, index=False)
        del dfs, combined_df
        gc.collect()
        print("   ✅ Combined temp file created")
    else:
        print(f"   ✅ Using existing combined temp file: {combined_temp_path}")

    # Create dataset with balanced partitions
    print("\n📦 Creating NVTabular Dataset...")
    print("   Using 64MB partitions for better throughput vs memory")
    clear_gpu_memory()

    dataset = Dataset(
        combined_temp_path,
        engine='parquet',
        part_size='64MB'
    )
    print("   ✅ Dataset created")

    # Create and fit workflow
    print("\n📊 Fitting workflow...")
    workflow = create_workflow_gbdt()
    workflow.fit(dataset)
    print("   ✅ Workflow fitted")

    # Transform and save
    print(f"\n💾 Transforming and saving to {CONFIG.output_dir}...")
    os.makedirs(CONFIG.output_dir, exist_ok=True)

    clear_gpu_memory()

    try:
        workflow.transform(dataset).to_parquet(
            output_path=CONFIG.output_dir,
            shuffle=None,  # ✅ 순서 보존을 위해 shuffle 제거
            out_files_per_proc=32
        )

        workflow_path = f'{CONFIG.output_dir}/workflow'
        workflow.save(workflow_path)
        print(f"   ✅ Data processed and saved")
        print(f"   ✅ Workflow saved to {workflow_path}")

    except (OSError, RuntimeError, MemoryError) as e:
        print(f"❌ Error during processing: {e}")
        if os.path.exists(CONFIG.output_dir):
            shutil.rmtree(CONFIG.output_dir, ignore_errors=True)
        raise

    elapsed = time.time() - start_time
    final_mem = print_memory()

    print("\n✅ Processing complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Memory increase: +{final_mem - initial_mem:.1f}%")

    clear_gpu_memory()
    return CONFIG.output_dir

# Process data
processed_dir = process_data()

def run_train(processed_dir):
    """Run training with pre-split train/validation/calibration data"""
    print("\n" + "="*70)
    print("🔄 Loading Pre-split Train/Validation/Calibration Data")
    print("="*70)
    print(f"   Train: {CONFIG.train_t_path}")
    print(f"   Val: {CONFIG.train_v_path}")
    print(f"   Cal: {CONFIG.train_c_path}")

    # Load processed data with smaller partitions
    print("\n📦 Loading processed data...")
    start_load = time.time()

    try:
        dataset = Dataset(processed_dir, engine='parquet', part_size='128MB')
        print("   Converting to GPU DataFrame...")
        gdf = dataset.to_ddf().compute()
        print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
        print(f"   Time: {time.time() - start_load:.1f}s")
    except (OSError, RuntimeError, MemoryError) as e:
        print(f"❌ Error loading data: {e}")
        print("   Trying with even smaller partitions...")
        try:
            dataset = Dataset(processed_dir, engine='parquet', part_size='64MB')
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded with 64MB partitions: {len(gdf):,} rows")
        except (OSError, RuntimeError, MemoryError) as e2:
            print(f"❌ Failed even with 64MB partitions: {e2}")
            return None

    print_memory()

    # Prepare data with memory optimization
    print("\n📊 Preparing data for GBDT...")
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)

    # 전체 컬럼 일괄 float32 변환
    print("   Converting all features to float32 (single pass)...")
    try:
        X = X.astype('float32', copy=False)
    except (ValueError, TypeError) as e:
        print(f"   ⚠️ astype(float32) failed with copy=False: {e}")
        X = X.astype('float32')

    # Convert to numpy
    print("   Converting to numpy...")
    X_np = X.to_numpy()
    print(f"   Shape: {X_np.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")

    # Class distribution
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print("\n📊 Overall class distribution:")
    print(f"   Positive ratio: {pos_ratio:.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    # Enhanced cleanup
    del X, gdf
    gc.collect()
    clear_gpu_memory()

    # Get model parameters
    params = get_model_params_dict(CONFIG, scale_pos_weight)

    print(f"\n🔧 Using {CONFIG.model_name} with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")

    # Split data based on original file sizes
    # ⚠️ CRITICAL: process_data()에서 순서대로 결합됨 (train → val → cal)
    # shuffle=None으로 설정되어 순서가 보존되므로 인덱스로 분할 가능
    print("\n🔄 Splitting data based on original file sizes...")
    
    # Load original files to get sizes
    df_train_orig = pd.read_parquet(CONFIG.train_t_path, columns=['clicked'])
    df_val_orig = pd.read_parquet(CONFIG.train_v_path, columns=['clicked'])
    df_cal_orig = pd.read_parquet(CONFIG.train_c_path, columns=['clicked'])
    
    train_size = len(df_train_orig)
    val_size = len(df_val_orig)
    cal_size = len(df_cal_orig)
    
    del df_train_orig, df_val_orig, df_cal_orig
    gc.collect()
    
    # Create index splits (순서 보존되어 있으므로 인덱스로 분할)
    train_idx = np.arange(0, train_size)  # [0:train_size]
    val_idx = np.arange(train_size, train_size + val_size)  # [train_size:train_size+val_size]
    cal_idx = np.arange(train_size + val_size, train_size + val_size + cal_size)  # [train_size+val_size:end]
    
    print(f"   Train: {len(train_idx):,} samples ({len(train_idx)/len(y):.1%})")
    print(f"   Val: {len(val_idx):,} samples ({len(val_idx)/len(y):.1%})")
    print(f"   Cal: {len(cal_idx):,} samples ({len(cal_idx)/len(y):.1%})")
    print(f"   Train positive ratio: {y[train_idx].mean():.4f}")
    print(f"   Val positive ratio: {y[val_idx].mean():.4f}")
    print(f"   Cal positive ratio: {y[cal_idx].mean():.4f}")
    
    # Apply MixUp augmentation if enabled
    X_train = X_np[train_idx]
    y_train = y[train_idx]
    sample_weight_train = None
    
    if CONFIG.use_mixup:
        print(f"\n🎨 Applying MixUp augmentation...")
        print(f"   Alpha: {CONFIG.mixup_alpha}")
        print(f"   Ratio: {CONFIG.mixup_ratio}")
        
        # Calculate class weight for base_weight
        pos_ratio = y_train.mean()
        class_weight = (1.0, scale_pos_weight)
        
        X_train, y_train, sample_weight_train = apply_mixup_to_dataset(
            X_train, y_train, 
            class_weight=class_weight,
            alpha=CONFIG.mixup_alpha,
            ratio=CONFIG.mixup_ratio,
            rng=np.random.default_rng(42)
        )
        
        print(f"   Original train samples: {len(train_idx):,}")
        print(f"   Augmented train samples: {len(X_train):,}")
        print(f"   Added {len(X_train) - len(train_idx):,} MixUp samples")
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
        dval = xgb.DMatrix(X_np[val_idx], label=y[val_idx])

        print("   Training XGBoost...")
        if CONFIG.use_mixup:
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
                eval_set=(X_np[val_idx], y[val_idx]),
                early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_np[val_idx], y[val_idx]),
                early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
                verbose=False
            )
        y_pred = model.predict_proba(X_np[val_idx])[:, 1]
        best_iteration = model.get_best_iteration()
    
    else:
        raise ValueError(f"Unknown model: {CONFIG.model_name}")

    # Validation results
    score, ap, wll = calculate_competition_score(y[val_idx], y_pred)

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
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'cal_samples': len(cal_idx),
        'use_mixup': CONFIG.use_mixup,
        'mixup_alpha': CONFIG.mixup_alpha if CONFIG.use_mixup else None,
        'mixup_ratio': CONFIG.mixup_ratio if CONFIG.use_mixup else None,
        'timestamp': timestamp,
        'model_params': CONFIG.model_params
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to {metadata_path}")

    # Cleanup
    print("\n🧹 Cleaning up memory...")
    del X_np, y, X_train, y_train, model, y_pred
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
val_score, val_ap, val_wll, save_dir = run_train(processed_dir)

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
    print("✅ Full dataset processed")
    print("✅ GBDT-optimized preprocessing (no normalization)")
    print(f"✅ Using pre-split data: train_t / train_v / train_c")
    print(f"✅ Note: Calibration will be performed during prediction (pred_gbdt.py)")
    print("="*70)
else:
    print("\n⚠️ Training did not complete. Please check for errors above.")

# Final cleanup
clear_gpu_memory()

# Clean up temporary files
print("\n🧹 Final cleanup...")
temp_files = [
    f'{CONFIG.temp_dir}/train_no_seq.parquet',
    f'{CONFIG.temp_dir}/val_no_seq.parquet',
    f'{CONFIG.temp_dir}/cal_no_seq.parquet',
    f'{CONFIG.temp_dir}/combined_no_seq.parquet'
]

for temp_path in temp_files:
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"   ✅ Removed {temp_path}")

if os.path.exists(CONFIG.temp_dir) and not os.listdir(CONFIG.temp_dir):
    os.rmdir(CONFIG.temp_dir)
    print(f"   ✅ Removed empty directory {CONFIG.temp_dir}")

print("🧹 Final cleanup complete")
print_memory()

