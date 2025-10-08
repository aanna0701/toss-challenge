# Environment setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# NOTE: GBDT는 단일 GPU 사용. 원하는 GPU 인덱스를 설정하거나 환경변수로 전달
# Example: export CUDA_VISIBLE_DEVICES=4 python train_and_predict_GBDT.py
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
    # cudf.set_allocator("managed")  # Not available in all cuDF versions
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
    train_path: str
    output_dir: str
    temp_dir: str
    val_ratio: float
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
    train_path = data_config.get('train_path', 'data/train.parquet')
    temp_dir = data_config.get('temp_dir', 'tmp')
    
    # Convert temp_dir to absolute path relative to script location
    if not os.path.isabs(temp_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, temp_dir)

    # Extract training config
    training_config = yaml_config.get('training', {})
    val_ratio = training_config.get('val_ratio', 0.1)
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
        train_path=train_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        val_ratio=val_ratio,
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

        # if memory_efficient:
        #     params['max_depth'] = min(params['max_depth'], 6)   # Depth 제한
        #     params['tree_method'] = 'approx'                    # gpu_hist 대비 메모리 적음

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
            'bootstrap_type': config.model_params['bootstrap_type']  # Use from config
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
    print(f"   Input: {config.train_path}")
    print(f"   Output: {config.output_dir}")
    print(f"   Validation ratio: {config.val_ratio}")
    print(f"   Force reprocess: {config.force_reprocess}")
    print(f"   MixUp enabled: {config.use_mixup}")
    if config.use_mixup:
        print(f"   MixUp alpha: {config.mixup_alpha}")
        print(f"   MixUp ratio: {config.mixup_ratio}")

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
    parser = argparse.ArgumentParser(description='GBDT Training and Prediction Script')

    parser.add_argument('--config', type=str, default='config_GBDT.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        help='Preset configuration (e.g., xgboost_fast, catboost_deep)')
    parser.add_argument('--val-ratio', type=float, default=None,
                        help='Validation split ratio (overrides config file, default: 0.1)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data even if processed data exists')

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration - use command line arguments
print("\n🔧 Command line arguments:")
print(f"   Config file: {args.config}")
print(f"   Preset: {args.preset}")
print(f"   Val ratio: {args.val_ratio}")
print(f"   Force reprocess: {args.force_reprocess}")

CONFIG = create_config_from_yaml(args.config, preset=args.preset)

# Override config with command line arguments if provided
if args.val_ratio is not None:
    CONFIG.val_ratio = args.val_ratio
if args.force_reprocess:
    CONFIG.force_reprocess = True

print_config(CONFIG)

# Test workflow creation
test_workflow = create_workflow_gbdt()
print("✅ Workflow creation tested successfully")

def process_data():
    """Process data with NVTabular"""
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

    # Prepare data without 'seq' column
    temp_path = f'{CONFIG.temp_dir}/train_no_seq.parquet'
    if not os.path.exists(temp_path):
        print("\n📋 Creating temp file without 'seq' column...")
        pf = pq.ParquetFile(CONFIG.train_path)
        cols = [c for c in pf.schema.names if c not in ['seq', '']]
        print(f"   Total columns: {len(pf.schema.names)}")
        print(f"   Using columns: {len(cols)} (excluded 'seq')")

        df = pd.read_parquet(CONFIG.train_path, columns=cols)
        print(f"   Loaded {len(df):,} rows")
        df.to_parquet(temp_path, index=False)
        del df
        gc.collect()
        print("   ✅ Temp file created")
    else:
        print(f"✅ Using existing temp file: {temp_path}")

    # Create dataset with balanced partitions
    print("\n📦 Creating NVTabular Dataset...")
    print("   Using 64MB partitions for better throughput vs memory")
    clear_gpu_memory()

    dataset = Dataset(
        temp_path,
        engine='parquet',
        part_size='64MB'  # 32~64MB 권장
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
            shuffle=nvt.io.Shuffle.PER_WORKER,  # 파티션/워커 기준 셔플
            out_files_per_proc=32               # 파일 수 증가로 병렬성/메모리 균형
        )

        workflow_path = f'{CONFIG.output_dir}/workflow'
        workflow.save(workflow_path)
        print(f"   ✅ Data processed and saved")
        print(f"   ✅ Workflow saved to {workflow_path}")

    except (OSError, RuntimeError, MemoryError) as e:
        print(f"❌ Error during processing: {e}")
        if os.path.exists(CONFIG.output_dir):
            # Fix "not empty" error by using ignore_errors=True
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

def run_train_val(processed_dir, val_ratio=0.1):
    """Run train/validation split and generate test predictions"""
    print("\n" + "="*70)
    print("🔄 Train/Validation Split with Test Predictions")
    print("="*70)
    print(f"   Validation ratio: {val_ratio:.1%}")
    
    # Test data path (must exist)
    test_path = 'data/test.parquet'
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    print(f"✅ Test data found: {test_path}")

    # Load processed data with smaller partitions
    print("\n📦 Loading processed data...")
    start_load = time.time()

    try:
        dataset = Dataset(processed_dir, engine='parquet', part_size='128MB')  # balanced
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

    # 전체 컬럼 일괄 float32 변환 — copy 최소화
    print("   Converting all features to float32 (single pass)...")
    try:
        X = X.astype('float32', copy=False)
    except (ValueError, TypeError) as e:
        print(f"   ⚠️ astype(float32) failed with copy=False: {e}")
        X = X.astype('float32')

    # Convert to numpy (메모리 부족 시 generator DMatrix로 변경 고려)
    print("   Converting to numpy...")
    X_np = X.to_numpy()
    print(f"   Shape: {X_np.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")

    # Class distribution
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print("\n📊 Class distribution:")
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

    # Load and preprocess test data
    print("\n📦 Loading and preprocessing test data...")
    try:
        # Load workflow
        workflow_path = f'{CONFIG.output_dir}/workflow'
        workflow = nvt.Workflow.load(workflow_path)
        
        # Create temp test file without 'seq' column to avoid CUDF string limit
        temp_test_path = f'{CONFIG.temp_dir}/test_no_seq.parquet'
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
        
        # Apply workflow to test data using the same method as process_data()
        print("   Applying workflow to test data...")
        test_dataset = Dataset(temp_test_path, engine='parquet', part_size='64MB')
        
        # Create test workflow without 'clicked' column (using same preprocessing as training)
        
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
        
        # Fit the test workflow on the test dataset to get the same preprocessing
        test_workflow.fit(test_dataset)
        gdf_test = test_workflow.transform(test_dataset).to_ddf().compute()
        
        # Get test IDs from original file
        test_df_original = pd.read_parquet(test_path, columns=['ID'])
        test_ids = test_df_original['ID'].values
        del test_df_original
        
        # Convert to float32 (excluding ID column)
        feature_cols = [col for col in gdf_test.columns if col != 'ID']
        X_test = gdf_test[feature_cols].astype('float32', copy=False)
        X_test_np = X_test.to_numpy()
        del X_test, gdf_test
        gc.collect()
        
        print(f"   ✅ Test data loaded: {X_test_np.shape}")
    except (OSError, RuntimeError, MemoryError, FileNotFoundError) as e:
        print(f"   ❌ Failed to load test data: {e}")
        raise RuntimeError(f"Test data loading failed: {e}") from e

    # Train/Validation split
    print("\n🔄 Splitting data into train and validation...")
    from sklearn.model_selection import train_test_split
    
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), 
        test_size=val_ratio, 
        random_state=42, 
        stratify=y
    )
    
    print(f"   Train: {len(train_idx):,} samples")
    print(f"   Val: {len(val_idx):,} samples")
    print(f"   Train positive ratio: {y[train_idx].mean():.4f}")
    print(f"   Val positive ratio: {y[val_idx].mean():.4f}")
    
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
        class_weight = (1.0, scale_pos_weight)  # (weight_class_0, weight_class_1)
        
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
    
    # Predict on test data
    print("\n🔮 Predicting on test data...")
    if CONFIG.model_name == 'xgboost':
        dtest = xgb.DMatrix(X_test_np)
        test_predictions = model.predict(dtest)
        del dtest
        if dtrain is not None:
            del dtrain
        if dval is not None:
            del dval
    else:
        test_predictions = model.predict_proba(X_test_np)[:, 1]
    
    print(f"   Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'clicked': test_predictions
    })
    
    submission_path = f'{CONFIG.output_dir}/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"   ✅ Submission saved to {submission_path}")
    print(f"   Samples: {len(submission_df):,}")
    
    # Save model
    print("\n💾 Saving trained model...")
    if CONFIG.model_name == 'xgboost':
        model_path = f'{CONFIG.output_dir}/model.json'
        model.save_model(model_path)
    else:
        model_path = f'{CONFIG.output_dir}/model.cbm'
        model.save_model(model_path)
    print(f"   ✅ Model saved to {model_path}")

    # Cleanup
    print("\n🧹 Cleaning up memory...")
    del X_np, y, X_train, y_train, model, y_pred, test_predictions, X_test_np
    if sample_weight_train is not None:
        del sample_weight_train
    clear_gpu_memory()
    print("   ✅ Memory cleaned")

    return score, ap, wll

# Run training and validation
val_score, val_ap, val_wll = run_train_val(processed_dir, CONFIG.val_ratio)

# Final summary
if val_score:
    print("\n" + "🎉"*35)
    print("TRAINING & PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Validation Score: {val_score:.6f}")
    print(f"✅ Validation AP: {val_ap:.6f}")
    print(f"✅ Validation WLL: {val_wll:.6f}")
    print(f"✅ Model used: {CONFIG.model_name}")
    print(f"✅ Output directory: {CONFIG.output_dir}")
    print("✅ Full dataset processed (10.7M rows)")
    print("✅ GBDT-optimized preprocessing (no normalization)")
    print(f"✅ Validation ratio: {CONFIG.val_ratio:.1%}")
    print(f"✅ Submission saved: {CONFIG.output_dir}/submission.csv")
    model_ext = 'json' if CONFIG.model_name == 'xgboost' else 'cbm'
    print(f"✅ Model saved: {CONFIG.output_dir}/model.{model_ext}")
    print("="*70)
else:
    print("\n⚠️ Training did not complete. Please check for errors above.")

# Final cleanup
clear_gpu_memory()

# Clean up any remaining temporary files
print("\n🧹 Final cleanup...")
temp_path = f'{CONFIG.temp_dir}/train_no_seq.parquet'
if os.path.exists(temp_path):
    os.remove(temp_path)
    print(f"   ✅ Removed {temp_path}")

temp_test_path = f'{CONFIG.temp_dir}/test_no_seq.parquet'
if os.path.exists(temp_test_path):
    os.remove(temp_test_path)
    print(f"   ✅ Removed {temp_test_path}")

if os.path.exists(CONFIG.temp_dir) and not os.listdir(CONFIG.temp_dir):
    os.rmdir(CONFIG.temp_dir)
    print(f"   ✅ Removed empty directory {CONFIG.temp_dir}")

print("🧹 Final cleanup complete")
print_memory()
