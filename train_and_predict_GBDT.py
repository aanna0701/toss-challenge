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
import warnings

# Suppress deprecation warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        import pkg_resources
    except:
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
        except:
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
import psutil

# GPU libraries
import cupy as cp

# ==== RMM / cuDF allocator 초기화 (NVTabular/ cuDF 사용 전 1회) ====
try:
    import rmm, cudf
    # RTX 3090 24GB 기준: 10~14GB 선할당 권장 (필요 시 조정)
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size="10GB",    # 8~14GB 사이에서 조정 가능
        managed_memory=True,         # 부족분 UVM 사용
    )
    cudf.set_allocator("managed")
    print("✅ RMM initialized (pool=10GB, managed_memory=True)")
except Exception as e:
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

# Configuration
import yaml
import argparse
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GBDTConfig:
    """Main GBDT configuration"""
    train_path: str
    output_dir: str
    temp_dir: str
    n_folds: int
    force_reprocess: bool
    model_name: str
    model_params: Dict[str, Any]

def load_yaml_config(config_path: str = 'GBDT_config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_config_from_yaml(config_path: str = 'GBDT_config.yaml', preset: str = None) -> GBDTConfig:
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

    # Extract CV config
    cv_config = yaml_config.get('cv', {})
    n_folds = cv_config.get('n_folds', 5)
    force_reprocess = cv_config.get('force_reprocess', False)

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
            'subsample': cb_config.get('subsample', 0.8),
            'colsample_bylevel': cb_config.get('colsample_bylevel', 0.8),
            'task_type': cb_config.get('task_type', 'GPU'),
            'devices': cb_config.get('devices', '0'),
            'verbose': cb_config.get('verbose', False),
            'early_stopping_rounds': cb_config.get('early_stopping_rounds', 20),
            'thread_count': cb_config.get('thread_count', -1),
            'random_state': cb_config.get('random_state', 42)
        }

    return GBDTConfig(
        train_path=train_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        n_folds=n_folds,
        force_reprocess=force_reprocess,
        model_name=final_model_name,
        model_params=model_params
    )

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update dictionary"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def get_model_params_dict(config: GBDTConfig, scale_pos_weight: float = None, memory_efficient: bool = True) -> Dict[str, Any]:
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

        if memory_efficient:
            params['max_depth'] = min(params['max_depth'], 6)   # Depth 제한
            params['tree_method'] = 'approx'                    # gpu_hist 대비 메모리 적음

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
            'subsample': config.model_params['subsample'],
            'colsample_bylevel': config.model_params['colsample_bylevel'],
            'verbose': config.model_params['verbose'],
            'random_seed': config.model_params['random_state'],
            'thread_count': config.model_params['thread_count']
        }
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
    print(f"   Folds: {config.n_folds}")
    print(f"   Force reprocess: {config.force_reprocess}")

    print(f"\n🔧 Model Parameters:")
    for key, value in config.model_params.items():
        print(f"   {key}: {value}")

print("✅ All libraries imported successfully")
print(f"NVTabular version: {nvt.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"CatBoost version: {cb.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Argument parser
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GBDT Training and Prediction Script')

    parser.add_argument('--config', type=str, default='GBDT_config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        help='Preset configuration (e.g., xgboost_fast, catboost_deep)')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of CV folds (overrides config file)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of data even if processed data exists')

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration - use command line arguments
print(f"\n🔧 Command line arguments:")
print(f"   Config file: {args.config}")
print(f"   Preset: {args.preset}")
print(f"   N folds: {args.n_folds}")
print(f"   Force reprocess: {args.force_reprocess}")

CONFIG = create_config_from_yaml(args.config, preset=args.preset)

# Override config with command line arguments if provided
if args.n_folds is not None:
    CONFIG.n_folds = args.n_folds
if args.force_reprocess:
    CONFIG.force_reprocess = True

print_config(CONFIG)

# Memory management functions
def print_memory():
    """Print current memory usage"""
    mem = psutil.virtual_memory()

    gpu_used = 0
    gpu_total = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        # CUDA_VISIBLE_DEVICES가 설정된 경우 가시 목록 내 0번
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = gpu_info.used / 1024**3
        gpu_total = gpu_info.total / 1024**3
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        print(f"💾 GPU ({gpu_name}): {gpu_used:.1f}GB/{gpu_total:.1f}GB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"💾 GPU: Error getting GPU info - {e}")

    print(f"💾 CPU: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")
    return mem.percent

def clear_gpu_memory():
    """Clear GPU memory with aggressive cleanup"""
    try:
        # Clear CuPy memory pools
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        # Force garbage collection
        gc.collect()

        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        print("🧹 GPU memory cleared")
    except Exception as e:
        print(f"⚠️ Error clearing GPU memory: {e}")
        gc.collect()

# Test memory functions
print("Testing memory functions:")
print_memory()
clear_gpu_memory()

# Metric functions
def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """Calculate competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

print("✅ Metric functions defined")

def create_workflow():
    """Create NVTabular workflow optimized for GBDT models"""
    print("\n🔧 Creating GBDT-optimized workflow...")

    # TRUE CATEGORICAL COLUMNS (only 5)
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

    # CONTINUOUS COLUMNS (110 total, l_feat_20, l_feat_23 제외)
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +   # 18
        [f'feat_b_{i}' for i in range(1, 7)] +    # 6
        [f'feat_c_{i}' for i in range(1, 9)] +    # 8
        [f'feat_d_{i}' for i in range(1, 7)] +    # 6
        [f'feat_e_{i}' for i in range(1, 11)] +   # 10
        [f'history_a_{i}' for i in range(1, 8)] +   # 7
        [f'history_b_{i}' for i in range(1, 31)] +  # 30
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]  # 25 (l_feat_20, l_feat_23 제외)
    )

    print(f"   Categorical: {len(true_categorical)} columns")
    print(f"   Continuous: {len(all_continuous)} columns")
    print(f"   Total features: {len(true_categorical) + len(all_continuous)}")

    # Minimal preprocessing for GBDT models
    cat_features = true_categorical >> ops.Categorify(
        freq_threshold=0,
        max_size=50000
    )
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)

    workflow = nvt.Workflow(cat_features + cont_features + ['clicked'])

    print("   ✅ Workflow created (no normalization for tree models)")
    return workflow

# Test workflow creation
test_workflow = create_workflow()
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
    workflow = create_workflow()
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

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if os.path.exists(CONFIG.output_dir):
            # Fix "not empty" error by using ignore_errors=True
            shutil.rmtree(CONFIG.output_dir, ignore_errors=True)
        raise

    elapsed = time.time() - start_time
    final_mem = print_memory()

    print(f"\n✅ Processing complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Memory increase: +{final_mem - initial_mem:.1f}%")

    clear_gpu_memory()
    return CONFIG.output_dir

# Process data
processed_dir = process_data()

def run_cv(processed_dir, n_folds=5):
    """Run stratified cross-validation and generate test predictions"""
    print("\n" + "="*70)
    print("🔄 Stratified KFold Cross-Validation with Test Predictions")
    print("="*70)
    
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
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Trying with even smaller partitions...")
        try:
            dataset = Dataset(processed_dir, engine='parquet', part_size='64MB')
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded with 64MB partitions: {len(gdf):,} rows")
        except Exception as e2:
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
    except Exception as e:
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
    print(f"\n📊 Class distribution:")
    print(f"   Positive ratio: {pos_ratio:.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    # Enhanced cleanup
    del X, gdf
    gc.collect()
    clear_gpu_memory()

    # Get model parameters with memory efficiency
    params = get_model_params_dict(CONFIG, scale_pos_weight, memory_efficient=True)

    print(f"\n🔧 Using {CONFIG.model_name} with parameters (memory optimized):")
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
            import pyarrow.parquet as pq
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
        
        # Create test workflow without 'clicked' column
        # Use the same preprocessing steps but without 'clicked'
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
    except Exception as e:
        print(f"   ❌ Failed to load test data: {e}")
        raise RuntimeError(f"Test data loading failed: {e}") from e

    # Cross-validation
    print("\n🔄 Starting cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []
    cv_ap = []
    cv_wll = []
    test_predictions = []  # Store predictions from each fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        print(f"\n📍 Fold {fold}/{n_folds}")
        fold_start = time.time()

        print(f"   Train: {len(train_idx):,} | Val: {len(val_idx):,}")

        if CONFIG.model_name == 'xgboost':
            dtrain = xgb.DMatrix(X_np[train_idx], label=y[train_idx])
            dval   = xgb.DMatrix(X_np[val_idx],   label=y[val_idx])

            print("   Training XGBoost...")
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
            model.fit(
                X_np[train_idx], y[train_idx],
                eval_set=(X_np[val_idx], y[val_idx]),
                early_stopping_rounds=CONFIG.model_params['early_stopping_rounds'],
                verbose=False
            )
            y_pred = model.predict_proba(X_np[val_idx])[:, 1]
            best_iteration = model.get_best_iteration()

        score, ap, wll = calculate_competition_score(y[val_idx], y_pred)

        cv_scores.append(score)
        cv_ap.append(ap)
        cv_wll.append(wll)

        print(f"   📊 Results:")
        print(f"      Score: {score:.6f}")
        print(f"      AP: {ap:.6f}")
        print(f"      WLL: {wll:.6f}")
        print(f"      Best iteration: {best_iteration}")
        print(f"   ⏱️ Time: {time.time() - fold_start:.1f}s")
        
        # Predict on test data
        print(f"   🔮 Predicting on test data...")
        if CONFIG.model_name == 'xgboost':
            dtest = xgb.DMatrix(X_test_np)
            fold_test_pred = model.predict(dtest)
            del dtest
        else:
            fold_test_pred = model.predict_proba(X_test_np)[:, 1]
        
        test_predictions.append(fold_test_pred)
        print(f"      Test predictions saved for fold {fold}")

        # 메모리 정리: 3 fold마다 큼직하게
        if CONFIG.model_name == 'xgboost':
            del dtrain, dval
        del model, y_pred, score, ap, wll, best_iteration, fold_test_pred
        gc.collect()
        if fold % 3 == 0:
            clear_gpu_memory()

    # Final results
    print("\n" + "="*70)
    print("📊 Final Cross-Validation Results")
    print("="*70)

    print(f"\n🏆 Competition Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"📈 Average Precision: {np.mean(cv_ap):.6f} ± {np.std(cv_ap):.6f}")
    print(f"📉 Weighted LogLoss: {np.mean(cv_wll):.6f} ± {np.std(cv_wll):.6f}")

    print(f"\nAll fold scores: {[f'{s:.6f}' for s in cv_scores]}")

    # Generate ensemble predictions
    print("\n" + "="*70)
    print("🔮 Generating Ensemble Test Predictions")
    print("="*70)
    
    # Average predictions from all folds
    avg_predictions = sum(test_predictions) / len(test_predictions)
    print(f"   Averaged {len(test_predictions)} fold predictions")
    print(f"   Prediction range: [{avg_predictions.min():.6f}, {avg_predictions.max():.6f}]")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'clicked': avg_predictions
    })
    
    submission_path = f'{CONFIG.output_dir}/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"   ✅ Submission saved to {submission_path}")
    print(f"   Samples: {len(submission_df):,}")

    # Enhanced memory cleanup after CV
    print("\n🧹 Cleaning up CV memory...")
    del X_np, y, cv_ap, cv_wll, test_predictions, avg_predictions, X_test_np
    clear_gpu_memory()
    print("   ✅ CV memory cleaned")

    return cv_scores

# Run cross-validation
cv_scores = run_cv(processed_dir, CONFIG.n_folds)

# Final summary
if cv_scores:
    print("\n" + "🎉"*35)
    print("CROSS-VALIDATION & PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Final CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"✅ Model used: {CONFIG.model_name}")
    print(f"✅ Output directory: {CONFIG.output_dir}")
    print("✅ Full dataset processed (10.7M rows)")
    print("✅ GBDT-optimized preprocessing (no normalization)")
    print(f"✅ Ensemble predictions from {CONFIG.n_folds} folds")
    print(f"✅ Submission saved: {CONFIG.output_dir}/submission.csv")
    print("="*70)
else:
    print("\n⚠️ Cross-validation did not complete. Please check for errors above.")

# Cleanup - removed train_final_model_and_predict() function
# Now using ensemble of CV fold predictions instead

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
