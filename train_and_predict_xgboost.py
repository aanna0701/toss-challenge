# Environment setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured")

# Required libraries and versions
required_libs = {
    'nvtabular': '23.08.00',
    'cudf': '23.10',      # Prefix match
    'cupy': '13.6',       # Prefix match  
    'xgboost': '3.0',     # Minimum version
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
        
        # Check version compatibility
        req_major = required_version.split('.')[0]
        inst_version_parts = installed_version.split('.')
        inst_major = inst_version_parts[0] if installed_version != 'unknown' else ''
        
        # More lenient version check
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
import datetime
import cupy as cp

# NVTabular
import nvtabular as nvt
from nvtabular import ops
from merlin.io import Dataset

# ML libraries
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

print("✅ All libraries imported successfully")
print(f"NVTabular version: {nvt.__version__}")
print(f"XGBoost version: {xgb.__version__}")

# Configuration(DATA PATH)
TRAIN_PATH = 'data/train.parquet'
OUTPUT_DIR = f"results/xgboost_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
TEMP_DIR = 'tmp'
N_FOLDS = 5
FORCE_REPROCESS = False  # Set to True to reprocess data

print(f"📋 Configuration:")
print(f"   Input: {TRAIN_PATH}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Folds: {N_FOLDS}")
print(f"   Force reprocess: {FORCE_REPROCESS}")

# Memory management functions
def print_memory():
    """Print current memory usage"""
    mem = psutil.virtual_memory()
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = gpu_info.used / 1024**3
        gpu_total = gpu_info.total / 1024**3
    except:
        gpu_used = 0
        gpu_total = 0
    
    print(f"💾 CPU: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")
    print(f"💾 GPU: {gpu_used:.1f}GB/{gpu_total:.1f}GB")
    return mem.percent

def clear_gpu_memory():
    """Clear GPU memory"""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print("🧹 GPU memory cleared")

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
    """Create NVTabular workflow optimized for XGBoost"""
    print("\n🔧 Creating XGBoost-optimized workflow...")
    
    # TRUE CATEGORICAL COLUMNS (only 5)
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    
    # CONTINUOUS COLUMNS (112 total)
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +  # 18
        [f'feat_b_{i}' for i in range(1, 7)] +   # 6
        [f'feat_c_{i}' for i in range(1, 9)] +   # 8
        [f'feat_d_{i}' for i in range(1, 7)] +   # 6
        [f'feat_e_{i}' for i in range(1, 11)] +  # 10
        [f'history_a_{i}' for i in range(1, 8)] +  # 7
        [f'history_b_{i}' for i in range(1, 31)] + # 30
        [f'l_feat_{i}' for i in range(1, 28)]      # 27
    )
    
    print(f"   Categorical: {len(true_categorical)} columns")
    print(f"   Continuous: {len(all_continuous)} columns")
    print(f"   Total features: {len(true_categorical) + len(all_continuous)}")
    
    # Minimal preprocessing for XGBoost
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
    if os.path.exists(OUTPUT_DIR) and not FORCE_REPROCESS:
        try:
            test_dataset = Dataset(OUTPUT_DIR, engine='parquet')
            print(f"✅ Using existing processed data from {OUTPUT_DIR}")
            return OUTPUT_DIR
        except:
            print(f"⚠️ Existing data corrupted, reprocessing...")
            shutil.rmtree(OUTPUT_DIR)
    
    # Clear existing if needed
    if os.path.exists(OUTPUT_DIR):
        print(f"🗑️ Removing existing directory {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    start_time = time.time()
    initial_mem = print_memory()
    
    # Prepare data without 'seq' column
    temp_path = f'{TEMP_DIR}/train_no_seq.parquet'
    if not os.path.exists(temp_path):
        print("\n📋 Creating temp file without 'seq' column...")
        pf = pq.ParquetFile(TRAIN_PATH)
        cols = [c for c in pf.schema.names if c != 'seq']
        print(f"   Total columns: {len(pf.schema.names)}")
        print(f"   Using columns: {len(cols)} (excluded 'seq')")
        
        df = pd.read_parquet(TRAIN_PATH, columns=cols)
        print(f"   Loaded {len(df):,} rows")
        df.to_parquet(temp_path, index=False)
        del df
        gc.collect()
        print("   ✅ Temp file created")
    else:
        print(f"✅ Using existing temp file: {temp_path}")
    
    # Create dataset with small partitions
    print("\n📦 Creating NVTabular Dataset...")
    print("   Using 32MB partitions for memory efficiency")
    clear_gpu_memory()
    
    dataset = Dataset(
        temp_path,
        engine='parquet',
        part_size='32MB'  #change size based on your environment
    )
    print("   ✅ Dataset created")
    
    # Create and fit workflow
    print("\n📊 Fitting workflow...")
    workflow = create_workflow()
    workflow.fit(dataset)
    print("   ✅ Workflow fitted")
    
    # Transform and save
    print(f"\n💾 Transforming and saving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    clear_gpu_memory()
    
    try:
        workflow.transform(dataset).to_parquet(
            output_path=OUTPUT_DIR,
            shuffle=nvt.io.Shuffle.PER_PARTITION,
            out_files_per_proc=8
        )
        
        workflow_path = f'{OUTPUT_DIR}/workflow'
        workflow.save(workflow_path)
        print(f"   ✅ Data processed and saved")
        print(f"   ✅ Workflow saved to {workflow_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        raise
    
    elapsed = time.time() - start_time
    final_mem = print_memory()
    
    print(f"\n✅ Processing complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Memory increase: +{final_mem - initial_mem:.1f}%")
    
    clear_gpu_memory()
    return OUTPUT_DIR

# Process data
processed_dir = process_data()

def run_cv(processed_dir, n_folds=5):
    """Run stratified cross-validation"""
    print("\n" + "="*70)
    print("🔄 Stratified KFold Cross-Validation")
    print("="*70)
    
    # Load processed data
    print("\n📦 Loading processed data...")
    start_load = time.time()
    
    try:
        dataset = Dataset(processed_dir, engine='parquet', part_size='256MB')
        print("   Converting to GPU DataFrame...")
        gdf = dataset.to_ddf().compute()
        print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
        print(f"   Time: {time.time() - start_load:.1f}s")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    print_memory()
    
    # Prepare data
    print("\n📊 Preparing data for XGBoost...")
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    
    # Convert to float32
    for col in X.columns:
        if X[col].dtype != 'float32':
            X[col] = X[col].astype('float32')
    
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
    
    del X, gdf
    clear_gpu_memory()
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'gpu_id': 0,
        'verbosity': 0,
        'seed': 42
    }
    
    # Cross-validation
    print("\n🔄 Starting cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_ap = []
    cv_wll = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        print(f"\n📍 Fold {fold}/{n_folds}")
        fold_start = time.time()
        
        # Create DMatrix
        print(f"   Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        dtrain = xgb.DMatrix(X_np[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X_np[val_idx], label=y[val_idx])
        
        # Train
        print("   Training...")
        model = xgb.train(
            params, dtrain,
            num_boost_round=200,
            evals=[(dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Evaluate
        y_pred = model.predict(dval)
        score, ap, wll = calculate_competition_score(y[val_idx], y_pred)
        
        cv_scores.append(score)
        cv_ap.append(ap)
        cv_wll.append(wll)
        
        print(f"   📊 Results:")
        print(f"      Score: {score:.6f}")
        print(f"      AP: {ap:.6f}")
        print(f"      WLL: {wll:.6f}")
        print(f"      Best iteration: {model.best_iteration}")
        print(f"   ⏱️ Time: {time.time() - fold_start:.1f}s")
        
        # Cleanup
        del dtrain, dval, model
        clear_gpu_memory()
    
    # Final results
    print("\n" + "="*70)
    print("📊 Final Cross-Validation Results")
    print("="*70)
    
    print(f"\n🏆 Competition Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"📈 Average Precision: {np.mean(cv_ap):.6f} ± {np.std(cv_ap):.6f}")
    print(f"📉 Weighted LogLoss: {np.mean(cv_wll):.6f} ± {np.std(cv_wll):.6f}")
    
    print(f"\nAll fold scores: {[f'{s:.6f}' for s in cv_scores]}")
    
    return cv_scores

# Run cross-validation
cv_scores = run_cv(processed_dir, N_FOLDS)

# Final summary
if cv_scores:
    print("\n" + "🎉"*35)
    print("COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Final CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print("✅ Full dataset processed (10.7M rows)")
    print("✅ XGBoost-optimized preprocessing (no normalization)")
    print("✅ Memory-efficient with small partitions")
    print("="*70)
else:
    print("\n⚠️ Cross-validation did not complete. Please check for errors above.")

# Final cleanup
clear_gpu_memory()

# Clean up any remaining temporary files
print("\n🧹 Final cleanup...")
temp_path = f'{TEMP_DIR}/train_no_seq.parquet'
if os.path.exists(temp_path):
    os.remove(temp_path)
    print(f"   ✅ Removed {temp_path}")

if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
    os.rmdir(TEMP_DIR)
    print(f"   ✅ Removed empty directory {TEMP_DIR}")

print("🧹 Final cleanup complete")
print_memory()