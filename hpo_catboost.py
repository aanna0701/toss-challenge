# CatBoost Hyperparameter Optimization using Optuna — GPU handle-safe version
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

for k in ("CATBOOST_LOGGING_LEVEL", "CATBOOST_VERBOSE", "CATBOOST_SILENT"):
    os.environ.pop(k, None)

LOG_KEYS = ("verbose", "logging_level", "verbose_eval", "silent")

def sanitize_logging(params: dict) -> dict:
    p = dict(params)
    for k in LOG_KEYS:
        p.pop(k, None)
    return p


import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
# CUDA/driver 충돌 완화: spawn 사용
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

print("✅ Environment configured for CatBoost HPO (GPU-handle-safe)")

# Core imports
import gc
import time
import numpy as np
import argparse
import yaml

# GPU libraries
import cupy as cp

# RMM / cuDF allocator initialization
try:
    import rmm, cudf
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size="10GB",
        managed_memory=False,   # 🔧 충돌 가능성 낮춤
    )
    cudf.set_allocator("managed")  # cuDF는 managed로 유지
    print("✅ RMM initialized (pool=10GB, managed_memory=False)")
except Exception as e:
    print(f"⚠️ RMM init skipped: {e}")

# 선택된 GPU로 컨텍스트 바인딩
try:
    cp.cuda.Device(0).use()
except Exception as e:
    print(f"⚠️ Could not select CUDA device 0: {e}")

# ML libraries
import catboost as cb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# Optuna
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# NVTabular
import nvtabular as nvt
from nvtabular import ops
from merlin.io import Dataset

# MixUp
from mixup import apply_mixup_to_dataset

print(f"✅ CatBoost version: {cb.__version__}")
print(f"✅ Optuna version: {optuna.__version__}")

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    ll_0 = -np.mean(np.log(np.clip(1 - y_pred[mask_0], eps, 1 - eps))) if mask_0.sum() > 0 else 0.0
    ll_1 = -np.mean(np.log(np.clip(y_pred[mask_1], eps, 1 - eps))) if mask_1.sum() > 0 else 0.0
    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

def clear_gpu_memory():
    """Clear GPU memory and reset CatBoost CUDA handles"""
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    # 🔧 CatBoost GPU 핸들 완전 초기화
    try:
        cb._catboost._reset_cuda_manager()
    except Exception as e:
        # 일부 버전에서 심볼이 없을 수 있음
        print(f"ℹ️ CatBoost CUDA manager reset not available: {e}")
    gc.collect()

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
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]  # 25
    )

    print(f"   Categorical: {len(true_categorical)} columns")
    print(f"   Continuous: {len(all_continuous)} columns")
    print(f"   Total features: {len(true_categorical) + len(all_continuous)}")

    # Minimal preprocessing for GBDT models
    cat_features = true_categorical >> ops.Categorify(freq_threshold=0, max_size=50000)
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)

    workflow = nvt.Workflow(cat_features + cont_features + ['clicked'])
    print("   ✅ Workflow created (no normalization for tree models)")
    return workflow

def process_data_with_nvtabular(data_path, temp_dir='tmp'):
    """Process data with NVTabular (matching train_and_predict_GBDT.py)"""
    import pandas as pd
    import pyarrow.parquet as pq

    print("\n" + "="*70)
    print("🚀 NVTabular Data Processing")
    print("="*70)

    os.makedirs(temp_dir, exist_ok=True)

    # Prepare data without 'seq' column
    temp_path = f'{temp_dir}/train_no_seq.parquet'
    if not os.path.exists(temp_path):
        print("\n📋 Creating temp file without 'seq' column...")
        pf = pq.ParquetFile(data_path)
        cols = [c for c in pf.schema.names if c not in ['seq', '']]
        print(f"   Total columns: {len(pf.schema.names)}")
        print(f"   Using columns: {len(cols)} (excluded 'seq')")

        df = pd.read_parquet(data_path, columns=cols)
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

    dataset = Dataset(temp_path, engine='parquet', part_size='64MB')
    print("   ✅ Dataset created")

    # Create and fit workflow
    print("\n📊 Fitting workflow...")
    workflow = create_workflow()
    workflow.fit(dataset)
    print("   ✅ Workflow fitted")

    # Transform and return processed data
    print(f"\n💾 Transforming data...")
    clear_gpu_memory()

    try:
        gdf = workflow.transform(dataset).to_ddf().compute()
        print(f"   ✅ Data processed: {len(gdf):,} rows x {len(gdf.columns)} columns")
        return gdf
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

def load_processed_data(data_path, subsample_ratio=1.0):
    """Load processed data (matching train_and_predict_GBDT.py exactly)"""
    print(f"\n📦 Loading data from {data_path}...")
    start_load = time.time()
    
    # Check if it's NVTabular processed data or raw parquet
    if os.path.isdir(data_path):
        try:
            dataset = Dataset(data_path, engine='parquet', part_size='128MB')
            print("   Converting to GPU DataFrame...")
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("   Trying with even smaller partitions...")
            dataset = Dataset(data_path, engine='parquet', part_size='64MB')
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded with 64MB partitions: {len(gdf):,} rows")
    else:
        gdf = process_data_with_nvtabular(data_path)
    
    # Subsample if needed
    if subsample_ratio < 1.0:
        n_samples = int(len(gdf) * subsample_ratio)
        if 'clicked' in gdf.columns:
            gdf_pos = gdf[gdf['clicked'] == 1]
            gdf_neg = gdf[gdf['clicked'] == 0]
            n_pos = int(len(gdf_pos) * subsample_ratio)
            n_neg = int(len(gdf_neg) * subsample_ratio)
            gdf = cudf.concat([
                gdf_pos.sample(n=min(n_pos, len(gdf_pos)), random_state=42),
                gdf_neg.sample(n=min(n_neg, len(gdf_neg)), random_state=42)
            ]).reset_index(drop=True)
            print(f"   📊 Stratified subsampled to {len(gdf):,} rows (ratio={subsample_ratio})")
        else:
            gdf = gdf.sample(n=n_samples, random_state=42).reset_index(drop=True)
            print(f"   📊 Subsampled to {len(gdf):,} rows (ratio={subsample_ratio})")
    
    # Prepare X and y
    print("\n📊 Preparing data for GBDT...")
    if 'clicked' not in gdf.columns:
        raise ValueError("'clicked' column not found in data")
    
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    
    print("   Converting all features to float32 (single pass)...")
    try:
        X = X.astype('float32', copy=False)
    except Exception as e:
        print(f"   ⚠️ astype(float32) failed with copy=False: {e}")
        X = X.astype('float32')
    
    print("   Converting to numpy...")
    X_np = X.to_numpy()
    print(f"   Shape: {X_np.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Time: {time.time() - start_load:.1f}s")
    
    del X, gdf
    gc.collect()
    clear_gpu_memory()
    
    return X_np, y

def objective(trial, X_train_orig, y_train_orig, X_val, y_val,
              early_stopping_rounds=20, task_type='GPU', use_mixup=True, scale_pos_weight=1.0):
    """Optuna objective function for CatBoost (GPU-handle-safe)"""
    # Hyperparameter space
    params = {
        'task_type': task_type,
        'devices': '0',
        'verbose': False,
        'random_seed': 42,
        'thread_count': -1,
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.08, 0.3),
        'depth': trial.suggest_int('depth', 3, 16),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
        # 🔧 GPU 안정화 파라미터
        'allow_writing_files': False,
        'gpu_ram_part': 0.85,
        'used_ram_limit': '8gb',
        'logging_level': 'Silent',
    }
    # GPU에서는 colsample_bylevel 대신 다른 방법 사용
    if task_type == 'GPU':image.png
        # GPU에서는 rsm 파라미터 사용 (pairwise 모드에서만 지원되므로 주의)
        params['rsm'] = trial.suggest_float('rsm', 0.5, 1)
    else:
        # CPU에서는 기존 방식 사용
        params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1)

    # Class imbalance handling
    params['auto_class_weights'] = 'Balanced'
    
    # MixUp
    if use_mixup:
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.01, 0.3)
        mixup_ratio = trial.suggest_float('mixup_ratio', 0.3, 0.7, step=0.1)
        class_weight = (1.0, scale_pos_weight)
        X_train, y_train, sample_weight = apply_mixup_to_dataset(
            X_train_orig, y_train_orig,
            class_weight=class_weight,
            alpha=mixup_alpha,
            ratio=mixup_ratio,
            rng=np.random.default_rng(42)
        )
    else:
        X_train = X_train_orig
        y_train = y_train_orig
        sample_weight = None
    
    params = sanitize_logging(params)
    model = cb.CatBoostClassifier(**params)

    try:
        if sample_weight is not None:
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=(X_val, y_val),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        y_pred = model.predict_proba(X_val)[:, 1]
        score, _, _ = calculate_competition_score(y_val, y_pred)
    except Exception as e:
        # GPU 핸들 오류 등 발생 시: 컨텍스트 정리 후 trial 낮은 점수 반환/Prune
        print(f"❌ Trial failed with exception: {e}")
        score = 0.0  # 또는: raise optuna.TrialPruned()
    finally:
        # Cleanup & 강제 리셋
        del model
        gc.collect()
        clear_gpu_memory()

    return score

def run_optimization(data_path, n_trials=100, val_ratio=0.2, subsample_ratio=1.0, 
                     early_stopping_rounds=20, timeout=None, task_type='GPU', use_mixup=True):
    """Run Optuna optimization (GPU-handle-safe)"""
    print("\n" + "="*70)
    print("🔍 CatBoost Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    
    # Load data
    X_np, y = load_processed_data(data_path, subsample_ratio)
    
    # Split
    print(f"\n📊 Splitting data (val_ratio={val_ratio})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y, test_size=val_ratio, random_state=42, stratify=y
    )
    
    # scale_pos_weight
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    print(f"\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Task type: {task_type}")
    print(f"   Total samples: {len(X_np):,}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Val samples: {len(X_val):,}")
    print(f"   Features: {X_np.shape[1]}")
    print(f"   Train positive ratio: {y_train.mean():.4f}")
    print(f"   Val positive ratio: {y_val.mean():.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    print(f"   Timeout: {timeout if timeout else 'None'}")
    
    # Study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    print("\n🚀 Starting optimization...")
    start_time = time.time()
    
    # Optimize (trial마다 GC 강제)
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val,
            early_stopping_rounds, task_type, use_mixup, scale_pos_weight
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Optimization Complete!")
    print("="*70)
    
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"🎯 Best score: {study.best_value:.6f}")
    print(f"📊 Number of finished trials: {len(study.trials)}")
    
    print("\n🏆 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    del X_np, y, X_train, X_val, y_train, y_val
    clear_gpu_memory()
    
    return study

def save_best_params_to_yaml(study, output_path='config_GBDT_optimized.yaml', 
                              original_config_path='config_GBDT.yaml'):
    """Save best parameters to YAML config"""
    print(f"\n💾 Saving best parameters to {output_path}...")
    
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    best_params = study.best_params
    
    config['catboost']['n_estimators'] = best_params['iterations']
    config['catboost']['learning_rate'] = best_params['learning_rate']
    config['catboost']['max_depth'] = best_params['depth']
    config['catboost']['bootstrap_type'] = best_params['bootstrap_type']
    if 'colsample_bylevel' in best_params:
        config['catboost']['colsample_bylevel'] = best_params['colsample_bylevel']
    
    # 기본값
    config['catboost']['task_type'] = 'GPU'
    config['catboost']['devices'] = '0'
    config['catboost']['verbose'] = False
    config['catboost']['early_stopping_rounds'] = 20
    config['catboost']['thread_count'] = -1
    config['catboost']['random_state'] = 42
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"   ✅ Saved to {output_path}")
    
    best_params_path = output_path.replace('.yaml', '_catboost_best.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'best_score': float(study.best_value),
            'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in best_params.items()}
        }, f, default_flow_style=False)
    print(f"   ✅ Best params saved to {best_params_path}")

def main():
    parser = argparse.ArgumentParser(description='CatBoost Hyperparameter Optimization')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed data directory or raw parquet file')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--subsample-ratio', type=float, default=1.0,
                        help='Ratio of data to use (default: 1.0 = use all)')
    parser.add_argument('--early-stopping-rounds', type=int, default=20,
                        help='Early stopping rounds (default: 20)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--task-type', type=str, default='GPU', choices=['GPU', 'CPU'],
                        help='Task type for CatBoost (default: GPU)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                        help='Enable MixUp data augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                        help='Disable MixUp data augmentation')
    parser.add_argument('--output-config', type=str, default='config_optimized.yaml',
                        help='Output config file path (default: config_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_GBDT.yaml',
                        help='Original config file path (default: config_GBDT.yaml)')
    
    args = parser.parse_args()
    
    print(f"\n🔧 HPO Configuration:")
    print(f"   Data path: {args.data_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Validation ratio: {args.val_ratio}")
    print(f"   Subsample ratio: {args.subsample_ratio}")
    print(f"   Early stopping: {args.early_stopping_rounds}")
    print(f"   Task type: {args.task_type}")
    print(f"   Use MixUp: {args.use_mixup}")
    print(f"   Timeout: {args.timeout if args.timeout else 'None'}")
    
    # Run optimization
    study = run_optimization(
        data_path=args.data_path,
        n_trials=args.n_trials,
        val_ratio=args.val_ratio,
        subsample_ratio=args.subsample_ratio,
        early_stopping_rounds=args.early_stopping_rounds,
        timeout=args.timeout,
        task_type=args.task_type,
        use_mixup=args.use_mixup
    )
    
    # Save results
    save_best_params_to_yaml(
        study, 
        output_path=args.output_config,
        original_config_path=args.original_config
    )
    
    print("\n" + "🎉"*35)
    print("OPTIMIZATION COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Best score: {study.best_value:.6f}")
    print(f"✅ Config saved to: {args.output_config}")
    print("="*70)

if __name__ == '__main__':
    main()
