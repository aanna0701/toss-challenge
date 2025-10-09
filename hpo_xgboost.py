# XGBoost Hyperparameter Optimization using Optuna
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured for XGBoost HPO")

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
        managed_memory=True,
    )
    cudf.set_allocator("managed")
    print("✅ RMM initialized (pool=10GB, managed_memory=True)")
except Exception as e:
    print(f"⚠️ RMM init skipped: {e}")

cp.cuda.Device(0).use()

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Optuna
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import common functions
from utils import calculate_competition_score, clear_gpu_memory
from data_loader import load_processed_data_gbdt
from mixup import apply_mixup_to_dataset

print(f"✅ XGBoost version: {xgb.__version__}")
print(f"✅ Optuna version: {optuna.__version__}")

def objective(trial, X_train_orig, y_train_orig, X_val, y_val, early_stopping_rounds=20, use_mixup=True, scale_pos_weight=1.0):
    """Optuna objective function for XGBoost"""
    
    # Hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'verbosity': 0,
        'predictor': 'gpu_predictor',
        
        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 100),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1e-5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1e-5, log=True),
        'max_bin': trial.suggest_int('max_bin', 128, 256),
        'seed': 42,
        'scale_pos_weight': scale_pos_weight,
    }
    
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    
    # MixUp hyperparameters (if enabled)
    if use_mixup:
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.01, 0.3)
        mixup_ratio = 0.6
        
        # Apply MixUp
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
    
    # Train model
    if sample_weight is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    y_pred = model.predict(dval)
    score, _, _ = calculate_competition_score(y_val, y_pred)
    
    # Cleanup
    del dtrain, dval, model
    gc.collect()
    clear_gpu_memory()
    
    return score

def run_optimization(data_path, n_trials=100, val_ratio=0.2, 
                     early_stopping_rounds=20, timeout=None, use_mixup=True):
    """Run Optuna optimization"""
    print("\n" + "="*70)
    print("🔍 XGBoost Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    
    # Load data
    X_np, y = load_processed_data_gbdt(data_path)
    
    # Split data into train/validation
    print(f"\n📊 Splitting data (val_ratio={val_ratio})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y, test_size=val_ratio, random_state=42, stratify=y
    )
    
    # Calculate scale_pos_weight
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    print(f"\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Total samples: {len(X_np):,}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Val samples: {len(X_val):,}")
    print(f"   Features: {X_np.shape[1]}")
    print(f"   Train positive ratio: {y_train.mean():.4f}")
    print(f"   Val positive ratio: {y_val.mean():.4f}")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    if timeout:
        print(f"   Timeout: {timeout}s")
    else:
        print("   Timeout: None")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    print("\n🚀 Starting optimization...")
    start_time = time.time()
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, early_stopping_rounds, use_mixup, scale_pos_weight),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    
    # Results
    print("\n" + "="*70)
    print("✅ Optimization Complete!")
    print("="*70)
    
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"🎯 Best score: {study.best_value:.6f}")
    print(f"📊 Number of finished trials: {len(study.trials)}")
    
    print("\n🏆 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    del X_np, y, X_train, X_val, y_train, y_val
    clear_gpu_memory()
    
    return study

def save_best_params_to_yaml(study, output_path='config_GBDT_optimized.yaml', 
                              original_config_path='config_GBDT.yaml'):
    """Save best parameters to YAML config"""
    print(f"\n💾 Saving best parameters to {output_path}...")
    
    # Load original config
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update XGBoost parameters
    best_params = study.best_params
    config['xgboost']['n_estimators'] = best_params['n_estimators']
    config['xgboost']['learning_rate'] = best_params['learning_rate']
    config['xgboost']['max_depth'] = best_params['max_depth']
    config['xgboost']['subsample'] = 1.0
    config['xgboost']['colsample_bytree'] = best_params['colsample_bytree']
    
    # Add additional parameters
    if 'min_child_weight' in best_params:
        config['xgboost']['min_child_weight'] = best_params['min_child_weight']
    if 'gamma' in best_params:
        config['xgboost']['gamma'] = best_params['gamma']
    if 'reg_alpha' in best_params:
        config['xgboost']['reg_alpha'] = best_params['reg_alpha']
    if 'reg_lambda' in best_params:
        config['xgboost']['reg_lambda'] = best_params['reg_lambda']
    if 'max_bin' in best_params:
        config['xgboost']['max_bin'] = best_params['max_bin']
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Saved to {output_path}")
    
    # Also save best params separately
    best_params_path = output_path.replace('.yaml', '_xgboost_best.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'best_score': float(study.best_value),
            'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in best_params.items()}
        }, f, default_flow_style=False)
    
    print(f"   ✅ Best params saved to {best_params_path}")

def main():
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Optimization')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed data directory or raw parquet file')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--early-stopping-rounds', type=int, default=20,
                        help='Early stopping rounds (default: 20)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                        help='Enable MixUp data augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                        help='Disable MixUp data augmentation')
    parser.add_argument('--output-config', type=str, default='config_xgboost_optimized.yaml',
                        help='Output config file path (default: config_xgboost_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_GBDT.yaml',
                        help='Original config file path (default: config_GBDT.yaml)')
    
    args = parser.parse_args()
    
    print(f"\n🔧 HPO Configuration:")
    print(f"   Data path: {args.data_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Validation ratio: {args.val_ratio}")
    print(f"   Early stopping: {args.early_stopping_rounds}")
    print(f"   Use MixUp: {args.use_mixup}")
    if args.timeout:
        print(f"   Timeout: {args.timeout}s")
    else:
        print("   Timeout: None")
    
    # Run optimization
    study = run_optimization(
        data_path=args.data_path,
        n_trials=args.n_trials,
        val_ratio=args.val_ratio,
        early_stopping_rounds=args.early_stopping_rounds,
        timeout=args.timeout,
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

