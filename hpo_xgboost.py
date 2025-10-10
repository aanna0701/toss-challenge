# XGBoost Hyperparameter Optimization using Optuna
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured for XGBoost HPO")

# Standard library
import argparse
import gc
import time

# Third-party libraries
import cupy as cp
import numpy as np
import optuna
import xgboost as xgb
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression

# RMM / cuDF allocator initialization
try:
    import cudf
    import rmm
    
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

# Custom modules
from data_loader import load_processed_data_gbdt
from mixup import apply_mixup_to_dataset
from utils import calculate_competition_score, clear_gpu_memory


print(f"✅ XGBoost version: {xgb.__version__}")
print(f"✅ Optuna version: {optuna.__version__}")

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
    
    def predict_proba(self, logits):
        """Apply temperature scaling"""
        scaled_logits = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))
        return probs

def objective(trial, X_train_orig, y_train_orig, X_val, y_val, X_cal, y_cal, early_stopping_rounds=20, use_mixup=True, scale_pos_weight=1.0):
    """Optuna objective function for XGBoost with calibration"""
    
    # Hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'verbosity': 0,
        'predictor': 'gpu_predictor',
        
        # GPU memory optimization
        'max_bin': trial.suggest_int('max_bin', 128, 256),  # Reduced from 512
        'gpu_page_size': 4096,  # Smaller page size for memory efficiency
        
        # Hyperparameters to optimize
        'max_depth': trial.suggest_int('max_depth', 3, 20),  # Reduced from 30
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # Start from 0.5
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # Reduced from 30
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),  # Reduced from 5
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),  # Reduced from 5
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),  # Reduced from 5
        'seed': 42,
        'scale_pos_weight': scale_pos_weight,
    }
    
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    
    # Calibration method
    calibration_method = trial.suggest_categorical('calibration_method', ['none', 'temperature', 'sigmoid'])
    
    # MixUp hyperparameters (if enabled)
    if use_mixup:
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.7, step=0.1)
        mixup_ratio = trial.suggest_float('mixup_ratio', 0.1, 0.7, step=0.1)
        
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
    
    # Train model with error handling and CPU fallback
    try:
        if sample_weight is not None:
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Clear GPU memory before training
        clear_gpu_memory()
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
    except Exception as e:
        print(f"⚠️ XGBoost GPU training error: {e}")
        print("   Trying CPU fallback...")
        
        # Cleanup GPU resources
        if 'dtrain' in locals():
            del dtrain
        if 'dval' in locals():
            del dval
        gc.collect()
        clear_gpu_memory()
        
        # Try CPU fallback
        try:
            params_cpu = params.copy()
            params_cpu['tree_method'] = 'hist'
            params_cpu['predictor'] = 'cpu_predictor'
            del params_cpu['gpu_id']
            del params_cpu['gpu_page_size']
            
            if sample_weight is not None:
                dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
            else:
                dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params_cpu, dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            print("   ✅ CPU fallback successful")
        except Exception as e2:
            print(f"⚠️ CPU fallback also failed: {e2}")
            # Final cleanup
            if 'dtrain' in locals():
                del dtrain
            if 'dval' in locals():
                del dval
            gc.collect()
            clear_gpu_memory()
            return -1.0  # Return poor score to skip this trial
    
    # Get predictions on validation and calibration sets
    y_pred_val = model.predict(dval)
    
    # Apply calibration if needed
    if calibration_method != 'none':
        # Get predictions on calibration data (train_c)
        dcal = xgb.DMatrix(X_cal)
        y_pred_cal = model.predict(dcal)
        
        # Balance calibration data: downsample negatives to match positives
        pos_idx = np.where(y_cal == 1)[0]
        neg_idx = np.where(y_cal == 0)[0]
        n_pos = len(pos_idx)
        
        # Randomly sample negatives to match positive count
        rng = np.random.default_rng(42)
        neg_sampled_idx = rng.choice(neg_idx, size=min(n_pos, len(neg_idx)), replace=False)
        
        # Combine indices
        balanced_idx = np.concatenate([pos_idx, neg_sampled_idx])
        rng.shuffle(balanced_idx)
        
        # Create balanced calibration set
        y_pred_cal_balanced = y_pred_cal[balanced_idx]
        y_cal_balanced = y_cal[balanced_idx]
        
        if calibration_method == 'temperature':
            # Convert probabilities to logits
            y_pred_cal_clipped = np.clip(y_pred_cal_balanced, 1e-7, 1 - 1e-7)
            logits_cal = np.log(y_pred_cal_clipped / (1 - y_pred_cal_clipped))
            
            # Fit temperature scaling on balanced train_c
            calibrator = TemperatureScaling()
            calibrator.fit(logits_cal, y_cal_balanced)
            
            # Apply to validation set
            y_pred_val_clipped = np.clip(y_pred_val, 1e-7, 1 - 1e-7)
            logits_val = np.log(y_pred_val_clipped / (1 - y_pred_val_clipped))
            y_pred_calibrated = calibrator.predict_proba(logits_val)
            
        else:  # sigmoid
            # Fit logistic regression on balanced train_c
            calibrator = LogisticRegression()
            calibrator.fit(y_pred_cal_balanced.reshape(-1, 1), y_cal_balanced)
            
            # Apply to validation set
            y_pred_calibrated = calibrator.predict_proba(y_pred_val.reshape(-1, 1))[:, 1]
        
        # Calculate score on validation set with calibration
        score, _, _ = calculate_competition_score(y_val, y_pred_calibrated)
        
        del dcal, calibrator
    else:
        # No calibration - use validation set predictions directly
        score, _, _ = calculate_competition_score(y_val, y_pred_val)
    
    # Cleanup
    del dtrain, dval, model
    gc.collect()
    clear_gpu_memory()
    
    return score

def run_optimization(train_t_path, train_v_path, train_c_path, n_trials=100, 
                     early_stopping_rounds=20, timeout=None, use_mixup=True):
    """Run Optuna optimization using pre-processed data"""
    print("\n" + "="*70)
    print("🔍 XGBoost Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    
    # Load train_t (training data, drop seq for GBDT)
    print(f"\n📦 Loading training data from {train_t_path}...")
    X_train, y_train = load_processed_data_gbdt(train_t_path, drop_seq=True)
    
    # Load train_v (validation data, drop seq for GBDT) 
    print(f"\n📦 Loading validation data from {train_v_path}...")
    X_val, y_val = load_processed_data_gbdt(train_v_path, drop_seq=True)
    
    # Load train_c (calibration data, drop seq for GBDT)
    print(f"\n📦 Loading calibration data from {train_c_path}...")
    X_cal, y_cal = load_processed_data_gbdt(train_c_path, drop_seq=True)
    
    # Calculate scale_pos_weight
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    # Calculate balanced calibration set info
    n_cal_pos = int(y_cal.sum())
    n_cal_neg = len(y_cal) - n_cal_pos
    n_cal_balanced = min(n_cal_pos, n_cal_neg) * 2  # positive + sampled negative
    
    print("\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Val samples: {len(X_val):,}")
    print(f"   Cal samples (original): {len(X_cal):,} (pos: {n_cal_pos:,}, neg: {n_cal_neg:,})")
    print(f"   Cal samples (balanced): {n_cal_balanced:,} (pos: {min(n_cal_pos, n_cal_neg):,}, neg: {min(n_cal_pos, n_cal_neg):,})")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Train positive ratio: {y_train.mean():.4f}")
    print(f"   Val positive ratio: {y_val.mean():.4f}")
    print(f"   Cal positive ratio (original): {y_cal.mean():.4f}")
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
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_cal, y_cal, early_stopping_rounds, use_mixup, scale_pos_weight),
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
    del X_train, X_val, X_cal, y_train, y_val, y_cal
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
    config['xgboost']['n_estimators'] = int(best_params['n_estimators'])
    config['xgboost']['learning_rate'] = best_params['learning_rate']
    config['xgboost']['max_depth'] = int(best_params['max_depth'])
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
        config['xgboost']['max_bin'] = int(best_params['max_bin'])
    
    # Add MixUp parameters if present
    if 'mixup_alpha' in best_params:
        config['xgboost']['mixup_alpha'] = best_params['mixup_alpha']
    if 'mixup_ratio' in best_params:
        config['xgboost']['mixup_ratio'] = best_params['mixup_ratio']
    
    # Add calibration method
    if 'calibration_method' in best_params:
        config['xgboost']['calibration_method'] = best_params['calibration_method']
        print(f"   Best calibration method: {best_params['calibration_method']}")
    
    # Save
    with open(output_path.replace('.yaml', '_xgboost_best.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='XGBoost Hyperparameter Optimization')
    
    parser.add_argument('--train-t-path', type=str, default='data/proc_train_hpo',
                        help='Path to training data (default: data/proc_train_hpo)')
    parser.add_argument('--train-v-path', type=str, default='data/proc_train_v',
                        help='Path to validation data (default: data/proc_train_v)')
    parser.add_argument('--train-c-path', type=str, default='data/proc_train_c',
                        help='Path to calibration data (default: data/proc_train_c)')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--early-stopping-rounds', type=int, default=20,
                        help='Early stopping rounds (default: 20)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                        help='Enable MixUp data augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                        help='Disable MixUp data augmentation')
    parser.add_argument('--output-config', type=str, default='config_optimized.yaml',
                        help='Output config file path (default: config_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_GBDT.yaml',
                        help='Original config file path (default: config_GBDT.yaml)')
    
    args = parser.parse_args()
    
    print("\n🔧 HPO Configuration:")
    print(f"   Train data: {args.train_t_path}")
    print(f"   Val data: {args.train_v_path}")
    print(f"   Cal data: {args.train_c_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Early stopping: {args.early_stopping_rounds}")
    print(f"   Use MixUp: {args.use_mixup}")
    if args.timeout:
        print(f"   Timeout: {args.timeout}s")
    else:
        print("   Timeout: None")
    
    # Run optimization
    study = run_optimization(
        train_t_path=args.train_t_path,
        train_v_path=args.train_v_path,
        train_c_path=args.train_c_path,
        n_trials=args.n_trials,
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

