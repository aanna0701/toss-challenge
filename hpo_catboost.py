# CatBoost Hyperparameter Optimization using Optuna — GPU handle-safe version
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# CatBoost logging 환경변수 제거
for k in ("CATBOOST_LOGGING_LEVEL", "CATBOOST_VERBOSE", "CATBOOST_SILENT"):
    os.environ.pop(k, None)

import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
# CUDA/driver 충돌 완화: spawn 사용
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

print("✅ Environment configured for CatBoost HPO (GPU-handle-safe)")

# Standard library
import argparse
import gc
import time

# Third-party libraries
import catboost as cb
import cupy as cp
import numpy as np
import optuna
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
        managed_memory=False,   # 🔧 CatBoost GPU 안정성
    )
    cudf.set_allocator("managed")
    print("✅ RMM initialized (pool=10GB, managed_memory=False)")
except Exception as e:
    print(f"⚠️ RMM init skipped: {e}")

# 선택된 GPU로 컨텍스트 바인딩
try:
    cp.cuda.Device(0).use()
except Exception as e:
    print(f"⚠️ Could not select CUDA device 0: {e}")

# Custom modules
from data_loader import load_processed_data_gbdt
from mixup import apply_mixup_to_dataset
from utils import calculate_competition_score, clear_gpu_memory

print(f"✅ CatBoost version: {cb.__version__}")
print(f"✅ Optuna version: {optuna.__version__}")

# Logging 파라미터 정리 함수
LOG_KEYS = ("verbose", "logging_level", "verbose_eval", "silent")

def sanitize_logging(params: dict) -> dict:
    """Remove logging parameters from CatBoost params"""
    p = dict(params)
    for k in LOG_KEYS:
        p.pop(k, None)
    return p

def clear_catboost_gpu_memory():
    """Clear GPU memory and reset CatBoost CUDA handles"""
    # 기본 GPU 메모리 정리
    clear_gpu_memory()
    
    # 🔧 CatBoost GPU 핸들 완전 초기화
    try:
        cb._catboost._reset_cuda_manager()
    except Exception:
        # 일부 버전에서 심볼이 없을 수 있음
        pass
    
    # 추가 메모리 정리
    gc.collect()
    gc.collect()  # 두 번 호출하여 순환 참조 제거
    
    # CUDA 컨텍스트 동기화
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    
    time.sleep(0.5)  # GPU 핸들 해제 대기

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

def objective(trial, X_train_orig, y_train_orig, X_val, y_val, X_cal, y_cal,
              early_stopping_rounds=20, task_type='GPU', use_mixup=True, scale_pos_weight=1.0,
              bootstrap_types=None):
    """Optuna objective function for CatBoost with calibration (GPU-optimized)"""
    
    # Default bootstrap types if not specified
    if bootstrap_types is None:
        bootstrap_types = ['Bernoulli', 'Bayesian', 'MVS']
    
    # 데이터 샘플링 없이 전체 사용 (파라미터 최적화로 메모리 관리)
    X_train_sampled = X_train_orig
    y_train_sampled = y_train_orig
    X_val_sampled = X_val
    y_val_sampled = y_val
    
    # Hyperparameter search space (전체 데이터 사용을 위한 메모리 최적화)
    params = {
        'task_type': task_type,
        'devices': '0',
        'verbose': False,
        'random_seed': 42,
        'thread_count': -1,
        'iterations': trial.suggest_int('iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # GPU 메모리 최적화: depth를 낮게 유지 (메모리에 가장 큰 영향)
        'depth': trial.suggest_int('depth', 4, 16),
        # Bootstrap type (user-specified types)
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', bootstrap_types),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),  # 트리당 샘플링
        
        # 🔧 GPU 메모리 최적화 파라미터 (전체 데이터용)
        'allow_writing_files': False,
        'border_count': trial.suggest_int('border_count', 32, 128),  # feature binning 줄임 (기본 254 → 64로 메모리 절약)
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 8),  # CTR 복잡도 제한
        
        # GPU 설정
        'gpu_ram_part': 0.95,  # 전체 데이터이므로 GPU RAM 충분히 사용
        'pinned_memory_size': '4gb',
        'gpu_cat_features_storage': 'CpuPinnedMemory',  # 안정성을 위해 CPU 메모리 사용
        
        'logging_level': 'Silent',
    }
    
    # GPU에서는 colsample_bylevel(rsm)이 지원되지 않음 (pairwise 모드만 지원)
    # CPU에서만 colsample_bylevel 사용
    if task_type == 'CPU':
        params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        
    if params['bootstrap_type'] == 'Bayesian':
        del params['subsample']
    
    # Calibration method
    calibration_method = trial.suggest_categorical('calibration_method', ['none', 'temperature', 'sigmoid'])
    
    # MixUp hyperparameters (if enabled)
    # GPU 메모리 고려: mixup_ratio를 낮게 유지 (데이터 증가 제한)
    if use_mixup:
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.01, 0.3)
        mixup_ratio = trial.suggest_float('mixup_ratio', 0.1, 0.3, step=0.1)  # 0.3~0.7 → 0.1~0.3로 낮춤
        
        # Apply MixUp (샘플링된 데이터 사용)
        class_weight = (1.0, scale_pos_weight)
        X_train, y_train, sample_weight = apply_mixup_to_dataset(
            X_train_sampled, y_train_sampled,
            class_weight=class_weight,
            alpha=mixup_alpha,
            ratio=mixup_ratio,
            rng=np.random.default_rng(42)
        )
    else:
        X_train = X_train_sampled
        y_train = y_train_sampled
        sample_weight = None
    
    # Train model
    params = sanitize_logging(params)
    model = cb.CatBoostClassifier(**params)

    try:
        if sample_weight is not None:
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=(X_val_sampled, y_val_sampled),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,
                eval_set=(X_val_sampled, y_val_sampled),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        
        # Get predictions on validation set
        y_pred_val = model.predict_proba(X_val_sampled)[:, 1]
        
        # Apply calibration if needed
        if calibration_method != 'none':
            # Get predictions on calibration data (train_c)
            y_pred_cal = model.predict_proba(X_cal)[:, 1]
            
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
            score, _, _ = calculate_competition_score(y_val_sampled, y_pred_calibrated)
            
            del calibrator
        else:
            # No calibration - use validation set predictions directly
            score, _, _ = calculate_competition_score(y_val_sampled, y_pred_val)
        
    except Exception as e:
        # GPU 핸들 오류 등 발생 시: 컨텍스트 정리 후 trial 낮은 점수 반환
        print(f"❌ Trial failed with exception: {e}")
        score = 0.0
    finally:
        # Cleanup & 강제 리셋
        del model
        gc.collect()
        clear_catboost_gpu_memory()

    return score

def run_optimization(train_t_path, train_v_path, train_c_path, n_trials=100,
                     early_stopping_rounds=20, timeout=None, task_type='GPU', use_mixup=True,
                     bootstrap_types=None):
    """Run Optuna optimization using pre-processed data (GPU-handle-safe)"""
    
    # Default bootstrap types if not specified
    if bootstrap_types is None:
        bootstrap_types = ['Bernoulli', 'Bayesian', 'MVS']
    
    print("\n" + "="*70)
    print("🔍 CatBoost Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    print(f"   Bootstrap types: {', '.join(bootstrap_types)}")
    
    # Load train_t (training data, drop seq for GBDT)
    print(f"\n📦 Loading training data from {train_t_path}...")
    X_train, y_train = load_processed_data_gbdt(train_t_path)
    
    # Load train_v (validation data, seq automatically excluded)
    print(f"\n📦 Loading validation data from {train_v_path}...")
    X_val, y_val = load_processed_data_gbdt(train_v_path)
    
    # Load train_c (calibration data, seq automatically excluded)
    print(f"\n📦 Loading calibration data from {train_c_path}...")
    X_cal, y_cal = load_processed_data_gbdt(train_c_path)
    
    # scale_pos_weight
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    # Calculate balanced calibration set info
    n_cal_pos = int(y_cal.sum())
    n_cal_neg = len(y_cal) - n_cal_pos
    n_cal_balanced = min(n_cal_pos, n_cal_neg) * 2
    
    print(f"\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Task type: {task_type}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Val samples: {len(X_val):,}")
    print(f"   Cal samples (original): {len(X_cal):,} (pos: {n_cal_pos:,}, neg: {n_cal_neg:,})")
    print(f"   Cal samples (balanced): {n_cal_balanced:,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Train positive ratio: {y_train.mean():.4f}")
    print(f"   Val positive ratio: {y_val.mean():.4f}")
    print(f"   Cal positive ratio (original): {y_cal.mean():.4f}")
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
            trial, X_train, y_train, X_val, y_val, X_cal, y_cal,
            early_stopping_rounds, task_type, use_mixup, scale_pos_weight,
            bootstrap_types
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
    
    # Cleanup
    del X_train, X_val, X_cal, y_train, y_val, y_cal
    clear_gpu_memory()
    
    return study

def save_best_params_to_yaml(study, output_path='config_GBDT_optimized.yaml', 
                              original_config_path='config_GBDT.yaml',
                              bootstrap_types=None):
    """Save best parameters to YAML config"""
    
    # Add bootstrap_types suffix to output path
    if bootstrap_types:
        bootstrap_suffix = '_'.join(bootstrap_types)
        # Insert bootstrap types before file extension
        path_parts = output_path.rsplit('.', 1)
        if len(path_parts) == 2:
            output_path = f"{path_parts[0]}_{bootstrap_suffix}.{path_parts[1]}"
        else:
            output_path = f"{output_path}_{bootstrap_suffix}"
    
    print(f"\n💾 Saving best parameters to {output_path}...")
    
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    best_params = study.best_params
    
    # Core parameters (always present in trials)
    config['catboost']['n_estimators'] = best_params['iterations']
    config['catboost']['learning_rate'] = best_params['learning_rate']
    config['catboost']['max_depth'] = best_params['depth']
    config['catboost']['bootstrap_type'] = best_params['bootstrap_type']
    
    # Add optional parameters if present
    if 'colsample_bylevel' in best_params:
        config['catboost']['colsample_bylevel'] = best_params['colsample_bylevel']
    
    # Add MixUp parameters if present
    if 'mixup_alpha' in best_params:
        config['catboost']['mixup_alpha'] = best_params['mixup_alpha']
    if 'mixup_ratio' in best_params:
        config['catboost']['mixup_ratio'] = best_params['mixup_ratio']
    
    # Add calibration method
    if 'calibration_method' in best_params:
        config['catboost']['calibration_method'] = best_params['calibration_method']
        print(f"   Best calibration method: {best_params['calibration_method']}")
    
    # 기본값
    config['catboost']['task_type'] = 'GPU'
    config['catboost']['devices'] = '0'
    config['catboost']['verbose'] = False
    config['catboost']['early_stopping_rounds'] = 20
    config['catboost']['thread_count'] = -1
    config['catboost']['random_state'] = 42
    
    final_output_path = output_path.replace('.yaml', '_catboost_best.yaml')
    with open(final_output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"   ✅ Saved to {final_output_path}")

def main():
    parser = argparse.ArgumentParser(description='CatBoost Hyperparameter Optimization')
    
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
    parser.add_argument('--task-type', type=str, default='GPU', choices=['GPU', 'CPU'],
                        help='Task type for CatBoost (default: GPU)')
    parser.add_argument('--use-mixup', action='store_true', default=False,
                        help='Enable MixUp data augmentation (default: False, GPU 메모리 절약)')
    parser.add_argument('--bootstrap-types', type=str, nargs='+', 
                        choices=['Bernoulli', 'Bayesian', 'MVS'],
                        default=['Bernoulli', 'Bayesian', 'MVS'],
                        help='Bootstrap types to explore (default: all three types)')
    parser.add_argument('--output-config', type=str, default='config_optimized.yaml',
                        help='Output config file path (default: config_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_GBDT.yaml',
                        help='Original config file path (default: config_GBDT.yaml)')
    
    args = parser.parse_args()
    
    print(f"\n🔧 HPO Configuration:")
    print(f"   Train data: {args.train_t_path}")
    print(f"   Val data: {args.train_v_path}")
    print(f"   Cal data: {args.train_c_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Early stopping: {args.early_stopping_rounds}")
    print(f"   Task type: {args.task_type}")
    print(f"   Use MixUp: {args.use_mixup}")
    print(f"   Bootstrap types: {', '.join(args.bootstrap_types)}")
    print(f"   Timeout: {args.timeout if args.timeout else 'None'}")
    
    # Run optimization
    study = run_optimization(
        train_t_path=args.train_t_path,
        train_v_path=args.train_v_path,
        train_c_path=args.train_c_path,
        n_trials=args.n_trials,
        early_stopping_rounds=args.early_stopping_rounds,
        timeout=args.timeout,
        task_type=args.task_type,
        use_mixup=args.use_mixup,
        bootstrap_types=args.bootstrap_types
    )
    
    # Save results
    save_best_params_to_yaml(
        study, 
        output_path=args.output_config,
        original_config_path=args.original_config,
        bootstrap_types=args.bootstrap_types
    )
    
    # Generate final config name with bootstrap types
    if args.bootstrap_types:
        bootstrap_suffix = '_'.join(args.bootstrap_types)
        path_parts = args.output_config.rsplit('.', 1)
        if len(path_parts) == 2:
            final_config_name = f"{path_parts[0]}_{bootstrap_suffix}_catboost_best.{path_parts[1]}"
        else:
            final_config_name = f"{args.output_config}_{bootstrap_suffix}_catboost_best"
    else:
        final_config_name = args.output_config.replace('.yaml', '_catboost_best.yaml')
    
    print("\n" + "🎉"*35)
    print("OPTIMIZATION COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Best score: {study.best_value:.6f}")
    print(f"✅ Config saved to: {final_config_name}")
    print("="*70)

if __name__ == '__main__':
    main()
