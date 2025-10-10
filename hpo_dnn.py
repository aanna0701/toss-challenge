# DNN (WideDeepCTR) Hyperparameter Optimization using Optuna
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Multi-GPU support: Use all available GPUs for HPO
# Default to GPUs 0-7 (8 GPUs) if not specified
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured for DNN HPO (Multi-GPU - 8 GPUs default)")

# Standard library
import argparse
import gc
import random
import time

# Third-party libraries
import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from lightning.fabric import Fabric
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom modules
from data_loader import ClickDatasetDNN, collate_fn_dnn_train, load_processed_dnn_data
from mixup import mixup_batch_torch

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Optuna version: {optuna.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    
    # Additional clipping to prevent log(0) or log(negative) issues
    if mask_0.sum() > 0:
        pred_0 = np.clip(1 - y_pred[mask_0], eps, 1 - eps)
        ll_0 = -np.mean(np.log(pred_0))
    else:
        ll_0 = 0
    
    if mask_1.sum() > 0:
        pred_1 = np.clip(y_pred[mask_1], eps, 1 - eps)
        ll_1 = -np.mean(np.log(pred_1))
    else:
        ll_1 = 0
    
    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """Calculate competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

def clear_gpu_memory():
    """Clear GPU memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"⚠️ Error clearing GPU memory: {e}")
        gc.collect()

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x

class WideDeepCTR(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64,
                 hidden_units=None, dropout=None, cross_layers=2):
        super().__init__()
        if hidden_units is None:
            hidden_units = [512, 256, 128]
        if dropout is None:
            dropout = [0.1, 0.2, 0.3]
        
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
        ])
        cat_input_dim = emb_dim * len(cat_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        seq_out_dim = lstm_hidden * 2
        self.cross = CrossNetwork(num_features + cat_input_dim + seq_out_dim, num_layers=cross_layers)
        input_dim = num_features + cat_input_dim + seq_out_dim
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout[i % len(dropout)])]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        num_x = self.bn_num(num_x)
        cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
        cat_feat = torch.cat(cat_embs, dim=1)
        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(seqs, seq_lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        z = torch.cat([num_x, cat_feat, h], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)

def get_categorical_cardinalities(train_df, val_df, cat_cols):
    """
    Calculate categorical cardinalities from pre-encoded data
    
    Args:
        train_df: Training dataframe (already Categorify-encoded)
        val_df: Validation dataframe (already Categorify-encoded)
        cat_cols: List of categorical column names
    
    Returns:
        dict: {column_name: cardinality}
    """
    print("\n🔧 Calculating categorical cardinalities...")
    cat_cardinalities = {}
    for col in cat_cols:
        # Get max value across train/val to determine cardinality
        max_val = max(train_df[col].max(), val_df[col].max())
        # Cardinality = max_index + 1 (since 0-based indexing)
        cat_cardinalities[col] = int(max_val) + 1
        print(f"   {col}: {cat_cardinalities[col]} unique categories")
    
    return cat_cardinalities

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

def objective(trial, train_df, val_df, num_cols, cat_cols, seq_col, 
              cat_cardinalities, target_col, num_devices, use_mixup=True):
    """Optuna objective function for DNN with Fabric
    
    Note: For Optuna compatibility, we use single-process execution.
    Each trial runs on a single GPU (or uses DataParallel on multiple GPUs).
    """
    
    # For Optuna compatibility, we use devices=1 and let PyTorch handle multi-GPU
    # Alternatively, we could use DataParallel within Fabric
    fabric = Fabric(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,  # Single process for Optuna compatibility
        strategy="auto",
        precision="32-true"
    )
    
    print(f"\n{'='*70}")
    print(f"🔍 Starting Trial {trial.number}")
    print(f"{'='*70}")
    
    # Hyperparameter search space
    batch_size = 512
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Model architecture hyperparameters
    emb_dim = trial.suggest_categorical('emb_dim', [16, 32, 64])
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [32, 64])
    cross_layers = trial.suggest_int('cross_layers', 1, 3)
    
    # MixUp hyperparameters (if enabled)
    if use_mixup:
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.5, step=0.1)
        mixup_prob = trial.suggest_float('mixup_prob', 0.3, 0.7, step=0.1)
    else:
        mixup_alpha = 0.0
        mixup_prob = 0.0
    
    # MLP architecture
    n_layers = trial.suggest_int('n_layers', 2, 4)
    
    # Generate hidden units and dropout rates based on n_layers
    hidden_units = [128*2**(n_layers-(j+1)) for j in range(n_layers)]
    dropout_rates = [0.1*(j+1) for j in range(n_layers)]
    
    # Print hyperparameters
    print("📋 Hyperparameters:")
    print(f"   batch_size: {batch_size}")
    print(f"   learning_rate: {learning_rate:.6f}")
    print(f"   weight_decay: {weight_decay:.6f}")
    print(f"   emb_dim: {emb_dim}")
    print(f"   lstm_hidden: {lstm_hidden}")
    print(f"   cross_layers: {cross_layers}")
    print(f"   n_layers: {n_layers}")
    print(f"   hidden_units: {hidden_units}")
    print(f"   dropout_rates: {dropout_rates}")
    if use_mixup:
        print(f"   mixup_alpha: {mixup_alpha:.3f}")
        print(f"   mixup_prob: {mixup_prob:.3f}")
    
    # Create datasets (data already preprocessed by dataset_split_and_preprocess.py)
    train_dataset = ClickDatasetDNN(train_df, num_cols, cat_cols, seq_col, target_col, True)
    val_dataset = ClickDatasetDNN(val_df, num_cols, cat_cols, seq_col, target_col, True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4,
                              worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4,
                            worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    
    # Create model
    # cat_cardinalities는 dict → list 변환
    cat_cardinalities_list = [cat_cardinalities[c] for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities_list,
        emb_dim=emb_dim,
        lstm_hidden=lstm_hidden,
        hidden_units=hidden_units,
        dropout=dropout_rates,
        cross_layers=cross_layers
    )
    
    # Setup training
    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Setup model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    
    # Apply DataParallel if multiple GPUs are available (for Optuna compatibility)
    if num_devices > 1 and torch.cuda.device_count() > 1:
        actual_devices = min(num_devices, torch.cuda.device_count())
        print(f"   Using DataParallel with {actual_devices} GPUs")
        model = nn.DataParallel(model.module if hasattr(model, 'module') else model, 
                                device_ids=list(range(actual_devices)))
    
    # Setup dataloaders with Fabric
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)
    
    # Move pos_weight to device and create criterion
    pos_weight = pos_weight.to(fabric.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training (single epoch)
    model.train()
    train_loss = 0
    train_samples = 0
    
    # Use tqdm for training progress
    train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} Training", leave=False)
    
    for num_x, cat_x, seqs, lens, ys in train_pbar:
        optimizer.zero_grad()
        
        # Apply MixUp with probability if enabled
        if use_mixup and np.random.rand() < mixup_prob:
            # MixUp for numerical features
            num_x_mixed, ys_mixed, _ = mixup_batch_torch(num_x, ys, alpha=mixup_alpha, device=fabric.device)
            
            # For categorical and sequence features, we keep them from the first sample
            logits = model(num_x_mixed, cat_x, seqs, lens)
            loss = criterion(logits, ys_mixed)
        else:
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
        
        # Use fabric.backward instead of loss.backward()
        fabric.backward(loss)
        optimizer.step()
        
        batch_loss = loss.item()
        train_loss += batch_loss * ys.size(0)
        train_samples += ys.size(0)
        
        # Update progress bar
        train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_train_loss = train_loss / train_samples
    
    # Get predictions on validation and calibration sets
    model.eval()
    
    # Validation predictions
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Trial {trial.number} Validation", leave=False)
        
        for num_x, cat_x, seqs, lens, ys in val_pbar:
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            val_loss += loss.item() * ys.size(0)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(ys.cpu().numpy())
    
    val_loss = val_loss / len(val_targets)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    # Calculate score on validation set
    score, ap, wll = calculate_competition_score(val_targets, val_preds)
    
    # Print results
    print(f"Trial {trial.number} - "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val Score: {score:.6f} (AP: {ap:.6f}, WLL: {wll:.6f})")
    
    # Report to Optuna
    trial.report(score, 0)
    
    # Cleanup
    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
    clear_gpu_memory()
    
    return score

def run_optimization(train_t_path, train_v_path, n_trials=50,
                     timeout=None, seed=42, use_mixup=True, num_devices=None):
    """Run Optuna optimization using pre-split data with Fabric DDP"""
    print("\n" + "="*70)
    print("🔍 DNN (WideDeepCTR) Hyperparameter Optimization with Optuna + Fabric DDP")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    
    # Set seed
    seed_everything(seed)
    
    # Setup device configuration
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if num_devices is None:
            num_devices = min(available_gpus, 8)  # Default to 8 GPUs
        else:
            num_devices = min(num_devices, available_gpus)
        
        print("\n🖥️  GPU Configuration:")
        print(f"   Available GPUs: {available_gpus}")
        print(f"   Using GPUs: {num_devices}")
        for i in range(num_devices):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print("   Strategy: Fabric + DataParallel (Optuna compatible)")
    else:
        num_devices = 1
        print("\n🖥️  Device: CPU (CUDA not available)")
    
    # Load data
    print(f"\n📦 Loading training data from {train_t_path}...")
    train_df = load_processed_dnn_data(train_t_path)
    print(f"   ✅ Loaded {len(train_df):,} rows x {len(train_df.columns)} columns")
    
    print(f"\n📦 Loading validation data from {train_v_path}...")
    val_df = load_processed_dnn_data(train_v_path)
    print(f"   ✅ Loaded {len(val_df):,} rows x {len(val_df.columns)} columns")
    
    # Define features
    target_col = "clicked"
    seq_col = "seq"
    # l_feat_20, l_feat_23은 dataset_split_and_preprocess.py에서 이미 제거됨
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    
    # Categorical columns (based on analysis/results/feature_classification.json)
    cat_cols = ["gender", "age_group", "inventory_id"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    print("\n📊 Features:")
    print(f"   Numerical: {len(num_cols)}")
    print(f"   Categorical: {len(cat_cols)}")
    
    # Calculate categorical cardinalities (data is already Categorify-encoded)
    cat_cardinalities = get_categorical_cardinalities(train_df, val_df, cat_cols)
    
    print("   ✅ Data already preprocessed by dataset_split_and_preprocess.py:")
    print("      - Categorical: Categorify-encoded (0-based indices)")
    print("      - Numerical: Normalized (mean=0, std=1)")
    print("      - Missing values: Filled with 0")
    
    print("\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Train samples: {len(train_df):,}")
    print(f"   Val samples: {len(val_df):,}")
    print(f"   Train positive ratio: {train_df[target_col].mean():.4f}")
    print(f"   Val positive ratio: {val_df[target_col].mean():.4f}")
    if timeout:
        print(f"   Timeout: {timeout}s")
    else:
        print("   Timeout: None")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    print("\n🚀 Starting optimization...")
    start_time = time.time()
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, train_df, val_df, num_cols, cat_cols, seq_col, 
            cat_cardinalities, target_col, num_devices, use_mixup
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False  # Using custom tqdm progress bars instead
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
    del train_df, val_df
    clear_gpu_memory()
    
    return study

def save_best_params_to_yaml(study, output_path='config_widedeep_optimized.yaml',
                              original_config_path='config_widedeep.yaml'):
    """Save best parameters to YAML config"""
    print(f"\n💾 Saving best parameters to {output_path}...")
    
    # Load original config
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update parameters
    best_params = study.best_params
    
    # Basic training parameters
    # Note: batch_size is fixed at 512 in HPO, not tuned
    config['BATCH_SIZE'] = best_params.get('batch_size', 512)
    config['LEARNING_RATE'] = best_params['learning_rate']
    config['WEIGHT_DECAY'] = best_params['weight_decay']
    
    # Model architecture parameters (always present)
    config['MODEL']['WIDEDEEP']['EMB_DIM'] = best_params['emb_dim']
    config['MODEL']['WIDEDEEP']['LSTM_HIDDEN'] = best_params['lstm_hidden']
    
    # Extract hidden units and dropout rates
    n_layers = best_params.get('n_layers', 3)
    hidden_units = [128*2**(n_layers-(j+1)) for j in range(n_layers)]
    dropout_rates = [0.1*(j+1) for j in range(n_layers)]
    
    config['MODEL']['WIDEDEEP']['HIDDEN_UNITS'] = hidden_units
    config['MODEL']['WIDEDEEP']['DROPOUT'] = dropout_rates
    
    # Add cross layers if in config structure
    if 'CROSS_LAYERS' not in config['MODEL']['WIDEDEEP']:
        config['MODEL']['WIDEDEEP']['CROSS_LAYERS'] = best_params.get('cross_layers', 2)
    else:
        config['MODEL']['WIDEDEEP']['CROSS_LAYERS'] = best_params.get('cross_layers', 2)
    
    # Add MixUp parameters if present
    if 'mixup_alpha' in best_params:
        config['MODEL']['WIDEDEEP']['MIXUP_ALPHA'] = best_params['mixup_alpha']
    if 'mixup_prob' in best_params:
        config['MODEL']['WIDEDEEP']['MIXUP_PROB'] = best_params['mixup_prob']
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Saved to {output_path}")
    
    # Also save best params separately
    best_params_path = output_path.replace('.yaml', '_best_params.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'best_score': float(study.best_value),
            'best_params': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in best_params.items()}
        }, f, default_flow_style=False)
    
    print(f"   ✅ Best params saved to {best_params_path}")

def main():
    parser = argparse.ArgumentParser(description='DNN (WideDeepCTR) Hyperparameter Optimization')
    
    parser.add_argument('--train-t-path', type=str, default='data/proc_train_hpo',
                        help='Path to training data (default: data/proc_train_hpo)')
    parser.add_argument('--train-v-path', type=str, default='data/proc_train_v',
                        help='Path to validation data (default: data/proc_train_v)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                        help='Enable MixUp data augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                        help='Disable MixUp data augmentation')
    parser.add_argument('--output-config', type=str, default='config_optimized.yaml',
                        help='Output config file path (default: config_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_widedeep.yaml',
                        help='Original config file path (default: config_widedeep.yaml)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3,4,5,6,7"). If not specified, uses CUDA_VISIBLE_DEVICES env var or default "0,1,2,3,4,5,6,7"')
    parser.add_argument('--num-devices', type=int, default=None,
                        help='Number of GPU devices to use with DataParallel (default: auto-detect, max 8)')
    
    args = parser.parse_args()
    
    # Override CUDA_VISIBLE_DEVICES if --gpus is specified
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f"\n🖥️  Using GPUs: {args.gpus}")
    
    print("\n🔧 HPO Configuration:")
    print(f"   Train data: {args.train_t_path}")
    print(f"   Val data: {args.train_v_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Seed: {args.seed}")
    print(f"   Use MixUp: {args.use_mixup}")
    if args.timeout:
        print(f"   Timeout: {args.timeout}s")
    else:
        print("   Timeout: None")
    
    # Run optimization
    study = run_optimization(
        train_t_path=args.train_t_path,
        train_v_path=args.train_v_path,
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
        use_mixup=args.use_mixup,
        num_devices=args.num_devices
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

