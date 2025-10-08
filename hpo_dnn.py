# DNN (WideDeepCTR) Hyperparameter Optimization using Optuna
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import warnings
warnings.filterwarnings('ignore')

print("✅ Environment configured for DNN HPO")

# Core imports
import gc
import time
import numpy as np
import pandas as pd
import argparse
import yaml
import json
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optuna
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# MixUp
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

class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col, norm_stats, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        
        # Standardize numerical features using normalization_stats.json
        num_data = self.df[self.num_cols].astype(float)
        for col in self.num_cols:
            if col in norm_stats:
                mean = norm_stats[col]['mean']
                std = norm_stats[col]['std']
                # Avoid division by zero
                if std > 0:
                    num_data[col] = (num_data[col] - mean) / std
        self.num_X = num_data.fillna(0).values
        
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        else:
            return num_x, cat_x, seq

def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys

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

def load_and_prepare_data(train_path, subsample_ratio=1.0, seed=42):
    """Load and prepare data for HPO"""
    print(f"\n📦 Loading data from {train_path}...")
    start_load = time.time()
    
    # Load data
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    print(f"   ✅ Loaded data: {len(train_df):,} rows x {len(train_df.columns)} columns")
    
    # Subsample if needed
    if subsample_ratio < 1.0:
        if 'clicked' in train_df.columns:
            train_pos = train_df[train_df['clicked'] == 1]
            train_neg = train_df[train_df['clicked'] == 0]
            
            n_pos = int(len(train_pos) * subsample_ratio)
            n_neg = int(len(train_neg) * subsample_ratio)
            
            train_df = pd.concat([
                train_pos.sample(n=min(n_pos, len(train_pos)), random_state=seed),
                train_neg.sample(n=min(n_neg, len(train_neg)), random_state=seed)
            ]).reset_index(drop=True)
            print(f"   📊 Stratified subsampled to {len(train_df):,} rows (ratio={subsample_ratio})")
    
    print(f"   Time: {time.time() - start_load:.1f}s")
    return train_df

def prepare_features(train_df, cat_cols):
    """Encode categorical features"""
    print("\n🔧 Encoding categorical features...")
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        print(f"   {col}: {len(le.classes_)} unique categories")
    
    return train_df, encoders

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

def objective(trial, train_df, val_df, num_cols, cat_cols, seq_col, norm_stats, 
              cat_encoders, target_col, device, use_mixup=True):
    """Optuna objective function for DNN"""
    
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
    
    # Create datasets
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, norm_stats, target_col, True)
    val_dataset = ClickDataset(val_df, num_cols, cat_cols, seq_col, norm_stats, target_col, True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, pin_memory=True, num_workers=24,
                              worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_train, pin_memory=True, num_workers=24,
                            worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    
    # Create model
    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
        lstm_hidden=lstm_hidden,
        hidden_units=hidden_units,
        dropout=dropout_rates,
        cross_layers=cross_layers
    ).to(device)
    
    # Setup training
    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training (single epoch)
    model.train()
    train_loss = 0
    train_samples = 0
    
    # Use tqdm for training progress
    train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} Training", leave=False)
    for num_x, cat_x, seqs, lens, ys in train_pbar:
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        seqs = seqs.to(device)
        lens = lens.to(device)
        ys = ys.to(device)
        
        optimizer.zero_grad()
        
        # Apply MixUp with probability if enabled
        if use_mixup and np.random.rand() < mixup_prob:
            # MixUp for numerical features
            num_x_mixed, ys_mixed, _ = mixup_batch_torch(num_x, ys, alpha=mixup_alpha, device=device)
            
            # For categorical and sequence features, we keep them from the first sample
            logits = model(num_x_mixed, cat_x, seqs, lens)
            loss = criterion(logits, ys_mixed)
        else:
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
        
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        train_loss += batch_loss * ys.size(0)
        train_samples += ys.size(0)
        
        # Update progress bar
        train_pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_train_loss = train_loss / train_samples
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Trial {trial.number} Validation", leave=False)
        for num_x, cat_x, seqs, lens, ys in val_pbar:
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            ys = ys.to(device)
            
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            val_loss += loss.item() * ys.size(0)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(ys.cpu().numpy())
    
    val_loss = val_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate competition score
    score, ap, wll = calculate_competition_score(all_targets, all_preds)
    
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

def run_optimization(train_path, n_trials=50, val_ratio=0.2, subsample_ratio=1.0,
                     timeout=None, seed=42, use_mixup=True):
    """Run Optuna optimization"""
    print("\n" + "="*70)
    print("🔍 DNN (WideDeepCTR) Hyperparameter Optimization with Optuna")
    print("="*70)
    print(f"   MixUp enabled: {use_mixup}")
    
    # Set seed
    seed_everything(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    train_df = load_and_prepare_data(train_path, subsample_ratio, seed)
    
    # Define features
    target_col = "clicked"
    seq_col = "seq"
    FEATURE_EXCLUDE = {target_col, seq_col, "ID", "l_feat_20", "l_feat_23"}
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    
    cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    print("\n📊 Features:")
    print(f"   Numerical: {len(num_cols)}")
    print(f"   Categorical: {len(cat_cols)}")
    
    # Encode categorical features
    train_df, cat_encoders = prepare_features(train_df, cat_cols)
    
    # Load normalization stats
    norm_stats_path = 'analysis/results/normalization_stats.json'
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r', encoding='utf-8') as f:
            norm_stats_data = json.load(f)
            norm_stats = norm_stats_data['statistics']
        print(f"   ✅ Loaded normalization stats from {norm_stats_path}")
    else:
        print(f"   ⚠️ Normalization stats not found at {norm_stats_path}, using raw features")
        norm_stats = {}
    
    # Split data
    print(f"\n📊 Splitting data (val_ratio={val_ratio})...")
    train_split, val_split = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df[target_col]
    )
    
    print("\n📊 Optimization settings:")
    print(f"   Trials: {n_trials}")
    print(f"   Total samples: {len(train_df):,}")
    print(f"   Train samples: {len(train_split):,}")
    print(f"   Val samples: {len(val_split):,}")
    print(f"   Train positive ratio: {train_split[target_col].mean():.4f}")
    print(f"   Val positive ratio: {val_split[target_col].mean():.4f}")
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
            trial, train_split, val_split, num_cols, cat_cols, seq_col, 
            norm_stats, cat_encoders, target_col, device, use_mixup
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
    del train_df, train_split, val_split
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
    config['BATCH_SIZE'] = best_params['batch_size']
    config['LEARNING_RATE'] = best_params['learning_rate']
    config['WEIGHT_DECAY'] = best_params['weight_decay']
    
    # Model architecture parameters
    config['MODEL']['WIDEDEEP']['EMB_DIM'] = best_params['emb_dim']
    config['MODEL']['WIDEDEEP']['LSTM_HIDDEN'] = best_params['lstm_hidden']
    
    # Extract hidden units and dropout rates
    n_layers = best_params.get('n_layers', 3)
    hidden_units = [best_params[f'hidden_size_{i}'] for i in range(n_layers)]
    dropout_rates = [best_params[f'dropout_{i}'] for i in range(n_layers)]
    
    config['MODEL']['WIDEDEEP']['HIDDEN_UNITS'] = hidden_units
    config['MODEL']['WIDEDEEP']['DROPOUT'] = dropout_rates
    
    # Add cross layers if in config structure
    if 'CROSS_LAYERS' not in config['MODEL']['WIDEDEEP']:
        config['MODEL']['WIDEDEEP']['CROSS_LAYERS'] = best_params.get('cross_layers', 2)
    else:
        config['MODEL']['WIDEDEEP']['CROSS_LAYERS'] = best_params.get('cross_layers', 2)
    
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
    
    parser.add_argument('--train-path', type=str, default='data/train.parquet',
                        help='Path to training data parquet file (default: data/train.parquet)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--subsample-ratio', type=float, default=1.0,
                        help='Ratio of data to use (default: 1.0 = use all)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                        help='Enable MixUp data augmentation (default: True)')
    parser.add_argument('--no-mixup', dest='use_mixup', action='store_false',
                        help='Disable MixUp data augmentation')
    parser.add_argument('--output-config', type=str, default='config_widedeep_optimized.yaml',
                        help='Output config file path (default: config_widedeep_optimized.yaml)')
    parser.add_argument('--original-config', type=str, default='config_widedeep.yaml',
                        help='Original config file path (default: config_widedeep.yaml)')
    
    args = parser.parse_args()
    
    print("\n🔧 HPO Configuration:")
    print(f"   Train path: {args.train_path}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Validation ratio: {args.val_ratio}")
    print(f"   Subsample ratio: {args.subsample_ratio}")
    print(f"   Seed: {args.seed}")
    if args.timeout:
        print(f"   Timeout: {args.timeout}s")
    else:
        print("   Timeout: None")
    
    # Run optimization
    study = run_optimization(
        train_path=args.train_path,
        n_trials=args.n_trials,
        val_ratio=args.val_ratio,
        subsample_ratio=args.subsample_ratio,
        timeout=args.timeout,
        seed=args.seed,
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

