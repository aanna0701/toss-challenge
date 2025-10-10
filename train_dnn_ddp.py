# Standard library
import argparse
import csv
import json
import os
import pickle
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from lightning.fabric import Fabric
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom modules
from data_loader import ClickDatasetDNN, collate_fn_dnn_train, load_processed_dnn_data
from mixup import mixup_batch_torch
from utils import calculate_competition_score, seed_everything

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
        print(f"      Optimal temperature: {self.temperature:.4f}")
    
    def predict_proba(self, logits):
        """Apply temperature scaling"""
        scaled_logits = logits / self.temperature
        probs = 1 / (1 + np.exp(-scaled_logits))
        return probs

# Default configuration
DEFAULT_CFG = {
    'BATCH_SIZE': 512,
    'EPOCHS': 3,
    'LEARNING_RATE': 0.0011213785126861335,
    'WEIGHT_DECAY': 1.812767196814977e-06,
    'SEED': 42,
    'NUM_DEVICES': 4,  # Use 4 GPUs out of 7 available
    'STRATEGY': 'ddp',
    'TRAIN_RATIO': 0.8,
    'VAL_RATIO': 0.1,
    'CAL_RATIO': 0.1,
    'LOG_INTERVAL': 100,
    'USE_MIXUP': True,
    'MIXUP_ALPHA': 0.4,
    'MIXUP_PROB': 0.3,
    'CALIBRATION_METHOD': 'sigmoid',  # Calibration method: 'none', 'isotonic', 'sigmoid', or 'temperature'
    # Model architecture
    'MODEL': {
        'EMB_DIM': 32,
        'LSTM_HIDDEN': 32,
        'HIDDEN_UNITS': [512, 256],
        'DROPOUT': [0.1, 0.2],
        'CROSS_LAYERS': 1
    }
}

def load_config_from_yaml(yaml_path):
    """Load hyperparameters from YAML config file (HPO results)"""
    if not os.path.exists(yaml_path):
        print(f"⚠️  Config file not found: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from: {yaml_path}")
    return config

def merge_configs(default_cfg, yaml_cfg):
    """Merge YAML config with default config"""
    cfg = default_cfg.copy()
    
    # Update basic training parameters
    if 'BATCH_SIZE' in yaml_cfg:
        cfg['BATCH_SIZE'] = yaml_cfg['BATCH_SIZE']
    if 'LEARNING_RATE' in yaml_cfg:
        cfg['LEARNING_RATE'] = yaml_cfg['LEARNING_RATE']
    if 'WEIGHT_DECAY' in yaml_cfg:
        cfg['WEIGHT_DECAY'] = yaml_cfg['WEIGHT_DECAY']
    if 'EPOCHS' in yaml_cfg:
        cfg['EPOCHS'] = yaml_cfg['EPOCHS']
    
    # Update MixUp parameters
    if 'USE_MIXUP' in yaml_cfg:
        cfg['USE_MIXUP'] = yaml_cfg['USE_MIXUP']
    if 'MIXUP_ALPHA' in yaml_cfg:
        cfg['MIXUP_ALPHA'] = yaml_cfg['MIXUP_ALPHA']
    if 'MIXUP_PROB' in yaml_cfg:
        cfg['MIXUP_PROB'] = yaml_cfg['MIXUP_PROB']
    
    # Update calibration method
    if 'CALIBRATION_METHOD' in yaml_cfg:
        cfg['CALIBRATION_METHOD'] = yaml_cfg['CALIBRATION_METHOD']
    
    # Update model architecture parameters
    if 'MODEL' in yaml_cfg:
        model_cfg = yaml_cfg['MODEL']
        if 'WIDEDEEP' in model_cfg:
            model_cfg = model_cfg['WIDEDEEP']
        
        if 'EMB_DIM' in model_cfg:
            cfg['MODEL']['EMB_DIM'] = model_cfg['EMB_DIM']
        if 'LSTM_HIDDEN' in model_cfg:
            cfg['MODEL']['LSTM_HIDDEN'] = model_cfg['LSTM_HIDDEN']
        if 'HIDDEN_UNITS' in model_cfg:
            cfg['MODEL']['HIDDEN_UNITS'] = model_cfg['HIDDEN_UNITS']
        if 'DROPOUT' in model_cfg:
            cfg['MODEL']['DROPOUT'] = model_cfg['DROPOUT']
        if 'CROSS_LAYERS' in model_cfg:
            cfg['MODEL']['CROSS_LAYERS'] = model_cfg['CROSS_LAYERS']
    
    # Handle HPO best_params format (if present)
    if 'best_params' in yaml_cfg:
        params = yaml_cfg['best_params']
        if 'learning_rate' in params:
            cfg['LEARNING_RATE'] = params['learning_rate']
        if 'weight_decay' in params:
            cfg['WEIGHT_DECAY'] = params['weight_decay']
        if 'emb_dim' in params:
            cfg['MODEL']['EMB_DIM'] = params['emb_dim']
        if 'lstm_hidden' in params:
            cfg['MODEL']['LSTM_HIDDEN'] = params['lstm_hidden']
        if 'cross_layers' in params:
            cfg['MODEL']['CROSS_LAYERS'] = params['cross_layers']
        if 'mixup_alpha' in params:
            cfg['MIXUP_ALPHA'] = params['mixup_alpha']
        if 'mixup_prob' in params:
            cfg['MIXUP_PROB'] = params['mixup_prob']
        
        # Reconstruct hidden_units and dropout from n_layers
        if 'n_layers' in params:
            n_layers = int(params['n_layers'])
            cfg['MODEL']['HIDDEN_UNITS'] = [128*2**(n_layers-(j+1)) for j in range(n_layers)]
            cfg['MODEL']['DROPOUT'] = [0.1*(j+1) for j in range(n_layers)]
    
    return cfg

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DNN (WideDeepCTR) with DDP')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g., from HPO results)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable MixUp augmentation')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "1,2,3,4"). If not specified, uses CUDA_VISIBLE_DEVICES env var or default "1,2,3,4,5,6,7"')
    parser.add_argument('--num-devices', type=int, default=None,
                        help='Number of GPU devices to use for DDP (overrides config, default: 4)')
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set CUDA_VISIBLE_DEVICES (must be done before any CUDA operations)
if args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f"\n🖥️  Using GPUs: {args.gpus}")
else:
    # Default to 1-7 if not specified
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '1,2,3,4,5,6,7')

# Load and merge configurations
CFG = DEFAULT_CFG.copy()
if args.config:
    yaml_cfg = load_config_from_yaml(args.config)
    CFG = merge_configs(CFG, yaml_cfg)
    print(f"\n📋 Using config from: {args.config}")

# Override with command line arguments
if args.epochs is not None:
    CFG['EPOCHS'] = args.epochs
if args.batch_size is not None:
    CFG['BATCH_SIZE'] = args.batch_size
if args.learning_rate is not None:
    CFG['LEARNING_RATE'] = args.learning_rate
if args.no_mixup:
    CFG['USE_MIXUP'] = False
if args.num_devices is not None:
    CFG['NUM_DEVICES'] = args.num_devices

seed_everything(CFG['SEED'])

# Print configuration
if 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0':
    print("\n" + "="*70)
    print("🔧 Training Configuration:")
    print("="*70)
    print(f"   Batch Size: {CFG['BATCH_SIZE']}")
    print(f"   Epochs: {CFG['EPOCHS']}")
    print(f"   Learning Rate: {CFG['LEARNING_RATE']}")
    print(f"   Weight Decay: {CFG['WEIGHT_DECAY']}")
    print(f"   Seed: {CFG['SEED']}")
    print(f"   NUM Devices: {CFG['NUM_DEVICES']}")
    print(f"   Strategy: {CFG['STRATEGY']}")
    print(f"   MixUp: {CFG['USE_MIXUP']}")
    if CFG['USE_MIXUP']:
        print(f"   MixUp Alpha: {CFG['MIXUP_ALPHA']}")
        print(f"   MixUp Prob: {CFG['MIXUP_PROB']}")
    print(f"\n🖥️  GPU Configuration:")
    import torch
    if torch.cuda.is_available():
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        for i in range(min(torch.cuda.device_count(), CFG['NUM_DEVICES'])):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   WARNING: CUDA not available!")
    print(f"\n📐 Model Architecture:")
    print(f"   Embedding Dim: {CFG['MODEL']['EMB_DIM']}")
    print(f"   LSTM Hidden: {CFG['MODEL']['LSTM_HIDDEN']}")
    print(f"   Cross Layers: {CFG['MODEL']['CROSS_LAYERS']}")
    print(f"   Hidden Units: {CFG['MODEL']['HIDDEN_UNITS']}")
    print(f"   Dropout: {CFG['MODEL']['DROPOUT']}")
    print("="*70 + "\n")

# Initialize Fabric
fabric = Fabric(
    accelerator="cuda",
    devices=CFG['NUM_DEVICES'],
    strategy=CFG['STRATEGY'],
    precision="32-true"
)
fabric.launch()

fabric.print(f"Fabric initialized - Global Rank: {fabric.global_rank}, Local Rank: {fabric.local_rank}, World Size: {fabric.world_size}")

def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))

# ============================================================================
# 모델 아키텍처
# ============================================================================

class CrossNetwork(nn.Module):
    """Cross Network for WideDeepCTR model"""
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
    """Wide & Deep CTR 모델"""
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

# ============================================================================
# Data Loading
# ============================================================================

fabric.print("데이터 로드 시작 (pre-processed 데이터 사용)")

# Load pre-processed data from dataset_split_and_preprocess.py
# These are already processed (continuous standardized, seq 결측치 처리됨)
train_t_df = load_processed_dnn_data("data/proc_train_t")
train_v_df = load_processed_dnn_data("data/proc_train_v")
train_c_df = load_processed_dnn_data("data/proc_train_c")

fabric.print(f"Train_t shape: {train_t_df.shape}")
fabric.print(f"Train_v shape: {train_v_df.shape}")
fabric.print(f"Train_c shape: {train_c_df.shape}")
fabric.print("데이터 로드 완료 (continuous standardized, seq 결측치 처리됨)")

target_col = "clicked"
seq_col = "seq"
# l_feat_20, l_feat_23은 dataset_split_and_preprocess.py에서 이미 제거됨
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train_t_df.columns if c not in FEATURE_EXCLUDE]

# Categorical columns (based on analysis/results/feature_classification.json)
cat_cols = ["gender", "age_group", "inventory_id"]
num_cols = [c for c in feature_cols if c not in cat_cols]
fabric.print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")

# Categorical features are already encoded by NVTabular Categorify (0-based indices)
# Calculate cardinalities for each categorical column
cat_cardinalities = {}
for col in cat_cols:
    # Get max value across all splits to determine cardinality
    max_val = max(
        train_t_df[col].max(),
        train_v_df[col].max(),
        train_c_df[col].max()
    )
    # Cardinality = max_index + 1 (since 0-based indexing)
    cat_cardinalities[col] = int(max_val) + 1

fabric.print("범주형 피처 이미 인코딩됨 (Categorify by NVTabular)")
fabric.print(f"Categorical cardinalities: {cat_cardinalities}")

# ============================================================================
# Split train_c for Calibration
# ============================================================================
if fabric.global_rank == 0:
    print("\n" + "="*70)
    print("🔀 Splitting train_c for Calibration")
    print("="*70)

# Split train_c into calibration set and additional training set
train_c_targets = train_c_df[target_col].values
pos_idx = np.where(train_c_targets == 1)[0]
neg_idx = np.where(train_c_targets == 0)[0]

if fabric.global_rank == 0:
    print(f"   train_c total: {len(train_c_df):,} samples")
    print(f"   Positive: {len(pos_idx):,}, Negative: {len(neg_idx):,}")

# Sample half of positive samples for calibration
np.random.seed(42)
n_pos_cal = len(pos_idx) // 2
pos_cal_idx = np.random.choice(pos_idx, size=n_pos_cal, replace=False)
pos_train_idx = np.setdiff1d(pos_idx, pos_cal_idx)

# Sample same number of negative samples for calibration
neg_cal_idx = np.random.choice(neg_idx, size=n_pos_cal, replace=False)
neg_train_idx = np.setdiff1d(neg_idx, neg_cal_idx)

cal_idx = np.concatenate([pos_cal_idx, neg_cal_idx])
train_c_remaining_idx = np.concatenate([pos_train_idx, neg_train_idx])

# Create calibration dataframe
cal_df = train_c_df.iloc[cal_idx].reset_index(drop=True)

# Create remaining train_c for training
train_c_remaining_df = train_c_df.iloc[train_c_remaining_idx].reset_index(drop=True)

if fabric.global_rank == 0:
    print(f"\n   → Calibration set: {len(cal_df):,} samples")
    print(f"      Positive: {len(pos_cal_idx):,}, Negative: {len(neg_cal_idx):,}")
    print(f"      Positive ratio: {cal_df[target_col].mean():.6f}")
    
    print(f"\n   → Remaining train_c: {len(train_c_remaining_df):,} samples")
    print(f"      Positive: {len(pos_train_idx):,}, Negative: {len(neg_train_idx):,}")
    print(f"      Positive ratio: {train_c_remaining_df[target_col].mean():.6f}")

# ============================================================================
# Combine train_t + train_v + remaining train_c for full training
# ============================================================================
if fabric.global_rank == 0:
    print("\n" + "="*70)
    print("🔗 Combining Training Data")
    print("="*70)

train_df = pd.concat([train_t_df, train_v_df, train_c_remaining_df], ignore_index=True)
# Use calibration set as validation for early stopping
val_df = cal_df

if fabric.global_rank == 0:
    total_samples = len(train_t_df) + len(train_v_df) + len(train_c_df)
    print(f"   Combined training set: {len(train_df):,} samples")
    print(f"      From train_t: {len(train_t_df):,}")
    print(f"      From train_v: {len(train_v_df):,}")
    print(f"      From train_c: {len(train_c_remaining_df):,}")
    print(f"   Combined positive ratio: {train_df[target_col].mean():.6f}")
    print(f"\n   Using calibration set ({len(cal_df):,} samples) as validation for early stopping")
    print(f"   Cal positive ratio: {cal_df[target_col].mean():.6f}")

# Normalization statistics not needed - data is already standardized
# Pass empty dict to ClickDatasetDNN (it will skip standardization)
norm_stats = {}
fabric.print("⚠️  Normalization skipped - data is already standardized by dataset_split_and_preprocess.py")

def evaluate_model(fabric, model, val_loader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for num_x, cat_x, seqs, lens, ys in val_loader:
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            total_loss += loss.item() * ys.size(0)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(ys.cpu().numpy())
    
    val_loss = total_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)
    
    # Calculate competition metrics
    score, ap, wll = calculate_competition_score(all_targets, all_preds)
    
    return val_loss, score, ap, wll, all_logits, all_preds, all_targets

def train_model(fabric, train_df, val_df, cal_df, num_cols, cat_cols, seq_col, target_col, 
                config, output_dir, cat_cardinalities):
    """Train model with validation
    
    Args:
        config: Dictionary containing all training configuration including:
            - BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
            - USE_MIXUP, MIXUP_ALPHA, MIXUP_PROB
            - MODEL: {EMB_DIM, LSTM_HIDDEN, HIDDEN_UNITS, DROPOUT, CROSS_LAYERS}
            - LOG_INTERVAL
        cat_cardinalities: dict of categorical feature cardinalities
    """
    batch_size = config['BATCH_SIZE']
    epochs = config['EPOCHS']
    lr = config['LEARNING_RATE']
    weight_decay = config['WEIGHT_DECAY']
    log_interval = config['LOG_INTERVAL']
    use_mixup = config['USE_MIXUP']
    mixup_alpha = config['MIXUP_ALPHA']
    mixup_prob = config['MIXUP_PROB']
    
    # Create datasets and loaders
    train_dataset = ClickDatasetDNN(train_df, num_cols, cat_cols, seq_col, target_col, True)
    val_dataset = ClickDatasetDNN(val_df, num_cols, cat_cols, seq_col, target_col, True)
    cal_dataset = ClickDatasetDNN(cal_df, num_cols, cat_cols, seq_col, target_col, True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4,
                              worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4,
                            worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4,
                            worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=8)
    
    # Create model with architecture from config
    # cat_cardinalities는 dict → list 변환
    cat_cardinalities_list = [cat_cardinalities[c] for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities_list,
        emb_dim=config['MODEL']['EMB_DIM'],
        lstm_hidden=config['MODEL']['LSTM_HIDDEN'],
        hidden_units=config['MODEL']['HIDDEN_UNITS'],
        dropout=config['MODEL']['DROPOUT'],
        cross_layers=config['MODEL']['CROSS_LAYERS']
    )
    
    # Setup training
    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    
    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader_device = fabric.setup_dataloaders(val_loader)
    cal_loader_device = fabric.setup_dataloaders(cal_loader)
    
    # Move pos_weight to device
    pos_weight = pos_weight.to(fabric.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create timestamped subdirectory (only on rank 0)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(output_dir, timestamp)
    
    if fabric.global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = f'{save_dir}/training_log.csv'
        
        # Initialize CSV file with headers
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'train_loss', 'val_loss', 'val_score', 'val_ap', 'val_wll'])
        
        print(f"학습 시작 (로그 저장: {log_file})")
        print(f"모델 저장 디렉토리: {save_dir}")
        print(f"Log interval: {log_interval} steps")
        if use_mixup:
            print(f"MixUp enabled: alpha={mixup_alpha}, prob={mixup_prob}")
        else:
            print("MixUp disabled")
    else:
        log_file = None
    
    global_step = 0
    best_score = 0.0
    
    fabric.print("학습 시작 (DDP Mode)")
    
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        step_losses = []
        total_samples = 0
        
        # Use tqdm only on rank 0
        if fabric.global_rank == 0:
            pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}]")
        else:
            pbar = train_loader
            
        for num_x, cat_x, seqs, lens, ys in pbar:
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
            scheduler.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * ys.size(0)
            step_losses.append(batch_loss)
            total_samples += ys.size(0)
            global_step += 1
            
            # Update progress bar (only on rank 0)
            if fabric.global_rank == 0:
                avg_loss = np.mean(step_losses[-10:]) if step_losses else 0
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
            
            # Log every N steps
            if global_step % log_interval == 0 and fabric.global_rank == 0:
                train_loss = np.mean(step_losses[-log_interval:])
                
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, 'a', newline='', buffering=1) as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp_str, epoch, global_step, train_loss, 0.0, 0.0, 0.0, 0.0])
                    f.flush()
        
        # Calculate average loss across all processes
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        avg_loss_tensor = torch.tensor([avg_loss], device=fabric.device)
        gathered_loss = fabric.all_reduce(avg_loss_tensor, reduce_op="mean")
        
        fabric.print(f"\n[Epoch {epoch}] Average Train Loss: {gathered_loss.item():.4f}")
        
        # Validation at the end of each epoch
        val_loss, val_score, val_ap, val_wll, _, _, _ = evaluate_model(fabric, model, val_loader_device, criterion)
        
        # Gather validation metrics from all ranks
        val_loss_tensor = torch.tensor([val_loss], device=fabric.device)
        val_score_tensor = torch.tensor([val_score], device=fabric.device)
        val_ap_tensor = torch.tensor([val_ap], device=fabric.device)
        val_wll_tensor = torch.tensor([val_wll], device=fabric.device)
        
        gathered_val_loss = fabric.all_reduce(val_loss_tensor, reduce_op="mean")
        gathered_val_score = fabric.all_reduce(val_score_tensor, reduce_op="mean")
        gathered_val_ap = fabric.all_reduce(val_ap_tensor, reduce_op="mean")
        gathered_val_wll = fabric.all_reduce(val_wll_tensor, reduce_op="mean")
        
        # Print metrics and save best model (only rank 0)
        if fabric.global_rank == 0:
            print(f"[Epoch {epoch}] Val Loss: {gathered_val_loss.item():.4f} | Val Score: {gathered_val_score.item():.6f} | Val AP: {gathered_val_ap.item():.6f} | Val WLL: {gathered_val_wll.item():.6f}")
            
            # Save best model based on gathered score
            if gathered_val_score.item() > best_score:
                best_score = gathered_val_score.item()
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'best_score': best_score
                }, f'{save_dir}/best_model_temp.pt')
                print(f"✅ Best model checkpoint saved (Score: {best_score:.6f})")
    
    fabric.print("학습 완료")
    
    # ========================================================================
    # Train Calibrator (only rank 0)
    # ========================================================================
    calibrator = None
    calibration_method = config.get('CALIBRATION_METHOD', 'none')
    
    if fabric.global_rank == 0 and calibration_method != 'none':
        print("\n" + "="*70)
        print("🎯 Training Calibrator")
        print("="*70)
        print(f"   Method: {calibration_method}")
        
        # Load best model for calibration
        checkpoint = torch.load(f'{save_dir}/best_model_temp.pt', map_location=fabric.device)
        model_for_cal = model.module if hasattr(model, 'module') else model
        model_for_cal.load_state_dict(checkpoint['model_state_dict'])
        model_for_cal.eval()
        
        # Collect predictions on calibration set
        print(f"   Collecting predictions on calibration set...")
        cal_logits = []
        cal_preds = []
        cal_targets = []
        
        with torch.no_grad():
            for num_x, cat_x, seqs, lens, ys in cal_loader:
                logits = model_for_cal(num_x, cat_x, seqs, lens)
                preds = torch.sigmoid(logits)
                
                cal_logits.append(logits.cpu().numpy())
                cal_preds.append(preds.cpu().numpy())
                cal_targets.append(ys.cpu().numpy())
        
        cal_logits = np.concatenate(cal_logits)
        cal_preds = np.concatenate(cal_preds)
        cal_targets = np.concatenate(cal_targets)
        
        # Train calibrator
        print(f"   Fitting {calibration_method} calibrator...")
        
        if calibration_method == 'temperature':
            calibrator = TemperatureScaling()
            calibrator.fit(cal_logits, cal_targets)
            
            calibrated_preds = calibrator.predict_proba(cal_logits)
            
        elif calibration_method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(cal_preds, cal_targets)
            
            calibrated_preds = calibrator.predict(cal_preds)
            
        elif calibration_method == 'sigmoid':
            calibrator = LogisticRegression()
            calibrator.fit(cal_preds.reshape(-1, 1), cal_targets)
            
            calibrated_preds = calibrator.predict_proba(cal_preds.reshape(-1, 1))[:, 1]
        
        else:
            print(f"   ⚠️  Unknown calibration method: {calibration_method}")
            calibrated_preds = cal_preds
        
        # Evaluate calibrated predictions
        if calibrator is not None:
            cal_score, cal_ap, cal_wll = calculate_competition_score(cal_targets, calibrated_preds)
            raw_score, raw_ap, raw_wll = calculate_competition_score(cal_targets, cal_preds)
            
            print(f"\n   📊 Calibrated Results:")
            print(f"      Score: {cal_score:.6f} (raw: {raw_score:.6f}, Δ: {cal_score - raw_score:+.6f})")
            print(f"      AP: {cal_ap:.6f} (raw: {raw_ap:.6f}, Δ: {cal_ap - raw_ap:+.6f})")
            print(f"      WLL: {cal_wll:.6f} (raw: {raw_wll:.6f}, Δ: {cal_wll - raw_wll:+.6f})")
        
        print("="*70)
    
    # Save model and metadata (only rank 0)
    if fabric.global_rank == 0:
        print(f"\n학습 완료! Best validation score: {best_score:.6f}")
        
        # Save final model with score in filename
        model_filename = f"dnn_{best_score:.6f}.pt"
        final_model_path = os.path.join(save_dir, model_filename)
        os.rename(f'{save_dir}/best_model_temp.pt', final_model_path)
        print(f"✅ Model saved to {final_model_path}")
        
        # Save cat_cardinalities (dict)
        cardinalities_path = os.path.join(save_dir, 'cat_cardinalities.json')
        with open(cardinalities_path, 'w') as f:
            json.dump(cat_cardinalities, f, indent=2)
        print(f"✅ Categorical cardinalities saved to {cardinalities_path}")
        print(f"   {cat_cardinalities}")
        
        # Save calibrator if exists
        if calibrator is not None:
            calibrator_path = os.path.join(save_dir, 'calibrator.pkl')
            with open(calibrator_path, 'wb') as f:
                pickle.dump(calibrator, f)
            print(f"✅ Calibrator saved to {calibrator_path}")
        
        # Save metadata (including model architecture config)
        metadata = {
            'model_name': 'dnn',
            'val_score': float(best_score),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'cal_samples': len(cal_df),
            'use_mixup': use_mixup,
            'mixup_alpha': mixup_alpha if use_mixup else None,
            'mixup_prob': mixup_prob if use_mixup else None,
            'calibration_method': calibration_method,
            'timestamp': timestamp,
            'num_features': len(num_cols),
            'cat_features': cat_cols,
            'cat_cardinalities': cat_cardinalities,  # dict 형태로 저장
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'model_architecture': {
                'emb_dim': config['MODEL']['EMB_DIM'],
                'lstm_hidden': config['MODEL']['LSTM_HIDDEN'],
                'hidden_units': config['MODEL']['HIDDEN_UNITS'],
                'dropout': config['MODEL']['DROPOUT'],
                'cross_layers': config['MODEL']['CROSS_LAYERS']
            }
        }
        
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to {metadata_path}")
        
        return model, best_score, save_dir
    else:
        return model, 0.0, None

# Train model
output_dir = 'result_dnn_ddp'
os.makedirs(output_dir, exist_ok=True)

fabric.print("모델 학습 실행")
model, best_score, save_dir = train_model(
    fabric=fabric,
    train_df=train_df,
    val_df=val_df,
    cal_df=cal_df,
    num_cols=num_cols,
    cat_cols=cat_cols,
    seq_col=seq_col,
    target_col=target_col,
    config=CFG,
    output_dir=output_dir,
    cat_cardinalities=cat_cardinalities
)

# Final summary (only on rank 0)
if fabric.global_rank == 0 and save_dir:
    print("\n" + "🎉"*35)
    print("TRAINING COMPLETE!")
    print("🎉"*35)
    print(f"\n✅ Best validation score: {best_score:.6f}")
    print(f"✅ Calibration method: {CFG['CALIBRATION_METHOD']}")
    print(f"✅ Models saved to: {save_dir}")
    print(f"✅ Training data: train_t + train_v + train_c (remaining)")
    print(f"✅ Calibration data: train_c (balanced subset)")
    print(f"✅ Training log: {save_dir}/training_log.csv")
    print(f"\n📐 Model Architecture Used:")
    print(f"   Embedding Dim: {CFG['MODEL']['EMB_DIM']}")
    print(f"   LSTM Hidden: {CFG['MODEL']['LSTM_HIDDEN']}")
    print(f"   Cross Layers: {CFG['MODEL']['CROSS_LAYERS']}")
    print(f"   Hidden Units: {CFG['MODEL']['HIDDEN_UNITS']}")
    print(f"   Dropout: {CFG['MODEL']['DROPOUT']}")
    print(f"\n⚙️  Training Hyperparameters:")
    print(f"   Learning Rate: {CFG['LEARNING_RATE']}")
    print(f"   Weight Decay: {CFG['WEIGHT_DECAY']}")
    print(f"   Batch Size: {CFG['BATCH_SIZE']}")
    print(f"   Epochs: {CFG['EPOCHS']}")
    if CFG['USE_MIXUP']:
        print(f"   MixUp Alpha: {CFG['MIXUP_ALPHA']}")
        print(f"   MixUp Prob: {CFG['MIXUP_PROB']}")
    print("="*70)

# Wait for all processes to finish
fabric.print(f"Rank {fabric.global_rank}: 완료")

