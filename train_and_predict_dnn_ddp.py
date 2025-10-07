import pandas as pd
import numpy as np
import os, random
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric

# Set CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

CFG = {
    'BATCH_SIZE': 1024,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'NUM_DEVICES': 4,  # Use 4 GPUs out of 7 available (will use GPUs 1,2,3,4)
    'STRATEGY': 'ddp',
    'VAL_RATIO': 0.1,
    'LOG_INTERVAL': 100  # Log every 100 steps
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# Initialize Fabric
fabric = Fabric(
    accelerator="cuda",
    devices=CFG['NUM_DEVICES'],
    strategy=CFG['STRATEGY'],
    precision="32-true"
)
fabric.launch()

fabric.print(f"Fabric initialized - Global Rank: {fabric.global_rank}, Local Rank: {fabric.local_rank}, World Size: {fabric.world_size}")

# Metric functions (from GBDT script)
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

fabric.print("데이터 로드 시작")
train = pd.read_parquet("data/train.parquet", engine="pyarrow")
test = pd.read_parquet("data/test.parquet", engine="pyarrow")
fabric.print(f"Train shape: {train.shape}")
fabric.print(f"Test shape: {test.shape}")
fabric.print("데이터 로드 완료")

target_col = "clicked"
seq_col = "seq"
FEATURE_EXCLUDE = {target_col, seq_col, "ID", "l_feat_20", "l_feat_23"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
fabric.print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")

def encode_categoricals(train_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col]  = le.transform(test_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        fabric.print(f"{col} unique categories: {len(le.classes_)}")
    return train_df, test_df, encoders

train, test, cat_encoders = encode_categoricals(train, test, cat_cols)

# Split train into train and validation
if fabric.global_rank == 0:
    print(f"\n데이터 분할 시작 (Validation ratio: {CFG['VAL_RATIO']:.1%})")
train_df, val_df = train_test_split(
    train, 
    test_size=CFG['VAL_RATIO'], 
    random_state=CFG['SEED'],
    stratify=train[target_col]
)
if fabric.global_rank == 0:
    print(f"Train set: {len(train_df):,} samples")
    print(f"Val set: {len(val_df):,} samples")
    print(f"Val positive ratio: {val_df[target_col].mean():.4f}")

# Load normalization statistics
with open('analysis/results/normalization_stats.json', 'r', encoding='utf-8') as f:
    norm_stats_data = json.load(f)
    norm_stats = norm_stats_data['statistics']
fabric.print("정규화 통계 로드 완료")

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

def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths

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
                 hidden_units=None, dropout=None):
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
        self.cross = CrossNetwork(num_features + cat_input_dim + seq_out_dim, num_layers=2)
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

def evaluate_model(fabric, model, val_loader, criterion):
    """Evaluate model on validation set (only on rank 0)"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for num_x, cat_x, seqs, lens, ys in val_loader:
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            total_loss += loss.item() * ys.size(0)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(ys.cpu().numpy())
    
    val_loss = total_loss / len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate competition metrics
    score, ap, wll = calculate_competition_score(all_targets, all_preds)
    
    return val_loss, score, ap, wll

def train_model(fabric, train_df, val_df, num_cols, cat_cols, seq_col, norm_stats, target_col, batch_size, epochs, lr, log_interval, save_dir):
    """Train model with validation and logging (only rank 0 logs)"""
    # Create datasets and loaders
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, norm_stats, target_col, True)
    val_dataset = ClickDataset(val_df, num_cols, cat_cols, seq_col, norm_stats, target_col, True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_train, pin_memory=True, num_workers=4)
    
    # Create model
    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        emb_dim=16,
        lstm_hidden=64,
        hidden_units=[512,256,128],
        dropout=[0.1,0.2,0.3]
    )
    
    # Setup training
    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    
    # Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    
    # Setup validation loader for all ranks
    val_loader_device = fabric.setup_dataloaders(val_loader)
    
    # Move pos_weight to device
    pos_weight = pos_weight.to(fabric.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Setup logging (only on rank 0)
    if fabric.global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = f'{save_dir}/training_log.csv'
        
        # Initialize CSV file with headers
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'step', 'train_loss', 'val_loss', 'val_score', 'val_ap', 'val_wll'])
        
        print(f"학습 시작 (로그 저장: {log_file})")
        print(f"Log interval: {log_interval} steps")
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
            
            # Log every N steps (simplified - no validation during training)
            if global_step % log_interval == 0 and fabric.global_rank == 0:
                # Calculate average train loss over last log_interval steps
                train_loss = np.mean(step_losses[-log_interval:])
                
                # Log to CSV with immediate flush for real-time saving
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, 'a', newline='', buffering=1) as f:  # Line buffering
                    writer = csv.writer(f)
                    writer.writerow([timestamp, epoch, global_step, train_loss, 0.0, 0.0, 0.0, 0.0])  # No validation during training
                    f.flush()  # Force write to disk immediately
        
        # Calculate average loss across all processes
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        avg_loss_tensor = torch.tensor([avg_loss], device=fabric.device)
        gathered_loss = fabric.all_reduce(avg_loss_tensor, reduce_op="mean")
        
        fabric.print(f"\n[Epoch {epoch}] Average Train Loss: {gathered_loss.item():.4f}")
        
        # Validation at the end of each epoch (all ranks evaluate, then gather)
        val_loss, val_score, val_ap, val_wll = evaluate_model(fabric, model, val_loader_device, criterion)
        
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
                }, f'{save_dir}/best_model.pt')
                print(f"✅ Best model saved (Score: {best_score:.6f})")
    
    fabric.print("학습 완료")
    
    if fabric.global_rank == 0:
        print(f"Best validation score: {best_score:.6f}")
        print("Best model will be used for inference")
    
    return model, best_score

save_dir = 'result_dnn_ddp'
fabric.print("모델 학습 실행")
model, best_score = train_model(
    fabric=fabric,
    train_df=train_df,
    val_df=val_df,
    num_cols=num_cols,
    cat_cols=cat_cols,
    seq_col=seq_col,
    norm_stats=norm_stats,
    target_col=target_col,
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr=CFG['LEARNING_RATE'],
    log_interval=CFG['LOG_INTERVAL'],
    save_dir=save_dir
)

# Inference only on rank 0
if fabric.global_rank == 0:
    fabric.print("추론 시작 (Rank 0에서만 실행)")
    test_dataset = ClickDataset(test, num_cols, cat_cols, seq_col, norm_stats, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False,
                             collate_fn=collate_fn_infer, pin_memory=True, num_workers=4)
    
    model.eval()
    outs = []
    with torch.no_grad():
        for num_x, cat_x, seqs, lens in tqdm(test_loader, desc="[Inference]"):
            num_x = num_x.to(fabric.device)
            cat_x = cat_x.to(fabric.device)
            seqs = seqs.to(fabric.device)
            lens = lens.to(fabric.device)
            outs.append(torch.sigmoid(model(num_x, cat_x, seqs, lens)).cpu())
    
    test_preds = torch.cat(outs).numpy()
    fabric.print("추론 완료")
    
    submit = pd.DataFrame({
        'ID': test_dataset.df['ID'],
        'clicked': test_preds
    })
    submit.to_csv(f'{save_dir}/submission.csv', index=False)
    fabric.print(f"제출 파일 저장 완료: {save_dir}/submission.csv")
    fabric.print(f"Training log: {save_dir}/training_log.csv")
    fabric.print(f"Best validation score: {best_score:.6f}")
else:
    fabric.print(f"Rank {fabric.global_rank}: 추론 건너뛰기 (Rank 0에서만 실행)")

# Wait for all processes to finish
fabric.barrier()
fabric.print(f"Rank {fabric.global_rank}: 완료")

