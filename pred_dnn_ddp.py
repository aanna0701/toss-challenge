import pandas as pd
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
import argparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import common functions
from utils import seed_everything, calculate_competition_score
from data_loader import (
    ClickDatasetDNN,
    collate_fn_dnn_infer,
    collate_fn_dnn_train
)

# Set seed for reproducibility
seed_everything(42)

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

# ============================================================================
# 모델 아키텍처 (동일하게 정의 필요)
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
# Helper Functions
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DNN Prediction Script')
    
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model (e.g., result_dnn_ddp/20231201_120000)')
    parser.add_argument('--test-path', type=str, default='data/test.parquet',
                        help='Path to test data')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save submission file (default: {model_dir}/submission.csv)')
    parser.add_argument('--use-calibration', action='store_true', default=True,
                        help='Use calibration model for predictions (default: True)')
    parser.add_argument('--no-calibration', dest='use_calibration', action='store_false',
                        help='Disable calibration')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for inference (default: 2048)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference (default: cuda)')
    
    return parser.parse_args()

def load_model_and_metadata(model_dir, device='cuda'):
    """Load model, encoders, and metadata"""
    print(f"\n📦 Loading model from {model_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"   Model: {metadata['model_name']}")
    print(f"   Validation Score: {metadata['val_score']:.6f}")
    
    # Load categorical encoders
    encoders_path = os.path.join(model_dir, 'cat_encoders.pkl')
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Categorical encoders not found: {encoders_path}")
    
    with open(encoders_path, 'rb') as f:
        cat_encoders = pickle.load(f)
    print("   ✅ Categorical encoders loaded")
    
    # Find model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('dnn_') and f.endswith('.pt')]
    
    if not model_files:
        raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    model_path = os.path.join(model_dir, model_files[0])
    print(f"   Loading model from {model_path}")
    
    # Create model architecture
    cat_cardinalities = metadata['cat_cardinalities']
    num_features = metadata['num_features']
    
    # Get model architecture from metadata (with fallback to defaults)
    model_arch = metadata.get('model_architecture', {})
    emb_dim = model_arch.get('emb_dim', 16)
    lstm_hidden = model_arch.get('lstm_hidden', 64)
    hidden_units = model_arch.get('hidden_units', [512, 256, 128])
    dropout = model_arch.get('dropout', [0.1, 0.2, 0.3])
    cross_layers = model_arch.get('cross_layers', 2)
    
    model = WideDeepCTR(
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        emb_dim=emb_dim,
        lstm_hidden=lstm_hidden,
        hidden_units=hidden_units,
        dropout=dropout,
        cross_layers=cross_layers
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("   ✅ Model loaded and ready for inference")
    
    return model, cat_encoders, metadata

def load_and_encode_test(test_path, cat_encoders, cat_cols):
    """Load and encode test data"""
    print(f"\n📦 Loading test data from {test_path}...")
    
    test = pd.read_parquet(test_path, engine="pyarrow")
    print(f"   Test shape: {test.shape}")
    
    # Apply categorical encoding
    print("   Encoding categorical features...")
    for col in cat_cols:
        if col in test.columns:
            # Use transform (not fit_transform) to use training encodings
            test[col] = cat_encoders[col].transform(test[col].astype(str).fillna("UNK"))
    
    print("   ✅ Test data loaded and encoded")
    
    return test

def find_best_calibration(model, cal_df, cat_cols, num_cols, seq_col, norm_stats, target_col, device='cuda', batch_size=2048, test_ratio=0.5):
    """
    Find best calibration method by comparing all methods on calibration test set
    
    Returns:
        best_method: Best calibration method name ('none', 'isotonic', 'sigmoid', or 'temperature')
        best_calibrator: Fitted calibrator (or None if 'none' is best)
        calibration_results: Dict of all results
    """
    print("\n" + "="*70)
    print("🎯 Finding Best Calibration Method")
    print("="*70)
    
    # Load calibration data
    print("\n📦 Loading calibration data (train_c)...")
    cal_dataset = ClickDatasetDNN(cal_df, num_cols, cat_cols, seq_col, norm_stats, target_col, True)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn_dnn_train, pin_memory=True, num_workers=4)
    
    # Collect predictions on train_c
    print("   Collecting predictions on train_c...")
    model.eval()
    cal_logits = []
    cal_preds = []
    cal_targets = []
    
    with torch.no_grad():
        for num_x, cat_x, seqs, lens, ys in tqdm(cal_loader, desc="[train_c]", leave=False):
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            
            logits = model(num_x, cat_x, seqs, lens)
            preds = torch.sigmoid(logits)
            
            cal_logits.append(logits.cpu().numpy())
            cal_preds.append(preds.cpu().numpy())
            cal_targets.append(ys.numpy())
    
    cal_logits = np.concatenate(cal_logits)
    cal_preds = np.concatenate(cal_preds)
    cal_targets = np.concatenate(cal_targets)
    
    # Split train_c: balanced set for fitting, rest for testing
    pos_idx = np.where(cal_targets == 1)[0]
    neg_idx = np.where(cal_targets == 0)[0]
    
    print(f"   train_c total: {len(cal_targets):,} samples (pos: {len(pos_idx):,}, neg: {len(neg_idx):,})")
    
    # Sample balanced training set
    np.random.seed(42)
    n_pos_train = int(len(pos_idx) * test_ratio)
    pos_train_idx = np.random.choice(pos_idx, size=n_pos_train, replace=False)
    pos_test_idx = np.setdiff1d(pos_idx, pos_train_idx)
    
    neg_train_idx = np.random.choice(neg_idx, size=n_pos_train, replace=False)
    neg_test_idx = np.setdiff1d(neg_idx, neg_train_idx)
    
    cal_train_idx = np.concatenate([pos_train_idx, neg_train_idx])
    cal_test_idx = np.concatenate([pos_test_idx, neg_test_idx])
    
    np.random.shuffle(cal_train_idx)
    np.random.shuffle(cal_test_idx)
    
    print(f"   → Calibration fit set: {len(cal_train_idx):,} samples (balanced, pos={len(pos_train_idx):,}, neg={len(neg_train_idx):,})")
    print(f"   → Calibration test set: {len(cal_test_idx):,} samples (imbalanced, pos={len(pos_test_idx):,}, neg={len(neg_test_idx):,})")
    
    # Test all calibration methods
    methods = ['none', 'isotonic', 'sigmoid', 'temperature']
    results = {}
    calibrators = {}
    
    print("\n🔬 Testing calibration methods...")
    
    for method in methods:
        print(f"\n   [{method.upper()}]")
        
        if method == 'none':
            # No calibration - use raw predictions
            cal_test_calibrated = cal_preds[cal_test_idx]
            calibrators[method] = None
        else:
            # Get training data for calibration
            cal_train_logits = cal_logits[cal_train_idx]
            cal_train_preds = cal_preds[cal_train_idx]
            cal_train_targets = cal_targets[cal_train_idx]
            
            # Fit calibrator
            if method == 'temperature':
                calibrator = TemperatureScaling()
                calibrator.fit(cal_train_logits, cal_train_targets)
                
                # Apply to test set
                cal_test_calibrated = calibrator.predict_proba(cal_logits[cal_test_idx])
                
            elif method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(cal_train_preds, cal_train_targets)
                
                cal_test_calibrated = calibrator.predict(cal_preds[cal_test_idx])
                
            else:  # sigmoid
                calibrator = LogisticRegression()
                calibrator.fit(cal_train_preds.reshape(-1, 1), cal_train_targets)
                
                cal_test_calibrated = calibrator.predict_proba(cal_preds[cal_test_idx].reshape(-1, 1))[:, 1]
            
            calibrators[method] = calibrator
        
        # Evaluate on calibration test set
        cal_test_targets = cal_targets[cal_test_idx]
        score, ap, wll = calculate_competition_score(cal_test_targets, cal_test_calibrated)
        
        # Store results
        results[method] = {
            'score': score,
            'ap': ap,
            'wll': wll
        }
        
        print(f"      Score: {score:.6f}, AP: {ap:.6f}, WLL: {wll:.6f}")
    
    # Find best method based on calibration test score
    best_method = max(results.keys(), key=lambda m: results[m]['score'])
    best_score = results[best_method]['score']
    
    print(f"\n🏆 Best Method: {best_method.upper()}")
    print(f"   Score: {best_score:.6f}")
    print(f"   AP: {results[best_method]['ap']:.6f}")
    print(f"   WLL: {results[best_method]['wll']:.6f}")
    
    # Compare with no calibration
    if best_method != 'none':
        improvement = best_score - results['none']['score']
        print(f"   Improvement over raw: {improvement:+.6f}")
    
    print("="*70)
    
    return best_method, calibrators[best_method], results

def predict_and_save(model, best_method, best_calibrator, test_df, cat_cols, num_cols, seq_col, norm_stats, output_path, batch_size=2048, device='cuda'):
    """Generate predictions and save submission file"""
    print(f"\n🔮 Predicting on test data...")
    print(f"   Using calibration method: {best_method.upper()}")
    
    # Create test dataset
    test_dataset = ClickDatasetDNN(test_df, num_cols, cat_cols, seq_col, norm_stats, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_dnn_infer, pin_memory=True, 
                             num_workers=4, prefetch_factor=8)
    
    # Run inference
    all_logits = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for num_x, cat_x, seqs, lens in tqdm(test_loader, desc="[Test Inference]"):
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            
            logits = model(num_x, cat_x, seqs, lens)
            preds = torch.sigmoid(logits)
            
            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_logits = np.concatenate(all_logits)
    all_preds = np.concatenate(all_preds)
    
    print(f"   Raw prediction range: [{all_preds.min():.6f}, {all_preds.max():.6f}]")
    
    # Apply calibration if needed
    if best_method != 'none' and best_calibrator is not None:
        print(f"   Applying {best_method} calibration...")
        
        if best_method == 'temperature':
            all_preds = best_calibrator.predict_proba(all_logits)
        elif best_method == 'isotonic':
            all_preds = best_calibrator.predict(all_preds)
        else:  # sigmoid
            all_preds = best_calibrator.predict_proba(all_preds.reshape(-1, 1))[:, 1]
        
        print(f"   Calibrated prediction range: [{all_preds.min():.6f}, {all_preds.max():.6f}]")
    else:
        print("   ✅ Using raw predictions (no calibration)")
    
    # Create submission file
    submission_df = pd.DataFrame({
        'ID': test_dataset.df['ID'],
        'clicked': all_preds
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to {output_path}")
    print(f"   Samples: {len(submission_df):,}")
    print(f"   Prediction stats:")
    print(f"      Mean: {all_preds.mean():.6f}")
    print(f"      Std: {all_preds.std():.6f}")
    print(f"      Min: {all_preds.min():.6f}")
    print(f"      Max: {all_preds.max():.6f}")

def main():
    """Main prediction pipeline"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🔮 DNN Prediction Pipeline")
    print("="*70)
    print(f"   Model directory: {args.model_dir}")
    print(f"   Test data: {args.test_path}")
    print(f"   Use calibration: {args.use_calibration}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {args.device}")
    
    # Verify paths
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test data not found: {args.test_path}")
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_dir, 'submission.csv')
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("   ⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and metadata
    model, cat_encoders, metadata = load_model_and_metadata(args.model_dir, device=args.device)
    
    # Load normalization statistics
    print("\n📊 Loading normalization statistics...")
    with open('analysis/results/normalization_stats.json', 'r', encoding='utf-8') as f:
        norm_stats_data = json.load(f)
        norm_stats = norm_stats_data['statistics']
    print("   ✅ Normalization statistics loaded")
    
    # Define feature columns
    cat_cols = metadata['cat_features']
    target_col = "clicked"
    seq_col = "seq"
    FEATURE_EXCLUDE = {target_col, seq_col, "ID", "l_feat_20", "l_feat_23"}
    
    # Find best calibration method if enabled
    best_method = 'none'
    best_calibrator = None
    
    if args.use_calibration:
        # Load calibration data
        print("\n📦 Loading data for calibration selection...")
        cal_df = pd.read_parquet("data/train_c.parquet", engine="pyarrow")
        
        # Encode categorical features
        for col in cat_cols:
            cal_df[col] = cat_encoders[col].transform(cal_df[col].astype(str).fillna("UNK"))
        
        # Get feature columns
        feature_cols = [c for c in cal_df.columns if c not in FEATURE_EXCLUDE]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        
        # Find best calibration method
        best_method, best_calibrator, _ = find_best_calibration(
            model, cal_df, cat_cols, num_cols, seq_col, norm_stats, target_col,
            device=args.device, batch_size=args.batch_size
        )
    else:
        print("\n⚠️  Calibration disabled by user (--no-calibration)")
    
    # Load and encode test data
    test_df = load_and_encode_test(args.test_path, cat_encoders, cat_cols)
    
    # Define numerical columns (if not already defined from calibration)
    if 'num_cols' not in locals():
        feature_cols = [c for c in test_df.columns if c not in FEATURE_EXCLUDE]
        num_cols = [c for c in feature_cols if c not in cat_cols]
    
    print(f"\n📊 Features: Num={len(num_cols)} | Cat={len(cat_cols)}")
    
    # Predict and save
    predict_and_save(
        model,
        best_method,
        best_calibrator,
        test_df,
        cat_cols,
        num_cols,
        seq_col,
        norm_stats,
        args.output_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\n" + "🎉"*35)
    print("PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"✅ Submission saved to: {args.output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
