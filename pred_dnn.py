import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Import common functions
from utils import seed_everything, calculate_competition_score
from data_loader import (
    ClickDatasetDNN,
    collate_fn_dnn_infer,
    collate_fn_dnn_train,
    load_processed_dnn_data
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
    parser.add_argument('--test-path', type=str, default='data/proc_test',
                        help='Path to preprocessed test data (default: data/proc_test)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save submission file (default: {model_dir}/submission.csv)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for inference (default: 2048)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference (default: cuda)')
    
    return parser.parse_args()

def load_model_and_metadata(model_dir, device='cuda'):
    """Load model, encoders, calibrator, and metadata"""
    print(f"\n📦 Loading model from {model_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"   Model: {metadata['model_name']}")
    print(f"   Validation Score: {metadata['val_score']:.6f}")
    print(f"   Calibration method: {metadata.get('calibration_method', 'none')}")
    
    # Load categorical encoders (if exists, may not exist for new preprocessing)
    cat_encoders = None
    encoders_path = os.path.join(model_dir, 'cat_encoders.pkl')
    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            cat_encoders = pickle.load(f)
        print("   ✅ Categorical encoders loaded")
    else:
        print("   ℹ️  No categorical encoders found (using preprocessed data)")
    
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
    
    # Convert cat_cardinalities from dict to list (based on cat_features order)
    cat_features = metadata['cat_features']
    cat_cardinalities_list = [cat_cardinalities[col] for col in cat_features]
    
    model = WideDeepCTR(
        num_features=num_features,
        cat_cardinalities=cat_cardinalities_list,
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
    
    # Load calibrator if exists
    calibrator = None
    calibrator_path = os.path.join(model_dir, 'calibrator.pkl')
    if os.path.exists(calibrator_path):
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"   ✅ Calibrator loaded from {calibrator_path}")
    else:
        print("   ℹ️  No calibrator found (will use raw predictions)")
    
    return model, cat_encoders, calibrator, metadata

def load_test_data(test_path, num_cols, cat_cols, seq_col, batch_size=2048):
    """Load preprocessed test data, return dataset and dataloader"""
    print(f"\n📦 Loading preprocessed test data from {test_path}...")
    
    # Load preprocessed test data using the same loader as HPO
    test_df = load_processed_dnn_data(test_path)
    print(f"   ✅ Test data loaded: {test_df.shape}")
    
    # Create dataset and dataloader (no normalization needed, already normalized)
    print("   Creating dataset and dataloader...")
    test_dataset = ClickDatasetDNN(test_df, num_cols, cat_cols, seq_col, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_dnn_infer, pin_memory=True, 
                            num_workers=4, prefetch_factor=8)
    
    print("   ✅ Dataset and dataloader created")
    
    return test_dataset, test_loader

def predict_and_save(model, calibrator, calibration_method, test_dataset, test_loader, output_path, device='cuda'):
    """Generate predictions and save submission file"""
    print("\n🔮 Predicting on test data...")
    
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
    
    # Apply calibration if calibrator exists
    if calibrator is not None:
        print(f"   Applying {calibration_method} calibration...")
        
        if calibration_method == 'temperature':
            all_preds = calibrator.predict_proba(all_logits)
        elif calibration_method == 'isotonic':
            all_preds = calibrator.predict(all_preds)
        elif calibration_method == 'sigmoid':
            all_preds = calibrator.predict_proba(all_preds.reshape(-1, 1))[:, 1]
        else:
            print(f"   ⚠️  Unknown calibration method: {calibration_method}")
            print("   Using raw predictions")
        
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
    print("   Prediction stats:")
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
    
    # Load model, calibrator, and metadata
    model, _, calibrator, metadata = load_model_and_metadata(args.model_dir, device=args.device)
    calibration_method = metadata.get('calibration_method', 'none')
    
    # Define feature columns
    cat_cols = metadata['cat_features']
    target_col = "clicked"
    seq_col = "seq"
    FEATURE_EXCLUDE = {target_col, seq_col, "ID", "l_feat_20", "l_feat_23"}
    
    # Get numerical columns from metadata or infer from test data
    # Load a sample of test data to determine feature columns
    test_df_sample = load_processed_dnn_data(args.test_path)
    feature_cols = [c for c in test_df_sample.columns if c not in FEATURE_EXCLUDE]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    del test_df_sample
    
    print(f"\n📊 Features: Num={len(num_cols)} | Cat={len(cat_cols)}")
    
    # Load preprocessed test data with dataset and dataloader
    test_dataset, test_loader = load_test_data(
        args.test_path, num_cols, cat_cols, seq_col, 
        batch_size=args.batch_size
    )
    
    # Predict and save
    predict_and_save(
        model,
        calibrator,
        calibration_method,
        test_dataset,
        test_loader,
        args.output_path,
        device=args.device
    )
    
    print("\n" + "🎉"*35)
    print("PREDICTION COMPLETE!")
    print("🎉"*35)
    print(f"✅ Submission saved to: {args.output_path}")
    print("="*70)

if __name__ == "__main__":
    main()
