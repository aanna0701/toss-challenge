import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ClickDataset(Dataset):
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target

        # 비-시퀀스 피처: 전부 연속값으로
        self.X = self.df[self.feature_cols].astype(float).fillna(0).values

        # 시퀀스: 문자열 그대로 보관 (lazy 파싱)
        self.seq_strings = self.df[self.seq_col].astype(str).values

        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float)

        # 전체 시퀀스 사용 (빈 시퀀스만 방어)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)

        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)  # 빈 시퀀스 방어

        seq = torch.from_numpy(arr)  # shape (seq_len,)

        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return x, seq, y
        else:
            return x, seq

def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 빈 시퀀스 방지
    return xs, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths

def create_data_loaders(train_df, val_df, test_df, feature_cols, seq_col, target_col, batch_size):
    """데이터로더 생성 함수"""
    # Train/Val datasets
    train_dataset = ClickDataset(train_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = ClickDataset(val_df, feature_cols, seq_col, target_col, has_target=True)
    
    # Test dataset
    test_dataset = ClickDataset(test_df, feature_cols, seq_col, has_target=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset

def load_and_preprocess_data(use_sampling=True, sample_size=1000000):
    """데이터 로드 및 전처리 함수"""
    from main import CFG
    
    def safe_load_parquet(file_path, sample_size=None):
        """안전한 parquet 로드 함수"""
        try:
            # 전체 데이터 로드 시도
            if not use_sampling:
                return pd.read_parquet(file_path, engine="pyarrow")
            else:
                raise Exception("샘플링 모드로 진행")
        except Exception:
            print(f"⚠️  {file_path} 대용량 데이터 - 샘플링 진행...")
            
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                
                if sample_size and total_rows > sample_size:
                    print(f"📊 {total_rows:,} 행 중 {sample_size:,} 행 샘플링")
                    sample_ratio = sample_size / total_rows
                    
                    chunks = []
                    for batch in parquet_file.iter_batches(batch_size=50000):
                        chunk_df = batch.to_pandas()
                        chunk_sample = chunk_df.sample(frac=sample_ratio, random_state=42)
                        chunks.append(chunk_sample)
                        
                        if sum(len(chunk) for chunk in chunks) >= sample_size:
                            break
                    
                    return pd.concat(chunks, ignore_index=True).head(sample_size)
                else:
                    return pd.read_parquet(file_path, engine="pyarrow")
                    
            except Exception as e:
                print(f"❌ {file_path} 로드 실패: {e}")
                raise
    
    # 데이터 로드
    print("📊 훈련 데이터 로드 중...")
    all_train = safe_load_parquet(CFG['PATHS']['TRAIN_DATA'], sample_size)
    
    print("📊 테스트 데이터 로드 중...")
    test = safe_load_parquet(CFG['PATHS']['TEST_DATA'], sample_size)
    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])

    print("Train shape:", all_train.shape)
    print("Test shape:", test.shape)

    # clicked == 1 데이터
    clicked_1 = all_train[all_train['clicked'] == 1]

    # clicked == 0 데이터에서 동일 개수x2 만큼 무작위 추출 (다운 샘플링)
    clicked_0 = all_train[all_train['clicked'] == 0].sample(n=len(clicked_1)*2, random_state=42)

    # 두 데이터프레임 합치기
    train = pd.concat([clicked_1, clicked_0], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Train shape:", train.shape)
    print("Train clicked:0:", train[train['clicked']==0].shape)
    print("Train clicked:1:", train[train['clicked']==1].shape)

    # Target / Sequence
    target_col = "clicked"
    seq_col = "seq"

    # 학습에 사용할 피처: ID/seq/target 제외, 나머지 전부
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
    feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

    print("Num features:", len(feature_cols))
    print("Sequence:", seq_col)
    print("Target:", target_col)

    return train, test, feature_cols, seq_col, target_col
