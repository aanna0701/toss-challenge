#!/usr/bin/env python3
"""
NVIDIA Merlin을 활용한 고성능 훈련 및 예측 워크플로우
"""

import argparse
import gc
import json
import logging
import os
import psutil
import time
import traceback
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='NVIDIA Merlin 기반 고성능 훈련 및 예측 워크플로우')
    parser.add_argument('--config', type=str, default='config_fold1.yaml',
                       help='설정 파일 경로 (기본값: config_fold1.yaml)')
    parser.add_argument('--use-merlin', action='store_true', default=True,
                       help='NVIDIA Merlin 사용 여부 (기본값: True)')
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer', 'xgboost'],
                       help='모델 타입 (기본값: transformer)')
    return parser.parse_args()

def load_config(config_path):
    """설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"📋 설정 파일 로드: {config_path}")
    return config

# 명령행 인수 파싱 및 설정 로드
args = parse_args()
CFG = load_config(args.config)

from utils import seed_everything, get_device
from data_loader_merlin import (
    MerlinFeatureProcessor, 
    MerlinClickDataset, 
    create_merlin_dataloader,
    load_train_data_merlin,
    load_test_data_merlin,
    MERLIN_AVAILABLE
)

DEVICE = get_device()

def create_results_directory():
    """결과 저장 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
    
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"📁 결과 디렉토리 생성: {results_dir}")
    
    return results_dir

def train_transformer_model_merlin(train_df, CFG, results_dir, use_merlin=True):
    """Merlin 기반 Transformer 모델 훈련"""
    logger.info(f"🚀 Merlin 기반 Transformer 모델 훈련 시작...")
    logger.info(f"   • Merlin 사용: {use_merlin and MERLIN_AVAILABLE}")
    
    # 데이터 분할
    tr_df, va_df = train_test_split(
        train_df, 
        test_size=CFG['VAL_SPLIT'], 
        random_state=42, 
        shuffle=True, 
        stratify=train_df['clicked']
    )
    
    logger.info(f"📊 데이터 분할 결과:")
    logger.info(f"   • 훈련 데이터: {len(tr_df):,}개")
    logger.info(f"   • 검증 데이터: {len(va_df):,}개")
    
    # 피처 프로세서 생성 및 학습
    logger.info("🔧 피처 프로세서 학습 중...")
    processor = MerlinFeatureProcessor(
        config=CFG,
        normalization_stats_path="analysis/results/normalization_stats.json",
        use_merlin=use_merlin and MERLIN_AVAILABLE
    )
    
    processor.fit(tr_df)
    
    # 데이터셋 생성
    logger.info("📊 데이터셋 생성 중...")
    train_dataset = MerlinClickDataset(
        df=tr_df,
        feature_processor=processor,
        target_col='clicked',
        has_target=True,
        has_id=False,
        use_merlin=use_merlin and MERLIN_AVAILABLE
    )
    
    val_dataset = MerlinClickDataset(
        df=va_df,
        feature_processor=processor,
        target_col='clicked',
        has_target=True,
        has_id=False,
        use_merlin=use_merlin and MERLIN_AVAILABLE
    )
    
    # DataLoader 생성
    logger.info("🚀 DataLoader 생성 중...")
    train_loader = create_merlin_dataloader(
        dataset=train_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = create_merlin_dataloader(
        dataset=val_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"✅ DataLoader 생성 완료:")
    logger.info(f"   • 훈련 배치: {len(train_loader)}개")
    logger.info(f"   • 검증 배치: {len(val_loader)}개")
    
    # 모델 생성
    from model import create_model
    model = create_model(CFG).to(DEVICE)
    
    # 손실 함수 및 옵티마이저
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['LEARNING_RATE'],
        weight_decay=CFG['WEIGHT_DECAY']
    )
    
    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS']
    )
    
    # 훈련 루프
    logger.info("🚀 훈련 시작...")
    best_val_score = 0.0
    patience_counter = 0
    
    for epoch in range(1, CFG['EPOCHS'] + 1):
        # 훈련
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # GPU로 이동
            x_categorical = batch['x_categorical'].to(DEVICE)
            x_numerical = batch['x_numerical'].to(DEVICE)
            seqs = batch['seqs'].to(DEVICE)
            seq_lengths = batch['seq_lengths'].to(DEVICE)
            ys = batch['ys'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(x_categorical, x_numerical, seqs, seq_lengths)
            loss = criterion(outputs.squeeze(), ys)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"   Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_categorical = batch['x_categorical'].to(DEVICE)
                x_numerical = batch['x_numerical'].to(DEVICE)
                seqs = batch['seqs'].to(DEVICE)
                seq_lengths = batch['seq_lengths'].to(DEVICE)
                ys = batch['ys'].to(DEVICE)
                
                outputs = model(x_categorical, x_numerical, seqs, seq_lengths)
                loss = criterion(outputs.squeeze(), ys)
                
                val_loss += loss.item()
                
                # 예측값 저장
                val_predictions.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy())
                val_targets.extend(ys.cpu().numpy())
        
        # 메트릭 계산
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        val_auc = roc_auc_score(val_targets, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        # 스케줄러 업데이트
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        logger.info(f"📊 Epoch {epoch} 결과:")
        logger.info(f"   • 훈련 손실: {avg_train_loss:.6f}")
        logger.info(f"   • 검증 손실: {avg_val_loss:.6f}")
        logger.info(f"   • 검증 AUC: {val_auc:.6f}")
        logger.info(f"   • 검증 RMSE: {val_rmse:.6f}")
        logger.info(f"   • 소요 시간: {epoch_time:.2f}초")
        
        # 조기 종료
        if CFG['EARLY_STOPPING']['ENABLED']:
            if val_auc > best_val_score + CFG['EARLY_STOPPING']['MIN_DELTA']:
                best_val_score = val_auc
                patience_counter = 0
                
                # 최고 모델 저장
                model_path = os.path.join(results_dir, f"best_model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_rmse': val_rmse,
                    'processor': processor
                }, model_path)
                logger.info(f"💾 최고 모델 저장: {model_path}")
            else:
                patience_counter += 1
                logger.info(f"⏳ 조기 종료 카운터: {patience_counter}/{CFG['EARLY_STOPPING']['PATIENCE']}")
                
                if patience_counter >= CFG['EARLY_STOPPING']['PATIENCE']:
                    logger.info("🛑 조기 종료 조건 만족. 훈련 종료.")
                    break
    
    logger.info("✅ Transformer 모델 훈련 완료!")
    return model, processor

def predict_transformer_model_merlin(test_df, model, processor, CFG, results_dir, use_merlin=True):
    """Merlin 기반 Transformer 모델 예측"""
    logger.info("🔮 Merlin 기반 Transformer 모델 예측 시작...")
    
    # 테스트 데이터셋 생성
    test_dataset = MerlinClickDataset(
        df=test_df,
        feature_processor=processor,
        target_col='clicked',
        has_target=False,
        has_id=True,
        use_merlin=use_merlin and MERLIN_AVAILABLE
    )
    
    # 테스트 DataLoader 생성
    test_loader = create_merlin_dataloader(
        dataset=test_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"📊 테스트 DataLoader 생성 완료: {len(test_loader)}개 배치")
    
    # 예측 수행
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_categorical = batch['x_categorical'].to(DEVICE)
            x_numerical = batch['x_numerical'].to(DEVICE)
            seqs = batch['seqs'].to(DEVICE)
            seq_lengths = batch['seq_lengths'].to(DEVICE)
            batch_ids = batch['ids']
            
            outputs = model(x_categorical, x_numerical, seqs, seq_lengths)
            probs = torch.sigmoid(outputs.squeeze())
            
            predictions.extend(probs.cpu().numpy())
            ids.extend(batch_ids)
            
            if batch_idx % 100 == 0:
                logger.info(f"   배치 {batch_idx}/{len(test_loader)} 처리 완료")
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'id': ids,
        'clicked': predictions
    })
    
    submission_path = os.path.join(results_dir, f"merlin_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"✅ 예측 완료!")
    logger.info(f"📊 예측 결과 통계:")
    logger.info(f"   • Shape: {submission_df.shape}")
    logger.info(f"   • Min: {submission_df['clicked'].min():.4f}")
    logger.info(f"   • Max: {submission_df['clicked'].max():.4f}")
    logger.info(f"   • Mean: {submission_df['clicked'].mean():.4f}")
    logger.info(f"💾 제출 파일 저장: {submission_path}")
    
    return submission_df

def main():
    """메인 실행 함수"""
    try:
        logger.info("🚀 NVIDIA Merlin 기반 고성능 워크플로우 시작")
        logger.info("=" * 60)
        
        # Merlin 사용 가능 여부 확인
        if args.use_merlin and not MERLIN_AVAILABLE:
            logger.warning("⚠️  NVIDIA Merlin이 설치되지 않았습니다. 표준 데이터로더를 사용합니다.")
            args.use_merlin = False
        
        logger.info(f"🔧 설정 정보:")
        logger.info(f"   • Merlin 사용: {args.use_merlin and MERLIN_AVAILABLE}")
        logger.info(f"   • 모델 타입: {args.model_type}")
        logger.info(f"   • 배치 크기: {CFG['BATCH_SIZE']}")
        logger.info(f"   • 에포크: {CFG['EPOCHS']}")
        
        # 시드 설정
        seed_everything(CFG['SEED'])
        logger.info(f"🎲 시드 설정: {CFG['SEED']}")
        
        # 결과 디렉토리 생성
        results_dir = create_results_directory()
        
        # 메모리 사용량 확인
        memory_info = psutil.virtual_memory()
        logger.info(f"💾 메모리 사용량: {memory_info.percent:.1f}% ({memory_info.used / 1024**3:.1f}GB / {memory_info.total / 1024**3:.1f}GB)")
        
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU 사용 가능: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 훈련 데이터 로드
        logger.info("📊 훈련 데이터 로드 중...")
        train_df, feature_cols, seq_col, target_col = load_train_data_merlin(CFG)
        logger.info(f"   • 훈련 데이터 크기: {train_df.shape}")
        logger.info(f"   • 클래스 분포: {train_df['clicked'].value_counts().to_dict()}")
        
        # 모델별 훈련 및 예측
        if args.model_type == 'transformer':
            # Transformer 모델 훈련
            model, processor = train_transformer_model_merlin(
                train_df, CFG, results_dir, args.use_merlin
            )
            
            # 테스트 데이터 로드
            logger.info("📊 테스트 데이터 로드 중...")
            test_df = load_test_data_merlin(CFG)
            logger.info(f"   • 테스트 데이터 크기: {test_df.shape}")
            
            # 예측 수행
            submission_df = predict_transformer_model_merlin(
                test_df, model, processor, CFG, results_dir, args.use_merlin
            )
            
        elif args.model_type == 'xgboost':
            logger.info("🚀 XGBoost 모델은 별도 스크립트를 사용하세요: train_and_predict_xgboost.py")
            return 0
        
        # 메타데이터 저장
        metadata = {
            'model_type': args.model_type,
            'use_merlin': args.use_merlin and MERLIN_AVAILABLE,
            'config': CFG,
            'timestamp': datetime.now().isoformat(),
            'submission_shape': submission_df.shape,
            'device': str(DEVICE)
        }
        
        metadata_path = os.path.join(results_dir, f"merlin_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 메타데이터 저장: {metadata_path}")
        
        logger.info("🎉 Merlin 기반 워크플로우 완료!")
        logger.info(f"📁 결과 저장 위치: {results_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {str(e)}")
        logger.error("📋 상세 오류 정보:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
