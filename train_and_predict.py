#!/usr/bin/env python3
"""
훈련 후 자동으로 예측을 실행하고 결과를 날짜별로 저장하는 워크플로우 스크립트
"""

import os
from datetime import datetime
import yaml

# config_debug.yaml 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

from main import device
from utils import seed_everything
from data_loader import load_and_preprocess_data
from model import *
from train import train_model, save_model
from predict import predict_test_data


def create_results_directory():
    """결과 저장 디렉토리 생성"""
    # {datetime}을 실제 타임스탬프로 치환
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 결과 디렉토리 생성: {results_dir}")
    return results_dir


def cleanup_temp_files(temp_model_path):
    """임시 웨이트 파일 삭제"""
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        print(f"🗑️  임시 웨이트 파일 삭제: {temp_model_path}")
    else:
        print(f"⚠️  임시 웨이트 파일이 존재하지 않음: {temp_model_path}")


def save_results_with_metadata(results_dir, submission_df, model_info):
    """결과와 메타데이터를 함께 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 예측 결과 저장
    submission_path = os.path.join(results_dir, f"submission_{timestamp}.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"📊 예측 결과 저장: {submission_path}")
    
    # 메타데이터 저장
    metadata = {
        'timestamp': timestamp,
        'model_info': model_info,
        'config': CFG,
        'submission_shape': submission_df.shape,
        'submission_stats': {
            'mean_prediction': float(submission_df['clicked'].mean()),
            'min_prediction': float(submission_df['clicked'].min()),
            'max_prediction': float(submission_df['clicked'].max()),
            'std_prediction': float(submission_df['clicked'].std())
        }
    }
    
    metadata_path = os.path.join(results_dir, f"metadata_{timestamp}.json")
    import json
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"📋 메타데이터 저장: {metadata_path}")
    
    return submission_path, metadata_path


def print_progress(step, total_steps, description):
    """진행상황 프린트 함수"""
    progress = f"[{step}/{total_steps}]"
    print(f"\n{'='*20} {progress} {description} {'='*20}")
    print(f"⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_step_summary(step_name, details):
    """단계별 요약 정보 프린트"""
    print(f"\n📋 {step_name} 완료:")
    for key, value in details.items():
        print(f"   • {key}: {value}")


def main():
    """메인 워크플로우 실행"""
    total_steps = 10
    print("🚀 훈련 → 예측 → 결과 저장 워크플로우 시작")
    print("=" * 80)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 총 단계: {total_steps}단계")
    
    # 1. 초기화
    print_progress(1, total_steps, "시스템 초기화")
    # 시드 고정
    seed_everything(CFG['SEED'])
    print(f"Device: {device}")
    print_step_summary("초기화", {
        "Device": device,
        "Epochs": CFG['EPOCHS'],
        "Batch Size": CFG['BATCH_SIZE'],
        "Learning Rate": CFG['LEARNING_RATE'],
        "Data Sampling": CFG['DATA']['USE_SAMPLING'],
        "Sample Size": CFG['DATA']['SAMPLE_SIZE']
    })
    
    # 2. 결과 디렉토리 생성
    print_progress(2, total_steps, "결과 디렉토리 생성")
    results_dir = create_results_directory()
    print_step_summary("디렉토리 생성", {"Results Dir": results_dir})
    
    # 3. 임시 웨이트 파일 경로 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_model_path = CFG['PATHS']['TEMP_MODEL'].replace('{datetime}', timestamp)
    print(f"📝 임시 웨이트 파일 경로: {temp_model_path}")
    
    try:
        # 4. 데이터 로드 및 전처리
        print_progress(3, total_steps, "데이터 로드 및 전처리")
        # 훈련 데이터: config.yaml의 USE_SAMPLING 설정에 따라 샘플링 또는 전체 로드
        # 테스트 데이터: 무조건 전체 데이터 로드 (force_full_load=True)
        train_data, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
        print_step_summary("데이터 로드", {
            "Train Shape": train_data.shape,
            "Test Shape": test_data.shape,
            "Features": len(feature_cols),
            "Sequence Column": seq_col,
            "Target Column": target_col,
            "Test ID Column": "ID" in test_data.columns,
            "Data Sampling": CFG['DATA']['USE_SAMPLING'],
            "Test Data": "전체 로드 (샘플링 없음)"
        })
        
        # 5. 모델 훈련
        print_progress(4, total_steps, "모델 훈련")
        print(f"🏋️ 모델 설정:")
        print(f"   • Type: {CFG['MODEL']['TYPE']}")
        if CFG['MODEL']['TYPE'] == 'tabular_seq':
            print(f"   • LSTM Hidden: {CFG['MODEL']['LSTM_HIDDEN']}")
            print(f"   • Hidden Units: {CFG['MODEL']['HIDDEN_UNITS']}")
            print(f"   • Dropout: {CFG['MODEL']['DROPOUT']}")
        elif CFG['MODEL']['TYPE'] == 'tabular_transformer':
            print(f"   • Hidden Dim: {CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM']}")
            print(f"   • N Heads: {CFG['MODEL']['TRANSFORMER']['N_HEADS']}")
            print(f"   • N Layers: {CFG['MODEL']['TRANSFORMER']['N_LAYERS']}")
            print(f"   • LSTM Hidden: {CFG['MODEL']['TRANSFORMER']['LSTM_HIDDEN']}")
        
        model = train_model(
            train_df=train_data,
            feature_cols=feature_cols,
            seq_col=seq_col,
            target_col=target_col,
            device=device
        )
        print_step_summary("모델 훈련", {
            "Model Type": CFG['MODEL']['TYPE'],
            "Epochs Completed": CFG['EPOCHS'],
            "Device Used": device
        })
        
        # 6. 임시 웨이트 파일 저장
        print_progress(5, total_steps, "임시 웨이트 파일 저장")
        save_model(model, temp_model_path)
        print_step_summary("웨이트 저장", {"File Path": temp_model_path})
        
        # 7. 모델 정보 수집
        print_progress(6, total_steps, "모델 정보 수집")
        model_info = {
            'model_type': CFG['MODEL']['TYPE'],
            'lstm_hidden': CFG['MODEL']['LSTM_HIDDEN'],
            'hidden_units': CFG['MODEL']['HIDDEN_UNITS'],
            'dropout': CFG['MODEL']['DROPOUT'],
            'epochs': CFG['EPOCHS'],
            'learning_rate': CFG['LEARNING_RATE'],
            'batch_size': CFG['BATCH_SIZE'],
            'train_shape': train_data.shape,
            'test_shape': test_data.shape,
            'num_features': len(feature_cols),
            'early_stopping': CFG['EARLY_STOPPING']['ENABLED'],
            'monitor_metric': CFG['EARLY_STOPPING']['MONITOR'],
            'patience': CFG['EARLY_STOPPING']['PATIENCE']
        }
        print_step_summary("정보 수집", {
            "Model Parameters": len(model_info),
            "Training Data Shape": train_data.shape,
            "Test Data Shape": test_data.shape
        })
        
        # 8. 예측 실행
        print_progress(7, total_steps, "테스트 데이터 예측")
        print(f"🔮 예측 설정:")
        print(f"   • Test Data Shape: {test_data.shape}")
        print(f"   • Features: {len(feature_cols)}")
        print(f"   • Batch Size: {CFG['BATCH_SIZE']}")
        
        submission_df = predict_test_data(
            test_data=test_data,
            feature_cols=feature_cols,
            seq_col=seq_col,
            model_path=temp_model_path,
            device=device
        )
        print_step_summary("예측 완료", {
            "Submission Shape": submission_df.shape,
            "Mean Prediction": f"{submission_df['clicked'].mean():.4f}",
            "Min Prediction": f"{submission_df['clicked'].min():.4f}",
            "Max Prediction": f"{submission_df['clicked'].max():.4f}"
        })
        
        # 9. 결과 저장
        print_progress(8, total_steps, "결과 및 메타데이터 저장")
        submission_path, metadata_path = save_results_with_metadata(
            results_dir, submission_df, model_info
        )
        print_step_summary("결과 저장", {
            "Submission File": submission_path,
            "Metadata File": metadata_path,
            "Results Directory": results_dir
        })
        
        # 10. 워크플로우 완료 요약
        print_progress(9, total_steps, "워크플로우 완료 요약")
        print("\n✅ 모든 단계 완료!")
        print(f"📁 결과 디렉토리: {results_dir}")
        print(f"📊 예측 결과: {submission_path}")
        print(f"📋 메타데이터: {metadata_path}")
        print(f"⏱️  총 소요 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        print(f"🔍 상세 오류:")
        traceback.print_exc()
        raise
    
    finally:
        # 11. 임시 웨이트 파일 삭제
        print_progress(10, total_steps, "정리 작업")
        cleanup_temp_files(temp_model_path)
    
    print("\n🎉 모든 작업 완료!")
    print("=" * 80)
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
