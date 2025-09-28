#!/usr/bin/env python3
"""
훈련 후 자동으로 예측을 실행하고 결과를 날짜별로 저장하는 워크플로우 스크립트
"""

import os
import gc
import psutil
import traceback
import logging
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

def setup_error_logging(results_dir):
    """에러 로깅 설정"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log_path = os.path.join(results_dir, f"error_log_{timestamp}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(error_log_path, encoding='utf-8'),
            logging.StreamHandler()  # 콘솔에도 출력
        ]
    )
    
    print(f"📋 에러 로그 설정 완료: {error_log_path}")
    return error_log_path

def log_error(error, error_log_path, step_info=None):
    """에러를 파일에 로깅"""
    error_info = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'step_info': step_info,
        'memory_usage': f"{get_memory_usage():.1f} MB",
        'traceback': traceback.format_exc()
    }
    
    # 상세 에러 정보를 파일에 저장
    with open(error_log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"ERROR OCCURRED\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {error_info['timestamp']}\n")
        f.write(f"Error Type: {error_info['error_type']}\n")
        f.write(f"Error Message: {error_info['error_message']}\n")
        f.write(f"Memory Usage: {error_info['memory_usage']}\n")
        if step_info:
            f.write(f"Step Info: {step_info}\n")
        f.write(f"\nTraceback:\n{error_info['traceback']}\n")
        f.write(f"{'='*80}\n")
    
    # 로깅 시스템에도 기록
    logging.error(f"Error occurred: {error_info['error_type']} - {error_info['error_message']}")
    
    print(f"📋 에러가 로그 파일에 저장되었습니다: {error_log_path}")

def save_error_summary(results_dir, error_log_path, error_count=1):
    """에러 요약 정보 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"error_summary_{timestamp}.json")
    
    summary = {
        'timestamp': timestamp,
        'total_errors': error_count,
        'error_log_file': os.path.basename(error_log_path),
        'results_directory': results_dir,
        'system_info': {
            'memory_usage': f"{get_memory_usage():.1f} MB",
            'python_version': f"{os.sys.version}",
            'platform': f"{os.sys.platform}"
        }
    }
    
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 에러 요약 저장: {summary_path}")
    return summary_path


def get_memory_usage():
    """현재 메모리 사용량 반환"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def cleanup_memory():
    """메모리 정리 함수"""
    print(f"🧹 메모리 정리 시작 (현재 사용량: {get_memory_usage():.1f} MB)")
    
    # Python 가비지 컬렉션
    collected = gc.collect()
    print(f"   • 가비지 컬렉션: {collected}개 객체 정리")
    
    # PyTorch 캐시 정리
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   • CUDA 캐시 정리 완료")
    
    print(f"🧹 메모리 정리 완료 (현재 사용량: {get_memory_usage():.1f} MB)")

def cleanup_train_data(train_data, test_data=None):
    """훈련 데이터 및 관련 변수 메모리에서 제거"""
    print(f"🗑️  훈련 데이터 메모리에서 제거 시작 (현재 사용량: {get_memory_usage():.1f} MB)")
    
    # 훈련 데이터 삭제
    del train_data
    print(f"   • train_data 변수 삭제")
    
    # 테스트 데이터는 예측에 필요하므로 유지
    if test_data is not None:
        print(f"   • test_data 유지 (예측에 필요)")
    
    # 메모리 정리
    cleanup_memory()

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

def safe_execute_step(step_func, step_name, error_log_path, *args, **kwargs):
    """안전하게 단계를 실행하고 에러 발생 시 로깅"""
    try:
        return step_func(*args, **kwargs)
    except Exception as e:
        step_info = f"단계 실행 중 오류: {step_name}"
        log_error(e, error_log_path, step_info)
        raise


def main():
    """메인 워크플로우 실행"""
    total_steps = 11
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
        "Data Loading": "전체 데이터 사용"
    })
    
    # 2. 결과 디렉토리 생성
    print_progress(2, total_steps, "결과 디렉토리 생성")
    results_dir = create_results_directory()
    
    # 에러 로깅 설정
    error_log_path = setup_error_logging(results_dir)
    
    print_step_summary("디렉토리 생성", {
        "Results Dir": results_dir,
        "Error Log": os.path.basename(error_log_path)
    })
    
    # 3. 임시 웨이트 파일 경로 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_model_path = CFG['PATHS']['TEMP_MODEL'].replace('{datetime}', timestamp)
    print(f"📝 임시 웨이트 파일 경로: {temp_model_path}")
    
    try:
        # 4. 데이터 로드 및 전처리
        print_progress(3, total_steps, "데이터 로드 및 전처리")
        print(f"💾 초기 메모리 사용량: {get_memory_usage():.1f} MB")
        
        # 전체 데이터 로드 (훈련 시 클래스 불균형 해결을 위한 다운샘플링 적용)
        train_data, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
        print(f"💾 데이터 로드 후 메모리 사용량: {get_memory_usage():.1f} MB")
        
        print_step_summary("데이터 로드", {
            "Train Shape": train_data.shape,
            "Test Shape": test_data.shape,
            "Features": len(feature_cols),
            "Sequence Column": seq_col,
            "Target Column": target_col,
            "Test ID Column": "ID" in test_data.columns,
            "Data Loading": "전체 데이터 사용",
            "Class Balancing": "다운샘플링 적용",
            "Memory Usage": f"{get_memory_usage():.1f} MB"
        })
        
        # 5. 모델 훈련
        print_progress(4, total_steps, "모델 훈련")
        print(f"🏋️ 모델 설정:")
        print(f"   • Type: TabularTransformer")
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
            "Model Type": "TabularTransformer",
            "Epochs Completed": CFG['EPOCHS'],
            "Device Used": device,
            "Checkpoint Interval": "Every 5 epochs"
        })
        
        # 6. 임시 웨이트 파일 저장
        print_progress(5, total_steps, "임시 웨이트 파일 저장")
        save_model(model, temp_model_path)
        print_step_summary("웨이트 저장", {"File Path": temp_model_path})
        
        # 6.5. 훈련 데이터 메모리에서 제거
        print_progress(6, total_steps, "훈련 데이터 메모리 정리")
        print(f"💾 훈련 완료 후 메모리 사용량: {get_memory_usage():.1f} MB")
        
        # 훈련에 필요한 정보만 저장
        train_shape = train_data.shape
        model_info_data = {
            'train_shape': train_shape,
            'test_shape': test_data.shape,
            'num_features': len(feature_cols)
        }
        
        # 훈련 데이터 메모리에서 제거
        cleanup_train_data(train_data, test_data)
        print_step_summary("메모리 정리", {
            "Train Data": "메모리에서 제거됨",
            "Test Data": "유지됨 (예측에 필요)",
            "Memory Usage": f"{get_memory_usage():.1f} MB"
        })
        
        # 7. 모델 정보 수집
        print_progress(7, total_steps, "모델 정보 수집")
        model_info = {
            'model_type': 'tabular_transformer',
            'hidden_dim': CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM'],
            'n_heads': CFG['MODEL']['TRANSFORMER']['N_HEADS'],
            'n_layers': CFG['MODEL']['TRANSFORMER']['N_LAYERS'],
            'lstm_hidden': CFG['MODEL']['TRANSFORMER']['LSTM_HIDDEN'],
            'epochs': CFG['EPOCHS'],
            'learning_rate': CFG['LEARNING_RATE'],
            'batch_size': CFG['BATCH_SIZE'],
            'train_shape': model_info_data['train_shape'],
            'test_shape': model_info_data['test_shape'],
            'num_features': model_info_data['num_features'],
            'early_stopping': CFG['EARLY_STOPPING']['ENABLED'],
            'monitor_metric': CFG['EARLY_STOPPING']['MONITOR'],
            'patience': CFG['EARLY_STOPPING']['PATIENCE']
        }
        print_step_summary("정보 수집", {
            "Model Parameters": len(model_info),
            "Training Data Shape": model_info_data['train_shape'],
            "Test Data Shape": model_info_data['test_shape']
        })
        
        # 8. 예측 실행
        print_progress(8, total_steps, "테스트 데이터 예측")
        print(f"🔮 예측 설정:")
        print(f"   • Test Data Shape: {test_data.shape}")
        print(f"   • Features: {len(feature_cols)}")
        print(f"   • Batch Size: {CFG['BATCH_SIZE']}")
        print(f"💾 예측 전 메모리 사용량: {get_memory_usage():.1f} MB")
        
        submission_df = predict_test_data(
            test_data=test_data,
            feature_cols=feature_cols,
            seq_col=seq_col,
            model_path=temp_model_path,
            device=device
        )
        print(f"💾 예측 후 메모리 사용량: {get_memory_usage():.1f} MB")
        
        print_step_summary("예측 완료", {
            "Submission Shape": submission_df.shape,
            "Mean Prediction": f"{submission_df['clicked'].mean():.4f}",
            "Min Prediction": f"{submission_df['clicked'].min():.4f}",
            "Max Prediction": f"{submission_df['clicked'].max():.4f}",
            "Memory Usage": f"{get_memory_usage():.1f} MB"
        })
        
        # 9. 결과 저장
        print_progress(9, total_steps, "결과 및 메타데이터 저장")
        submission_path, metadata_path = save_results_with_metadata(
            results_dir, submission_df, model_info
        )
        print_step_summary("결과 저장", {
            "Submission File": submission_path,
            "Metadata File": metadata_path,
            "Results Directory": results_dir
        })
        
        # 10. 워크플로우 완료 요약
        print_progress(10, total_steps, "워크플로우 완료 요약")
        print(f"💾 최종 메모리 사용량: {get_memory_usage():.1f} MB")
        print("\n✅ 모든 단계 완료!")
        print(f"📁 결과 디렉토리: {results_dir}")
        print(f"📊 예측 결과: {submission_path}")
        print(f"📋 메타데이터: {metadata_path}")
        print(f"⏱️  총 소요 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        
        # 에러 로깅
        step_info = f"워크플로우 실행 중 오류 발생"
        log_error(e, error_log_path, step_info)
        
        # 에러 요약 저장
        save_error_summary(results_dir, error_log_path)
        
        print(f"🔍 상세 오류:")
        traceback.print_exc()
        print(f"\n📋 에러 로그 파일 위치: {error_log_path}")
        raise
    
    finally:
        # 11. 임시 웨이트 파일 삭제
        print_progress(11, total_steps, "정리 작업")
        cleanup_temp_files(temp_model_path)
    
    print("\n🎉 모든 작업 완료!")
    print("=" * 80)
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
