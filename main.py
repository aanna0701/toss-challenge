import os
from utils import seed_everything, load_config, print_config, print_valid_keys, get_device, apply_datetime_to_paths

# Default Configuration
CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'VAL_SPLIT': 0.2,
    'DATA': {
        # 전체 데이터 사용 (샘플링 없음)
    },
    'EARLY_STOPPING': {
        'ENABLED': True,
        'PATIENCE': 5,
        'MIN_DELTA': 0.001,
        'MONITOR': 'val_score',
        'MODE': 'max',
        'RESTORE_BEST_WEIGHTS': True
    },
    'METRICS': {
        'SAVE_LOGS': True,
        'LOG_FILE': 'train_logs.csv'
    },
    'GRADIENT_NORM': {
        'ENABLED': True,
        'SAVE_LOGS': True,
        'LOG_FILE': 'gradient_norms.csv',
        'COMPONENTS': ['lstm', 'mlp', 'total']
    },
    'MODEL': {
        'TYPE': 'tabular_seq',
        'LSTM_HIDDEN': 64,
        'HIDDEN_UNITS': [256, 128],
        'DROPOUT': 0.2
    },
    'PATHS': {
        'MODEL_SAVE': 'trained_model.pth',
        'TEMP_MODEL': 'temp_model.pth',
        'SUBMISSION': 'baseline_submit.csv',
        'RESULTS_DIR': 'results',
        'TRAIN_DATA': 'train.parquet',
        'TEST_DATA': 'test.parquet',
        'SAMPLE_SUBMISSION': 'sample_submission.csv'
    }
}

device = get_device()

def load_project_config(config_path="config.yaml"):
    """프로젝트 설정을 로드하는 함수"""
    global CFG
    # 현재 스크립트의 디렉토리 기준으로 상대경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(script_dir, config_path)
    
    CFG = load_config(config_full_path, CFG)
    # datetime 플레이스홀더를 실제 datetime으로 교체
    CFG = apply_datetime_to_paths(CFG)
    return CFG

def initialize(config_path="config.yaml"):
    """초기화 함수"""
    # YAML 설정 파일 로드
    load_project_config(config_path)
    
    # 시드 고정
    seed_everything(CFG['SEED'])
    
    print(f"Device: {device}")
    print("Current Configuration:")
    print_config(CFG)

if __name__ == "__main__":
    # 정상적인 초기화
    initialize()
    print("Main initialization complete!")
    print()
    
    # 유효한 키들 출력
    print_valid_keys(CFG)
