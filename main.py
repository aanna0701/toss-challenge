from utils import seed_everything, load_config, print_config, print_valid_keys, get_device

# Default Configuration
CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'VAL_SPLIT': 0.2,
    'MODEL': {
        'TYPE': 'tabular_seq',
        'LSTM_HIDDEN': 64,
        'HIDDEN_UNITS': [256, 128],
        'DROPOUT': 0.2
    },
    'PATHS': {
        'MODEL_SAVE': 'trained_model.pth',
        'SUBMISSION': 'baseline_submit.csv',
        'TRAIN_DATA': './train.parquet',
        'TEST_DATA': './test.parquet',
        'SAMPLE_SUBMISSION': './sample_submission.csv'
    }
}

device = get_device()

def load_project_config(config_path="config.yaml"):
    """프로젝트 설정을 로드하는 함수"""
    global CFG
    CFG = load_config(config_path, CFG)
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
