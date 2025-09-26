import os
import random
import numpy as np
import torch
import yaml
from datetime import datetime

def seed_everything(seed):
    """시드 고정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="config.yaml", base_config=None):
    """YAML 설정 파일을 읽어서 기본 설정을 업데이트하는 함수"""
    if base_config is None:
        raise ValueError("base_config must be provided")
    
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 딕셔너리 업데이트 (재귀적으로 중첩된 딕셔너리도 처리)
        # YAML에 없는 키가 들어오면 에러 발생
        def update_dict(base_dict, new_dict, path=""):
            for key, value in new_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in base_dict:
                    raise KeyError(f"Unknown configuration key: '{current_path}'. Valid keys are: {list(base_dict.keys())}")
                
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict(base_dict[key], value, current_path)
                else:
                    base_dict[key] = value
        
        try:
            update_dict(base_config, yaml_config)
            print("Configuration updated from YAML file")
        except KeyError as e:
            print(f"Configuration Error: {e}")
            print("Please check your YAML file and ensure all keys are valid.")
            raise
    else:
        print(f"Configuration file {config_path} not found. Using default configuration.")
    
    return base_config

def print_config(config, indent=0):
    """설정을 예쁘게 출력하는 함수"""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

def get_valid_keys(config, prefix=""):
    """유효한 설정 키들을 모두 가져오는 함수"""
    keys = []
    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else key
        keys.append(current_key)
        if isinstance(value, dict):
            keys.extend(get_valid_keys(value, current_key))
    return keys

def print_valid_keys(config):
    """유효한 설정 키들을 출력하는 함수"""
    print("Valid configuration keys:")
    valid_keys = get_valid_keys(config)
    for key in sorted(valid_keys):
        print(f"  - {key}")

def get_device():
    """사용 가능한 디바이스 반환"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def apply_datetime_to_paths(config):
    """설정에서 {datetime} 플레이스홀더를 실제 datetime으로 교체하는 함수"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def replace_datetime_in_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                replace_datetime_in_dict(value)
            elif isinstance(value, str) and "{datetime}" in value:
                d[key] = value.replace("{datetime}", current_time)
    
    replace_datetime_in_dict(config)
    return config
