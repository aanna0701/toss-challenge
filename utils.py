import os
import random
import gc
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

def seed_everything(seed):
    """시드 고정 함수"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path, base_config=None):
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


# ============================================================================
# GBDT 관련 공통 함수들
# ============================================================================

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """Calculate Weighted LogLoss with 50:50 class weights"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    
    # Additional clipping to prevent log(0) or log(negative) issues
    if mask_0.sum() > 0:
        pred_0 = np.clip(1 - y_pred[mask_0], eps, 1 - eps)
        ll_0 = -np.mean(np.log(pred_0))
    else:
        ll_0 = 0
    
    if mask_1.sum() > 0:
        pred_1 = np.clip(y_pred[mask_1], eps, 1 - eps)
        ll_1 = -np.mean(np.log(pred_1))
    else:
        ll_1 = 0
    
    return 0.5 * ll_0 + 0.5 * ll_1


def calculate_competition_score(y_true, y_pred):
    """Calculate competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll


def clear_gpu_memory():
    """Clear GPU memory with aggressive cleanup"""
    try:
        # Clear CuPy memory pools
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        # Force garbage collection
        gc.collect()
        
        # Try to clear CUDA cache if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        print("🧹 GPU memory cleared")
    except Exception as e:
        print(f"⚠️ Error clearing GPU memory: {e}")
        gc.collect()


def print_memory():
    """Print current memory usage"""
    import psutil
    
    mem = psutil.virtual_memory()
    
    gpu_used = 0
    gpu_total = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        # CUDA_VISIBLE_DEVICES가 설정된 경우 가시 목록 내 0번
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = gpu_info.used / 1024**3
        gpu_total = gpu_info.total / 1024**3
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        print(f"💾 GPU ({gpu_name}): {gpu_used:.1f}GB/{gpu_total:.1f}GB")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"💾 GPU: Error getting GPU info - {e}")
    
    print(f"💾 CPU: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")
    return mem.percent
