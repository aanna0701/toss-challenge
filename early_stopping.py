"""
Early Stopping 구현
"""

import copy
import torch


class EarlyStopping:
    """Early Stopping 클래스"""
    
    def __init__(self, patience=5, min_delta=0.001, monitor='val_loss', mode='min', restore_best_weights=True):
        """
        Early Stopping 초기화
        
        Args:
            patience (int): 개선이 없는 에포크 수
            min_delta (float): 개선으로 인정할 최소 변화량
            monitor (str): 모니터링할 메트릭 이름
            mode (str): 'min' 또는 'max' - 모니터링 메트릭의 방향
            restore_best_weights (bool): 최고 성능 가중치 복원 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # 초기화
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        
        # 모드에 따른 비교 함수 설정
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score_init = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score_init = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        self.best_score = self.best_score_init
    
    def __call__(self, current_score, model):
        """
        Early stopping 체크
        
        Args:
            current_score (float): 현재 에포크의 모니터링 메트릭 값
            model (torch.nn.Module): 모델 객체
            
        Returns:
            bool: True면 훈련 중단, False면 계속 훈련
        """
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        # 개선 여부 체크
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            print(f"🎯 Early Stopping: {self.monitor} 개선됨 ({self.best_score:.6f})")
        else:
            self.wait += 1
            print(f"⏳ Early Stopping: {self.monitor} 개선 없음 ({self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                print(f"🛑 Early Stopping: 훈련 중단 (patience={self.patience} 도달)")
                
                # 최고 성능 가중치 복원
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"🔄 최고 성능 가중치 복원됨 ({self.monitor}={self.best_score:.6f})")
                
                return True
        
        return False
    
    def get_best_score(self):
        """최고 성능 점수 반환"""
        return self.best_score
    
    def get_best_weights(self):
        """최고 성능 가중치 반환"""
        return self.best_weights
    
    def reset(self):
        """Early stopping 상태 초기화"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = self.best_score_init
        self.best_weights = None
        print("🔄 Early Stopping 상태 초기화됨")


def create_early_stopping_from_config(config):
    """Config에서 Early Stopping 객체 생성"""
    if not config['EARLY_STOPPING']['ENABLED']:
        return None
    
    es_config = config['EARLY_STOPPING']
    return EarlyStopping(
        patience=es_config['PATIENCE'],
        min_delta=es_config['MIN_DELTA'],
        monitor=es_config['MONITOR'],
        mode=es_config['MODE'],
        restore_best_weights=es_config['RESTORE_BEST_WEIGHTS']
    )
