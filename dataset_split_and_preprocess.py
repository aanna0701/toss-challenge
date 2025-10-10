#!/usr/bin/env python3
"""
데이터셋 분할 및 NVTabular 전처리 통합 스크립트

Pipeline:
1. train.parquet를 train_t(80%), train_v(10%), train_c(10%)로 분할
2. train_t에서 10% stratified sampling하여 train_hpo 생성
3. ⚠️  train_t로만 NVTabular workflow fit (Data Leakage 방지!)
4. 나머지(val/cal/hpo/test)는 train_t 통계로 transform만 수행
5. l_feat_20, l_feat_23 제거 (상수 피처), seq는 유지
6. 전처리 결과 저장:
   - data/proc_train_t, data/proc_train_v, data/proc_train_c, 
   - data/proc_train_hpo, data/proc_test
7. 임시 파일 자동 정리 (data/tmp 삭제)

제외 컬럼:
- l_feat_20, l_feat_23: 상수 피처 (정보 없음)

유지 컬럼:
- seq: 시퀀스 데이터 (DNN용, GBDT는 로더에서 제거)

Features:
- 3 categorical features (gender, age_group, inventory_id)
  → Categorify 적용 (결측치 자동 처리, 인덱스 인코딩)
- 112 continuous features (l_feat_14 포함)
  → Normalize(mean=0, std=1) + FillMissing(0) 적용 (순서 중요!)
- seq: string (LSTM 입력)
- Total: 3 categorical + 112 continuous + seq + clicked (target)

전처리 상세:
- Categorical: Categorify (결측치 자동 처리, 0-based 인덱스 인코딩)
- Continuous: 
  1. Normalize: Standardization (mean=0, std=1) 먼저 수행
     - 결측치 없는 실제 데이터로 mean/std 계산
     - ⚠️  train_t로만 통계 계산 (Data Leakage 방지!)
     - 모든 split에 train_t 통계 적용
  2. FillMissing(0): 표준화 후 결측치를 0으로 대체
     - 표준화 공간에서 0 = 원래 평균값
     - 실질적으로 평균값 imputation 효과
- seq: 결측치 처리만 적용 (빈 문자열/NaN → '0.0')

생성 디렉토리:
- data/proc_train_t/    (80%, 전처리 완료, seq 포함)
- data/proc_train_v/    (10%, 전처리 완료, seq 포함)
- data/proc_train_c/    (10%, 전처리 완료, seq 포함)
- data/proc_train_hpo/  (~10%, HPO용 샘플, seq 포함)
- data/proc_test/       (test, 전처리 완료, seq 포함)
- data/tmp/             (임시, 자동 삭제)

사용 방법:
- GBDT: 로더에서 seq 제거 후 사용
- DNN: seq 포함 그대로 사용

장점:
- ✅ 공통 전처리 데이터 1벌로 GBDT/DNN 모두 사용
- ✅ Data Leakage 방지 (train_t로만 통계 계산)
- ✅ 모든 split이 train_t 통계로 일관성 유지
- ✅ Test도 미리 전처리되어 prediction 빠름
- ✅ HPO용 작은 dataset으로 빠른 실험
- ✅ Stratified sampling으로 분포 유지
- ✅ 상수 피처 자동 제거
- ✅ 임시 파일 자동 정리
- ✅ 결측치 처리 및 Standardization 자동 적용
- ✅ 디스크 절약 (중복 데이터 없음)
"""

import os
import gc
import shutil
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

def fill_missing_seq(df, seq_col='seq', fill_value='0.0'):
    """
    seq 컬럼의 결측치 처리
    
    seq는 "1.0,2.0,3.0" 형태의 문자열
    빈 문자열, NaN, None 등을 fill_value로 대체
    """
    print(f"\n  🔧 seq 컬럼 결측치 처리 중...")
    
    def clean_seq(seq_str):
        if seq_str is None or str(seq_str).strip() == '' or str(seq_str) == 'nan':
            return fill_value
        return str(seq_str)
    
    missing_count = df[seq_col].isna().sum() + (df[seq_col] == '').sum()
    df[seq_col] = df[seq_col].apply(clean_seq)
    
    print(f"  ✅ seq 결측치 처리 완료 (결측치 {missing_count}개 → '{fill_value}'로 대체)")
    
    return df


def create_workflow_common():
    """Create NVTabular workflow (seq 제외 - cudf string limit 회피)"""
    import nvtabular as nvt
    from nvtabular import ops
    
    print("\n🔧 Creating common workflow (seq 제외)...")

    # CONTINUOUS COLUMNS (112 total, l_feat_20, l_feat_23 제외)
    # l_feat_14는 continuous (float 값 존재 확인됨)
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +   # 18
        [f'feat_b_{i}' for i in range(1, 7)] +    # 6
        [f'feat_c_{i}' for i in range(1, 9)] +    # 8
        [f'feat_d_{i}' for i in range(1, 7)] +    # 6
        [f'feat_e_{i}' for i in range(1, 11)] +   # 10
        [f'history_a_{i}' for i in range(1, 8)] +   # 7
        [f'history_b_{i}' for i in range(1, 31)] +  # 30
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]  # 25 (l_feat_14 포함!)
    )
    
    # CATEGORICAL COLUMNS (Categorify 적용 - 결측치 자동 처리, 인덱스 인코딩)
    # l_feat_14는 제외 (float 값 존재하여 continuous로 처리)
    categorical_cols = ['gender', 'age_group', 'inventory_id']
    
    print(f"   Categorical: {len(categorical_cols)} columns")
    print(f"   Continuous: {len(all_continuous)} columns")
    print(f"   ⚠️  seq: workflow 제외 (cudf string limit 회피, 나중에 병합)")

    # Preprocessing pipeline:
    # 1. Categorical: Categorify 적용 (결측치 자동 처리, 인덱스 인코딩)
    cat_features = categorical_cols >> ops.Categorify(
        freq_threshold=0,
        max_size=50000
    )
    
    # 2. Normalize + FillMissing for continuous features
    cont_features = all_continuous >> ops.Normalize() >> ops.FillMissing(fill_val=0)

    # 3. _row_id는 그대로 유지 (순서 보장용)
    workflow = nvt.Workflow(cat_features + cont_features + ['clicked', '_row_id'])

    print("   ✅ Workflow created (seq 제외):")
    print("      - Categorical: Categorify (결측치 자동 처리, 인덱스 인코딩)")
    print("      - Continuous: Normalize + FillMissing(0)")
    print("      - _row_id: passthrough (순서 보장)")
    print("      - seq: transform 후 원본 병합")
    return workflow


def process_all_data(train_ratio=0.8, val_ratio=0.1, cal_ratio=0.1, 
                     hpo_ratio=0.1, random_state=42):
    """
    전체 데이터 분할 및 전처리
    
    Args:
        train_ratio: Training 데이터 비율 (0.8 = 80%)
        val_ratio: Validation 데이터 비율 (0.1 = 10%)
        cal_ratio: Calibration 데이터 비율 (0.1 = 10%)
        hpo_ratio: HPO용 샘플링 비율 (0.1 = train의 10%)
        random_state: 랜덤 시드
    """
    print("=" * 80)
    print("🚀 데이터셋 분할 및 NVTabular 전처리 시작")
    print("=" * 80)
    print(f"📅 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 분할 비율: train={train_ratio}, val={val_ratio}, cal={cal_ratio}")
    print(f"🔬 HPO 샘플링 비율: {hpo_ratio} (from train)")
    
    # 비율 검증
    assert abs(train_ratio + val_ratio + cal_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
    
    # 제외할 컬럼 목록 (상수 피처만 제외, seq는 유지)
    EXCLUDE_COLS = ['l_feat_20', 'l_feat_23', '']
    print(f"🗑️  제외 컬럼: {', '.join([c for c in EXCLUDE_COLS if c])}")
    print(f"✅ seq 유지 (DNN용, GBDT는 로더에서 제거)")
    
    # 출력 디렉토리 (공통 전처리 데이터, seq 포함, continuous standardization)
    output_dirs = {
        # 공통 전처리 데이터 (seq 포함, continuous standardization 적용)
        # GBDT: 로더에서 seq 제거 후 사용
        # DNN: seq 포함 그대로 사용 (categorical은 자체 LabelEncoder)
        'train_t': 'data/proc_train_t',
        'train_v': 'data/proc_train_v',
        'train_c': 'data/proc_train_c',
        'train_hpo': 'data/proc_train_hpo',
        'test': 'data/proc_test',
        'temp': 'data/tmp'
    }
    
    # 기존 디렉토리 확인
    existing = [d for d in output_dirs.values() if os.path.exists(d)]
    if existing:
        print("\n⚠️  기존 전처리 디렉토리가 존재합니다:")
        for d in existing:
            print(f"  - {d}")
        response = input("\n🔄 기존 디렉토리를 삭제하고 다시 처리하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("❌ 작업을 취소했습니다.")
            return
        print("\n🗑️  기존 디렉토리 삭제 중...")
        for d in existing:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"  ✅ 삭제: {d}")
    
    # 임시 디렉토리 생성
    os.makedirs(output_dirs['temp'], exist_ok=True)
    print(f"\n📁 임시 디렉토리 생성: {output_dirs['temp']}")
    
    # =================================================================
    # Step 1: 전체 train.parquet 로드 및 분할
    # =================================================================
    print("\n" + "=" * 80)
    print("📂 Step 1: 전체 train.parquet 로드 및 분할")
    print("=" * 80)
    
    train_path = 'data/train.parquet'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} 파일이 없습니다!")
    
    print(f"\n📖 로딩: {train_path}")
    df_full = pd.read_parquet(train_path)
    total_rows = len(df_full)
    print(f"  ✅ 로드 완료: {total_rows:,} rows")
    print(f"  📊 Positive ratio: {df_full['clicked'].mean():.6f}")
    
    # 데이터 섞기
    print("\n🔀 데이터 셔플링...")
    df_full = df_full.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 분할 크기 계산
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)
    cal_size = total_rows - train_size - val_size
    
    print(f"\n✂️  분할 크기:")
    print(f"  - train_t: {train_size:,} ({train_size/total_rows:.1%})")
    print(f"  - train_v: {val_size:,} ({val_size/total_rows:.1%})")
    print(f"  - train_c: {cal_size:,} ({cal_size/total_rows:.1%})")
    
    # 분할
    df_train_t = df_full.iloc[:train_size].copy()
    df_train_v = df_full.iloc[train_size:train_size+val_size].copy()
    df_train_c = df_full.iloc[train_size+val_size:].copy()
    
    # 분포 확인
    print(f"\n  📊 분할 후 분포:")
    print(f"    train_t positive ratio: {df_train_t['clicked'].mean():.6f}")
    print(f"    train_v positive ratio: {df_train_v['clicked'].mean():.6f}")
    print(f"    train_c positive ratio: {df_train_c['clicked'].mean():.6f}")
    
    # train_hpo 생성 (train_t에서 stratified sampling)
    print(f"\n🔬 train_hpo 생성 (train_t의 {hpo_ratio:.1%} stratified sampling)...")
    _, df_train_hpo = train_test_split(
        df_train_t,
        test_size=hpo_ratio,
        random_state=random_state,
        stratify=df_train_t['clicked']
    )
    df_train_hpo = df_train_hpo.reset_index(drop=True)
    print(f"  ✅ train_hpo: {len(df_train_hpo):,} rows")
    print(f"  📊 train_hpo positive ratio: {df_train_hpo['clicked'].mean():.6f}")
    
    # =================================================================
    # Step 2: 불필요한 컬럼 제거 및 임시 저장 (seq는 유지!)
    # =================================================================
    print("\n" + "=" * 80)
    print("🗑️  Step 2: 불필요한 컬럼 제거 및 임시 저장")
    print("=" * 80)
    print("   제외: l_feat_20, l_feat_23 (상수 피처)")
    print("   유지: seq (DNN용, GBDT는 로더에서 제거)")
    
    splits = {
        'train_t': df_train_t,  # workflow fit용 (통계 계산)
        'train_v': df_train_v,
        'train_c': df_train_c,
        'train_hpo': df_train_hpo
    }
    
    temp_files = {}
    seq_files = {}  # seq를 별도로 저장
    
    for name, df in splits.items():
        # 불필요한 컬럼 제거
        cols_to_keep = [c for c in df.columns if c not in EXCLUDE_COLS]
        df_clean = df[cols_to_keep].copy()
        
        # ✅ 순서 보장을 위해 임시 row_id 추가
        df_clean['_row_id'] = range(len(df_clean))
        
        # seq 결측치 처리
        if 'seq' in df_clean.columns:
            df_clean = fill_missing_seq(df_clean, seq_col='seq', fill_value='0.0')
            
            # seq를 row_id와 함께 별도 파일로 저장
            seq_path = os.path.join(output_dirs['temp'], f'{name}_seq.parquet')
            df_clean[['_row_id', 'seq']].to_parquet(seq_path, index=False)
            seq_files[name] = seq_path
            
            # workflow용 데이터에서 seq 제거 (row_id는 유지)
            df_no_seq = df_clean.drop('seq', axis=1)
        else:
            df_no_seq = df_clean
        
        # 임시 파일로 저장 (seq 제외, row_id 포함)
        temp_path = os.path.join(output_dirs['temp'], f'{name}.parquet')
        df_no_seq.to_parquet(temp_path, index=False)
        temp_files[name] = temp_path
        print(f"  ✅ {name}: {len(df_no_seq):,} rows, {len(df_no_seq.columns)} cols (seq 별도 저장, row_id 추가)")
        
        del df_clean, df_no_seq
    
    # 메모리 정리 (df_full은 이미 사용 완료)
    del df_full, df_train_t, df_train_v, df_train_c, df_train_hpo, splits
    gc.collect()
    print(f"\n  🧹 메모리 정리 완료")
    
    # =================================================================
    # Step 3: NVTabular Workflow Fit (train_t만 사용)
    # =================================================================
    print("\n" + "=" * 80)
    print("🔧 Step 3: NVTabular Workflow Fit (train_t ONLY - Data Leakage 방지)")
    print("=" * 80)
    print("   ✅ train_t로만 통계 계산 (val/cal은 transform만)")
    print("   ✅ seq 포함, continuous만 standardization")
    print("   ✅ categorical은 raw 유지 (DNN 자체 LabelEncoder 사용)")
    
    # GPU 메모리 관리 초기화
    try:
        import cupy as cp
        import rmm
        
        # RMM 메모리 풀 설정 (더 큰 풀)
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=8 * 1024**3,  # 8GB
            managed_memory=True,
        )
        cp.cuda.Device(0).use()
        print("✅ GPU 활성화 (RMM pool: 8GB)")
    except Exception as e:
        print(f"⚠️  GPU 초기화 경고: {e}")
    
    from merlin.io import Dataset
    
    # train_t로만 workflow fit (data leakage 방지)
    print(f"\n📊 Workflow fitting on train_t ONLY ({train_size:,} rows)...")
    print("   ⚡ Part size: 128MB (메모리 효율)")
    print("   ⚠️  중요: train_t로만 통계 계산 (val/cal/test 정보 누출 방지)")
    
    train_dataset = Dataset(temp_files['train_t'], engine='parquet', part_size='128MB')
    
    workflow = create_workflow_common()
    
    # Fit with memory cleanup
    print("   🔧 Fitting workflow on train_t (seq 제외)...")
    workflow.fit(train_dataset)
    print("  ✅ Workflow fitted on train_t only (val/cal/test는 transform만)")
    
    # Workflow는 메모리에만 유지 (저장 안 함)
    
    del train_dataset
    gc.collect()
    
    # GPU 메모리 정리
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("  ✅ GPU 메모리 정리 완료")
    except:
        pass
    
    # =================================================================
    # Step 4: Transform 각 split + seq 병합
    # =================================================================
    print("\n" + "=" * 80)
    print("⚙️  Step 4: Transform 각 split + seq 병합")
    print("=" * 80)
    
    # Transform 순서: train_t, train_v, train_c, train_hpo
    splits_to_transform = ['train_t', 'train_v', 'train_c', 'train_hpo']
    
    for split_name in splits_to_transform:
        print(f"\n🔄 Processing {split_name}...")
        
        # Dataset 생성 (큰 파티션으로 메모리 효율 개선)
        dataset = Dataset(temp_files[split_name], engine='parquet', part_size='128MB')
        
        # Transform (seq 제외)
        temp_output_dir = os.path.join(output_dirs['temp'], f'{split_name}_transformed')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        workflow.transform(dataset).to_parquet(
            output_path=temp_output_dir,
            shuffle=None,
            out_files_per_proc=4
        )
        print(f"  ✅ {split_name} transformed (seq 제외)")
        
        del dataset
        gc.collect()
        
        # GPU 메모리 정리
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        
        # seq 병합 (pandas로 처리 - row_id로 순서 보장)
        print(f"  🔗 Merging seq back (using row_id for order preservation)...")
        from merlin.io import Dataset as MerlinDataset
        
        # Transform된 데이터 로드
        dataset_transformed = MerlinDataset(temp_output_dir, engine='parquet')
        df_transformed = dataset_transformed.to_ddf().compute().to_pandas()
        
        # 원본 seq 로드 및 병합
        if split_name in seq_files:
            df_seq = pd.read_parquet(seq_files[split_name])
            
            # row_id로 정렬 (순서 보장)
            df_transformed = df_transformed.sort_values('_row_id').reset_index(drop=True)
            df_seq = df_seq.sort_values('_row_id').reset_index(drop=True)
            
            # 순서 검증
            assert len(df_transformed) == len(df_seq), f"Length mismatch: {len(df_transformed)} vs {len(df_seq)}"
            assert (df_transformed['_row_id'].values == df_seq['_row_id'].values).all(), "row_id mismatch!"
            
            # seq 병합 (row_id 기준으로 정렬되었으므로 안전)
            df_final = df_transformed.copy()
            df_final['seq'] = df_seq['seq'].values  # numpy array로 직접 할당 (순서 보장)
            
            # row_id 제거
            df_final = df_final.drop('_row_id', axis=1)
            
            print(f"  ✅ seq merged: {len(df_final)} rows, {len(df_final.columns)} cols (row_id removed)")
        else:
            df_final = df_transformed.drop('_row_id', axis=1) if '_row_id' in df_transformed.columns else df_transformed
            print(f"  ⚠️  No seq found for {split_name}")
        
        # 최종 저장
        output_dir = output_dirs[split_name]
        os.makedirs(output_dir, exist_ok=True)
        df_final.to_parquet(
            os.path.join(output_dir, '0.parquet'),
            index=False,
            engine='pyarrow'
        )
        print(f"  ✅ Saved to {output_dir}")
        
        # 정리
        del df_transformed, df_final
        if split_name in seq_files:
            del df_seq
        gc.collect()
        
        # 임시 변환 디렉토리 삭제
        shutil.rmtree(temp_output_dir)
        print(f"  🧹 Cleaned temp: {temp_output_dir}")
    
    # =================================================================
    # Step 5: Test 데이터 전처리
    # =================================================================
    print("\n" + "=" * 80)
    print("🧪 Step 5: Test 데이터 전처리")
    print("=" * 80)
    
    test_path = 'data/test.parquet'
    if os.path.exists(test_path):
        print(f"\n📖 로딩: {test_path}")
        df_test = pd.read_parquet(test_path)
        print(f"  ✅ 로드 완료: {len(df_test):,} rows")
        
        # 불필요한 컬럼 제거 (l_feat_20, l_feat_23만 제거, seq는 유지)
        cols_to_keep = [c for c in df_test.columns if c not in EXCLUDE_COLS]
        df_test_clean = df_test[cols_to_keep].copy()
        print(f"  🗑️  제외 컬럼: {[c for c in EXCLUDE_COLS if c in df_test.columns]}")
        
        # ✅ 순서 보장을 위해 임시 row_id 추가
        df_test_clean['_row_id'] = range(len(df_test_clean))
        
        # seq 결측치 처리
        if 'seq' in df_test_clean.columns:
            df_test_clean = fill_missing_seq(df_test_clean, seq_col='seq', fill_value='0.0')
            
            # seq를 row_id와 함께 별도 파일로 저장
            test_seq_path = os.path.join(output_dirs['temp'], 'test_seq.parquet')
            df_test_clean[['_row_id', 'seq']].to_parquet(test_seq_path, index=False)
            
            # workflow용 데이터에서 seq 제거 (row_id는 유지)
            df_test_no_seq = df_test_clean.drop('seq', axis=1)
        else:
            df_test_no_seq = df_test_clean
            test_seq_path = None
        
        # clicked 컬럼 추가 (dummy, NVTabular workflow 호환용)
        if 'clicked' not in df_test_no_seq.columns:
            df_test_no_seq['clicked'] = 0
            print("  ⚠️  Test에 'clicked' 컬럼 추가 (dummy)")
        
        # 임시 저장 (seq 제외, row_id 포함)
        test_temp_path = os.path.join(output_dirs['temp'], 'test.parquet')
        df_test_no_seq.to_parquet(test_temp_path, index=False)
        print(f"  ✅ Test 준비 완료 (seq 별도 저장, row_id 추가)")
        
        del df_test, df_test_clean, df_test_no_seq
        gc.collect()
        
        # Transform (seq 제외)
        print(f"\n🔄 Processing test (seq 제외)...")
        test_dataset = Dataset(test_temp_path, engine='parquet', part_size='128MB')
        
        temp_test_output_dir = os.path.join(output_dirs['temp'], 'test_transformed')
        os.makedirs(temp_test_output_dir, exist_ok=True)
        
        workflow.transform(test_dataset).to_parquet(
            output_path=temp_test_output_dir,
            shuffle=None,
            out_files_per_proc=4
        )
        print(f"  ✅ test transformed (seq 제외)")
        
        del test_dataset
        gc.collect()
        
        # GPU 메모리 정리
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass
        
        # seq 병합 (pandas로 처리 - row_id로 순서 보장)
        if test_seq_path:
            print(f"  🔗 Merging seq back to test...")
            from merlin.io import Dataset as MerlinDataset
            
            # Transform된 데이터 로드
            dataset_transformed = MerlinDataset(temp_test_output_dir, engine='parquet')
            df_test_transformed = dataset_transformed.to_ddf().compute().to_pandas()
            
            # 원본 seq 로드
            df_test_seq = pd.read_parquet(test_seq_path)
            
            # row_id로 정렬 (순서 보장)
            df_test_transformed = df_test_transformed.sort_values('_row_id').reset_index(drop=True)
            df_test_seq = df_test_seq.sort_values('_row_id').reset_index(drop=True)
            
            # 순서 검증
            assert len(df_test_transformed) == len(df_test_seq), f"Length mismatch: {len(df_test_transformed)} vs {len(df_test_seq)}"
            assert (df_test_transformed['_row_id'].values == df_test_seq['_row_id'].values).all(), "row_id mismatch!"
            
            # seq 병합
            df_test_final = df_test_transformed.copy()
            df_test_final['seq'] = df_test_seq['seq'].values
            
            # row_id 제거
            df_test_final = df_test_final.drop('_row_id', axis=1)
            
            print(f"  ✅ seq merged: {len(df_test_final)} rows, {len(df_test_final.columns)} cols")
        else:
            # seq가 없는 경우
            from merlin.io import Dataset as MerlinDataset
            dataset_transformed = MerlinDataset(temp_test_output_dir, engine='parquet')
            df_test_final = dataset_transformed.to_ddf().compute().to_pandas()
            df_test_final = df_test_final.drop('_row_id', axis=1) if '_row_id' in df_test_final.columns else df_test_final
        
        # 최종 저장
        output_dir = output_dirs['test']
        os.makedirs(output_dir, exist_ok=True)
        df_test_final.to_parquet(
            os.path.join(output_dir, '0.parquet'),
            index=False,
            engine='pyarrow'
        )
        print(f"  ✅ Test saved to {output_dir}")
        print(f"     (GBDT: seq 제거해서 사용, DNN: seq 포함 사용)")
        
        # 정리
        del df_test_final
        if test_seq_path:
            del df_test_transformed, df_test_seq
        gc.collect()
        
        # 임시 변환 디렉토리 삭제
        shutil.rmtree(temp_test_output_dir)
        print(f"  🧹 Cleaned temp: {temp_test_output_dir}")
    else:
        print(f"\n⚠️  {test_path} 파일이 없습니다. Test 전처리를 건너뜁니다.")
    
    # =================================================================
    # Step 6: 임시 파일 정리
    # =================================================================
    print("\n" + "=" * 80)
    print("🧹 Step 6: 임시 파일 정리")
    print("=" * 80)
    
    if os.path.exists(output_dirs['temp']):
        # 임시 파일 개수 확인
        temp_files_count = len([f for f in os.listdir(output_dirs['temp']) if os.path.isfile(os.path.join(output_dirs['temp'], f))])
        print(f"  🗑️  삭제할 임시 파일: {temp_files_count}개")
        
        shutil.rmtree(output_dirs['temp'])
        print(f"  ✅ 임시 디렉토리 삭제 완료: {output_dirs['temp']}")
    
    # =================================================================
    # 최종 요약
    # =================================================================
    print("\n" + "=" * 80)
    print("✅ 전체 전처리 완료!")
    print("=" * 80)
    print(f"📅 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 생성된 전처리 디렉토리 (공통):")
    print("  🔷 공통 전처리 데이터 (seq 포함, continuous standardization):")
    for key in ['train_t', 'train_v', 'train_c', 'train_hpo', 'test']:
        path = output_dirs.get(key)
        if path and os.path.exists(path):
            print(f"    ✅ {key:12} → {path}")
    
    print("\n📊 전처리 정보:")
    print(f"  - seq 포함 (DNN용, GBDT는 로더에서 제거)")
    print(f"    * seq 결측치 처리: 빈 문자열/NaN → '0.0'")
    print(f"  - l_feat_20, l_feat_23 제거 (상수 피처)")
    print(f"  ⚠️  중요: train_t로만 workflow fit (Data Leakage 방지!)")
    print(f"    * val/cal/test는 train_t 통계로 transform만 수행")
    print(f"  - Categorical: 3개 (gender, age_group, inventory_id)")
    print(f"    * Categorify: 결측치 자동 처리, 0-based 인덱스")
    print(f"  - Continuous: 112개 (l_feat_14 포함)")
    print(f"    * Normalize → FillMissing(0)")
    print(f"    * Normalize 먼저: train_t의 실제 분포로 mean/std 계산")
    print(f"    * FillMissing(0) 나중: 표준화 공간에서 평균값으로 imputation")
    print(f"  - Normalization: Standardization (mean=0, std=1)")
    print(f"  - 모든 split에 train_t 통계 적용 (일관성 보장)")
    
    print("\n💡 사용 방법:")
    print("\n  [GBDT 모델 - train_gbdt.py, hpo_xgboost.py]")
    print("  from data_loader import load_processed_data_gbdt")
    print("  X, y = load_processed_data_gbdt('data/proc_train_t', drop_seq=True)")
    print("  # ✅ seq 제거")
    print("  # ✅ categorical은 이미 인덱스 인코딩됨 (0-based)")
    print("  # ✅ continuous는 이미 standardization 적용됨 (mean=0, std=1)")
    print()
    print("  [DNN 모델 - train_dnn_ddp.py, hpo_dnn.py]")
    print("  from merlin.io import Dataset")
    print("  dataset = Dataset('data/proc_train_t', engine='parquet')")
    print("  gdf = dataset.to_ddf().compute()")
    print("  df = gdf.to_pandas()  # seq 포함")
    print("  # ✅ l_feat_20, l_feat_23 이미 제거됨")
    print("  # ✅ seq 결측치 이미 처리됨 (빈값 → '0.0')")
    print("  # ✅ categorical은 이미 인덱스 인코딩됨 (0-based)")
    print("  # ✅ continuous는 이미 standardization 적용됨")
    
    print("\n" + "=" * 80)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='데이터셋 분할 및 NVTabular 전처리',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 실행 (0.8/0.1/0.1 분할, HPO 10%)
  python dataset_split_and_preprocess.py
  
  # 커스텀 비율
  python dataset_split_and_preprocess.py --train-ratio 0.85 --val-ratio 0.075 --cal-ratio 0.075
  
  # HPO 샘플링 비율 변경
  python dataset_split_and_preprocess.py --hpo-ratio 0.15
        """
    )
    
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training 데이터 비율 (기본: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation 데이터 비율 (기본: 0.1)')
    parser.add_argument('--cal-ratio', type=float, default=0.1,
                        help='Calibration 데이터 비율 (기본: 0.1)')
    parser.add_argument('--hpo-ratio', type=float, default=0.1,
                        help='HPO 샘플링 비율 (train의 %, 기본: 0.1)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    
    args = parser.parse_args()
    
    try:
        process_all_data(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            cal_ratio=args.cal_ratio,
            hpo_ratio=args.hpo_ratio,
            random_state=args.random_state
        )
        
        print("\n🎉 모든 작업 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

