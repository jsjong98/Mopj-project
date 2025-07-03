from flask import Flask, request, jsonify, send_file, make_response, Blueprint
from flask_cors import CORS, cross_origin
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, precision_score, recall_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import os
import json
import warnings
import random
import traceback
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # 서버에서 GUI 백엔드 사용 안 함
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from werkzeug.utils import secure_filename
import io
import base64
import tempfile
import time
from threading import Thread
import logging
import calendar
import shutil
import optuna
import csv
from pathlib import Path
import math
import logging
import glob
import time
import xlwings as xw

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)  # ← 이 줄이 있는지 확인

# VARMAX 관련 import (선택적 가져오기)
try:
    from statsmodels.tsa.statespace.varmax import VARMAX
    VARMAX_AVAILABLE = True
except ImportError:
    VARMAX_AVAILABLE = False
    logger.warning("VARMAX not available. Please install statsmodels.")


# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 랜덤 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # GPU 사용 시
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed=SEED):
    """
    모든 라이브러리의 시드를 고정하여 일관된 예측 결과 보장
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch의 deterministic 동작 강제
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optuna 시드 설정 (하이퍼파라미터 최적화용)
    try:
        import optuna
        # Optuna 2.x 버전 호환
        if hasattr(optuna.samplers, 'RandomSampler'):
            optuna.samplers.RandomSampler(seed=seed)
        # 레거시 지원
        if hasattr(optuna.samplers, '_random'):
            optuna.samplers._random.seed(seed)
    except Exception as e:
        logger.debug(f"Optuna 시드 설정 실패: {e}")
    
    logger.debug(f"🎯 랜덤 시드 {seed}로 고정됨")

# 디렉토리 설정 - 파일별 캐시 시스템
UPLOAD_FOLDER = 'uploads'
HOLIDAY_DIR = 'holidays'
CACHE_ROOT_DIR = 'cache'  # 🔑 새로운 파일별 캐시 루트
PREDICTIONS_DIR = 'predictions'  # 기본 예측 디렉토리 (호환성용)

# 기본 디렉토리 생성 (최소한만 유지)
for d in [UPLOAD_FOLDER, CACHE_ROOT_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

def get_file_cache_dirs(file_path=None):
    """
    파일별 캐시 디렉토리 구조를 반환하는 함수
    🎯 각 파일마다 독립적인 모델, 예측, 시각화 캐시 제공
    """
    try:
        if not file_path:
            file_path = prediction_state.get('current_file', None)
        
        # Debug: file cache directory setup
        
        if not file_path:
            logger.warning(f"⚠️ No file path provided and no current_file in prediction_state")
            # 기본 캐시 디렉토리 반환 (파일별 캐시 없이)
            default_cache_root = Path(CACHE_ROOT_DIR) / 'default'
            dirs = {
                'root': default_cache_root,
                'models': default_cache_root / 'models',
                'predictions': default_cache_root / 'predictions',
                'plots': default_cache_root / 'static' / 'plots',
                'ma_plots': default_cache_root / 'static' / 'ma_plots',
                'accumulated': default_cache_root / 'accumulated'
            }
            
            # Create default directories
            for name, dir_path in dirs.items():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"❌ Failed to create default {name} directory {dir_path}: {str(e)}")
            
            logger.warning(f"⚠️ Using default cache directory")
            return dirs
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File does not exist: {file_path}")
            raise ValueError(f"File does not exist: {file_path}")
        
        # Generate file cache directory
        file_content_hash = get_data_content_hash(file_path)
        
        if not file_content_hash:
            logger.error(f"❌ Failed to get content hash for file: {file_path}")
            raise ValueError(f"Unable to generate content hash for file: {file_path}")
        
        file_name = Path(file_path).stem
        file_dir_name = f"{file_content_hash[:12]}_{file_name}"
        file_cache_root = Path(CACHE_ROOT_DIR) / file_dir_name
        
        dirs = {
            'root': file_cache_root,
            'models': file_cache_root / 'models',
            'predictions': file_cache_root / 'predictions',
            'plots': file_cache_root / 'static' / 'plots',
            'ma_plots': file_cache_root / 'static' / 'ma_plots',
            'accumulated': file_cache_root / 'accumulated'
        }
        
        # Create cache directories
        for name, dir_path in dirs.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Failed to create {name} directory {dir_path}: {str(e)}")
        
        return dirs
        
    except Exception as e:
        logger.error(f"❌ Error in get_file_cache_dirs: {str(e)}")
        logger.error(traceback.format_exc())
        raise e  # 오류 발생 시 예외 전파

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 더 간결한 로그 포맷
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """GPU 사용 가능성 및 현재 디바이스 정보를 확인하고 로깅하는 함수"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 GPU 및 디바이스 정보 확인")
        logger.info("=" * 60)
        
        # CUDA 사용 가능성 확인
        cuda_available = torch.cuda.is_available()
        logger.info(f"🔧 CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            # GPU 개수 및 정보
            gpu_count = torch.cuda.device_count()
            logger.info(f"🎮 사용 가능한 GPU 개수: {gpu_count}")
            
            # 각 GPU 정보 출력
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory = gpu_props.total_memory / 1024**3  # GB
                    
                    # 추가 정보 수집 (안전한 방법)
                    compute_capability = f"{getattr(gpu_props, 'major', 0)}.{getattr(gpu_props, 'minor', 0)}"
                    
                    logger.info(f"  📱 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB, Compute {compute_capability})")
                    
                    # 멀티프로세서 개수 (존재하는 경우)
                    if hasattr(gpu_props, 'multiprocessor_count'):
                        mp_count = gpu_props.multiprocessor_count
                        logger.info(f"    🔧 멀티프로세서: {mp_count}개")
                    elif hasattr(gpu_props, 'multi_processor_count'):
                        mp_count = gpu_props.multi_processor_count
                        logger.info(f"    🔧 멀티프로세서: {mp_count}개")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ GPU {i} 정보 수집 실패: {str(e)}")
                    logger.info(f"  📱 GPU {i}: 정보 확인 불가")
            
            # 현재 GPU 디바이스
            current_device = torch.cuda.current_device()
            current_gpu_name = torch.cuda.get_device_name(current_device)
            logger.info(f"🎯 현재 사용 중인 GPU: {current_device} ({current_gpu_name})")
            
                    # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            cached = torch.cuda.memory_reserved(current_device) / 1024**3
            logger.info(f"💾 GPU 메모리 사용량: {allocated:.2f}GB (할당) / {cached:.2f}GB (캐시)")
            
            # 간단한 GPU 테스트 수행
            try:
                logger.info("🧪 GPU 기능 테스트 시작...")
                test_tensor = torch.randn(1000, 1000, device=current_device)
                test_result = torch.matmul(test_tensor, test_tensor.T)
                
                # 테스트 후 메모리 사용량 재확인
                allocated_after = torch.cuda.memory_allocated(current_device) / 1024**3
                cached_after = torch.cuda.memory_reserved(current_device) / 1024**3
                logger.info(f"✅ GPU 테스트 완료! 테스트 후 메모리: {allocated_after:.2f}GB (할당) / {cached_after:.2f}GB (캐시)")
                
                # 테스트 텐서 정리
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
                # 정리 후 메모리 상태
                allocated_final = torch.cuda.memory_allocated(current_device) / 1024**3
                cached_final = torch.cuda.memory_reserved(current_device) / 1024**3
                logger.info(f"🧹 메모리 정리 후: {allocated_final:.2f}GB (할당) / {cached_final:.2f}GB (캐시)")
                
            except Exception as e:
                logger.error(f"❌ GPU 테스트 실패: {str(e)}")
        
        # 사용할 디바이스 결정
        device = torch.device('cuda' if cuda_available else 'cpu')
        logger.info(f"⚡ 모델 학습/예측에 사용할 디바이스: {device}")
        
        # PyTorch 버전 정보
        logger.info(f"🔢 PyTorch 버전: {torch.__version__}")
        
        # CUDNN 정보 (CUDA 사용 가능한 경우)
        if cuda_available:
            try:
                logger.info(f"🔧 cuDNN 버전: {torch.backends.cudnn.version()}")
                logger.info(f"🔧 cuDNN 활성화: {torch.backends.cudnn.enabled}")
            except Exception as e:
                logger.warning(f"⚠️ cuDNN 정보 확인 실패: {str(e)}")
                
            # GPU 속성 디버깅 정보 (첫 번째 GPU만)
            if gpu_count > 0:
                try:
                    props = torch.cuda.get_device_properties(0)
                    available_attrs = [attr for attr in dir(props) if not attr.startswith('_')]
                    logger.info(f"🔍 사용 가능한 GPU 속성들: {available_attrs}")
                except Exception as e:
                    logger.warning(f"⚠️ GPU 속성 확인 실패: {str(e)}")
        
        logger.info("=" * 60)
        
        return device, cuda_available
        
    except Exception as e:
        logger.error(f"❌ GPU 정보 확인 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return torch.device('cpu'), False

def get_detailed_gpu_utilization():
    """nvidia-smi를 사용하여 상세한 GPU 활용률을 확인하는 함수"""
    try:
        import subprocess
        
        # 기본 활용률 정보
        basic_result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        # 상세 활용률 정보 (Encoder, Decoder 등)
        detailed_result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        # 실행 중인 프로세스 정보
        process_result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        gpu_stats = []
        
        if basic_result.returncode == 0 and basic_result.stdout.strip():
            basic_lines = basic_result.stdout.strip().split('\n')
            detailed_lines = detailed_result.stdout.strip().split('\n') if detailed_result.returncode == 0 else []
            
            for i, line in enumerate(basic_lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_util = parts[0].strip()
                    mem_util = parts[1].strip()
                    temp = parts[2].strip()
                    power_draw = parts[3].strip() if len(parts) > 3 else 'N/A'
                    power_limit = parts[4].strip() if len(parts) > 4 else 'N/A'
                    
                    # 상세 정보 추가
                    encoder_util = 'N/A'
                    decoder_util = 'N/A'
                    
                    if i < len(detailed_lines):
                        detailed_parts = detailed_lines[i].split(', ')
                        if len(detailed_parts) >= 4:
                            encoder_util = detailed_parts[2].strip()
                            decoder_util = detailed_parts[3].strip()
                    
                    gpu_stat = {
                        'gpu_id': i,
                        'gpu_utilization': gpu_util,
                        'memory_utilization': mem_util,
                        'encoder_utilization': encoder_util,
                        'decoder_utilization': decoder_util,
                        'temperature': temp,
                        'power_draw': power_draw,
                        'power_limit': power_limit,
                        'measurement_method': 'nvidia-smi',
                        'timestamp': time.time()
                    }
                    
                    gpu_stats.append(gpu_stat)
        
        # 실행 중인 프로세스 정보 추가
        if process_result.returncode == 0 and process_result.stdout.strip():
            process_lines = process_result.stdout.strip().split('\n')
            compute_processes = []
            for line in process_lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    compute_processes.append({
                        'pid': parts[0].strip(),
                        'name': parts[1].strip(),
                        'gpu_memory_mb': parts[2].strip()
                    })
            
            # 첫 번째 GPU에 프로세스 정보 추가
            if gpu_stats:
                gpu_stats[0]['compute_processes'] = compute_processes
        
        return gpu_stats
        
    except Exception as e:
        logger.warning(f"⚠️ 상세 GPU 활용률 확인 실패: {str(e)}")
        return None

def get_gpu_utilization():
    """nvidia-smi를 사용하여 GPU 활용률을 확인하는 함수 (기존 호환성 유지)"""
    detailed_stats = get_detailed_gpu_utilization()
    if detailed_stats:
        # 기존 형식으로 변환
        return [{
            'gpu_id': stat['gpu_id'],
            'gpu_utilization': stat['gpu_utilization'],
            'memory_utilization': stat['memory_utilization'],
            'temperature': stat['temperature'],
            'power_draw': stat['power_draw'],
            'power_limit': stat['power_limit']
        } for stat in detailed_stats]
    return None

def compare_gpu_monitoring_methods():
    """다양한 GPU 모니터링 방법을 비교하는 함수"""
    comparison_results = {
        'nvidia_smi': None,
        'torch_cuda': None,
        'monitoring_notes': []
    }
    
    try:
        # nvidia-smi 결과
        nvidia_stats = get_detailed_gpu_utilization()
        if nvidia_stats:
            comparison_results['nvidia_smi'] = nvidia_stats[0]  # 첫 번째 GPU
            comparison_results['monitoring_notes'].append(
                "nvidia-smi: CUDA 연산 활용률 측정 (ML/AI 작업에 정확)"
            )
        
        # PyTorch CUDA 정보
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            cached = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            comparison_results['torch_cuda'] = {
                'allocated_memory_gb': round(allocated, 3),
                'cached_memory_gb': round(cached, 3),
                'total_memory_gb': round(total, 1),
                'memory_usage_percent': round((allocated / total) * 100, 2)
            }
            comparison_results['monitoring_notes'].append(
                "PyTorch CUDA: 실제 PyTorch 텐서 메모리 사용량"
            )
        
        comparison_results['monitoring_notes'].extend([
            "Windows 작업 관리자: 주로 3D 그래픽 엔진 활용률 (CUDA와 다름)",
            "nvidia-smi GPU 활용률: CUDA 연산 활용률 (ML/AI 작업)",
            "nvidia-smi Encoder/Decoder: 비디오 인코딩/디코딩 활용률",
            "측정 시점에 따라 순간적인 변화가 클 수 있음"
        ])
        
    except Exception as e:
        comparison_results['error'] = str(e)
    
    return comparison_results

def log_device_usage(device, context=""):
    """특정 상황에서의 디바이스 사용 정보를 로깅하는 함수"""
    try:
        context_str = f"[{context}] " if context else ""
        logger.info(f"🎯 {context_str}사용 중인 디바이스: {device}")
        
        if device.type == 'cuda' and torch.cuda.is_available():
            device_id = device.index if device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            cached = torch.cuda.memory_reserved(device_id) / 1024**3
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            logger.info(f"💾 {context_str}GPU 메모리: {allocated:.3f}GB 사용 / {total:.1f}GB 전체 (캐시: {cached:.3f}GB)")
            
            # 메모리 사용률 계산 및 상태 표시
            usage_percentage = (allocated / total) * 100
            cache_percentage = (cached / total) * 100
            
            if allocated > 0.001:  # 1MB 이상 사용 중인 경우
                logger.info(f"📊 {context_str}메모리 사용률: {usage_percentage:.2f}% (캐시: {cache_percentage:.2f}%)")
                
                if usage_percentage > 80:
                    logger.warning(f"⚠️ {context_str}GPU 메모리 사용률이 높습니다: {usage_percentage:.1f}%")
                elif usage_percentage > 50:
                    logger.info(f"📈 {context_str}GPU 메모리 사용률: {usage_percentage:.1f}% (정상)")
            else:
                logger.info(f"💭 {context_str}현재 GPU 메모리 사용량 없음 (대기 상태)")
            
            # GPU 활용률 확인 (상세)
            detailed_stats = get_detailed_gpu_utilization()
            if detailed_stats and len(detailed_stats) > device_id:
                stat = detailed_stats[device_id]
                logger.info(f"⚡ {context_str}CUDA 활용률: {stat['gpu_utilization']}% (메모리: {stat['memory_utilization']}%)")
                logger.info(f"🎬 {context_str}Encoder: {stat['encoder_utilization']}%, Decoder: {stat['decoder_utilization']}%")
                logger.info(f"🌡️ {context_str}GPU 온도: {stat['temperature']}°C, 전력: {stat['power_draw']}/{stat['power_limit']}W")
                
                # 실행 중인 프로세스 정보
                if 'compute_processes' in stat and stat['compute_processes']:
                    process_count = len(stat['compute_processes'])
                    logger.info(f"🔄 {context_str}CUDA 프로세스: {process_count}개")
                    for proc in stat['compute_processes'][:3]:  # 최대 3개까지만 표시
                        logger.info(f"    📱 PID {proc['pid']}: {proc['name']} ({proc['gpu_memory_mb']}MB)")
                
                # 낮은 활용률 분석 및 설명
                try:
                    gpu_util_num = float(stat['gpu_utilization'])
                    if gpu_util_num < 10:
                        logger.warning(f"⚠️ {context_str}CUDA 활용률이 매우 낮습니다: {gpu_util_num}%")
                        logger.info(f"💡 {context_str}참고: 작업 관리자의 GPU는 3D 그래픽을, nvidia-smi는 CUDA 연산을 측정합니다")
                        logger.info(f"💡 {context_str}ML/AI 작업에서는 nvidia-smi의 CUDA 활용률이 정확합니다")
                    elif gpu_util_num < 30:
                        logger.info(f"📉 {context_str}CUDA 활용률이 낮습니다: {gpu_util_num}% - 배치 크기 증가 고려")
                    else:
                        logger.info(f"✅ {context_str}CUDA 활용률이 양호합니다: {gpu_util_num}%")
                except:
                    pass
                
            # GPU 활성 프로세스 수 확인 (가능한 경우)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    active_processes = len(result.stdout.strip().split('\n'))
                    logger.info(f"🔄 {context_str}GPU에서 실행 중인 프로세스: {active_processes}개")
            except:
                pass  # nvidia-smi가 없거나 실행 실패 시 무시
                
        elif device.type == 'cpu':
            logger.info(f"🖥️ {context_str}CPU 모드로 실행 중")
            
    except Exception as e:
        logger.error(f"❌ 디바이스 사용 정보 로깅 중 오류: {str(e)}")

# GPU 정보 확인 및 기본 디바이스 설정
DEFAULT_DEVICE, CUDA_AVAILABLE = check_gpu_availability()

# Flask 설정
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 최대 파일 크기 32MB로 증가

# 전역 상태 변수에 새 필드 추가
prediction_state = {
    'current_data': None,
    'latest_predictions': None,
    'latest_interval_scores': None,
    'latest_attention_data': None,
    'latest_ma_results': None,
    'latest_plots': None,  # 추가
    'latest_metrics': None,  # 추가
    'current_date': None,
    'current_file': None,  # 추가: 현재 파일 경로
    'is_predicting': False,  # LSTM 예측 상태
    'prediction_progress': 0,
    'prediction_start_time': None,  # 예측 시작 시간
    'error': None,
    'selected_features': None,
    'feature_importance': None,
    'semimonthly_period': None,
    'next_semimonthly_period': None,
    'accumulated_predictions': [],
    'accumulated_metrics': {},
    'prediction_dates': [],
    'accumulated_consistency_scores': {},
    # VARMAX 관련 상태 변수 (독립적인 상태 관리)
    'varmax_predictions': None,
    'varmax_metrics': None,
    'varmax_ma_results': None,
    'varmax_selected_features': None,
    'varmax_current_date': None,
    'varmax_model_info': None,
    'varmax_plots': None,
    'varmax_is_predicting': False,  # 🆕 VARMAX 독립 예측 상태
    'varmax_prediction_progress': 0,  # 🆕 VARMAX 독립 진행률
    'varmax_prediction_start_time': None,  # 🆕 VARMAX 예측 시작 시간
    'varmax_error': None,  # 🆕 VARMAX 독립 에러 상태
}

# 데이터 로더의 워커 시드 고정을 위한 함수
def seed_worker(worker_id):
    """DataLoader worker 시드 고정"""
    # 기존 시드 고정 방식 유지하되 강화
    set_seed(SEED)

# 데이터 로더의 생성자 시드 고정
g = torch.Generator()
g.manual_seed(SEED)

def calculate_estimated_time_remaining(start_time, current_progress):
    """
    예측 시작 시간과 현재 진행률을 기반으로 남은 시간을 계산합니다.
    
    Args:
        start_time: 예측 시작 시간 (time.time() 값)
        current_progress: 현재 진행률 (0-100)
    
    Returns:
        dict: {
            'estimated_remaining_seconds': int,
            'estimated_remaining_text': str,
            'elapsed_time_seconds': int,
            'elapsed_time_text': str
        }
    """
    if not start_time or current_progress <= 0:
        return {
            'estimated_remaining_seconds': None,
            'estimated_remaining_text': '계산 중...',
            'elapsed_time_seconds': 0,
            'elapsed_time_text': '0초'
        }
    
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # 진행률이 100% 이상이면 완료
    if current_progress >= 100:
        return {
            'estimated_remaining_seconds': 0,
            'estimated_remaining_text': '완료',
            'elapsed_time_seconds': int(elapsed_time),
            'elapsed_time_text': format_time_duration(int(elapsed_time))
        }
    
    # 진행률을 기반으로 총 예상 시간 계산
    estimated_total_time = elapsed_time * (100 / current_progress)
    estimated_remaining_time = estimated_total_time - elapsed_time
    
    # 음수가 되지 않도록 보정
    estimated_remaining_time = max(0, estimated_remaining_time)
    
    return {
        'estimated_remaining_seconds': int(estimated_remaining_time),
        'estimated_remaining_text': format_time_duration(int(estimated_remaining_time)),
        'elapsed_time_seconds': int(elapsed_time),
        'elapsed_time_text': format_time_duration(int(elapsed_time))
    }

def format_time_duration(seconds):
    """시간을 사람이 읽기 쉬운 형태로 포맷팅"""
    if seconds < 60:
        return f"{seconds}초"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes}분 {remaining_seconds}초"
        else:
            return f"{minutes}분"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        if remaining_minutes > 0:
            return f"{hours}시간 {remaining_minutes}분"
        else:
            return f"{hours}시간"

#######################################################################
# 모델 및 유틸리티 함수
#######################################################################

# 날짜 포맷팅 유틸리티 함수
def format_date(date_obj, format_str='%Y-%m-%d'):
    """날짜 객체를 문자열로 안전하게 변환"""
    try:
        # pandas Timestamp 또는 datetime.datetime
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime(format_str)
        
        # numpy.datetime64
        elif isinstance(date_obj, np.datetime64):
            # 날짜 포맷이 'YYYY-MM-DD'인 경우
            return str(date_obj)[:10]
        
        # 문자열인 경우 이미 날짜 형식이라면 추가 처리
        elif isinstance(date_obj, str):
            # GMT 형식이면 파싱하여 변환
            if 'GMT' in date_obj:
                parsed_date = datetime.strptime(date_obj, '%a, %d %b %Y %H:%M:%S GMT')
                return parsed_date.strftime(format_str)
            return date_obj[:10] if len(date_obj) > 10 else date_obj
        
        # 그 외 경우
        else:
            return str(date_obj)
    
    except Exception as e:
        logger.warning(f"날짜 포맷팅 오류: {str(e)}")
        return str(date_obj)

# 🔧 스마트 파일 캐시 시스템 함수들
def calculate_file_hash(file_path, chunk_size=8192):
    """파일 내용의 SHA256 해시를 계산"""
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"File hash calculation failed: {str(e)}")
        return None

# 파일 해시 캐시 추가 (메모리 캐싱으로 성능 최적화)
_file_hash_cache = {}
_cache_lookup_index = {}  # 빠른 캐시 검색을 위한 인덱스

# 🔧 DataFrame 메모리 캐시 (중복 로딩 방지)
_dataframe_cache = {}
_cache_expiry_seconds = 120  # 2분간 캐시 유지

def get_data_content_hash(file_path):
    """데이터 파일(CSV/Excel)의 전처리된 내용으로 해시 생성 (캐싱 최적화)"""
    import hashlib
    import os
    
    try:
        # 파일 수정 시간 기반 캐시 확인
        if file_path in _file_hash_cache:
            cached_mtime, cached_hash = _file_hash_cache[file_path]
            current_mtime = os.path.getmtime(file_path)
            
            # 파일이 수정되지 않았다면 캐시된 해시 반환
            if abs(current_mtime - cached_mtime) < 1.0:  # 1초 이내 차이는 무시
                logger.debug(f"📋 Using cached hash for {os.path.basename(file_path)}")
                return cached_hash
        
        # 파일이 수정되었거나 캐시가 없는 경우 새로 계산
        logger.info(f"🔄 Calculating new hash for {os.path.basename(file_path)}")
        
        # 파일 형식에 맞게 로드
        file_ext = os.path.splitext(file_path.lower())[1]
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            # Excel 파일인 경우 load_data 함수 사용 (전처리된 데이터로 해시 생성)
            df = load_data(file_path)
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                df = df.reset_index()
        
        if 'Date' in df.columns:
            # 날짜를 기준으로 정렬하여 일관된 해시 생성
            df = df.sort_values('Date')
        
        # 🔧 DataFrame을 안전하게 문자열로 변환
        # 무한대나 NaN 값을 처리하여 해시 계산 오류 방지
        df_for_hash = df.copy()
        
        # 무한대와 NaN 값을 문자열로 변환
        df_for_hash = df_for_hash.replace([np.inf, -np.inf], 'inf')
        df_for_hash = df_for_hash.fillna('nan')
        
        # 모든 컬럼을 문자열로 변환하여 안전한 해시 계산
        try:
            content_str = df_for_hash.to_string()
        except Exception as str_error:
            logger.warning(f"DataFrame to_string failed, using alternative method: {str(str_error)}")
            # 대안: 각 컬럼을 개별적으로 문자열로 변환
            content_parts = []
            for col in df_for_hash.columns:
                try:
                    col_str = str(col) + ":" + str(df_for_hash[col].tolist())
                    content_parts.append(col_str)
                except Exception:
                    content_parts.append(f"{col}:error")
            content_str = "|".join(content_parts)
        
        file_hash = hashlib.sha256(content_str.encode('utf-8', errors='ignore')).hexdigest()[:16]  # 짧은 해시 사용
        
        # 캐시 저장
        _file_hash_cache[file_path] = (os.path.getmtime(file_path), file_hash)
        
        return file_hash
    except Exception as e:
        logger.error(f"Data content hash calculation failed: {str(e)}")
        # 해시 계산에 실패하면 파일 기본 해시를 사용
        try:
            return calculate_file_hash(file_path)[:16]
        except Exception:
            return None

def build_cache_lookup_index():
    """캐시 디렉토리의 인덱스를 빌드하여 빠른 검색 가능"""
    global _cache_lookup_index
    
    try:
        _cache_lookup_index = {}
        cache_root = Path(CACHE_ROOT_DIR)
        
        if not cache_root.exists():
            return
        
        for file_dir in cache_root.iterdir():
            if not file_dir.is_dir() or file_dir.name == "default":
                continue
            
            predictions_dir = file_dir / 'predictions'
            if not predictions_dir.exists():
                continue
            
            prediction_files = list(predictions_dir.glob("prediction_start_*_meta.json"))
            
            for meta_file in prediction_files:
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    file_hash = meta_data.get('file_content_hash')
                    data_end_date = meta_data.get('data_end_date')
                    
                    if file_hash and data_end_date:
                        semimonthly = get_semimonthly_period(pd.to_datetime(data_end_date))
                        cache_key = f"{file_hash}_{semimonthly}"
                        
                        _cache_lookup_index[cache_key] = {
                            'meta_file': str(meta_file),
                            'predictions_dir': str(predictions_dir),
                            'data_end_date': data_end_date,
                            'semimonthly': semimonthly
                        }
                        
                except Exception:
                    continue
                    
        logger.info(f"📊 Built cache lookup index with {len(_cache_lookup_index)} entries")
        
    except Exception as e:
        logger.error(f"Failed to build cache lookup index: {str(e)}")
        _cache_lookup_index = {}

def refresh_cache_index():
    """캐시 인덱스를 새로고침 (새로운 캐시 파일이 생성된 후 호출)"""
    global _cache_lookup_index
    logger.info("🔄 Refreshing cache lookup index...")
    build_cache_lookup_index()

def clear_cache_memory():
    """메모리 캐시를 클리어 (메모리 절약용)"""
    global _file_hash_cache, _cache_lookup_index
    _file_hash_cache.clear()
    _cache_lookup_index.clear()
    logger.info("🧹 Cleared memory cache")

def check_data_extension(old_file_path, new_file_path):
    """
    새 파일이 기존 파일의 순차적 확장(기존 데이터 이후에만 새 행 추가)인지 엄격하게 확인
    
    ⚠️ 중요: 다음 경우만 확장으로 인정:
    1. 기존 데이터와 정확히 동일한 부분이 있음
    2. 새 데이터가 기존 데이터의 마지막 날짜 이후에만 추가됨
    3. 기존 데이터의 시작/중간 날짜가 변경되지 않음
    
    Returns:
    --------
    dict: {
        'is_extension': bool,
        'new_rows_count': int,
        'base_hash': str,  # 기존 데이터 부분의 해시
        'old_start_date': str,
        'old_end_date': str,
        'new_start_date': str,
        'new_end_date': str,
        'validation_details': dict
    }
    """
    try:
        # 파일 형식에 맞게 로드 (🔧 캐시 활용)
        def load_file_safely(filepath, is_new_file=False):
            file_ext = os.path.splitext(filepath.lower())[1]
            if file_ext == '.csv':
                return pd.read_csv(filepath)
            else:
                # Excel 파일인 경우 load_data 함수 사용 (캐시 활용)
                df = load_data(filepath, use_cache=True)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if df.index.name == 'Date':
                    df = df.reset_index()
                return df
        
        logger.info(f"🔍 [EXTENSION_CHECK] Loading data files for comparison...")
        old_df = load_file_safely(old_file_path, is_new_file=False)
        new_df = load_file_safely(new_file_path, is_new_file=True)
        
        # 날짜 컬럼이 있는지 확인
        if 'Date' not in old_df.columns or 'Date' not in new_df.columns:
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'validation_details': {'error': 'No Date column found'}
            }
        
        # 날짜로 정렬
        old_df = old_df.sort_values('Date').reset_index(drop=True)
        new_df = new_df.sort_values('Date').reset_index(drop=True)
        
        # 날짜를 datetime으로 변환
        old_df['Date'] = pd.to_datetime(old_df['Date'])
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # 기본 정보 추출
        old_start_date = old_df['Date'].iloc[0]
        old_end_date = old_df['Date'].iloc[-1]
        new_start_date = new_df['Date'].iloc[0]
        new_end_date = new_df['Date'].iloc[-1]
        
        logger.info(f"🔍 [EXTENSION_CHECK] Old data: {old_start_date.strftime('%Y-%m-%d')} ~ {old_end_date.strftime('%Y-%m-%d')} ({len(old_df)} rows)")
        logger.info(f"🔍 [EXTENSION_CHECK] New data: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')} ({len(new_df)} rows)")
        
        # ✅ 검증 1: 새 파일이 더 길어야 함
        if len(new_df) <= len(old_df):
            logger.info(f"❌ [EXTENSION_CHECK] New file is not longer ({len(new_df)} <= {len(old_df)})")
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New file is not longer than old file'}
            }
        
        # ✅ 검증 2: 새 파일이 더 길거나 최소한 같은 길이여야 함 (과거 데이터 허용)
        # 과거 데이터가 포함된 경우도 허용하도록 변경
        logger.info(f"📅 [EXTENSION_CHECK] Date ranges - Old: {old_start_date} ~ {old_end_date}, New: {new_start_date} ~ {new_end_date}")
        
        # ✅ 검증 3: 새 데이터가 기존 데이터보다 더 많은 정보를 포함해야 함 (완화된 조건)
        # 과거 데이터 확장 또는 미래 데이터 확장 둘 다 허용
        has_more_data = (new_start_date < old_start_date) or (new_end_date > old_end_date) or (len(new_df) > len(old_df))
        if not has_more_data:
            logger.info(f"❌ [EXTENSION_CHECK] New data doesn't provide additional information")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New data does not provide additional information beyond existing data'}
            }
        
        # ✅ 검증 4: 기존 데이터의 모든 날짜가 새 데이터에 포함되어야 함
        old_dates = set(old_df['Date'].dt.strftime('%Y-%m-%d'))
        new_dates = set(new_df['Date'].dt.strftime('%Y-%m-%d'))
        
        missing_dates = old_dates - new_dates
        if missing_dates:
            logger.info(f"❌ [EXTENSION_CHECK] Some old dates are missing in new data: {missing_dates}")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': f'Missing dates from old data: {list(missing_dates)}'}
            }
        
        # ✅ 검증 5: 컬럼이 동일해야 함
        if list(old_df.columns) != list(new_df.columns):
            logger.info(f"❌ [EXTENSION_CHECK] Column structure differs")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'Column structure differs'}
            }
        
        # ✅ 검증 6: 기존 데이터 부분이 정확히 동일한지 확인 (관대한 조건으로 완화)
        logger.info(f"🔍 [EXTENSION_CHECK] Comparing overlapping data...")
        logger.info(f"  📊 Checking {len(old_df)} existing dates...")
        
        # 🔧 관대한 확장 검증: 샘플링 방식으로 변경 (전체 데이터가 아닌 일부만 검사)
        sample_size = min(50, len(old_df))  # 최대 50개 날짜만 검사
        sample_indices = list(range(0, len(old_df), max(1, len(old_df) // sample_size)))
        
        logger.info(f"  🔬 Sampling {len(sample_indices)} dates out of {len(old_df)} for validation...")
        
        # 기존 데이터의 각 날짜에 해당하는 새 데이터 행 찾기
        data_matches = True
        mismatch_details = []
        checked_dates = 0
        mismatched_dates = 0
        allowed_mismatches = max(1, len(sample_indices) // 10)  # 10% 정도의 미스매치는 허용
        
        for idx in sample_indices:
            if idx >= len(old_df):
                continue
                
            old_row = old_df.iloc[idx]
            old_date = old_row['Date']
            old_date_str = old_date.strftime('%Y-%m-%d')
            checked_dates += 1
            
            # 새 데이터에서 해당 날짜 찾기
            new_matching_rows = new_df[new_df['Date'] == old_date]
            
            if len(new_matching_rows) == 0:
                data_matches = False
                mismatch_details.append(f"Date {old_date_str} missing in new data")
                break
            elif len(new_matching_rows) > 1:
                data_matches = False
                mismatch_details.append(f"Duplicate date {old_date_str} in new data")
                break
            
            new_row = new_matching_rows.iloc[0]
            
            # 수치 컬럼 비교 (Date 제외) - 완화된 조건
            numeric_cols = old_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                old_val = old_row[col]
                new_val = new_row[col]
                
                # NaN 값 처리
                if pd.isna(old_val) and pd.isna(new_val):
                    continue
                elif pd.isna(old_val) or pd.isna(new_val):
                    data_matches = False
                    mismatch_details.append(f"NaN mismatch on {old_date_str}, column {col}: {old_val} != {new_val}")
                    break
                
                # 수치 비교 - 상대적으로 관대한 조건 (0.01% 오차 허용)
                if not np.allclose([old_val], [new_val], rtol=1e-4, atol=1e-6, equal_nan=True):
                    # 추가 검증: 정수값이 소수점으로 변환된 경우 허용 (예: 100 vs 100.0)
                    try:
                        if abs(float(old_val) - float(new_val)) < 1e-6:
                            continue
                    except:
                        pass
                    
                    mismatch_details.append(f"Value mismatch on {old_date_str}, column {col}: {old_val} != {new_val}")
                    mismatched_dates += 1
                    # 🔧 관대한 조건: 즉시 중단하지 않고 허용 한도까지 계속 검사
                    if mismatched_dates > allowed_mismatches:
                        data_matches = False
                        break
            
            if not data_matches:
                break
            
            # 문자열 컬럼 비교 (Date 제외) - 완화된 조건
            str_cols = old_df.select_dtypes(include=['object']).columns
            str_cols = [col for col in str_cols if col != 'Date']
            for col in str_cols:
                old_str = str(old_row[col]).strip() if not pd.isna(old_row[col]) else ''
                new_str = str(new_row[col]).strip() if not pd.isna(new_row[col]) else ''
                
                if old_str != new_str:
                    mismatch_details.append(f"String mismatch on {old_date_str}, column {col}: '{old_str}' != '{new_str}'")
                    mismatched_dates += 1
                    # 🔧 관대한 조건: 허용 한도까지 계속 검사
                    if mismatched_dates > allowed_mismatches:
                        data_matches = False
                        break
            
            if not data_matches:
                break
        
        # 🔧 관대한 검증 결과 평가
        logger.info(f"  ✅ Checked {checked_dates} sample dates, {mismatched_dates} mismatches found (allowed: {allowed_mismatches})")
        if mismatch_details:
            logger.info(f"  ⚠️ Sample mismatches: {mismatch_details[:3]}...")
        
        if not data_matches:
            logger.info(f"❌ [EXTENSION_CHECK] Too many data mismatches ({mismatched_dates} > {allowed_mismatches})")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {
                    'reason': f'Too many data mismatches: {mismatched_dates} > {allowed_mismatches}',
                    'mismatches_found': mismatched_dates,
                    'allowed_mismatches': allowed_mismatches,
                    'sample_details': mismatch_details[:5]
                }
            }
        elif mismatched_dates > 0:
            logger.info(f"⚠️ [EXTENSION_CHECK] Minor mismatches found but within tolerance ({mismatched_dates} <= {allowed_mismatches})")
        
        # ✅ 검증 7: 새로 추가된 데이터 분석 (과거/미래 데이터 모두 허용)
        new_only_dates = new_dates - old_dates
        
        # 확장 유형 분석
        extension_type = []
        if new_start_date < old_start_date:
            past_dates = len([d for d in new_only_dates if pd.to_datetime(d) < old_start_date])
            extension_type.append(f"과거 데이터 {past_dates}개 추가")
        if new_end_date > old_end_date:
            future_dates = len([d for d in new_only_dates if pd.to_datetime(d) > old_end_date])
            extension_type.append(f"미래 데이터 {future_dates}개 추가")
        
        extension_desc = " + ".join(extension_type) if extension_type else "데이터 보완"
        
        # ✅ 모든 검증 통과: 데이터 확장으로 인정 (과거/미래 모두 허용)
        new_rows_count = len(new_only_dates)
        base_hash = get_data_content_hash(old_file_path)
        
        logger.info(f"✅ [EXTENSION_CHECK] Valid data extension: {extension_desc} (+{new_rows_count} new dates)")
        
        return {
            'is_extension': True,
            'new_rows_count': new_rows_count,
            'base_hash': base_hash,
            'old_start_date': old_start_date.strftime('%Y-%m-%d'),
            'old_end_date': old_end_date.strftime('%Y-%m-%d'),
            'new_start_date': new_start_date.strftime('%Y-%m-%d'),
            'new_end_date': new_end_date.strftime('%Y-%m-%d'),
            'validation_details': {
                'reason': f'Valid data extension: {extension_desc}',
                'new_dates_added': sorted(list(new_only_dates)),
                'extension_type': extension_type
            }
        }
        
    except Exception as e:
        logger.error(f"Data extension check failed: {str(e)}")
        return {
            'is_extension': False, 
            'new_rows_count': 0,
            'old_start_date': None,
            'old_end_date': None,
            'new_start_date': None,
            'new_end_date': None,
            'validation_details': {'error': str(e)}
        }

def find_existing_cache_range(file_path):
    """
    기존 파일의 캐시에서 사용된 데이터 범위 정보를 찾는 함수
    
    Returns:
    --------
    dict or None: {'start_date': 'YYYY-MM-DD', 'cutoff_date': 'YYYY-MM-DD'} 또는 None
    """
    try:
        # 파일에 대응하는 캐시 디렉토리 찾기
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        
        if not predictions_dir.exists():
            return None
            
        # 최근 메타 파일에서 데이터 범위 정보 확인
        meta_files = list(predictions_dir.glob("*_meta.json"))
        if not meta_files:
            return None
            
        # 가장 최근 메타 파일 선택
        latest_meta = max(meta_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_meta, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            
        # 데이터 범위 정보 추출
        model_config = meta_data.get('model_config', {})
        lstm_config = model_config.get('lstm', {})
        
        start_date = lstm_config.get('data_start_date')
        cutoff_date = lstm_config.get('data_cutoff_date') or meta_data.get('data_end_date')
        
        if start_date and cutoff_date:
            return {
                'start_date': start_date,
                'cutoff_date': cutoff_date,
                'meta_file': str(latest_meta)
            }
            
        return None
        
    except Exception as e:
        logger.warning(f"Failed to find cache range for {file_path}: {str(e)}")
        return None

def find_compatible_cache_file(new_file_path, intended_data_range=None, cached_df=None):
    """
    새 파일과 호환되는 기존 캐시를 찾는 함수 (데이터 범위 고려)
    
    🔧 핵심 개선:
    - 파일 내용 + 사용 데이터 범위를 모두 고려
    - 같은 파일이라도 다른 데이터 범위면 새 예측으로 인식
    - 사용자 의도를 반영한 스마트 캐시 매칭
    - 중복 로딩 방지를 위한 캐시된 DataFrame 재사용
    
    Parameters:
    -----------
    new_file_path : str
        새 파일 경로
    intended_data_range : dict, optional
        사용자가 의도한 데이터 범위 {'start_date': 'YYYY-MM-DD', 'cutoff_date': 'YYYY-MM-DD'}
    cached_df : DataFrame, optional
        이미 로딩된 DataFrame (중복 로딩 방지)
    
    Returns:
    --------
    dict: {
        'found': bool,
        'cache_type': str,  # 'exact_with_range', 'extension', 'partial', 'range_mismatch'
        'cache_files': list,
        'compatibility_info': dict
    }
    """
    try:
        # 🔧 캐시된 DataFrame이 있으면 재사용, 없으면 새로 로딩
        if cached_df is not None:
            logger.info(f"🔄 [CACHE_OPTIMIZATION] Using cached DataFrame (avoiding duplicate load)")
            new_df = cached_df.copy()
        else:
            logger.info(f"📁 [CACHE_COMPATIBILITY] Loading data for cache check...")
            # 새 파일의 데이터 분석 (파일 형식에 맞게)
            file_ext = os.path.splitext(new_file_path.lower())[1]
            if file_ext == '.csv':
                new_df = pd.read_csv(new_file_path)
            else:
                # Excel 파일인 경우 load_data 함수 사용
                new_df = load_data(new_file_path)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if new_df.index.name == 'Date':
                    new_df = new_df.reset_index()
        
        if 'Date' not in new_df.columns:
            return {'found': False, 'cache_type': None, 'reason': 'No Date column'}
            
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_start_date = new_df['Date'].min()
        new_end_date = new_df['Date'].max()
        new_hash = get_data_content_hash(new_file_path)
        
        logger.info(f"🔍 [ENHANCED_CACHE] Analyzing new file:")
        logger.info(f"  📅 Full date range: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📊 Records: {len(new_df)}")
        logger.info(f"  🔑 Hash: {new_hash[:12] if new_hash else 'None'}...")
        
        # 사용자 의도 데이터 범위 확인
        if intended_data_range:
            intended_start = pd.to_datetime(intended_data_range.get('start_date', new_start_date))
            intended_cutoff = pd.to_datetime(intended_data_range.get('cutoff_date', new_end_date))
            logger.info(f"  🎯 Intended range: {intended_start.strftime('%Y-%m-%d')} ~ {intended_cutoff.strftime('%Y-%m-%d')}")
        else:
            intended_start = new_start_date
            intended_cutoff = new_end_date
            logger.info(f"  🎯 Using full range (no specific intention provided)")
        
        compatible_caches = []
        
        # 1. uploads 폴더의 파일들 검사 (데이터 범위 고려)
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = list(upload_dir.glob('*.csv')) + list(upload_dir.glob('*.xlsx')) + list(upload_dir.glob('*.xls'))
        
        logger.info(f"🔍 [ENHANCED_CACHE] Checking {len(existing_files)} upload files with range consideration...")
        
        for existing_file in existing_files:
            if existing_file.name == os.path.basename(new_file_path):
                continue
                
            try:
                # 파일 해시 확인
                existing_hash = get_data_content_hash(str(existing_file))
                if existing_hash == new_hash:
                    logger.info(f"📄 [ENHANCED_CACHE] Same file content found: {existing_file.name}")
                    
                    # 🔑 같은 파일이지만 데이터 범위 의도 확인
                    # 기존 캐시의 데이터 범위 정보를 찾아야 함
                    existing_cache_range = find_existing_cache_range(str(existing_file))
                    
                    if existing_cache_range and intended_data_range:
                        cache_start = existing_cache_range.get('start_date')
                        cache_cutoff = existing_cache_range.get('cutoff_date') 
                        
                        if cache_start and cache_cutoff:
                            cache_start = pd.to_datetime(cache_start)
                            cache_cutoff = pd.to_datetime(cache_cutoff)
                            
                            # 데이터 범위 비교
                            range_match = (
                                abs((intended_start - cache_start).days) <= 30 and 
                                abs((intended_cutoff - cache_cutoff).days) <= 30
                            )
                            
                            if range_match:
                                logger.info(f"✅ [ENHANCED_CACHE] Exact match with same intended range!")
                                return {
                                    'found': True,
                                    'cache_type': 'exact_with_range',
                                    'cache_files': [str(existing_file)],
                                    'compatibility_info': {
                                        'match_type': 'file_hash_and_range',
                                        'cache_range': existing_cache_range,
                                        'intended_range': intended_data_range
                                    }
                                }
                            else:
                                logger.info(f"⚠️ [ENHANCED_CACHE] Same file but different intended range:")
                                logger.info(f"    💾 Cached range: {cache_start.strftime('%Y-%m-%d')} ~ {cache_cutoff.strftime('%Y-%m-%d')}")
                                logger.info(f"    🎯 Intended range: {intended_start.strftime('%Y-%m-%d')} ~ {intended_cutoff.strftime('%Y-%m-%d')}")
                                logger.info(f"    🔄 Will create new cache for different range")
                                # 같은 파일이지만 다른 범위 의도 → 새 예측 필요
                                continue
                    
                    # 범위 정보가 없으면 기존 로직 적용
                    logger.info(f"✅ [ENHANCED_CACHE] Exact file match (no range info): {existing_file.name}")
                    return {
                        'found': True,
                        'cache_type': 'exact',
                        'cache_files': [str(existing_file)],
                        'compatibility_info': {'match_type': 'file_hash_only'}
                    }
                
                # 확장 파일 확인 (기존 로직 유지) - 디버깅 강화
                logger.info(f"🔍 [EXTENSION_CHECK] Testing extension: {existing_file.name} → {os.path.basename(new_file_path)}")
                extension_info = check_data_extension(str(existing_file), new_file_path)
                
                logger.info(f"📊 [EXTENSION_RESULT] is_extension: {extension_info['is_extension']}")
                if extension_info.get('validation_details'):
                    logger.info(f"📊 [EXTENSION_RESULT] reason: {extension_info['validation_details'].get('reason', 'N/A')}")
                
                if extension_info['is_extension']:
                    logger.info(f"📈 [ENHANCED_CACHE] Found extension base: {existing_file.name} (+{extension_info.get('new_rows_count', 0)} rows)")
                    return {
                        'found': True,
                        'cache_type': 'extension', 
                        'cache_files': [str(existing_file)],
                        'compatibility_info': extension_info
                    }
                else:
                    logger.info(f"❌ [EXTENSION_CHECK] Not an extension: {extension_info['validation_details'].get('reason', 'Unknown reason')}")
                    
            except Exception as e:
                logger.warning(f"Error checking upload file {existing_file}: {str(e)}")
                continue
        
        # 2. 🔧 캐시 디렉토리의 모든 예측 결과 검사 (신규)
        cache_root = Path(CACHE_ROOT_DIR)
        if not cache_root.exists():
            logger.info("❌ [ENHANCED_CACHE] No cache directory found")
            return {'found': False, 'cache_type': None}
            
        logger.info(f"🔍 [ENHANCED_CACHE] Scanning cache directories...")
        
        for file_cache_dir in cache_root.iterdir():
            if not file_cache_dir.is_dir():
                continue
                
            predictions_dir = file_cache_dir / 'predictions'
            if not predictions_dir.exists():
                continue
                
            # predictions_index.csv 파일에서 캐시된 예측들의 날짜 범위 확인
            index_file = predictions_dir / 'predictions_index.csv'
            if not index_file.exists():
                continue
                
            try:
                cache_index = pd.read_csv(index_file)
                if 'data_end_date' not in cache_index.columns:
                    continue
                    
                cache_index['data_end_date'] = pd.to_datetime(cache_index['data_end_date'])
                cache_start = cache_index['data_end_date'].min()
                cache_end = cache_index['data_end_date'].max()
                
                logger.info(f"  📁 {file_cache_dir.name}: {cache_start.strftime('%Y-%m-%d')} ~ {cache_end.strftime('%Y-%m-%d')} ({len(cache_index)} predictions)")
                
                # 날짜 범위 중복 확인
                overlap_start = max(new_start_date, cache_start)
                overlap_end = min(new_end_date, cache_end)
                
                if overlap_start <= overlap_end:
                    overlap_days = (overlap_end - overlap_start).days + 1
                    new_total_days = (new_end_date - new_start_date).days + 1
                    coverage_ratio = overlap_days / new_total_days
                    
                    logger.info(f"    📊 Overlap: {overlap_days} days ({coverage_ratio:.1%} coverage)")
                    
                    if coverage_ratio >= 0.7:  # 70% 이상 겹치면 호환 가능
                        compatible_caches.append({
                            'cache_dir': str(file_cache_dir),
                            'predictions_dir': str(predictions_dir),
                            'coverage_ratio': coverage_ratio,
                            'overlap_days': overlap_days,
                            'cache_range': (cache_start, cache_end),
                            'prediction_count': len(cache_index)
                        })
                        
            except Exception as e:
                logger.warning(f"Error analyzing cache {file_cache_dir.name}: {str(e)}")
                continue
        
        # 3. 호환 가능한 캐시 결과 처리
        if compatible_caches:
            # 커버리지 비율로 정렬 (높은 순)
            compatible_caches.sort(key=lambda x: x['coverage_ratio'], reverse=True)
            best_cache = compatible_caches[0]
            
            logger.info(f"🎯 [ENHANCED_CACHE] Found {len(compatible_caches)} compatible cache(s)")
            logger.info(f"  🥇 Best: {Path(best_cache['cache_dir']).name} ({best_cache['coverage_ratio']:.1%} coverage)")
            
            if best_cache['coverage_ratio'] >= 0.95:  # 95% 이상이면 거의 완전
                cache_type = 'near_complete'
            elif len(compatible_caches) > 1:  # 여러 캐시 조합 가능
                cache_type = 'multi_cache' 
            else:
                cache_type = 'partial'
                
            return {
                'found': True,
                'cache_type': cache_type,
                'cache_files': [cache['predictions_dir'] for cache in compatible_caches],
                'compatibility_info': {
                    'best_coverage': best_cache['coverage_ratio'],
                    'total_compatible_caches': len(compatible_caches),
                    'date_ranges': [(c['cache_range'][0].strftime('%Y-%m-%d'), 
                                   c['cache_range'][1].strftime('%Y-%m-%d')) for c in compatible_caches]
                }
            }
        
        logger.info("❌ [ENHANCED_CACHE] No compatible cache found")
        return {'found': False, 'cache_type': None}
        
    except Exception as e:
        logger.error(f"Enhanced cache compatibility check failed: {str(e)}")
        return {'found': False, 'cache_type': None, 'error': str(e)}

def create_proper_column_names(file_path, sheet_name):
    """헤더 3행을 읽어서 적절한 열 이름 생성"""
    # 헤더 3행을 읽어옴
    header_rows = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=3)
    
    # 각 열별로 적절한 이름 생성
    column_names = []
    prev_main_category = None  # 이전 메인 카테고리 저장
    
    for col_idx in range(header_rows.shape[1]):
        values = [str(header_rows.iloc[i, col_idx]).strip() 
                 for i in range(3) 
                 if pd.notna(header_rows.iloc[i, col_idx]) and str(header_rows.iloc[i, col_idx]).strip() != 'nan']
        
        # 첫 번째 행의 값이 있으면 메인 카테고리로 저장
        if pd.notna(header_rows.iloc[0, col_idx]) and str(header_rows.iloc[0, col_idx]).strip() != 'nan':
            prev_main_category = str(header_rows.iloc[0, col_idx]).strip()
        
        # 열 이름 생성 로직
        if 'Date' in values:
            column_names.append('Date')
        else:
            # 값이 하나도 없는 경우
            if not values:
                column_names.append(f'Unnamed_{col_idx}')
                continue
                
            # 메인 카테고리가 있고, 현재 값들에 포함되지 않은 경우 추가
            if prev_main_category and prev_main_category not in values:
                values.insert(0, prev_main_category)
            
            # 특수 케이스 처리 (예: WS, Naphtha 등)
            if 'WS' in values and 'SG-Korea' in values:
                column_names.append('WS_SG-Korea')
            elif 'Naphtha' in values and 'Platts' in values:
                column_names.append('Naphtha_Platts_' + '_'.join([v for v in values if v not in ['Naphtha', 'Platts']]))
            else:
                column_names.append('_'.join(values))
    
    return column_names

def remove_high_missing_columns(data, threshold=70):
    """높은 결측치 비율을 가진 열 제거"""
    missing_ratio = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_ratio[missing_ratio >= threshold].index
    
    print(f"\n=== {threshold}% 이상 결측치가 있어 제거될 열 목록 ===")
    for col in high_missing_cols:
        print(f"- {col}: {missing_ratio[col]:.1f}%")
    
    cleaned_data = data.drop(columns=high_missing_cols)
    print(f"\n원본 데이터 형태: {data.shape}")
    print(f"정제된 데이터 형태: {cleaned_data.shape}")
    
    return cleaned_data

def clean_text_values_advanced(data):
    """고급 텍스트 값 정제 (쉼표 소수점 처리 포함)"""
    cleaned_data = data.copy()
    
    def fix_comma_decimal(value_str):
        """쉼표로 된 소수점을 점으로 변경하는 함수"""
        if not isinstance(value_str, str) or ',' not in value_str:
            return value_str
            
        import re
        
        # 패턴 1: 단순 소수점 쉼표 (예: "123,45")
        if re.match(r'^-?\d+,\d{1,3}$', value_str):
            return value_str.replace(',', '.')
            
        # 패턴 2: 천 단위 구분자 + 소수점 쉼표 (예: "1.234,56")
        if re.match(r'^-?\d{1,3}(\.\d{3})*,\d{1,3}$', value_str):
            # 마지막 쉼표만 소수점으로 변경
            last_comma_pos = value_str.rfind(',')
            return value_str[:last_comma_pos] + '.' + value_str[last_comma_pos+1:]
            
        # 패턴 3: 쉼표만 천 단위 구분자로 사용 (예: "1,234,567")
        if re.match(r'^-?\d{1,3}(,\d{3})+$', value_str):
            return value_str.replace(',', '')
            
        return value_str
    
    def process_value(x):
        if pd.isna(x):  # 이미 NaN인 경우
            return x
        
        # 문자열로 변환하여 처리
        x_str = str(x).strip()
        
        # 1. 먼저 쉼표 소수점 문제 해결
        x_str = fix_comma_decimal(x_str)
        
        # 2. 휴일/미발표 데이터 처리
        if x_str.upper() in ['NOP', 'NO PUBLICATION', 'NO PUB']:
            return np.nan
            
        # 3. TBA (To Be Announced) 값 처리 - 특별 마킹하여 나중에 전날값으로 대체
        if x_str.upper() in ['TBA', 'TO BE ANNOUNCED']:
            return 'TBA_REPLACE'
            
        # 4. '*' 포함된 계산식 처리
        if '*' in x_str:
            try:
                # 계산식 실행
                return float(eval(x_str.replace(' ', '')))
            except:
                return x
        
        # 5. 숫자로 변환 시도
        try:
            return float(x_str)
        except:
            return x

    # 쉼표 처리 통계를 위한 변수
    comma_fixes = 0
    
    # 각 열에 대해 처리
    for column in cleaned_data.columns:
        if column != 'Date':  # Date 열 제외
            # 처리 전 쉼표가 있는 값들 확인
            before_comma_count = cleaned_data[column].astype(str).str.contains(',', na=False).sum()
            
            cleaned_data[column] = cleaned_data[column].apply(process_value)
            
            # 처리 후 쉼표가 있는 값들 확인
            after_comma_count = cleaned_data[column].astype(str).str.contains(',', na=False).sum()
            
            if before_comma_count > after_comma_count:
                fixed_count = before_comma_count - after_comma_count
                comma_fixes += fixed_count
                print(f"열 '{column}': {fixed_count}개의 쉼표 소수점을 수정했습니다.")
    
    if comma_fixes > 0:
        print(f"\n총 {comma_fixes}개의 쉼표 소수점을 점으로 수정했습니다.")
    
    # MOPJ 변수 처리 (결측치가 있는 행 제거)
    mopj_columns = [col for col in cleaned_data.columns if 'MOPJ' in col or 'Naphtha_Platts_MOPJ' in col]
    if mopj_columns:
        mopj_col = mopj_columns[0]  # 첫 번째 MOPJ 관련 열 사용
        print(f"\n=== {mopj_col} 변수 처리 전 데이터 크기 ===")
        print(f"행 수: {len(cleaned_data)}")
        
        # 결측치가 있는 행 제거
        cleaned_data = cleaned_data.dropna(subset=[mopj_col])
        
        # 문자열 값이 있는 행 제거
        try:
            pd.to_numeric(cleaned_data[mopj_col], errors='raise')
        except:
            # 숫자로 변환할 수 없는 행 찾기
            numeric_mask = pd.to_numeric(cleaned_data[mopj_col], errors='coerce').notna()
            cleaned_data = cleaned_data[numeric_mask]
        
        print(f"\n=== {mopj_col} 변수 처리 후 데이터 크기 ===")
        print(f"행 수: {len(cleaned_data)}")
    
    # 🔧 TBA 값을 전날 값으로 대체
    tba_replacements = 0
    if 'Date' in cleaned_data.columns:
        # 날짜순으로 정렬 (중요: 전날 값 참조를 위해)
        cleaned_data = cleaned_data.sort_values('Date').reset_index(drop=True)
        
        for column in cleaned_data.columns:
            if column != 'Date':  # Date 열 제외
                # TBA_REPLACE 마킹된 값들 찾기
                tba_mask = cleaned_data[column] == 'TBA_REPLACE'
                tba_indices = cleaned_data[tba_mask].index.tolist()
                
                if tba_indices:
                    print(f"\n[TBA 처리] 열 '{column}'에서 {len(tba_indices)}개의 TBA 값 발견")
                    
                    for idx in tba_indices:
                        # 🔧 개선: 가장 최근의 유효한 값 찾기 (연속 TBA 처리)
                        replacement_value = None
                        source_description = ""
                        
                        # 이전 행들을 거슬러 올라가면서 유효한 값 찾기
                        for prev_idx in range(idx-1, -1, -1):
                            candidate_value = cleaned_data.loc[prev_idx, column]
                            try:
                                if pd.notna(candidate_value) and candidate_value != 'TBA_REPLACE':
                                    replacement_value = float(candidate_value)
                                    days_back = idx - prev_idx
                                    if days_back == 1:
                                        source_description = "전날 값"
                                    else:
                                        source_description = f"{days_back}일 전 값"
                                    break
                            except (ValueError, TypeError):
                                continue
                        
                        # 값 대체 수행
                        if replacement_value is not None:
                            cleaned_data.loc[idx, column] = replacement_value
                            tba_replacements += 1
                            print(f"  - 행 {idx+1}: TBA → {replacement_value} ({source_description})")
                        else:
                            # 유효한 이전 값을 찾을 수 없는 경우
                            cleaned_data.loc[idx, column] = np.nan
                            print(f"  - 행 {idx+1}: TBA → NaN (유효한 이전 값 없음)")
    
    if tba_replacements > 0:
        print(f"\n✅ 총 {tba_replacements}개의 TBA 값을 전날 값으로 대체했습니다.")
    
    return cleaned_data

def fill_missing_values_advanced(data):
    """고급 결측치 채우기 (forward fill + backward fill)"""
    filled_data = data.copy()
    
    # Date 열 제외한 모든 수치형 열에 대해
    numeric_cols = filled_data.select_dtypes(include=[np.number]).columns
    
    # 이전 값으로 결측치 채우기 (forward fill)
    filled_data[numeric_cols] = filled_data[numeric_cols].ffill()
    
    # 남은 결측치가 있는 경우 다음 값으로 채우기 (backward fill)
    filled_data[numeric_cols] = filled_data[numeric_cols].bfill()
    
    return filled_data

def rename_columns_to_standard(data):
    """열 이름을 표준 형태로 변경"""
    column_mapping = {
        'Date': 'Date',
        'Crude Oil_WTI': 'WTI',
        'Crude Oil_Brent': 'Brent',
        'Crude Oil_Dubai': 'Dubai',
        'WS_AG-SG_55': 'WS_55',
        'WS_75.0': 'WS_75',
        'Naphtha_Platts_MOPJ': 'MOPJ',
        'Naphtha_MOPAG': 'MOPAG',
        'Naphtha_MOPS': 'MOPS',
        'Naphtha_Monthly Spread': 'Monthly Spread',
        'LPG_Argus FEI_C3': 'C3_LPG',
        'LPG_C4': 'C4_LPG',
        'Gasoline_FOB SP_92RON': 'Gasoline_92RON',
        'Gasoline_95RON': 'Gasoline_95RON',
        'Ethylene_Platts_CFR NEA': 'EL_CRF NEA',
        'Ethylene_CFR SEA': 'EL_CRF SEA',
        'Propylene_Platts_FOB Korea': 'PL_FOB Korea',
        'Benzene_Platts_FOB Korea': 'BZ_FOB Korea',
        'Benzene_Platts_FOB SEA': 'BZ_FOB SEA',
        'Benzene_Platts_FOB US M1': 'BZ_FOB US M1',
        'Benzene_Platts_FOB US M2': 'BZ_FOB US M2',
        'Benzene_Platts_H2-TIME SPREAD': 'BZ_H2-TIME SPREAD',
        'Toluene_Platts_FOB Korea': 'TL_FOB Korea',
        'Toluene_Platts_FOB US M1': 'TL_FOB US M1',
        'Toluene_Platts_FOB US M2': 'TL_FOB US M2',
        'MX_Platts FE_FOB K': 'MX_FOB Korea',
        'PX_FOB   Korea': 'PX_FOB Korea',
        'SM_FOB   Korea': 'SM_FOB Korea',
        'RPG Value_Calculated_FOB PG': 'RPG Value_FOB PG',
        'FO_Platts_HSFO 180 CST': 'FO_HSFO 180 CST',
        'MTBE_Platts_FOB S\'pore': 'MTBE_FOB Singapore',
        'MTBE_Dow_Jones': 'Dow_Jones',
        'MTBE_Euro': 'Euro',
        'MTBE_Gold': 'Gold',
        'PP (ICIS)_CIF NWE': 'Europe_CIF NWE',
        'PP (ICIS)_M.G.\n10ppm': 'Europe_M.G_10ppm',
        'PP (ICIS)_RBOB (NYMEX)_M1': 'RBOB (NYMEX)_M1',
        'Brent_WTI': 'Brent_WTI',
        'MOPJ_Mopag_Nap': 'MOPJ_MOPAG',
        'MOPJ_MOPS_Nap': 'MOPJ_MOPS',
        'Naphtha_Spread': 'Naphtha_Spread',
        'MG92_E Nap': 'MG92_E Nap',
        'C3_MOPJ': 'C3_MOPJ',
        'C4_MOPJ': 'C4_MOPJ',
        'Nap_Dubai': 'Nap_Dubai',
        'MG92_Nap_mops': 'MG92_Nap_MOPS',
        '95R_92R_Asia': '95R_92R_Asia',
        'M1_M2_RBOB': 'M1_M2_RBOB',
        'RBOB_Brent_m1': 'RBOB_Brent_m1',
        'RBOB_Brent_m2': 'RBOB_Brent_m2',
        'EL': 'EL_MOPJ',
        'PL': 'PL_MOPJ',
        'BZ_MOPJ': 'BZ_MOPJ',
        'TL': 'TL_MOPJ',
        'PX': 'PX_MOPJ',
        'HD': 'HD_EL',
        'LD_EL': 'LD_EL',
        'LLD': 'LLD_EL',
        'PP_PL': 'PP_PL',
        'SM_EL+BZ_Margin': 'SM_EL+BZ',
        'US_FOBK_BZ': 'US_FOBK_BZ',
        'NAP_HSFO_180': 'NAP_HSFO_180',
        'MTBE_MOPJ': 'MTBE_MOPJ',
        'MTBE_PG': 'Freight_55_PG',
        'MTBE_Maili': 'Freight_55_Maili',
        'Freight (55)_Ruwais_Yosu': 'Freight_55_Yosu',
        'Freight (55)_Daes\'': 'Freight_55_Daes',
        'Freight (55)_Chiba': 'Freight_55_Chiba',
        'Freight (55)_PG': 'Freight_75_PG',
        'Freight (55)_Maili': 'Freight_75_Maili',
        'Freight (75)_Ruwais_Yosu': 'Freight_75_Yosu',
        'Freight (75)_Daes\'': 'Freight_75_Daes',
        'Freight (75)_Chiba': 'Freight_75_Chiba',
        'Freight (75)_PG': 'Flat Rate_PG',
        'Freight (75)_Maili': 'Flat Rate_Maili',
        'Flat Rate_Ruwais_Yosu': 'Flat Rate_Yosu',
        'Flat Rate_Daes\'': 'Flat Rate_Daes',
        'Flat Rate_Chiba': 'Flat Rate_Chiba'
    }
    
    # 실제 존재하는 열만 매핑
    existing_columns = data.columns.tolist()
    final_mapping = {}
    
    for old_name, new_name in column_mapping.items():
        if old_name in existing_columns:
            final_mapping[old_name] = new_name
    
    # 매핑되지 않은 열들 확인
    unmapped_columns = [col for col in existing_columns if col not in column_mapping.keys()]
    if unmapped_columns:
        print(f"\n=== 매핑되지 않은 열들 ===")
        for col in unmapped_columns:
            print(f"- {col}")
    
    # 열 이름 변경
    renamed_data = data.rename(columns=final_mapping)
    
    print(f"\n=== 열 이름 변경 완료 ===")
    print(f"변경된 열 개수: {len(final_mapping)}")
    print(f"최종 데이터 형태: {renamed_data.shape}")
    
    return renamed_data

# process_data_250620.py의 추가 함수들
def remove_missing_and_analyze(data, threshold=10):
    """
    중간 수준의 결측치 비율을 가진 열을 제거하고 분석하는 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 결측치 비율 계산
    missing_ratio = (data.isnull().sum() / len(data)) * 100
    
    # threshold% 이상 결측치가 있는 열 식별
    high_missing_cols = missing_ratio[missing_ratio >= threshold]
    
    if len(high_missing_cols) > 0:
        logger.info(f"\n=== {threshold}% 이상 결측치가 있어 제거될 열 목록 ===")
        for col, ratio in high_missing_cols.items():
            logger.info(f"- {col}: {ratio:.1f}%")
        
        # 결측치가 threshold% 이상인 열 제거
        cleaned_data = data.drop(columns=high_missing_cols.index)
        logger.info(f"\n원본 데이터 형태: {data.shape}")
        logger.info(f"정제된 데이터 형태: {cleaned_data.shape}")
    else:
        cleaned_data = data
        logger.info(f"\n제거할 {threshold}% 이상 결측치 열 없음: {data.shape}")
    
    return cleaned_data

def find_text_missings(data, text_patterns=['NOP', 'No Publication']):
    """
    문자열 형태의 결측치를 찾는 함수
    (process_data_250620.py에서 가져온 함수)
    """
    logger.info("\n=== 문자열 형태의 결측치 분석 ===")
    
    # 각 패턴별로 검사
    for pattern in text_patterns:
        logger.info(f"\n['{pattern}' 포함된 데이터 확인]")
        
        # 모든 열에 대해 검사
        for column in data.columns:
            # 문자열 데이터만 검사
            if data[column].dtype == 'object':
                # 해당 패턴이 포함된 데이터 찾기
                mask = data[column].astype(str).str.contains(pattern, na=False, case=False)
                matches = data[mask]
                
                if len(matches) > 0:
                    logger.info(f"\n열: {column}")
                    logger.info(f"발견된 횟수: {len(matches)}")

def final_clean_data_improved(data):
    """
    최종 데이터 정제 함수 (process_data_250620.py에서 가져온 함수)
    M1_M2_RBOB 컬럼의 결측치나 'Q' 값을 RBOB_Brent_m1 - RBOB_Brent_m2로 계산해서 채움
    """
    # 데이터 복사본 생성
    cleaned_data = data.copy()
    
    # MTBE_Dow_Jones 열 특별 처리
    for col in ['MTBE_Dow_Jones']:
        if col in cleaned_data.columns:
            # 숫자로 변환 시도
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # 🔧 M1_M2_RBOB 열 특별 처리: 결측치와 'Q' 값을 계산으로 채우기
    if 'M1_M2_RBOB' in cleaned_data.columns and 'RBOB_Brent_m1' in cleaned_data.columns and 'RBOB_Brent_m2' in cleaned_data.columns:
        logger.info(f"\n=== M1_M2_RBOB 열 처리 시작 ===")
        logger.info(f"처리 전 데이터 타입: {cleaned_data['M1_M2_RBOB'].dtype}")
        logger.info(f"처리 전 결측치 개수: {cleaned_data['M1_M2_RBOB'].isnull().sum()}")
        
        # 'Q' 값들과 기타 문자열 값들을 NaN으로 변환
        original_values = cleaned_data['M1_M2_RBOB'].copy()
        q_count = 0
        other_string_count = 0
        
        # 'Q' 값 개수 확인
        if cleaned_data['M1_M2_RBOB'].dtype == 'object':
            q_mask = cleaned_data['M1_M2_RBOB'].astype(str).str.upper() == 'Q'
            q_count = q_mask.sum()
            
            # 기타 문자열 값들 확인
            numeric_convertible = pd.to_numeric(cleaned_data['M1_M2_RBOB'], errors='coerce')
            string_mask = pd.isna(numeric_convertible) & cleaned_data['M1_M2_RBOB'].notna()
            other_string_count = string_mask.sum() - q_count
            
            if q_count > 0:
                logger.info(f"'Q' 값 {q_count}개 발견")
            if other_string_count > 0:
                logger.info(f"기타 문자열 값 {other_string_count}개 발견")
        
        # 'Q' 값들과 기타 문자열을 NaN으로 변환
        cleaned_data['M1_M2_RBOB'] = cleaned_data['M1_M2_RBOB'].replace('Q', np.nan)
        cleaned_data['M1_M2_RBOB'] = cleaned_data['M1_M2_RBOB'].replace('q', np.nan)
        
        # 문자열로 저장된 숫자들을 실제 숫자로 변환
        cleaned_data['M1_M2_RBOB'] = pd.to_numeric(cleaned_data['M1_M2_RBOB'], errors='coerce')
        
        # 결측치와 'Q' 값들을 계산으로 채우기: M1_M2_RBOB = RBOB_Brent_m1 - RBOB_Brent_m2
        missing_mask = cleaned_data['M1_M2_RBOB'].isnull()
        missing_count_before = missing_mask.sum()
        
        if missing_count_before > 0:
            logger.info(f"결측치 {missing_count_before}개를 계산으로 채웁니다: M1_M2_RBOB = RBOB_Brent_m1 - RBOB_Brent_m2")
            
            # 계산 가능한 행들만 선택 (m1, m2 둘 다 유효한 값이 있는 경우)
            can_calculate = (missing_mask & 
                           cleaned_data['RBOB_Brent_m1'].notna() & 
                           cleaned_data['RBOB_Brent_m2'].notna())
            calculated_count = can_calculate.sum()
            
            if calculated_count > 0:
                # 계산 수행
                calculated_values = (cleaned_data.loc[can_calculate, 'RBOB_Brent_m1'] - 
                                   cleaned_data.loc[can_calculate, 'RBOB_Brent_m2'])
                
                cleaned_data.loc[can_calculate, 'M1_M2_RBOB'] = calculated_values
                logger.info(f"실제로 계산된 값: {calculated_count}개")
                
                # 계산 검증 (처음 5개 값 출력)
                logger.info(f"=== 계산 검증 (처음 5개 계산된 값) ===")
                calculated_rows = cleaned_data[can_calculate].head(5)
                for idx, row in calculated_rows.iterrows():
                    m1_val = row['RBOB_Brent_m1']
                    m2_val = row['RBOB_Brent_m2']
                    calculated_val = row['M1_M2_RBOB']
                    logger.info(f"인덱스 {idx}: {m1_val:.6f} - {m2_val:.6f} = {calculated_val:.6f}")
                    
            else:
                logger.warning("계산 가능한 행이 없습니다 (RBOB_Brent_m1 또는 RBOB_Brent_m2에 결측치가 있음)")
        
        # 처리 후 결과 확인
        missing_count_after = cleaned_data['M1_M2_RBOB'].isnull().sum()
        valid_count = cleaned_data['M1_M2_RBOB'].notna().sum()
        
        logger.info(f"\n=== M1_M2_RBOB 열 처리 후 ===")
        logger.info(f"데이터 타입: {cleaned_data['M1_M2_RBOB'].dtype}")
        logger.info(f"결측치 개수: {missing_count_after}")
        logger.info(f"유효 데이터 개수: {valid_count}")
        logger.info(f"처리된 결측치 개수: {missing_count_before - missing_count_after}")
        
        if valid_count > 0:
            logger.info(f"최소값: {cleaned_data['M1_M2_RBOB'].min():.6f}")
            logger.info(f"최대값: {cleaned_data['M1_M2_RBOB'].max():.6f}")
            logger.info(f"평균값: {cleaned_data['M1_M2_RBOB'].mean():.6f}")
    
    else:
        # 필요한 컬럼이 없는 경우
        missing_cols = []
        for col in ['M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2']:
            if col not in cleaned_data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"M1_M2_RBOB 계산에 필요한 컬럼이 없습니다: {missing_cols}")
    
    return cleaned_data

def clean_and_trim_data(data, start_date='2013-02-06'):
    """
    데이터 정제 및 날짜 범위 조정 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 시작 날짜 이후의 데이터만 선택
    cleaned_data = data[data['Date'] >= pd.to_datetime(start_date)].copy()
    
    # 기본 정보 출력
    logger.info(f"=== 데이터 처리 결과 ===")
    logger.info(f"원본 데이터 기간: {data['Date'].min()} ~ {data['Date'].max()}")
    logger.info(f"처리된 데이터 기간: {cleaned_data['Date'].min()} ~ {cleaned_data['Date'].max()}")
    logger.info(f"원본 데이터 행 수: {len(data)}")
    logger.info(f"처리된 데이터 행 수: {len(cleaned_data)}")
    
    return cleaned_data

def load_and_process_data_improved(file_path, sheet_name, start_date):
    """
    개선된 데이터 로드 및 처리 함수
    (process_data_250620.py에서 가져온 함수)
    """
    # 열 이름 생성
    column_names = create_proper_column_names(file_path, sheet_name)
    
    # 실제 데이터 읽기
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=3)
    data.columns = column_names
    
    # Date 열 변환
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # 시작 날짜 이후 데이터만 필터링
    data = data[data['Date'] >= start_date]
    
    # 불필요한 열 제거
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    
    return data

def process_excel_data_complete(file_path, sheet_name='29 Nov 2010 till todate', start_date='2013-01-04'):
    """
    Excel 데이터를 완전히 처리하는 통합 함수
    (process_data_250620.py의 메인 처리 파이프라인을 함수화)
    """
    try:
        logger.info("=== Excel 데이터 완전 처리 시작 === 📊")
        
        # 1. 데이터 로드 및 기본 처리
        cleaned_data = load_and_process_data_improved(file_path, sheet_name, pd.Timestamp(start_date))
        logger.info(f"초기 데이터 형태: {cleaned_data.shape}")
        
        # 2. 70% 이상 결측치가 있는 열 제거
        final_data = remove_high_missing_columns(cleaned_data, threshold=70)
        
        # 3. 10% 이상 결측치가 있는 열 제거  
        final_cleaned_data = remove_missing_and_analyze(final_data, threshold=10)
        
        # 4. 텍스트 형태의 결측치 처리
        text_patterns = ['NOP', 'No Publication', 'N/A', 'na', 'NA', 'none', 'None', '-']
        find_text_missings(final_cleaned_data, text_patterns)
        
        # 5. 텍스트 값들 정제
        final_cleaned_data_v2 = clean_text_values_advanced(final_cleaned_data)
        
        # 6. 최종 정제
        final_data_clean = final_clean_data_improved(final_cleaned_data_v2)
        
        # 7. 결측치 채우기
        filled_final_data = fill_missing_values_advanced(final_data_clean)
        
        # 8. 날짜 범위 조정
        trimmed_data = clean_and_trim_data(filled_final_data, start_date='2013-02-06')
        
        # 9. 열 이름을 최종 형태로 변경
        final_renamed_data = rename_columns_to_standard(trimmed_data)
        
        logger.info(f"\n=== 최종 결과 ===")
        logger.info(f"최종 데이터 형태: {final_renamed_data.shape}")
        logger.info(f"최종 열 이름들: {len(final_renamed_data.columns)}개")
        
        return final_renamed_data
        
    except Exception as e:
        logger.error(f"Excel 데이터 처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# xlwings 대안 로더 (보안프로그램이 파일을 잠그는 경우 사용)
try:
    import xlwings as xw
    XLWINGS_AVAILABLE = True
    logger.info("✅ xlwings library available - Excel security bypass enabled")
except ImportError:
    XLWINGS_AVAILABLE = False
    logger.warning("⚠️ xlwings not available - falling back to pandas only")

def load_data_with_xlwings(file_path, model_type=None):
    """
    xlwings를 사용하여 보안프로그램이 파일을 잠그는 상황에서도 안정적으로 Excel 파일을 읽는 함수
    
    Args:
        file_path (str): Excel 파일 경로
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available. Please install it with: pip install xlwings")
    
    logger.info(f"🔓 [XLWINGS] Loading Excel file with security bypass: {os.path.basename(file_path)}")
    
    app = None
    wb = None
    
    try:
        # Excel 애플리케이션을 백그라운드에서 시작
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False  # 경고창 비활성화
        app.screen_updating = False  # 화면 업데이트 비활성화 (성능 향상)
        
        logger.info(f"📱 [XLWINGS] Excel app started (PID: {app.pid})")
        
        # Excel 파일 열기
        wb = app.books.open(file_path, read_only=True, update_links=False)
        logger.info(f"📖 [XLWINGS] Workbook opened: {wb.name}")
        
        # 적절한 시트 찾기
        sheet_names = [sheet.name for sheet in wb.sheets]
        logger.info(f"📋 [XLWINGS] Available sheets: {sheet_names}")
        
        # 기본 시트명 또는 첫 번째 시트 사용
        target_sheet_name = '29 Nov 2010 till todate'
        if target_sheet_name in sheet_names:
            sheet = wb.sheets[target_sheet_name]
            logger.info(f"🎯 [XLWINGS] Using target sheet: {target_sheet_name}")
        else:
            sheet = wb.sheets[0]  # 첫 번째 시트 사용
            logger.info(f"🎯 [XLWINGS] Using first sheet: {sheet.name}")
        
        # 사용된 범위 확인
        used_range = sheet.used_range
        if used_range is None:
            raise ValueError("Sheet appears to be empty")
        
        logger.info(f"📏 [XLWINGS] Used range: {used_range.address}")
        
        # 데이터를 DataFrame으로 읽기 (헤더 포함)
        # xlwings의 expand='table' 옵션으로 자동으로 전체 데이터 범위 감지
        df = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
        
        logger.info(f"📊 [XLWINGS] Raw data loaded: {df.shape}")
        logger.info(f"📋 [XLWINGS] Columns: {list(df.columns)}")
        
        # 데이터 검증
        if df is None or df.empty:
            raise ValueError("No data found in the Excel file")
        
        # Date 컬럼 확인 및 처리
        if 'Date' not in df.columns:
            # 첫 번째 컬럼이 날짜일 가능성 확인
            first_col = df.columns[0]
            if 'date' in first_col.lower() or df[first_col].dtype == 'datetime64[ns]':
                df = df.rename(columns={first_col: 'Date'})
                logger.info(f"🔄 [XLWINGS] Renamed '{first_col}' to 'Date'")
            else:
                raise ValueError("Date column not found in the data")
        
        # Date 컬럼을 datetime으로 변환
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        logger.info(f"📅 [XLWINGS] Date range: {df.index.min()} to {df.index.max()}")
        
        # 모델 타입별 데이터 필터링 (기존 load_data와 동일)
        if model_type == 'lstm':
            cutoff_date = pd.to_datetime('2022-01-01')
            original_shape = df.shape
            df = df[df.index >= cutoff_date]
            logger.info(f"🔍 [XLWINGS] LSTM filter: {original_shape[0]} -> {df.shape[0]} records")
            
            if df.empty:
                raise ValueError("No data available after 2022-01-01 filter for LSTM model")
        
        # 기본 데이터 정제
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ [XLWINGS] Data loaded successfully: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ [XLWINGS] Error loading file: {str(e)}")
        raise e
        
    finally:
        # 리소스 정리
        try:
            if wb is not None:
                wb.close()
                logger.info("📖 [XLWINGS] Workbook closed")
        except:
            pass
        
        try:
            if app is not None:
                app.quit()
                logger.info("📱 [XLWINGS] Excel app closed")
        except:
            pass

def load_csv_with_xlwings(csv_path):
    """
    xlwings를 사용하여 CSV 파일을 읽는 함수 - 보안프로그램 우회
    
    Args:
        csv_path (str): CSV 파일 경로
    
    Returns:
        pd.DataFrame: CSV 데이터프레임
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available for CSV loading")
    
    logger.info(f"🔓 [XLWINGS_CSV] Loading CSV file with security bypass: {os.path.basename(csv_path)}")
    
    app = None
    wb = None
    
    try:
        # Excel 애플리케이션을 백그라운드에서 시작
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False
        
        logger.info(f"📱 [XLWINGS_CSV] Excel app started for CSV")
        
        # CSV 파일을 Excel로 열기 (CSV는 자동으로 파싱됨)
        wb = app.books.open(csv_path, read_only=True, update_links=False)
        logger.info(f"📖 [XLWINGS_CSV] CSV workbook opened: {wb.name}")
        
        # 첫 번째 시트 사용 (CSV는 항상 하나의 시트만 가짐)
        sheet = wb.sheets[0]
        
        # 사용된 범위 확인
        used_range = sheet.used_range
        if used_range is None:
            raise ValueError("CSV file appears to be empty")
        
        logger.info(f"📏 [XLWINGS_CSV] Used range: {used_range.address}")
        
        # 데이터를 DataFrame으로 읽기 (헤더 포함)
        df = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
        
        logger.info(f"📊 [XLWINGS_CSV] CSV data loaded: {df.shape}")
        logger.info(f"📋 [XLWINGS_CSV] Columns: {list(df.columns)}")
        
        # 데이터 검증
        if df is None or df.empty:
            raise ValueError("No data found in the CSV file")
        
        logger.info(f"✅ [XLWINGS_CSV] CSV loaded successfully: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ [XLWINGS_CSV] Error loading CSV file: {str(e)}")
        raise e
        
    finally:
        # 리소스 정리
        try:
            if wb is not None:
                wb.close()
                logger.info("📖 [XLWINGS_CSV] CSV workbook closed")
        except:
            pass
        
        try:
            if app is not None:
                app.quit()
                logger.info("📱 [XLWINGS_CSV] Excel app closed")
        except:
            pass

def load_data_safe_holidays(file_path):
    """
    휴일 파일 전용 xlwings 로딩 함수 - 보안프로그램 우회
    
    Args:
        file_path (str): 휴일 Excel 파일 경로
    
    Returns:
        pd.DataFrame: 휴일 데이터프레임 (date, description 컬럼)
    """
    if not XLWINGS_AVAILABLE:
        raise ImportError("xlwings is not available for holiday file loading")
    
    logger.info(f"🔓 [HOLIDAYS_XLWINGS] Loading holiday file with security bypass: {os.path.basename(file_path)}")
    
    app = None
    wb = None
    
    try:
        # Excel 애플리케이션을 백그라운드에서 시작
        app = xw.App(visible=False, add_book=False)
        app.display_alerts = False
        app.screen_updating = False
        
        logger.info(f"📱 [HOLIDAYS_XLWINGS] Excel app started for holidays")
        
        # Excel 파일 열기
        wb = app.books.open(file_path, read_only=True, update_links=False)
        logger.info(f"📖 [HOLIDAYS_XLWINGS] Holiday workbook opened: {wb.name}")
        
        # 첫 번째 시트 사용 (휴일 파일은 보통 단순 구조)
        sheet = wb.sheets[0]
        logger.info(f"🎯 [HOLIDAYS_XLWINGS] Using sheet: {sheet.name}")
        
        # 사용된 범위 확인
        used_range = sheet.used_range
        if used_range is None:
            raise ValueError("Holiday sheet appears to be empty")
        
        logger.info(f"📏 [HOLIDAYS_XLWINGS] Used range: {used_range.address}")
        
        # 데이터를 DataFrame으로 읽기 (헤더 포함)
        df = sheet['A1'].options(pd.DataFrame, index=False, expand='table').value
        
        logger.info(f"📊 [HOLIDAYS_XLWINGS] Holiday data loaded: {df.shape}")
        logger.info(f"📋 [HOLIDAYS_XLWINGS] Columns: {list(df.columns)}")
        
        # 데이터 검증
        if df is None or df.empty:
            raise ValueError("No holiday data found in the Excel file")
        
        # 컬럼명 정규화 (case-insensitive)
        df.columns = df.columns.str.lower()
        
        # 필수 컬럼 확인
        if 'date' not in df.columns:
            # 첫 번째 컬럼을 날짜로 가정
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'date'})
            logger.info(f"🔄 [HOLIDAYS_XLWINGS] Renamed '{first_col}' to 'date'")
        
        # description 컬럼이 없으면 추가
        if 'description' not in df.columns:
            df['description'] = 'Holiday'
            logger.info(f"➕ [HOLIDAYS_XLWINGS] Added default 'description' column")
        
        logger.info(f"✅ [HOLIDAYS_XLWINGS] Holiday data loaded successfully: {len(df)} holidays")
        return df
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAYS_XLWINGS] Error loading holiday file: {str(e)}")
        raise e
        
    finally:
        # 리소스 정리
        try:
            if wb is not None:
                wb.close()
                logger.info("📖 [HOLIDAYS_XLWINGS] Holiday workbook closed")
        except:
            pass
        
        try:
            if app is not None:
                app.quit()
                logger.info("📱 [HOLIDAYS_XLWINGS] Excel app closed")
        except:
            pass

def load_data_safe(file_path, model_type=None, use_cache=True, use_xlwings_fallback=True):
    """
    안전한 데이터 로딩 함수 - 보안 문제 시 xlwings로 자동 전환
    
    Args:
        file_path (str): 데이터 파일 경로
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
        use_cache (bool): 메모리 캐시 사용 여부
        use_xlwings_fallback (bool): 실패 시 xlwings 사용 여부
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    try:
        # 먼저 기본 load_data 함수 시도
        return load_data(file_path, model_type, use_cache)
        
    except (PermissionError, OSError, pd.errors.ExcelFileError) as e:
        # 파일 접근 오류 시 xlwings로 대체 시도
        if use_xlwings_fallback and XLWINGS_AVAILABLE and file_path.endswith(('.xlsx', '.xls')):
            logger.warning(f"⚠️ [SECURITY_BYPASS] Standard loading failed: {str(e)}")
            logger.info("🔓 [SECURITY_BYPASS] Attempting xlwings bypass...")
            
            try:
                return load_data_with_xlwings(file_path, model_type)
            except Exception as xlwings_error:
                logger.error(f"❌ [SECURITY_BYPASS] xlwings also failed: {str(xlwings_error)}")
                raise e  # 원래 오류를 다시 발생
        else:
            raise e

# 데이터 로딩 및 전처리 함수
def load_data(file_path, model_type=None, use_cache=True):
    """
    데이터 로드 및 기본 전처리
    
    Args:
        file_path (str): 데이터 파일 경로
        model_type (str): 모델 타입 ('lstm', 'varmax', None)
                         - 'lstm': 단일/누적 예측용, 2022년 이전 데이터 제거
                         - 'varmax': 장기예측용, 모든 데이터 유지
                         - None: 기본 동작 (모든 데이터 유지)
        use_cache (bool): 메모리 캐시 사용 여부 (default: True)
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 🔧 메모리 캐시 확인 (중복 로딩 방지)
    cache_key = f"{file_path}|{model_type}|{os.path.getmtime(file_path)}"
    current_time = time.time()
    
    if use_cache and cache_key in _dataframe_cache:
        cached_data, cache_time = _dataframe_cache[cache_key]
        if (current_time - cache_time) < _cache_expiry_seconds:
            logger.info(f"🚀 [CACHE_HIT] Using cached DataFrame for {os.path.basename(file_path)} (saved {current_time - cache_time:.1f}s ago)")
            return cached_data.copy()  # 복사본 반환으로 원본 보호
        else:
            # 만료된 캐시 제거
            del _dataframe_cache[cache_key]
            logger.info(f"🗑️ [CACHE_EXPIRED] Removed expired cache for {os.path.basename(file_path)}")
    
    logger.info(f"📁 [LOAD_DATA] Loading data with model_type: {model_type} from {os.path.basename(file_path)}")

    
    # 파일 확장자에 따라 다른 로드 방법 사용
    if file_path.endswith('.csv'):
        logger.info("Loading CSV file with xlwings fallback support")
        # CSV 파일도 xlwings 우선 시도
        try:
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [XLWINGS_CSV] Attempting to load CSV with xlwings: {file_path}")
                df = load_csv_with_xlwings(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"⚠️ [XLWINGS_CSV] xlwings failed, falling back to pandas: {str(e)}")
            df = pd.read_csv(file_path)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 기본적인 결측치 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
    elif file_path.endswith(('.xlsx', '.xls')):
        logger.info("Loading Excel file with advanced processing pipeline")
        
        # process_data_250620.py의 완전한 Excel 처리 파이프라인 사용
        try:
            # 1. Excel 파일의 적절한 시트 이름 찾기
            sheet_name = '29 Nov 2010 till todate'  # 기본 시트명
            try:
                # 파일의 시트 목록 확인
                excel_file = pd.ExcelFile(file_path)
                available_sheets = excel_file.sheet_names
                logger.info(f"Available sheets: {available_sheets}")
                
                # 적절한 시트 찾기
                if sheet_name not in available_sheets:
                    sheet_name = available_sheets[0]  # 첫 번째 시트 사용
                    logger.info(f"Default sheet not found, using '{sheet_name}' sheet")
            except:
                sheet_name = 0  # 인덱스로 첫 번째 시트 사용
            
            # 2. process_data_250620.py의 완전한 처리 파이프라인 실행
            df = process_excel_data_complete(file_path, sheet_name, start_date='2013-01-04')
            
            if df is None:
                logger.error("Excel 데이터 처리에 실패했습니다. 기본 방식으로 다시 시도합니다.")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                logger.info(f"Excel file processed successfully with advanced pipeline: {df.shape}")
                
            # Date를 인덱스로 설정
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
                
        except Exception as e:
            logger.error(f"Advanced Excel processing failed: {e}")
            logger.info("Falling back to standard Excel loading method")
            
            # 실패시 기본 방식으로 로드
            df = pd.read_excel(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Original data shape: {df.shape} (from {df.index.min()} to {df.index.max()})")
    
    # 🔑 모델 타입별 데이터 필터링
    if model_type == 'lstm':
        # LSTM 모델용: 2022년 이전 데이터 제거
        cutoff_date = pd.to_datetime('2022-01-01')
        original_shape = df.shape
        df = df[df.index >= cutoff_date]
        
        logger.info(f"📊 LSTM model: Filtered data from 2022-01-01")
        logger.info(f"  Original: {original_shape[0]} records")
        logger.info(f"  Filtered: {df.shape[0]} records (removed {original_shape[0] - df.shape[0]} records)")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        if df.empty:
            raise ValueError("No data available after 2022-01-01 filter for LSTM model")
            
    elif model_type == 'varmax':
        # VARMAX 모델용: 모든 데이터 사용
        logger.info(f"📊 VARMAX model: Using all available data")
        logger.info(f"  Full date range: {df.index.min()} to {df.index.max()}")
        
    else:
        # 기본 동작: 모든 데이터 사용
        logger.info(f"📊 Default mode: Using all available data")
        logger.info(f"  Full date range: {df.index.min()} to {df.index.max()}")
    
    # 모든 inf 값을 NaN으로 변환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 결측치 처리 - 모든 컬럼에 동일하게 적용
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 처리 후 남아있는 inf나 nan 확인
    # 숫자 컬럼만 선택해서 isinf 검사
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_nan = df.isnull().any().any()
    has_inf = False
    if len(numeric_cols) > 0:
        has_inf = np.isinf(df[numeric_cols].values).any()
    
    if has_nan or has_inf:
        logger.warning("Dataset still contains NaN or inf values after preprocessing")
        
        # 📊 상세한 컬럼 분석 및 문제 진단
        logger.warning("=" * 60)
        logger.warning("📊 DATA QUALITY ANALYSIS")
        logger.warning("=" * 60)
        
        # 1. 데이터 타입 정보
        logger.warning(f"📋 Total columns: {len(df.columns)}")
        logger.warning(f"🔢 Numeric columns: {len(numeric_cols)} - {list(numeric_cols)}")
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        logger.warning(f"🔤 Non-numeric columns: {len(non_numeric_cols)} - {list(non_numeric_cols)}")
        
        # 2. NaN 값 분석
        problematic_cols_nan = df.columns[df.isnull().any()]
        if len(problematic_cols_nan) > 0:
            logger.warning(f"⚠️ Columns with NaN values: {len(problematic_cols_nan)}")
            for col in problematic_cols_nan:
                nan_count = df[col].isnull().sum()
                total_count = len(df[col])
                percentage = (nan_count / total_count) * 100
                logger.warning(f"   • {col}: {nan_count}/{total_count} ({percentage:.1f}%) NaN")
        
        # 3. inf 값 분석 (숫자 컬럼만)
        problematic_cols_inf = []
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    problematic_cols_inf.append(col)
                    inf_count = np.isinf(df[col]).sum()
                    total_count = len(df[col])
                    percentage = (inf_count / total_count) * 100
                    logger.warning(f"   • {col}: {inf_count}/{total_count} ({percentage:.1f}%) inf values")
        
        if len(problematic_cols_inf) > 0:
            logger.warning(f"⚠️ Columns with inf values: {len(problematic_cols_inf)} - {problematic_cols_inf}")
        
        # 4. 각 컬럼의 데이터 타입과 샘플 값
        logger.warning("📝 Column details:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            sample_values = df[col].dropna().head(3).tolist()
            logger.warning(f"   • {col}: {dtype} ({non_null_count} non-null) - Sample: {sample_values}")
        
        problematic_cols = list(set(list(problematic_cols_nan) + problematic_cols_inf))
        logger.warning("=" * 60)
        logger.warning(f"🎯 SUMMARY: {len(problematic_cols)} problematic columns found: {problematic_cols}")
        logger.warning("=" * 60)
        
        # 추가적인 전처리: 남은 inf/nan 값을 해당 컬럼의 평균값으로 대체 (숫자 컬럼만)
        for col in problematic_cols:
            if col in numeric_cols:
                # 숫자 컬럼에 대해서만 inf 처리
                col_mean = df[col].replace([np.inf, -np.inf], np.nan).mean()
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(col_mean)
            else:
                # 비숫자 컬럼에 대해서는 NaN만 처리
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Final shape after preprocessing: {df.shape}")
    
    # 🔧 메모리 캐시에 저장 (성공적으로 로딩된 경우)
    if use_cache:
        _dataframe_cache[cache_key] = (df.copy(), current_time)
        logger.info(f"💾 [CACHE_SAVE] Saved DataFrame to cache for {os.path.basename(file_path)} (expires in {_cache_expiry_seconds}s)")
        
        # 메모리 관리: 오래된 캐시 정리
        expired_keys = []
        for key, (cached_df, cache_time) in _dataframe_cache.items():
            if (current_time - cache_time) >= _cache_expiry_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del _dataframe_cache[key]
        
        if expired_keys:
            logger.info(f"🗑️ [CACHE_CLEANUP] Removed {len(expired_keys)} expired cache entries")
    
    return df

# 변수 그룹 정의
variable_groups = {
    'crude_oil': ['WTI', 'Brent', 'Dubai'],
    'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
    'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
    'lpg': ['C3_LPG', 'C4_LPG'],
    'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
    'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
    'spread': ['biweekly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
    'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
    'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
    'economics': ['Dow_Jones', 'Euro', 'Gold'],
    'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
    'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
    'Flat Rate_Chiba'],
    'ETF': ['DIG', 'DUG', 'IYE', 'VDE', 'XLE']
}

def load_holidays_from_file(filepath=None):
    """
    CSV 또는 Excel 파일에서 휴일 목록을 로드하는 함수
    
    Args:
        filepath (str): 휴일 목록 파일 경로, None이면 기본 경로 사용
    
    Returns:
        set: 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    # 기본 파일 경로 - holidays 폴더로 변경
    if filepath is None:
        holidays_dir = Path('holidays')
        holidays_dir.mkdir(exist_ok=True)
        filepath = str(holidays_dir / 'holidays.csv')
    
    # 파일 확장자 확인
    _, ext = os.path.splitext(filepath)
    
    # 파일이 존재하지 않으면 기본 휴일 목록 생성
    if not os.path.exists(filepath):
        logger.warning(f"Holiday file {filepath} not found. Creating default holiday file.")
        
        # 기본 2025년 싱가폴 공휴일
        default_holidays = [
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18", 
            "2025-05-01", "2025-05-12", "2025-06-07", "2025-08-09", "2025-10-20", 
            "2025-12-25", "2026-01-01"
        ]
        
        # 기본 파일 생성
        df = pd.DataFrame({'date': default_holidays, 'description': ['Singapore Holiday']*len(default_holidays)})
        
        if ext.lower() == '.xlsx':
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        logger.info(f"Created default holiday file at {filepath}")
        return set(default_holidays)
    
    try:
        # 파일 로드 - 보안 문제를 고려한 안전한 로딩 사용
        if ext.lower() == '.xlsx':
            # Excel 파일의 경우 xlwings 보안 우회 기능 사용
            try:
                df = load_data_safe_holidays(filepath)
            except Exception as e:
                logger.warning(f"⚠️ [HOLIDAYS] xlwings loading failed, using pandas: {str(e)}")
                df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # 'date' 컬럼이 있는지 확인
        if 'date' not in df.columns:
            logger.error(f"Holiday file {filepath} does not have 'date' column")
            return set()
        
        # 날짜 형식 표준화
        holidays = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format: {date_str}")
        
        logger.info(f"Loaded {len(holidays)} holidays from {filepath}")
        return holidays
        
    except Exception as e:
        logger.error(f"Error loading holiday file: {str(e)}")
        logger.error(traceback.format_exc())
        return set()

# 전역 변수로 휴일 집합 관리
holidays = load_holidays_from_file()

def is_holiday(date):
    """주어진 날짜가 휴일인지 확인하는 함수"""
    date_str = format_date(date, '%Y-%m-%d')
    return date_str in holidays

# 데이터에서 평일 빈 날짜를 휴일로 감지하는 함수
def detect_missing_weekdays_as_holidays(df, date_column='Date'):
    """
    데이터프레임에서 평일(월~금)인데 데이터가 없는 날짜들을 휴일로 감지하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        date_column (str): 날짜 컬럼명
    
    Returns:
        set: 감지된 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    if df.empty or date_column not in df.columns:
        return set()
    
    try:
        # 날짜 컬럼을 datetime으로 변환
        df_dates = pd.to_datetime(df[date_column]).dt.date
        date_set = set(df_dates)
        
        # 데이터 범위의 첫 날과 마지막 날
        start_date = min(df_dates)
        end_date = max(df_dates)
        
        # 전체 기간의 모든 평일 생성
        current_date = start_date
        missing_weekdays = set()
        
        while current_date <= end_date:
            # 평일인지 확인 (월요일=0, 일요일=6)
            if current_date.weekday() < 5:  # 월~금
                if current_date not in date_set:
                    missing_weekdays.add(current_date.strftime('%Y-%m-%d'))
            current_date += pd.Timedelta(days=1)
        
        logger.info(f"Detected {len(missing_weekdays)} missing weekdays as potential holidays")
        if missing_weekdays:
            logger.info(f"Missing weekdays sample: {list(missing_weekdays)[:10]}")
        
        return missing_weekdays
        
    except Exception as e:
        logger.error(f"Error detecting missing weekdays: {str(e)}")
        return set()

# 휴일 정보와 데이터 빈 날짜를 결합하는 함수
def get_combined_holidays(df=None, filepath=None):
    """
    휴일 파일의 휴일과 데이터에서 감지된 휴일을 결합하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임 (빈 날짜 감지용)
        filepath (str): 휴일 파일 경로
    
    Returns:
        set: 결합된 휴일 날짜 집합
    """
    # 휴일 파일에서 휴일 로드
    file_holidays = load_holidays_from_file(filepath)
    
    # 데이터에서 빈 평일 감지
    data_holidays = set()
    if df is not None:
        data_holidays = detect_missing_weekdays_as_holidays(df)
    
    # 두 세트 결합
    combined_holidays = file_holidays.union(data_holidays)
    
    logger.info(f"Combined holidays: {len(file_holidays)} from file + {len(data_holidays)} from data = {len(combined_holidays)} total")
    
    return combined_holidays

# 휴일 정보 업데이트 함수
def update_holidays(filepath=None, df=None):
    """휴일 정보를 재로드하는 함수 (데이터 빈 날짜 포함)"""
    global holidays
    holidays = get_combined_holidays(df, filepath)
    return holidays

def update_holidays_safe(filepath=None, df=None):
    """
    안전한 휴일 정보 업데이트 함수 - xlwings 보안 우회 기능 포함
    
    Args:
        filepath (str): 휴일 파일 경로
        df (pd.DataFrame): 데이터 분석용 데이터프레임
    
    Returns:
        set: 업데이트된 휴일 날짜 집합
    """
    global holidays
    
    try:
        # 기본 방식으로 휴일 로드 시도
        holidays = get_combined_holidays(df, filepath)
        logger.info(f"✅ [HOLIDAY_SAFE] Standard holiday loading successful: {len(holidays)} holidays")
        return holidays
        
    except (PermissionError, OSError, pd.errors.ExcelFileError) as e:
        # 파일 접근 오류 시 xlwings로 대체 시도 (Excel 파일만)
        if filepath and filepath.endswith(('.xlsx', '.xls')) and XLWINGS_AVAILABLE:
            logger.warning(f"⚠️ [HOLIDAY_BYPASS] Standard holiday loading failed: {str(e)}")
            logger.info("🔓 [HOLIDAY_BYPASS] Attempting xlwings bypass for holiday file...")
            
            try:
                # xlwings로 휴일 파일 로드
                file_holidays = load_holidays_from_file_safe(filepath)
                
                # 데이터에서 빈 평일 감지 (기존 방식)
                data_holidays = set()
                if df is not None:
                    data_holidays = detect_missing_weekdays_as_holidays(df)
                
                # 두 세트 결합
                holidays = file_holidays.union(data_holidays)
                
                logger.info(f"✅ [HOLIDAY_BYPASS] xlwings holiday loading successful: {len(file_holidays)} from file + {len(data_holidays)} from data = {len(holidays)} total")
                return holidays
                
            except Exception as xlwings_error:
                logger.error(f"❌ [HOLIDAY_BYPASS] xlwings holiday loading also failed: {str(xlwings_error)}")
                # 기본 휴일로 폴백
                logger.info("🔄 [HOLIDAY_FALLBACK] Using default holidays")
                holidays = load_holidays_from_file()  # 기본 파일에서 로드
                return holidays
        else:
            # xlwings를 사용할 수 없으면 기본 휴일로 폴백
            logger.warning(f"⚠️ [HOLIDAY_FALLBACK] Cannot use xlwings, using default holidays: {str(e)}")
            holidays = load_holidays_from_file()  # 기본 파일에서 로드
            return holidays

def load_holidays_from_file_safe(filepath):
    """
    xlwings를 사용한 안전한 휴일 파일 로딩
    
    Args:
        filepath (str): 휴일 파일 경로
    
    Returns:
        set: 휴일 날짜 집합
    """
    try:
        # xlwings로 휴일 파일 로드
        df = load_data_safe_holidays(filepath)
        
        # 날짜 형식 표준화
        holidays_set = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays_set.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format in xlwings holiday data: {date_str}")
        
        logger.info(f"🔓 [HOLIDAY_XLWINGS] Loaded {len(holidays_set)} holidays with xlwings")
        return holidays_set
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAY_XLWINGS] xlwings holiday loading failed: {str(e)}")
        raise e

# TimeSeriesDataset 및 평가 메트릭스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device, prev_values=None):
        if isinstance(X, torch.Tensor):
            self.X = X
            self.y = y
        else:
            self.X = torch.FloatTensor(X).to(device)
            self.y = torch.FloatTensor(y).to(device)
        
        if prev_values is not None:
            if isinstance(prev_values, torch.Tensor):
                self.prev_values = prev_values
            else:
                self.prev_values = torch.FloatTensor(prev_values).to(device)
        else:
            self.prev_values = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.prev_values is not None:
            return self.X[idx], self.y[idx], self.prev_values[idx]
        return self.X[idx], self.y[idx]

# 복합 손실 함수
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_value=None):
        # 차원 맞추기
        if len(target.shape) == 1:
            target = target.view(-1, 1)
        if len(pred.shape) == 1:
            pred = pred.view(-1, 1)
        
        # MSE Loss
        mse_loss = self.mse(pred, target)
        
        # Directional Loss (차원 확인)
        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            directional_loss = -torch.mean(torch.sign(pred_diff) * torch.sign(target_diff))
        else:
            directional_loss = torch.tensor(0.0).to(pred.device)
        
        # Continuity Loss
        continuity_loss = 0
        if prev_value is not None:
            if len(prev_value.shape) == 1:
                prev_value = prev_value.view(-1, 1)
            continuity_loss = self.mse(pred[:, 0:1], prev_value)
        
        return self.alpha * mse_loss + (1 - self.alpha) * directional_loss + self.beta * continuity_loss

# 학습률 스케줄러 클래스 (Purchase_decision_5days.py 방식)

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        # warmup 단계 동안 선형 증가
        lr = self.max_lr * self.current_step / self.warmup_steps
        # warmup 단계를 초과하면 max_lr로 고정
        if self.current_step > self.warmup_steps:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 개선된 LSTM 예측 모델
class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # hidden_size를 8의 배수로 조정
        self.adjusted_hidden = (hidden_size // 8) * 8
        if self.adjusted_hidden < 32:
            self.adjusted_hidden = 32
        
        # LSTM dropout 설정
        self.lstm_dropout = 0.0 if num_layers == 1 else dropout
        
        # 계층적 LSTM 구조
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else self.adjusted_hidden,
                hidden_size=self.adjusted_hidden,
                num_layers=1,
                batch_first=True
            ) for i in range(num_layers)
        ])
        
        # 듀얼 어텐션 메커니즘
        self.temporal_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.adjusted_hidden) for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(self.adjusted_hidden)
        
        # Dropout 레이어
        self.dropout_layer = nn.Dropout(dropout)
        
        # 이전 값 정보를 결합하기 위한 레이어
        self.prev_value_encoder = nn.Sequential(
            nn.Linear(1, self.adjusted_hidden // 4),
            nn.ReLU(),
            nn.Linear(self.adjusted_hidden // 4, self.adjusted_hidden)
        )
        
        # 시계열 특성 추출을 위한 컨볼루션 레이어
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 출력 레이어 - 계층적 구조
        self.output_layers = nn.ModuleList([
            nn.Linear(self.adjusted_hidden, self.adjusted_hidden // 2),
            nn.Linear(self.adjusted_hidden // 2, self.adjusted_hidden // 4),
            nn.Linear(self.adjusted_hidden // 4, output_size)
        ])
        
        # 잔차 연결을 위한 프로젝션 레이어
        self.residual_proj = nn.Linear(self.adjusted_hidden, output_size)
        
    def forward(self, x, prev_value=None, return_attention=False):
        batch_size = x.size(0)
        
        # 계층적 LSTM 처리
        lstm_out = x
        skip_connections = []
        
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = layer_norm(lstm_out)
            lstm_out = self.dropout_layer(lstm_out)
            skip_connections.append(lstm_out)
        
        # 시간적 어텐션
        temporal_context, temporal_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        temporal_context = self.dropout_layer(temporal_context)
        
        # 특징 어텐션
        # 특징 차원으로 변환 (B, L, H) -> (B, H, L)
        feature_input = lstm_out.transpose(1, 2)
        feature_input = self.conv_layers(feature_input)
        feature_input = feature_input.transpose(1, 2)
        
        feature_context, feature_weights = self.feature_attention(feature_input, feature_input, feature_input)
        feature_context = self.dropout_layer(feature_context)
        
        # 컨텍스트 결합
        combined_context = temporal_context + feature_context
        for skip in skip_connections:
            combined_context = combined_context + skip
        
        combined_context = self.final_layer_norm(combined_context)
        
        # 이전 값 정보 처리
        if prev_value is not None:
            prev_value = prev_value.unsqueeze(1) if len(prev_value.shape) == 1 else prev_value
            prev_encoded = self.prev_value_encoder(prev_value)
            combined_context = combined_context + prev_encoded.unsqueeze(1)
        
        # 최종 특징 추출 (마지막 시퀀스)
        final_features = combined_context[:, -1, :]
        
        # 계층적 출력 처리
        out = final_features
        residual = self.residual_proj(final_features)
        
        for i, layer in enumerate(self.output_layers):
            out = layer(out)
            if i < len(self.output_layers) - 1:
                out = F.relu(out)
                out = self.dropout_layer(out)
        
        # 잔차 연결 추가
        out = out + residual
        
        if return_attention:
            attention_weights = {
                'temporal_weights': temporal_weights,
                'feature_weights': feature_weights
            }
            return out, attention_weights
        
        return out
        
    def get_attention_maps(self, x, prev_value=None):
        """어텐션 가중치 맵을 반환하는 함수"""
        with torch.no_grad():
            # forward 메서드에 return_attention=True 전달
            _, attention_weights = self.forward(x, prev_value, return_attention=True)
            
            # 어텐션 가중치 평균 계산 (multi-head -> single map)
            temporal_weights = attention_weights['temporal_weights'].mean(dim=1)  # 헤드 평균
            feature_weights = attention_weights['feature_weights'].mean(dim=1)    # 헤드 평균
            
            return {
                'temporal_weights': temporal_weights.cpu().numpy(),
                'feature_weights': feature_weights.cpu().numpy()
            }

# VolatileLoss 클래스 제거됨 - 단순화를 위해 DirectionalLoss만 사용

# VolatileAwareLSTMPredictor 클래스 제거됨 - 단순화를 위해 ImprovedLSTMPredictor만 사용

#######################################################################
# 반월 기간 관련 함수
#######################################################################

# 1. 반월 기간 계산 함수
def get_semimonthly_period(date):
    """
    날짜를 반월 기간으로 변환하는 함수
    - 1일~15일: "YYYY-MM-SM1"
    - 16일~말일: "YYYY-MM-SM2"
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        semimonthly = f"{year}-{month:02d}-SM1"
    else:
        semimonthly = f"{year}-{month:02d}-SM2"
    
    return semimonthly

# 2. 특정 날짜 이후의 다음 반월 기간 계산 함수
def get_next_semimonthly_period(date):
    """
    주어진 날짜 이후의 다음 반월 기간을 계산하는 함수
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        # 현재 상반월이면 같은 달의 하반월
        semimonthly = f"{year}-{month:02d}-SM2"
    else:
        # 현재 하반월이면 다음 달의 상반월
        if month == 12:
            # 12월 하반월이면 다음 해 1월 상반월
            semimonthly = f"{year+1}-01-SM1"
        else:
            semimonthly = f"{year}-{(month+1):02d}-SM1"
    
    return semimonthly

# 3. 반월 기간의 시작일과 종료일 계산 함수
def get_semimonthly_date_range(semimonthly_period):
    """
    반월 기간 문자열을 받아 시작일과 종료일을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    tuple
        (시작일, 종료일) - datetime 객체
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월 (1일~15일)
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = pd.Timestamp(year=year, month=month, day=15)
    else:
        # 하반월 (16일~말일)
        start_date = pd.Timestamp(year=year, month=month, day=16)
        _, last_day = calendar.monthrange(year, month)
        end_date = pd.Timestamp(year=year, month=month, day=last_day)
    
    return start_date, end_date

# 4. 다음 반월의 모든 날짜 목록 생성 함수
def get_next_semimonthly_dates(reference_date, original_df):
    """
    참조 날짜 기준으로 다음 반월 기간에 속하는 모든 영업일 목록을 반환하는 함수
    """
    # 다음 반월 기간 계산
    next_period = get_next_semimonthly_period(reference_date)
    
    logger.info(f"Calculating next semimonthly dates from reference: {format_date(reference_date)} → target period: {next_period}")
    
    # 반월 기간의 시작일과 종료일 계산
    start_date, end_date = get_semimonthly_date_range(next_period)
    
    logger.info(f"Target period date range: {format_date(start_date)} ~ {format_date(end_date)}")
    
    # 이 기간에 속하는 영업일(월~금, 휴일 제외) 선택
    business_days = []
    
    # 원본 데이터에서 찾기
    future_dates = original_df.index[original_df.index > reference_date]
    for date in future_dates:
        if start_date <= date <= end_date and date.weekday() < 5 and not is_holiday(date):
            business_days.append(date)
    
    # 원본 데이터에 없는 경우, 날짜 범위에서 직접 생성
    if len(business_days) == 0:
        logger.info(f"No business days found in original data for period {next_period}. Generating from date range.")
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5 and not is_holiday(current_date):
                business_days.append(current_date)
            current_date += pd.Timedelta(days=1)
    
    # 날짜가 없거나 부족하면 추가 로직
    min_required_days = 5
    if len(business_days) < min_required_days:
        logger.warning(f"Only {len(business_days)} business days found in period {next_period}. Creating synthetic dates.")
        
        if business_days:
            synthetic_date = business_days[-1] + pd.Timedelta(days=1)
        else:
            synthetic_date = max(reference_date, start_date) + pd.Timedelta(days=1)
        
        while len(business_days) < 15 and synthetic_date <= end_date:
            if synthetic_date.weekday() < 5 and not is_holiday(synthetic_date):
                business_days.append(synthetic_date)
            synthetic_date += pd.Timedelta(days=1)
        
        logger.info(f"Created synthetic dates. Total business days: {len(business_days)} for period {next_period}")
    
    business_days.sort()
    logger.info(f"Final business days for purchase interval: {len(business_days)} days in {next_period}")
    
    return business_days, next_period

# 5. 다음 N 영업일 계산 함수
def get_next_n_business_days(current_date, original_df, n_days=23):
    """
    현재 날짜 이후의 n_days 영업일을 반환하는 함수 - 원본 데이터에 없는 미래 날짜도 생성
    휴일(주말 및 공휴일)은 제외
    """
    # 현재 날짜 이후의 데이터프레임에서 영업일 찾기
    future_df = original_df[original_df.index > current_date]
    
    # 필요한 수의 영업일 선택
    business_days = []
    
    # 먼저 데이터프레임에 있는 영업일 추가
    for date in future_df.index:
        if date.weekday() < 5 and not is_holiday(date):  # 월~금이고 휴일이 아닌 경우만 선택
            business_days.append(date)
        
        if len(business_days) >= n_days:
            break
    
    # 데이터프레임에서 충분한 날짜를 찾지 못한 경우 합성 날짜 생성
    if len(business_days) < n_days:
        # 마지막 날짜 또는 현재 날짜에서 시작
        last_date = business_days[-1] if business_days else current_date
        
        # 필요한 만큼 추가 날짜 생성
        current = last_date + pd.Timedelta(days=1)
        while len(business_days) < n_days:
            if current.weekday() < 5 and not is_holiday(current):  # 월~금이고 휴일이 아닌 경우만 포함
                business_days.append(current)
            current += pd.Timedelta(days=1)
    
    logger.info(f"Generated {len(business_days)} business days, excluding holidays")
    return business_days

# 6. 구간별 평균 가격 계산 및 점수 부여 함수
def calculate_interval_averages_and_scores(predictions, business_days, min_window_size=5):
    """
    다음 반월 기간에 대해 다양한 크기의 구간별 평균 가격을 계산하고 점수를 부여하는 함수
    - 반월 전체 영업일 수에 맞춰 윈도우 크기 범위 조정
    - global_rank 방식: 모든 구간을 비교해 전역적으로 가장 저렴한 구간에 점수 부여
    
    Parameters:
    -----------
    predictions : list
        날짜별 예측 가격 정보 (딕셔너리 리스트)
    business_days : list
        다음 반월의 영업일 목록
    min_window_size : int
        최소 고려할 윈도우 크기 (기본값: 3)
    
    Returns:
    -----------
    tuple
        (구간별 평균 가격 정보, 구간별 점수 정보, 분석 추가 정보)
    """
    import numpy as np
    
    # 예측 데이터를 날짜별로 정리
    predictions_dict = {pred['Date']: pred['Prediction'] for pred in predictions if pred['Date'] in business_days}
    
    # 날짜 순으로 정렬된 영업일 목록
    sorted_days = sorted(business_days)
    
    # 다음 반월 총 영업일 수 계산
    total_days = len(sorted_days)
    
    # 최소 윈도우 크기와 최대 윈도우 크기 설정 (최대는 반월 전체 일수)
    max_window_size = total_days
    
    # 고려할 모든 윈도우 크기 범위 생성
    window_sizes = range(min_window_size, max_window_size + 1)
    
    print(f"다음 반월 영업일: {total_days}일, 고려할 윈도우 크기: {list(window_sizes)}")
    
    # 각 윈도우 크기별 결과 저장
    interval_averages = {}
    
    # 모든 구간을 저장할 리스트
    all_intervals = []
    
    # 각 윈도우 크기에 대해 모든 가능한 구간 계산
    for window_size in window_sizes:
        window_results = []
        
        # 가능한 모든 시작점에 대해 윈도우 평균 계산
        for i in range(len(sorted_days) - window_size + 1):
            interval_days = sorted_days[i:i+window_size]
            
            # 모든 날짜에 예측 가격이 있는지 확인
            if all(day in predictions_dict for day in interval_days):
                avg_price = np.mean([predictions_dict[day] for day in interval_days])
                
                interval_info = {
                    'start_date': interval_days[0],
                    'end_date': interval_days[-1],
                    'days': window_size,
                    'avg_price': avg_price,
                    'dates': interval_days.copy()
                }
                
                window_results.append(interval_info)
                all_intervals.append(interval_info)  # 모든 구간 목록에도 추가
        
        # 해당 윈도우 크기에 대한 결과 저장 (참고용)
        if window_results:
            # 평균 가격 기준으로 정렬
            window_results.sort(key=lambda x: x['avg_price'])
            interval_averages[window_size] = window_results
    
    # 구간 점수 계산을 위한 딕셔너리
    interval_scores = {}
    
    # Global Rank 전략: 모든 구간을 통합하여 가격 기준으로 정렬
    all_intervals.sort(key=lambda x: x['avg_price'])
    
    # 상위 3개 구간에만 점수 부여 (전체 중에서)
    for i, interval in enumerate(all_intervals[:min(3, len(all_intervals))]):
        score = 3 - i  # 1등: 3점, 2등: 2점, 3등: 1점
        
        # 구간 식별을 위한 키 생성 (문자열 키로 변경)
        interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
        
        # 점수 정보 저장
        interval_scores[interval_key] = {
            'start_date': format_date(interval['start_date']),  # 형식 적용
            'end_date': format_date(interval['end_date']),      # 형식 적용
            'days': interval['days'],
            'avg_price': interval['avg_price'],
            'dates': [format_date(d) for d in interval['dates']],  # 날짜 목록도 형식 적용
            'score': score,
            'rank': i + 1
        }
    
    # 분석 정보 추가
    analysis_info = {
        'total_days': total_days,
        'window_sizes': list(window_sizes),
        'total_intervals': len(all_intervals),
        'min_avg_price': min([interval['avg_price'] for interval in all_intervals]) if all_intervals else None,
        'max_avg_price': max([interval['avg_price'] for interval in all_intervals]) if all_intervals else None
    }
    
    # 결과 출력 (참고용)
    if interval_scores:
        top_interval = max(interval_scores.values(), key=lambda x: x['score'])
        print(f"\n최고 점수 구간: {top_interval['days']}일 구간 ({format_date(top_interval['start_date'])} ~ {format_date(top_interval['end_date'])})")
        print(f"점수: {top_interval['score']}, 순위: {top_interval['rank']}, 평균가: {top_interval['avg_price']:.2f}")
    
    return interval_averages, interval_scores, analysis_info

# 7. 두 구매 방법의 결과 비교 함수
def decide_purchase_interval(interval_scores):
    """
    점수가 부여된 구간들 중에서 최종 구매 구간을 결정하는 함수
    - 점수가 가장 높은 구간 선택
    - 동점인 경우 평균 가격이 더 낮은 구간 선택
    
    Parameters:
    -----------
    interval_scores : dict
        구간별 점수 정보
    
    Returns:
    -----------
    dict
        최종 선택된 구매 구간 정보
    """
    if not interval_scores:
        return None
    
    # 점수가 가장 높은 구간 선택
    max_score = max(interval['score'] for interval in interval_scores.values())
    
    # 최고 점수를 가진 모든 구간 찾기
    top_intervals = [interval for interval in interval_scores.values() 
                    if interval['score'] == max_score]
    
    # 동점이 있는 경우, 평균 가격이 더 낮은 구간 선택
    if len(top_intervals) > 1:
        best_interval = min(top_intervals, key=lambda x: x['avg_price'])
        best_interval['selection_reason'] = "최고 점수 중 최저 평균가 구간"
    else:
        best_interval = top_intervals[0]
        best_interval['selection_reason'] = "최고 점수 구간"
    
    return best_interval

#######################################################################
# 특성 선택 함수
#######################################################################

def calculate_group_vif(df, variables):
    """그룹 내 변수들의 VIF 계산"""
    # 변수가 한 개 이하면 VIF 계산 불가
    if len(variables) <= 1:
        return pd.DataFrame({
            "Feature": variables,
            "VIF": [1.0] * len(variables)
        })
    
    # 모든 변수가 데이터프레임에 존재하는지 확인
    available_vars = [var for var in variables if var in df.columns]
    if len(available_vars) <= 1:
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [1.0] * len(available_vars)
        })
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = available_vars
        vif_data["VIF"] = [variance_inflation_factor(df[available_vars].values, i) 
                          for i in range(len(available_vars))]
        return vif_data.sort_values('VIF', ascending=False)
    except Exception as e:
        logger.error(f"Error calculating VIF: {str(e)}")
        # 오류 발생 시 기본값 반환
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [float('nan')] * len(available_vars)
        })

def analyze_group_correlations(df, variable_groups, target_col='MOPJ'):
    """그룹별 상관관계 분석"""
    logger.info("Analyzing correlations for each group:")
    group_correlations = {}
    
    for group_name, variables in variable_groups.items():
        # 각 그룹의 변수들과 타겟 변수의 상관관계 계산
        # 해당 그룹의 변수들이 데이터프레임에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
            
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        group_correlations[group_name] = correlations
        
        logger.info(f"\n{group_name} group correlations with {target_col}:")
        logger.info(str(correlations))
    
    return group_correlations

def select_features_from_groups(df, variable_groups, target_col='MOPJ', vif_threshold=50.0, corr_threshold=0.8):
    """각 그룹에서 대표 변수 선택"""
    selected_features = []
    selection_process = {}
    
    logger.info(f"\nCorrelation threshold: {corr_threshold}")
    
    for group_name, variables in variable_groups.items():
        logger.info(f"\nProcessing {group_name} group:")
        
        # 해당 그룹의 변수들이 df에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
        
        # 그룹 내 상관관계 계산
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        logger.info(f"\nCorrelations with {target_col}:")
        logger.info(str(correlations))
        
        # 상관관계가 임계값 이상인 변수만 필터링
        high_corr_vars = correlations[correlations >= corr_threshold].index.tolist()
        
        if not high_corr_vars:
            logger.warning(f"Warning: No variables in {group_name} group meet the correlation threshold of {corr_threshold}")
            continue
        
        # 상관관계 임계값을 만족하는 변수들에 대해 VIF 계산
        if len(high_corr_vars) > 1:
            vif_data = calculate_group_vif(df[high_corr_vars], high_corr_vars)
            logger.info(f"\nVIF values for {group_name} group (high correlation vars only):")
            logger.info(str(vif_data))
            
            # VIF 기준 적용하여 다중공선성 낮은 변수 선택
            low_vif_vars = vif_data[vif_data['VIF'] < vif_threshold]['Feature'].tolist()
            
            if low_vif_vars:
                # 낮은 VIF 변수들 중 상관관계가 가장 높은 변수 선택
                for var in correlations.index:
                    if var in low_vif_vars:
                        selected_var = var
                        break
                else:
                    selected_var = high_corr_vars[0]
            else:
                selected_var = high_corr_vars[0]
        else:
            selected_var = high_corr_vars[0]
            vif_data = pd.DataFrame({"Feature": [selected_var], "VIF": [1.0]})
        
        # 선택된 변수가 상관관계 임계값을 만족하는지 확인 (안전장치)
        if correlations[selected_var] >= corr_threshold:
            selected_features.append(selected_var)
            
            selection_process[group_name] = {
                'selected_variable': selected_var,
                'correlation': correlations[selected_var],
                'all_correlations': correlations.to_dict(),
                'vif_data': vif_data.to_dict() if not vif_data.empty else {},
                'high_corr_vars': high_corr_vars
            }
            
            logger.info(f"\nSelected variable from {group_name}: {selected_var} (corr: {correlations[selected_var]:.4f})")
        else:
            logger.info(f"\nNo variable selected from {group_name}: correlation threshold not met")
    
    # 상관관계 기준 재확인 (최종 안전장치)
    final_features = []
    for feature in selected_features:
        corr = abs(df[feature].corr(df[target_col]))
        if corr >= corr_threshold:
            final_features.append(feature)
            logger.info(f"Final selection: {feature} (corr: {corr:.4f})")
        else:
            logger.info(f"Excluded: {feature} (corr: {corr:.4f}) - below threshold")
    
    # 타겟 컬럼이 포함되어 있지 않으면 추가
    if target_col not in final_features:
        final_features.append(target_col)
        logger.info(f"Added target column: {target_col}")
    
    # 최소 특성 수 확인
    if len(final_features) < 3:
        logger.warning(f"Selected features ({len(final_features)}) < 3, lowering threshold to 0.5")
        return select_features_from_groups(df, variable_groups, target_col, vif_threshold, 0.5)
    
    return final_features, selection_process

def find_compatible_hyperparameters(current_file_path, current_period):
    """
    현재 파일이 기존 파일의 확장인 경우, 기존 파일의 호환 가능한 하이퍼파라미터를 찾는 함수
    
    Parameters:
    -----------
    current_file_path : str
        현재 파일 경로
    current_period : str
        현재 예측 기간
        
    Returns:
    --------
    dict or None: {
        'hyperparams': dict,
        'source_file': str,
        'extension_info': dict
    } 또는 None (호환 가능한 하이퍼파라미터가 없을 경우)
    """
    try:
        # uploads 폴더의 다른 파일들을 확인 (🔧 수정: xlsx 파일도 포함)
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != current_file_path]
        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 탐색할 기존 파일 수: {len(existing_files)}")
        for i, file in enumerate(existing_files):
            logger.info(f"    {i+1}. {file.name}")
        
        for existing_file in existing_files:
            try:
                # 🔧 수정: 확장 관계 확인 + 단순 파일명 유사성 확인
                extension_result = check_data_extension(str(existing_file), current_file_path)
                is_extension = extension_result.get('is_extension', False)
                
                # 📝 확장 관계가 인식되지 않는 경우 파일명 유사성으로 대체 확인
                if not is_extension:
                    existing_name = existing_file.stem.lower()
                    current_name = Path(current_file_path).stem.lower()
                    # 기본 이름이 같거나 하나가 다른 하나를 포함하는 경우
                    if (existing_name in current_name or current_name in existing_name or 
                        existing_name.replace('_', '') == current_name.replace('_', '')):
                        is_extension = True
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 파일명 유사성으로 확장 관계 인정: {existing_file.name} -> {Path(current_file_path).name}")
                
                if is_extension:
                    if extension_result.get('is_extension', False):
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 확장 관계 발견: {existing_file.name} -> {Path(current_file_path).name}")
                        logger.info(f"    📈 Extension type: {extension_result.get('validation_details', {}).get('extension_type', 'Unknown')}")
                        logger.info(f"    ➕ New rows: {extension_result.get('new_rows_count', 0)}")
                    else:
                        logger.info(f"🔍 [HYPERPARAMS_SEARCH] 파일명 유사성 기반 호환성 인정: {existing_file.name} -> {Path(current_file_path).name}")
                    
                    # 기존 파일의 하이퍼파라미터 캐시 확인
                    existing_cache_dirs = get_file_cache_dirs(str(existing_file))
                    existing_models_dir = existing_cache_dirs['models']
                    
                    if os.path.exists(existing_models_dir):
                        # 해당 기간의 하이퍼파라미터 파일 찾기
                        hyperparams_pattern = f"hyperparams_kfold_{current_period.replace('-', '_')}.json"
                        hyperparams_file = os.path.join(existing_models_dir, hyperparams_pattern)
                        
                        if os.path.exists(hyperparams_file):
                            try:
                                with open(hyperparams_file, 'r') as f:
                                    hyperparams = json.load(f)
                                
                                logger.info(f"✅ [HYPERPARAMS_SEARCH] 기존 파일에서 호환 하이퍼파라미터 발견!")
                                logger.info(f"    📁 Source file: {existing_file.name}")
                                logger.info(f"    📊 Hyperparams file: {hyperparams_pattern}")
                                
                                return {
                                    'hyperparams': hyperparams,
                                    'source_file': str(existing_file),
                                    'extension_info': extension_result,
                                    'period': current_period
                                }
                                
                            except Exception as e:
                                logger.warning(f"기존 하이퍼파라미터 파일 로드 실패 ({existing_file.name}): {str(e)}")
                        else:
                            # ❌ 삭제된 부분: 다른 기간의 하이퍼파라미터를 대체로 사용하는 로직 제거
                            logger.info(f"🔍 [HYPERPARAMS_SEARCH] {current_period} 기간의 하이퍼파라미터가 없습니다. 새로운 최적화가 필요합니다.")
                    
            except Exception as e:
                logger.warning(f"파일 확장 관계 확인 실패 ({existing_file.name}): {str(e)}")
                continue
        
        logger.info(f"❌ [HYPERPARAMS_SEARCH] 호환 가능한 하이퍼파라미터를 찾지 못했습니다.")
        return None
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 호환성 탐색 중 오류: {str(e)}")
        return None

def optimize_hyperparameters_semimonthly_kfold(train_data, input_size, target_col_idx, device, current_period, file_path=None, n_trials=30, k_folds=10, use_cache=True):
    """
    시계열 K-fold 교차 검증을 사용하여 반월별 데이터에 대한 하이퍼파라미터 최적화 (Purchase_decision_5days.py 방식)
    """
    # 일관된 하이퍼파라미터 최적화를 위한 시드 고정
    set_seed()
    
    logger.info(f"\n===== {current_period} 하이퍼파라미터 최적화 시작 (시계열 {k_folds}-fold 교차 검증) =====")
    
    # 🔧 확장된 하이퍼파라미터 캐시 로직 - 기존 파일의 하이퍼파라미터도 탐색
    file_cache_dir = get_file_cache_dirs(file_path)['models']
    cache_file = os.path.join(file_cache_dir, f"hyperparams_kfold_{current_period.replace('-', '_')}.json")
    logger.info(f"📁 하이퍼파라미터 캐시 파일: {cache_file}")
    
    # models 디렉토리 생성
    os.makedirs(file_cache_dir, exist_ok=True)
    
    # 🔍 1단계: 하이퍼파라미터 캐시 확인
    if use_cache:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_params = json.load(f)
                logger.info(f"✅ [{current_period}] 하이퍼파라미터 로드 완료")
                return cached_params
            except Exception as e:
                logger.error(f"캐시 파일 로드 오류: {str(e)}")
    
    # 🔍 2단계: 데이터 확장 시 기존 파일의 동일 기간 하이퍼파라미터 탐색
    if use_cache:
        logger.info(f"🔍 [{current_period}] 현재 파일에 캐시가 없습니다. 기존 파일에서 동일 기간의 하이퍼파라미터를 탐색합니다...")
        compatible_hyperparams = find_compatible_hyperparameters(file_path, current_period)
        if compatible_hyperparams:
            logger.info(f"✅ [{current_period}] 동일 기간의 호환 가능한 하이퍼파라미터를 발견했습니다!")
            logger.info(f"    📁 Source: {compatible_hyperparams['source_file']}")
            logger.info(f"    📊 Extension info: {compatible_hyperparams['extension_info']}")
            
            # 🔧 수정: 캐시 저장에 실패해도 기존 하이퍼파라미터 반환
            try:
                with open(cache_file, 'w') as f:
                    json.dump(compatible_hyperparams['hyperparams'], f, indent=2)
                logger.info(f"💾 [{current_period}] 호환 하이퍼파라미터를 현재 파일에 저장했습니다.")
            except Exception as e:
                logger.warning(f"⚠️ 하이퍼파라미터 저장 실패, 하지만 기존 하이퍼파라미터를 사용합니다: {str(e)}")
            
            # 🔑 핵심: 저장 성공/실패 여부와 관계없이 기존 하이퍼파라미터 반환
            logger.info(f"🚀 [{current_period}] 기존 하이퍼파라미터를 사용하여 최적화를 건너뜁니다.")
            return compatible_hyperparams['hyperparams']
                
        logger.info(f"🆕 [{current_period}] 동일 기간의 기존 하이퍼파라미터가 없습니다. 새로운 최적화를 진행합니다.")
    
            # 기본 하이퍼파라미터 정의 (최적화 실패 시 사용)
    default_params = {
        'sequence_length': 46,
        'hidden_size': 224,
        'num_layers': 6,
        'dropout': 0.318369281841675,
        'batch_size': 49,
        'learning_rate': 0.0009452489017042499,
        'num_epochs': 72,
        'loss_alpha': 0.7,  # DirectionalLoss alpha
        'loss_beta': 0.2,   # DirectionalLoss beta
        'patience': 14
    }
    
    # 데이터 길이 확인 - 충분하지 않으면 바로 기본값 반환
    MIN_DATA_SIZE = 100
    if len(train_data) < MIN_DATA_SIZE:
        logger.warning(f"훈련 데이터가 너무 적습니다 ({len(train_data)} 데이터 포인트 < {MIN_DATA_SIZE}). 기본 파라미터를 사용합니다.")
        return default_params
    
    # K-fold 분할 로직
    predict_window = 23  # 예측 윈도우 크기
    min_fold_size = 20 + predict_window + 5  # 최소 시퀀스 길이 + 예측 윈도우 + 여유
    max_possible_folds = len(train_data) // min_fold_size
    
    if max_possible_folds < 2:
        logger.warning(f"데이터가 충분하지 않아 k-fold를 수행할 수 없습니다 (가능한 fold: {max_possible_folds} < 2). 기본 파라미터를 사용합니다.")
        return default_params
    
    # 실제 사용 가능한 fold 수 조정
    k_folds = min(k_folds, max_possible_folds)
    fold_size = len(train_data) // (k_folds + 1)  # +1은 예측 윈도우를 위한 추가 부분

    logger.info(f"데이터 크기: {len(train_data)}, Fold 수: {k_folds}, 각 Fold 크기: {fold_size}")

    # fold 분할을 위한 인덱스 생성
    folds = []
    for i in range(k_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        train_indices = list(range(0, test_start)) + list(range(test_end, len(train_data)))
        test_indices = list(range(test_start, test_end))
        
        folds.append((train_indices, test_indices))
    
    # Optuna 목적 함수 정의
    def objective(trial):
        # 일관된 하이퍼파라미터 최적화를 위한 시드 고정
        set_seed(SEED + trial.number)  # trial마다 다른 시드로 다양성 보장하면서도 재현 가능
        
        # 하이퍼파라미터 범위 수정 - 시퀀스 길이 최대값 제한
        max_seq_length = min(fold_size - predict_window - 5, 60)
        
        # 최소 시퀀스 길이도 제한
        min_seq_length = min(10, max_seq_length)
        
        if max_seq_length <= min_seq_length:
            logger.warning(f"시퀀스 길이 범위가 너무 제한적입니다 (min={min_seq_length}, max={max_seq_length}). 해당 trial 건너뛰기.")
            return float('inf')
        
        params = {
            'sequence_length': trial.suggest_int('sequence_length', min_seq_length, max_seq_length),
            'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=8),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 16, min(128, fold_size)),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'num_epochs': trial.suggest_int('num_epochs', 50, 200),
            'patience': trial.suggest_int('patience', 10, 30),
            'loss_alpha': trial.suggest_float('loss_alpha', 0.5, 0.9),
            'loss_beta': trial.suggest_float('loss_beta', 0.1, 0.3)
        }
        
        # loss_gamma 제거됨 - 단순화를 위해 DirectionalLoss만 사용
        
        # K-fold 교차 검증
        fold_losses = []
        valid_fold_count = 0
        
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            try:
                # 시퀀스 길이가 fold 크기보다 크면 건너뛰기
                if params['sequence_length'] >= len(test_indices):
                    logger.warning(f"Fold {fold_idx+1}: 시퀀스 길이({params['sequence_length']})가 테스트 데이터({len(test_indices)})보다 큽니다.")
                    continue
                
                # fold별 훈련/테스트 데이터 준비
                fold_train_data = train_data[train_indices]
                fold_test_data = train_data[test_indices]
                
                # 데이터 준비
                X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
                    fold_train_data, fold_test_data, params['sequence_length'],
                    predict_window, target_col_idx, augment=False
                )
                
                # 데이터가 충분한지 확인
                if len(X_train) < params['batch_size'] or len(X_val) < 1:
                    logger.warning(f"Fold {fold_idx+1}: 데이터 불충분 (훈련: {len(X_train)}, 검증: {len(X_val)})")
                    continue
                
                # 데이터셋 및 로더 생성 (CPU에서 생성, 학습 시 GPU로 이동)
                train_dataset = TimeSeriesDataset(X_train, y_train, torch.device('cpu'), prev_train)
                batch_size = min(params['batch_size'], len(X_train))
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=g
                )
                
                val_dataset = TimeSeriesDataset(X_val, y_val, torch.device('cpu'), prev_val)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                
                # 모델 생성
                model = ImprovedLSTMPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    output_size=predict_window
                ).to(device)

                # 손실 함수 생성
                criterion = DirectionalLoss(
                    alpha=params['loss_alpha'],
                    beta=params['loss_beta']
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                
                # 스케줄러 설정
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5,
                    patience=params['patience']//2
                )

                # best_val_loss 변수 명시적 정의
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(params['num_epochs']):
                    # 학습
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch, prev_batch in train_loader:
                        optimizer.zero_grad()
                        
                        # 모델과 같은 디바이스로 데이터 이동
                        X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                        y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                        prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                        
                        # 모델 예측 및 손실 계산
                        y_pred = model(X_batch, prev_batch)
                        loss = criterion(y_pred, y_batch, prev_batch)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # 검증
                    model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for X_batch, y_batch, prev_batch in val_loader:
                            # 모델과 같은 디바이스로 데이터 이동
                            X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                            y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                            prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                            
                            # 모델 예측 및 손실 계산
                            y_pred = model(X_batch, prev_batch)
                            loss = criterion(y_pred, y_batch, prev_batch)
                            
                            val_loss += loss.item()
                        
                        val_loss /= len(val_loader)
                        
                        # 스케줄러 업데이트
                        scheduler.step(val_loss)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= params['patience']:
                            break
                
                valid_fold_count += 1
                fold_losses.append(best_val_loss)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx+1}: {str(e)}")
                continue
        
        # 모든 fold가 실패한 경우 매우 큰 손실값 반환
        if not fold_losses:
            logger.warning("모든 fold가 실패했습니다. 이 파라미터 조합은 건너뜁니다.")
            return float('inf')
        
        # 성공한 fold의 평균 손실값 반환
        return sum(fold_losses) / len(fold_losses)
    
    # Optuna 최적화 시도
    try:
        import optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 하이퍼파라미터
        if study.best_trial.value == float('inf'):
            logger.warning(f"모든 trial이 실패했습니다. 기본 하이퍼파라미터를 사용합니다.")
            return default_params
            
        best_params = study.best_params
        logger.info(f"\n{current_period} 최적 하이퍼파라미터 (K-fold):")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 모든 필수 키가 있는지 확인
        required_keys = ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                        'batch_size', 'learning_rate', 'num_epochs', 'patience',
                        'warmup_steps', 'lr_factor', 'lr_patience', 'min_lr',
                        'loss_alpha', 'loss_beta', 'loss_gamma', 'loss_delta']
        
        for key in required_keys:
            if key not in best_params:
                # 누락된 키가 있으면 기본값 할당
                if key == 'warmup_steps':
                    best_params[key] = 382
                elif key == 'lr_factor':
                    best_params[key] = 0.49
                elif key == 'lr_patience':
                    best_params[key] = 8
                elif key == 'min_lr':
                    best_params[key] = 1e-7
                elif key == 'loss_gamma':
                    best_params[key] = 0.07
                elif key == 'loss_delta':
                    best_params[key] = 0.07
                else:
                    best_params[key] = default_params[key]
        
        # 캐시에 저장
        with open(cache_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"하이퍼파라미터가 {cache_file}에 저장되었습니다.")
        
        return best_params
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 최적화 오류: {str(e)}")
        traceback.print_exc()
        
        # 오류 시 기본 하이퍼파라미터 반환
        return default_params

#######################################################################
# 예측 결과 저장/로드 함수들
#######################################################################

def save_prediction_simple(prediction_results: dict, prediction_date):
    """리스트·딕트 어떤 구조든 저장 가능한 안전 버전 - 파일명 통일"""
    try:
        preds_root = prediction_results.get("predictions")

        # ── 첫 예측 레코드 추출 ─────────────────────────
        if isinstance(preds_root, dict) and preds_root:
            preds_seq = preds_root.get("future") or []
        else:                                   # list 혹은 None
            preds_seq = preds_root or prediction_results.get("predictions_flat", [])

        if not preds_seq:
            raise ValueError("prediction_results 안에 예측 데이터가 비어 있습니다.")

        first_rec = preds_seq[0]
        first_date = pd.to_datetime(first_rec.get("date") or first_rec.get("Date"))
        if pd.isna(first_date):
            raise ValueError("첫 예측 레코드에 날짜 정보가 없습니다.")

        # 🎯 파일별 캐시 디렉토리 사용
        cache_dirs = get_file_cache_dirs()
        file_predictions_dir = cache_dirs['predictions']
        
        # ✅ 파일 경로 설정 (파일별 디렉토리 내)
        json_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}.json"
        csv_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}.csv"
        meta_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_meta.json"
        attention_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_attention.json"
        
        logger.info(f"📁 Using file cache directory: {cache_dirs['root'].name}")
        logger.info(f"  📄 Predictions: {file_predictions_dir.name}")
        logger.info(f"  📄 CSV: {csv_path.name}")
        logger.info(f"  📄 Meta: {meta_path.name}")

        # ── validation 개수 계산 ──────────────────────
        if isinstance(preds_root, dict):
            validation_cnt = len(preds_root.get("validation", []))
        else:
            validation_cnt = 0

        # ── 메타 + 본문 구성 (파일 캐시 정보 포함) ──────────────────────────
        current_file_path = prediction_state.get('current_file', None)
        file_content_hash = get_data_content_hash(current_file_path) if current_file_path else None
        
        meta = {
            "prediction_start_date": first_date.strftime("%Y-%m-%d"),
            "data_end_date": str(prediction_date)[:10],
            "created_at": datetime.now().isoformat(),
            "semimonthly_period": prediction_results.get("semimonthly_period"),
            "next_semimonthly_period": prediction_results.get("next_semimonthly_period"),
            "selected_features": prediction_results.get("selected_features", []),
            "total_predictions": len(prediction_results.get("predictions_flat", preds_seq)),
            "validation_points": validation_cnt,
            "is_pure_future_prediction": prediction_results.get("summary", {}).get(
                "is_pure_future_prediction", validation_cnt == 0
            ),
            "metrics": prediction_results.get("metrics"),
            "interval_scores": prediction_results.get("interval_scores", {}),
            # 🔑 캐시 연동을 위한 파일 정보
            "file_path": current_file_path,
            "file_content_hash": file_content_hash,
            "model_type": prediction_results.get("model_type", "ImprovedLSTMPredictor"),
            "loss_function": prediction_results.get("loss_function", "DirectionalLoss"),
            "prediction_mode": "일반 모드"
        }

        # ✅ CSV 파일 저장
        predictions_data = clean_predictions_data(
            prediction_results.get("predictions_flat", preds_seq)
        )
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"✅ CSV saved: {csv_path}")

        # ✅ 메타데이터 저장
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info(f"✅ Metadata saved: {meta_path}")

        # ✅ Attention 데이터 저장 (있는 경우)
        attention_data = prediction_results.get("attention_data")
        if attention_data:
            attention_save_data = {
                "image_base64": attention_data.get("image", ""),
                "feature_importance": attention_data.get("feature_importance", {}),
                "temporal_importance": attention_data.get("temporal_importance", {})
            }
            
            with open(attention_path, "w", encoding="utf-8") as fp:
                json.dump(attention_save_data, fp, ensure_ascii=False, indent=2)
            logger.info(f"✅ Attention saved: {attention_path}")

        # ✅ 이동평균 데이터 저장 (있는 경우)
        ma_results = prediction_results.get("ma_results")
        ma_file = None
        if ma_results:
            ma_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_ma.json"
            try:
                with open(ma_path, "w", encoding="utf-8") as fp:
                    json.dump(ma_results, fp, ensure_ascii=False, indent=2, default=str)
                logger.info(f"✅ MA results saved: {ma_path}")
                ma_file = str(ma_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to save MA results: {str(e)}")

        # ✅ 인덱스 업데이트
        update_predictions_index_simple(meta)
        
        logger.info(f"✅ Complete prediction save → start date: {meta['prediction_start_date']}")
        return {
            "success": True, 
            "csv_file": str(csv_path),
            "meta_file": str(meta_path),
            "attention_file": str(attention_path) if attention_data else None,
            "ma_file": ma_file,
            "prediction_start_date": meta["prediction_start_date"]
        }

    except Exception as e:
        logger.error(f"❌ save_prediction_simple 오류: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

# 2. Attention 데이터를 포함한 로드 함수
def load_prediction_simple(prediction_start_date):
    """
    단순화된 예측 결과 로드 함수
    """
    try:
        predictions_dir = Path(PREDICTIONS_DIR)
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        csv_filepath = predictions_dir / f"prediction_{date_str}.csv"
        meta_filepath = predictions_dir / f"prediction_{date_str}_meta.json"
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드 - xlwings 우선 시도
        try:
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [XLWINGS_CSV] Attempting to load CSV with xlwings: {csv_filepath}")
                predictions_df = load_csv_with_xlwings(csv_filepath)
            else:
                predictions_df = pd.read_csv(csv_filepath)
        except Exception as e:
            logger.warning(f"⚠️ [XLWINGS_CSV] xlwings failed, falling back to pandas: {str(e)}")
            predictions_df = pd.read_csv(csv_filepath)
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        if 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Simple prediction load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': {
                'image': metadata.get('attention_map', {}).get('image_base64', ''),
                'feature_importance': metadata.get('attention_map', {}).get('feature_importance', {}),
                'temporal_importance': metadata.get('attention_map', {}).get('temporal_importance', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction: {str(e)}")
        return {'success': False, 'error': str(e)}

def update_predictions_index_simple(metadata):
    """단순화된 예측 인덱스 업데이트 - 파일별 캐시 디렉토리 사용"""
    try:
        # 🔧 metadata가 None인 경우 처리
        if metadata is None:
            logger.warning("⚠️ [INDEX] metadata가 None입니다. 인덱스 업데이트를 건너뜁니다.")
            return False
            
        # 🎯 파일별 캐시 디렉토리 사용
        cache_dirs = get_file_cache_dirs()
        predictions_index_file = cache_dirs['predictions'] / 'predictions_index.csv'
        
        # 기존 인덱스 읽기
        index_data = []
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                index_data = list(reader)
        
        # 중복 제거
        prediction_start_date = metadata.get('prediction_start_date')
        if not prediction_start_date:
            logger.warning("⚠️ [INDEX] metadata에 prediction_start_date가 없습니다.")
            return False
            
        index_data = [row for row in index_data 
                     if row.get('prediction_start_date') != prediction_start_date]
        
        # metrics가 None일 수도 있으므로 안전하게 처리
        metrics = metadata.get('metrics') or {}
        
        # 새 데이터 추가 (🔧 필드명 수정)
        new_row = {
            'prediction_start_date': metadata.get('prediction_start_date', ''),
            'data_end_date': metadata.get('data_end_date', ''),
            'created_at': metadata.get('created_at', ''),
            'semimonthly_period': metadata.get('semimonthly_period', ''),
            'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
            'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),  # 🔧 수정
            'f1_score': metrics.get('f1', 0) if isinstance(metrics, dict) else 0,
            'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else 0,
            'mape': metrics.get('mape', 0) if isinstance(metrics, dict) else 0,
            'weighted_score': metrics.get('weighted_score', 0) if isinstance(metrics, dict) else 0
        }
        index_data.append(new_row)
        
        # 날짜순 정렬 후 저장
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        if index_data:
            fieldnames = new_row.keys()  # 🔧 일관된 필드명 사용
            with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(index_data)
            
            logger.info(f"✅ Predictions index updated successfully: {len(index_data)} entries")
            logger.info(f"📄 Index file: {predictions_index_file}")
            return True
        else:
            logger.warning("⚠️ No data to write to index file")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error updating simple predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def rebuild_predictions_index_from_existing_files():
    """
    기존 예측 파일들로부터 predictions_index.csv를 재생성하는 함수
    🔧 누적 예측이 기존 단일 예측 캐시를 인식할 수 있도록 함
    """
    try:
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.warning("⚠️ No current file set, cannot rebuild index")
            return False
        
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        predictions_index_file = predictions_dir / 'predictions_index.csv'
        
        logger.info(f"🔄 Rebuilding predictions index from existing files in: {predictions_dir}")
        
        # 기존 메타 파일들 찾기
        meta_files = list(predictions_dir.glob("*_meta.json"))
        logger.info(f"📋 Found {len(meta_files)} meta files")
        
        if not meta_files:
            logger.warning("⚠️ No meta files found to rebuild index")
            return False
        
        index_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 인덱스 레코드 생성 (동일한 필드명 사용)
                new_row = {
                    'prediction_start_date': metadata.get('prediction_start_date', ''),
                    'data_end_date': metadata.get('data_end_date', ''),
                    'created_at': metadata.get('created_at', ''),
                    'semimonthly_period': metadata.get('semimonthly_period', ''),
                    'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
                    'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
                    'f1_score': metadata.get('metrics', {}).get('f1', 0),
                    'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                    'mape': metadata.get('metrics', {}).get('mape', 0),
                    'weighted_score': metadata.get('metrics', {}).get('weighted_score', 0)
                }
                
                index_data.append(new_row)
                logger.info(f"  ✅ {meta_file.name}: {new_row['prediction_start_date']}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error reading {meta_file.name}: {str(e)}")
                continue
        
        if not index_data:
            logger.error("❌ No valid metadata found")
            return False
        
        # 날짜순 정렬
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        # CSV 파일 생성
        fieldnames = index_data[0].keys()
        
        with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_data)
        
        logger.info(f"✅ Successfully rebuilt predictions_index.csv with {len(index_data)} entries")
        logger.info(f"📄 Index file: {predictions_index_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_cached_prediction_actual_values(prediction_start_date, update_latest_only=True):
    """
    캐시된 예측의 실제값만 선택적으로 업데이트하는 최적화된 함수
    
    Args:
        prediction_start_date: 예측 시작 날짜
        update_latest_only: True면 최신 데이터만 체크하여 성능 최적화
    
    Returns:
        dict: 업데이트 결과
    """
    try:
        current_file = prediction_state.get('current_file')
        if not current_file:
            return {'success': False, 'error': 'No current file context available'}
        
        # 캐시된 예측 로드 (실제값 업데이트 없이)
        cached_result = load_prediction_with_attention_from_csv(prediction_start_date)
        if not cached_result['success']:
            return cached_result
        
        predictions = cached_result['predictions']
        
        # 데이터 로드 (캐시 활용)
        logger.info(f"🔄 [ACTUAL_UPDATE] Loading data for actual value update...")
        df = load_data(current_file, use_cache=True)
        
        if df is None or df.empty:
            logger.warning(f"⚠️ [ACTUAL_UPDATE] Could not load data file")
            return {'success': False, 'error': 'Could not load data file'}
        
        last_data_date = df.index.max()
        updated_count = 0
        
        # 각 예측에 대해 실제값 확인 및 설정
        for pred in predictions:
            pred_date = pd.to_datetime(pred['Date'])
            
            # 최신 데이터만 체크하는 경우 성능 최적화
            if update_latest_only and pred_date < last_data_date - pd.Timedelta(days=30):
                continue
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, 'MOPJ']) and 
                pred_date <= last_data_date):
                actual_val = float(df.loc[pred_date, 'MOPJ'])
                pred['Actual'] = actual_val
                updated_count += 1
                logger.debug(f"  📊 Updated actual value for {pred_date.strftime('%Y-%m-%d')}: {actual_val:.2f}")
            elif 'Actual' not in pred or pred['Actual'] is None:
                pred['Actual'] = None
        
        logger.info(f"✅ [ACTUAL_UPDATE] Updated {updated_count} actual values")
        
        # 업데이트된 결과 반환
        cached_result['predictions'] = predictions
        cached_result['actual_values_updated'] = True
        cached_result['updated_count'] = updated_count
        
        return cached_result
        
    except Exception as e:
        logger.error(f"❌ [ACTUAL_UPDATE] Error updating actual values: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_prediction_from_csv(prediction_start_date_or_data_end_date):
    """
    하위 호환성을 위한 함수 - 자동으로 새로운 함수로 리다이렉트
    """
    logger.info("Using compatibility wrapper - redirecting to new smart cache function")
    return load_prediction_with_attention_from_csv(prediction_start_date_or_data_end_date)

def load_prediction_with_attention_from_csv_in_dir(prediction_start_date, file_predictions_dir):
    """
    파일별 디렉토리에서 저장된 예측 결과와 attention 데이터를 함께 불러오는 함수
    """
    try:
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일별 디렉토리에서 파일 경로 설정
        csv_filepath = file_predictions_dir / f"prediction_start_{date_str}.csv"
        meta_filepath = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = file_predictions_dir / f"prediction_start_{date_str}_attention.json"
        ma_filepath = file_predictions_dir / f"prediction_start_{date_str}_ma.json"
        
        logger.info(f"📂 Loading from file directory: {file_predictions_dir.name}")
        logger.info(f"  📄 CSV: {csv_filepath.name}")
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            logger.warning(f"  ❌ Required files missing in {file_predictions_dir.name}")
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드 - xlwings 우선 시도
        try:
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [XLWINGS_CSV] Attempting to load CSV with xlwings: {csv_filepath}")
                predictions_df = load_csv_with_xlwings(csv_filepath)
            else:
                predictions_df = pd.read_csv(csv_filepath)
        except Exception as e:
            logger.warning(f"⚠️ [XLWINGS_CSV] xlwings failed, falling back to pandas: {str(e)}")
            predictions_df = pd.read_csv(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ JSON 직렬화를 위해 Timestamp 객체들을 문자열로 안전하게 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    # 예측값과 실제값은 모두 float로 유지
                    pred[key] = float(value)
                elif hasattr(value, 'item'):  # numpy scalars
                    pred[key] = value.item()
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (선택적 - 성능 최적화)
        # 💡 캐시된 예측을 빠르게 불러오기 위해 실제값 업데이트를 스킵
        # 필요시에만 별도 API로 실제값 업데이트 수행
        logger.info(f"📦 [CACHE_FAST] Skipping actual value update for faster cache loading")
        logger.info(f"💡 [CACHE_FAST] Use separate API endpoint if actual value update is needed")
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 로드
        attention_data = {}
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    attention_raw = json.load(f)
                attention_data = {
                    'image': attention_raw.get('image_base64', ''),
                    'feature_importance': attention_raw.get('feature_importance', {}),
                    'temporal_importance': attention_raw.get('temporal_importance', {})
                }
                logger.info(f"  🧠 Attention data loaded successfully")
                logger.info(f"  🧠 Image data length: {len(attention_data['image']) if attention_data['image'] else 0}")
                logger.info(f"  🧠 Feature importance keys: {len(attention_data['feature_importance'])}")
                logger.info(f"  🧠 Temporal importance keys: {len(attention_data['temporal_importance'])}")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load attention data: {str(e)}")
                attention_data = {}
        
        # 이동평균 데이터 로드
        ma_results = {}
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"  📊 MA results loaded successfully")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load MA results: {str(e)}")
        
        logger.info(f"✅ File directory cache load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading prediction from file directory: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_prediction_with_attention_from_csv(prediction_start_date):
    """
    저장된 예측 결과와 attention 데이터를 함께 불러오는 함수 - 파일별 캐시 시스템 사용
    """
    try:
        # 🎯 파일별 캐시 디렉토리 사용
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.error("❌ No current file set in prediction_state")
            return {'success': False, 'error': 'No current file context available'}
            
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일 경로들
        csv_filepath = predictions_dir / f"prediction_start_{date_str}.csv"
        meta_filepath = predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = predictions_dir / f"prediction_start_{date_str}_attention.json"
        
        # 필수 파일 존재 확인
        if not csv_filepath.exists() or not meta_filepath.exists():
            return {
                'success': False,
                'error': f'Prediction files not found for start date {start_date.strftime("%Y-%m-%d")}'
            }
        
        # CSV 파일 읽기 - xlwings 우선 시도
        try:
            if XLWINGS_AVAILABLE:
                logger.info(f"🔓 [XLWINGS_CSV] Attempting to load CSV with xlwings: {csv_filepath}")
                predictions_df = load_csv_with_xlwings(csv_filepath)
            else:
                predictions_df = pd.read_csv(csv_filepath)
        except Exception as e:
            logger.warning(f"⚠️ [XLWINGS_CSV] xlwings failed, falling back to pandas: {str(e)}")
            predictions_df = pd.read_csv(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ JSON 직렬화를 위해 Timestamp 객체들을 문자열로 안전하게 변환
        for pred in predictions:
            for key, value in list(pred.items()):
                if pd.isna(value):
                    pred[key] = None
                elif isinstance(value, pd.Timestamp):
                    pred[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.int64, np.float64)):
                    # 예측값과 실제값은 모두 float로 유지
                    pred[key] = float(value)
                elif hasattr(value, 'item'):  # numpy scalars
                    pred[key] = value.item()
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (선택적 - 성능 최적화)
        # 💡 캐시된 예측을 빠르게 불러오기 위해 실제값 업데이트를 스킵
        # 필요시에만 별도 API로 실제값 업데이트 수행
        logger.info(f"📦 [CACHE_FAST] Skipping actual value update for faster cache loading")
        logger.info(f"💡 [CACHE_FAST] Use separate API endpoint if actual value update is needed")
        
        # 메타데이터 읽기
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 읽기 (있는 경우)
        attention_data = None
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    stored_attention = json.load(f)
                
                attention_data = {
                    'image': stored_attention.get('image_base64', ''),
                    'file_path': None,  # 이미지는 base64로 저장됨
                    'feature_importance': stored_attention.get('feature_importance', {}),
                    'temporal_importance': stored_attention.get('temporal_importance', {})
                }
                logger.info(f"Attention data loaded from: {attention_filepath}")
            except Exception as e:
                logger.warning(f"Failed to load attention data: {str(e)}")
                attention_data = None

        # 🔄 이동평균 데이터 읽기 (있는 경우)
        ma_filepath = predictions_dir / f"prediction_start_{date_str}_ma.json"
        ma_results = None
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"MA results loaded from: {ma_filepath} ({len(ma_results)} windows)")
            except Exception as e:
                logger.warning(f"Failed to load MA results: {str(e)}")
                ma_results = None
        
        logger.info(f"Complete prediction data loaded: {csv_filepath} ({len(predictions)} predictions)")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results,  # 🔑 이동평균 데이터 추가
            'prediction_start_date': start_date.strftime('%Y-%m-%d'),
            'data_end_date': metadata.get('data_end_date'),
            'semimonthly_period': metadata['semimonthly_period'],
            'next_semimonthly_period': metadata['next_semimonthly_period'],
            'metrics': metadata['metrics'],
            'interval_scores': metadata['interval_scores'],
            'selected_features': metadata['selected_features'],
            'has_cached_attention': attention_data is not None,
            'has_cached_ma': ma_results is not None  # 🔑 MA 캐시 여부 추가
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction with attention: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def get_saved_predictions_list_for_file(file_path, limit=100):
    """
    특정 파일의 캐시 디렉토리에서 저장된 예측 결과 목록을 조회하는 함수
    
    Parameters:
    -----------
    file_path : str
        현재 파일 경로
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        predictions_list = []
        
        # 파일별 캐시 디렉토리 경로 구성
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        predictions_index_file = predictions_dir / 'predictions_index.csv'
        
        logger.info(f"🔍 [CACHE] Searching predictions in: {predictions_dir}")
        
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if len(predictions_list) >= limit:
                        break
                    
                    prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                    data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                    
                    if prediction_start_date and data_end_date:
                        pred_info = {
                            'prediction_start_date': prediction_start_date,
                            'data_end_date': data_end_date,
                            'prediction_date': data_end_date,
                            'first_prediction_date': prediction_start_date,
                            'created_at': row.get('created_at'),
                            'semimonthly_period': row.get('semimonthly_period'),
                            'next_semimonthly_period': row.get('next_semimonthly_period'),
                            'prediction_count': row.get('prediction_count'),
                            'actual_business_days': row.get('actual_business_days'),
                            'csv_file': row.get('csv_file'),
                            'meta_file': row.get('meta_file'),
                            'f1_score': float(row.get('f1_score', 0)),
                            'accuracy': float(row.get('accuracy', 0)),
                            'mape': float(row.get('mape', 0)),
                            'weighted_score': float(row.get('weighted_score', 0)),
                            'naming_scheme': row.get('naming_scheme', 'file_based'),
                            'source_file': os.path.basename(file_path),
                            'cache_system': 'file_based'
                        }
                        predictions_list.append(pred_info)
            
            logger.info(f"🎯 [CACHE] Found {len(predictions_list)} predictions in file-specific cache")
        else:
            logger.info(f"📂 [CACHE] No predictions index found in {predictions_index_file}")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error reading file-specific predictions list: {str(e)}")
        return []

def get_saved_predictions_list(limit=100):
    """
    저장된 예측 결과 목록을 조회하는 함수 (새로운 파일 체계 호환)
    
    Parameters:
    -----------
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        predictions_list = []
        
        # 1. 파일별 캐시 시스템에서 예측 검색
        cache_root = Path(CACHE_ROOT_DIR)
        if cache_root.exists():
            for file_dir in cache_root.iterdir():
                if not file_dir.is_dir():
                    continue
                
                predictions_dir = file_dir / 'predictions'
                predictions_index_file = predictions_dir / 'predictions_index.csv'
                
                if predictions_index_file.exists():
                    with open(predictions_index_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if len(predictions_list) >= limit:
                                break
                            
                            prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                            data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                            
                            if prediction_start_date and data_end_date:
                                pred_info = {
                                    'prediction_start_date': prediction_start_date,
                                    'data_end_date': data_end_date,
                                    'prediction_date': data_end_date,
                                    'first_prediction_date': prediction_start_date,
                                    'created_at': row.get('created_at'),
                                    'semimonthly_period': row.get('semimonthly_period'),
                                    'next_semimonthly_period': row.get('next_semimonthly_period'),
                                    'prediction_count': row.get('prediction_count'),
                                    'actual_business_days': row.get('actual_business_days'),
                                    'csv_file': row.get('csv_file'),
                                    'meta_file': row.get('meta_file'),
                                    'f1_score': float(row.get('f1_score', 0)),
                                    'accuracy': float(row.get('accuracy', 0)),
                                    'mape': float(row.get('mape', 0)),
                                    'weighted_score': float(row.get('weighted_score', 0)),
                                    'naming_scheme': row.get('naming_scheme', 'file_based'),
                                    'source_file': file_dir.name,
                                    'cache_system': 'file_based'
                                }
                                predictions_list.append(pred_info)
        
        if len(predictions_list) == 0:
            logger.info("No predictions found in file-based cache system")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        logger.info(f"Retrieved {len(predictions_list)} predictions from cache systems")
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error reading predictions list: {str(e)}")
        return []

def load_accumulated_predictions_from_csv(start_date, end_date=None, limit=None, file_path=None):
    """
    CSV에서 누적 예측 결과를 빠르게 불러오는 함수 (최적화됨)
    새로운 파일명 체계와 스마트 캐시 시스템 사용
    
    Parameters:
    -----------
    start_date : str or datetime
        시작 날짜 (데이터 기준일)
    end_date : str or datetime, optional
        종료 날짜 (데이터 기준일)
    limit : int, optional
        최대 로드할 예측 개수
    file_path : str, optional
        현재 파일 경로 (해당 파일의 캐시 디렉토리에서만 검색)
    
    Returns:
    --------
    list : 누적 예측 결과 리스트
    """
    try:
        logger.info(f"🔍 [CACHE_LOAD] Loading predictions from {start_date} to {end_date or 'latest'}")
        
        # 날짜 형식 통일
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 🔧 수정: 저장된 예측 목록 조회 (현재 파일 + 호환 가능한 파일들)
        all_predictions = []
        if file_path:
            try:
                # 1. 현재 파일의 캐시
                all_predictions = get_saved_predictions_list_for_file(file_path, limit=1000)
                logger.info(f"🎯 [CACHE_LOAD] Current file: Found {len(all_predictions)} prediction files")
                
                # 2. 다른 호환 가능한 파일들의 캐시
                upload_dir = Path(UPLOAD_FOLDER)
                existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != file_path]
                
                for existing_file in existing_files:
                    try:
                        # 확장 관계 또는 파일명 유사성 확인
                        extension_result = check_data_extension(str(existing_file), file_path)
                        is_extension = extension_result.get('is_extension', False)
                        
                        # 파일명 유사성 확인 (확장 관계 확인 실패 시)
                        if not is_extension:
                            existing_name = Path(existing_file).stem.lower()
                            current_name = Path(file_path).stem.lower()
                            if existing_name in current_name or current_name in existing_name:
                                is_extension = True
                        
                        if is_extension:
                            compatible_predictions = get_saved_predictions_list_for_file(str(existing_file), limit=500)
                            all_predictions.extend(compatible_predictions)
                            logger.info(f"🔗 [CACHE_LOAD] Compatible file {existing_file.name}: Found {len(compatible_predictions)} additional predictions")
                            
                    except Exception as file_error:
                        logger.warning(f"⚠️ [CACHE_LOAD] Error checking file {existing_file.name}: {str(file_error)}")
                        continue
                
                logger.info(f"🎯 [CACHE_LOAD] Total predictions found: {len(all_predictions)}")
                
            except Exception as e:
                logger.warning(f"⚠️ [CACHE_LOAD] Error in file-specific search: {str(e)}")
                return []
        else:
            try:
                all_predictions = get_saved_predictions_list(limit=1000)
                logger.info(f"🎯 [CACHE_LOAD] Found {len(all_predictions)} prediction files (global)")
            except Exception as e:
                logger.warning(f"⚠️ [CACHE_LOAD] Error in global search: {str(e)}")
                return []
        
        # 날짜 범위 필터링 (데이터 기준일 기준)
        filtered_predictions = []
        for pred_info in all_predictions:
            # 인덱스에서 데이터 기준일 확인
            data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
            
            # 날짜 범위 확인
            if data_end_date >= start_date:
                if end_date is None or data_end_date <= end_date:
                    filtered_predictions.append(pred_info)
            
            # 제한 개수 확인
            if limit and len(filtered_predictions) >= limit:
                break
        
        logger.info(f"📋 [CACHE] Found {len(filtered_predictions)} matching prediction files in date range")
        if len(filtered_predictions) > 0:
            logger.info(f"📅 [CACHE] Available cached dates:")
            for pred in filtered_predictions:
                data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
                logger.info(f"    - {data_end_date}")
        
        # 각 예측 결과 로드
        accumulated_results = []
        for i, pred_info in enumerate(filtered_predictions):
            try:
                # 데이터 기준일을 사용하여 예측 시작일 계산
                data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
                
                # 데이터 기준일로부터 예측 시작일 계산
                prediction_start_date = data_end_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)
                
                # 파일별 캐시 디렉토리 사용
                if file_path:
                    cache_dirs = get_file_cache_dirs(file_path)
                    loaded_result = load_prediction_with_attention_from_csv_in_dir(prediction_start_date, cache_dirs['predictions'])
                else:
                    loaded_result = load_prediction_with_attention_from_csv(prediction_start_date)
                
                if loaded_result['success']:
                    logger.info(f"  ✅ [CACHE] Successfully loaded cached prediction for {data_end_date.strftime('%Y-%m-%d')}")
                    # 누적 예측 형식에 맞게 변환
                    # 안전한 데이터 구조 생성
                    predictions = loaded_result.get('predictions', [])
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not isinstance(predictions, list):
                        logger.warning(f"Loaded predictions is not a list for {data_end_date.strftime('%Y-%m-%d')}: {type(predictions)}")
                        predictions = []
                    
                    metadata = loaded_result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # 🔧 metrics 안전성 처리: None이면 기본값 설정
                    cached_metrics = metadata.get('metrics')
                    if not cached_metrics or not isinstance(cached_metrics, dict):
                        cached_metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    accumulated_item = {
                        'date': data_end_date.strftime('%Y-%m-%d'),  # 데이터 기준일
                        'prediction_start_date': loaded_result.get('prediction_start_date'),  # 예측 시작일
                        'predictions': predictions,
                        'metrics': cached_metrics,
                        'interval_scores': metadata.get('interval_scores', {}),
                        'next_semimonthly_period': metadata.get('next_semimonthly_period'),
                        'actual_business_days': metadata.get('actual_business_days'),
                        'original_interval_scores': metadata.get('interval_scores', {}),
                        'has_attention': loaded_result.get('has_cached_attention', False)
                    }
                    accumulated_results.append(accumulated_item)
                    logger.info(f"  ✅ [CACHE] Added to results {i+1}/{len(filtered_predictions)}: {data_end_date.strftime('%Y-%m-%d')}")
                else:
                    logger.warning(f"  ❌ [CACHE] Failed to load prediction {i+1}/{len(filtered_predictions)}: {loaded_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error loading prediction {i+1}/{len(filtered_predictions)}: {str(e)}")
                continue
        
        logger.info(f"🎯 [CACHE] Successfully loaded {len(accumulated_results)} predictions from CSV cache files")
        return accumulated_results
        
    except Exception as e:
        logger.error(f"Error loading accumulated predictions from CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def delete_saved_prediction(prediction_date):
    """
    저장된 예측 결과를 삭제하는 함수
    
    Parameters:
    -----------
    prediction_date : str or datetime
        삭제할 예측 날짜
    
    Returns:
    --------
    dict : 삭제 결과
    """
    try:
        # 날짜 형식 통일
        if isinstance(prediction_date, str):
            pred_date = pd.to_datetime(prediction_date)
        else:
            pred_date = prediction_date
        
        date_str = pred_date.strftime('%Y%m%d')
        
        # 파일 경로들 (TARGET_DATE 방식)
        csv_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}.csv")
        meta_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}_meta.json")
        
        # 파일 삭제
        deleted_files = []
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
            deleted_files.append(csv_filepath)
        
        if os.path.exists(meta_filepath):
            os.remove(meta_filepath)
            deleted_files.append(meta_filepath)
        
        # 🚫 레거시 인덱스 제거 기능은 파일별 캐시 시스템에서 제거됨
        # 파일별 캐시에서는 각 파일의 predictions_index.csv가 자동으로 관리됨
        logger.info("⚠️ Legacy delete_saved_prediction function called - not supported in file-based cache system")
        
        return {
            'success': True,
            'deleted_files': deleted_files,
            'message': f'Prediction for {pred_date.strftime("%Y-%m-%d")} deleted successfully'
        }
        
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

#######################################################################
# 예측 신뢰도 및 구매 신뢰도 계산 함수
#######################################################################

def calculate_prediction_consistency(accumulated_predictions, target_period):
    """
    다음 반월에 대한 여러 날짜의 예측 일관성을 계산
    
    Parameters:
    -----------
    accumulated_predictions: list
        여러 날짜에 수행한 예측 결과 목록
    target_period: str
        다음 반월 기간 (예: "2025-01-SM1")
    
    Returns:
    -----------
    dict: 일관성 점수와 관련 메트릭
    """
    import numpy as np
    
    # 날짜별 예측 데이터 추출
    period_predictions = {}
    
    for prediction in accumulated_predictions:
        # 안전한 데이터 접근
        if not isinstance(prediction, dict):
            continue
            
        prediction_date = prediction.get('date')
        next_period = prediction.get('next_semimonthly_period')
        predictions_list = prediction.get('predictions', [])
        
        if next_period != target_period:
            continue
            
        if prediction_date not in period_predictions:
            period_predictions[prediction_date] = []
        
        # predictions_list가 배열인지 확인
        if not isinstance(predictions_list, list):
            logger.warning(f"predictions_list is not a list for {prediction_date}: {type(predictions_list)}")
            continue
            
        for pred in predictions_list:
            # pred가 딕셔너리인지 확인
            if not isinstance(pred, dict):
                logger.warning(f"Prediction item is not a dict for {prediction_date}: {type(pred)}")
                continue
                
            pred_date = pred.get('Date') or pred.get('date')
            pred_value = pred.get('Prediction') or pred.get('prediction')
            
            # 값이 유효한지 확인
            if pred_date and pred_value is not None:
                period_predictions[prediction_date].append({
                    'date': pred_date,
                    'value': pred_value
                })
    
    # 날짜별로 정렬
    prediction_dates = sorted(period_predictions.keys())
    
    if len(prediction_dates) < 2:
        return {
            "consistency_score": None,
            "message": "Insufficient prediction data (min 2 required)",
            "period": target_period,
            "dates_count": len(prediction_dates)
        }
    
    # 일관성 분석을 위한 날짜 매핑
    date_predictions = {}
    
    for pred_date in prediction_dates:
        for p in period_predictions[pred_date]:
            target_date = p['date']
            if target_date not in date_predictions:
                date_predictions[target_date] = []
            
            date_predictions[target_date].append({
                'prediction_date': pred_date,
                'value': p['value']
            })
    
    # 각 타겟 날짜별 예측값 변동성 계산
    overall_variations = []
    
    for target_date, predictions in date_predictions.items():
        if len(predictions) >= 2:
            # 예측값 추출 (None 값 필터링)
            values = [p['value'] for p in predictions if p['value'] is not None]
            
            if len(values) < 2:
                continue
                
            # 값이 모두 같은 경우 CV를 0으로 처리
            if all(v == values[0] for v in values):
                cv = 0.0
                overall_variations.append(cv)
                continue
            
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # 변동 계수 (Coefficient of Variation)
            cv = std_value / abs(mean_value) if mean_value != 0 else float('inf')
            overall_variations.append(cv)
    
    # 전체 일관성 점수 계산 (변동 계수 평균을 0-100 점수로 변환)
    if overall_variations:
        avg_cv = np.mean(overall_variations)
        consistency_score = max(0, min(100, 100 - (avg_cv * 100)))
    else:
        consistency_score = None
    
    # 신뢰도 등급 부여
    if consistency_score is not None:
        if consistency_score >= 90:
            grade = "Very High"
        elif consistency_score >= 75:
            grade = "High"
        elif consistency_score >= 60:
            grade = "Medium"
        elif consistency_score >= 40:
            grade = "Low"
        else:
            grade = "Very Low"
    else:
        grade = "Unable to determine"
    
    return {
        "consistency_score": consistency_score,
        "consistency_grade": grade,
        "target_period": target_period,
        "prediction_count": len(prediction_dates),
        "average_variation": avg_cv * 100 if overall_variations else None,
        "message": f"Consistency for period {target_period} based on {len(prediction_dates)} predictions"
    }

# 누적 예측의 구매 신뢰도 계산 함수 (올바른 버전)
def calculate_accumulated_purchase_reliability(accumulated_predictions):
    """
    누적 예측의 구매 신뢰도 계산 (올바른 방식)
    
    각 예측마다 상위 3개 구간(1등:3점, 2등:2점, 3등:1점)을 선정하고,
    같은 구간이 여러 예측에서 선택되면 점수를 누적하여,
    최고 누적 점수 구간의 점수 / (예측 횟수 × 3점) × 100%로 계산
    
    Returns:
        tuple: (reliability_percentage, debug_info)
    """
    print(f"🔍 [RELIABILITY] Function called with {len(accumulated_predictions) if accumulated_predictions else 0} predictions")
    
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        print(f"⚠️ [RELIABILITY] Invalid input: accumulated_predictions is empty or not a list")
        return 0.0, {}
    
    try:
        prediction_count = len(accumulated_predictions)
        print(f"📊 [RELIABILITY] Processing {prediction_count} predictions...")
        
        # 🔑 구간별 누적 점수를 저장할 딕셔너리
        interval_accumulated_scores = {}
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            interval_scores = pred.get('interval_scores', {})
            pred_date = pred.get('date')
            
            if interval_scores and isinstance(interval_scores, dict):
                # 모든 구간을 평균 가격 순으로 정렬 (가격이 낮을수록 좋음)
                valid_intervals = []
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'avg_price' in score_data:
                        valid_intervals.append(score_data)
                
                if valid_intervals:
                    # 평균 가격 기준으로 정렬 (낮은 가격이 우선)
                    valid_intervals.sort(key=lambda x: x.get('avg_price', float('inf')))
                    
                    # 상위 3개 구간에 점수 부여
                    for rank, interval in enumerate(valid_intervals[:3]):
                        score = 3 - rank  # 1등: 3점, 2등: 2점, 3등: 1점
                        
                        # 구간 식별키 생성 (시작일-종료일)
                        interval_key = f"{interval.get('start_date')} ~ {interval.get('end_date')} ({interval.get('days')}일)"
                        
                        # 누적 점수 계산
                        if interval_key not in interval_accumulated_scores:
                            interval_accumulated_scores[interval_key] = {
                                'total_score': 0,
                                'appearances': 0,
                                'details': [],
                                'avg_price': interval.get('avg_price', 0),
                                'days': interval.get('days', 0)
                            }
                        
                        interval_accumulated_scores[interval_key]['total_score'] += score
                        interval_accumulated_scores[interval_key]['appearances'] += 1
                        interval_accumulated_scores[interval_key]['details'].append({
                            'date': pred_date,
                            'rank': rank + 1,
                            'score': score,
                            'avg_price': interval.get('avg_price', 0)
                        })
                        
                        print(f"📊 [RELIABILITY] 날짜 {pred_date}: {rank+1}등 {interval_key} → {score}점 (평균가: {interval.get('avg_price', 0):.2f})")
        
        # 최고 누적 점수 구간 찾기
        if interval_accumulated_scores:
            best_interval_key = max(interval_accumulated_scores.keys(), 
                                  key=lambda k: interval_accumulated_scores[k]['total_score'])
            best_total_score = interval_accumulated_scores[best_interval_key]['total_score']
            
            # 만점 계산 (각 예측마다 최대 3점씩)
            max_possible_total_score = prediction_count * 3
            
            # 구매 신뢰도 계산
            reliability_percentage = (best_total_score / max_possible_total_score) * 100 if max_possible_total_score > 0 else 0.0
            
            print(f"\n🎯 [RELIABILITY] === 구간별 누적 점수 분석 ===")
            print(f"📊 예측 횟수: {prediction_count}개")
            print(f"📊 구간별 누적 점수:")
            
            # 누적 점수 순으로 정렬하여 표시
            sorted_intervals = sorted(interval_accumulated_scores.items(), 
                                    key=lambda x: x[1]['total_score'], reverse=True)
            
            for interval_key, data in sorted_intervals[:5]:  # 상위 5개만 표시
                print(f"   - {interval_key}: {data['total_score']}점 ({data['appearances']}회 선택)")
            
            print(f"\n🏆 최고 점수 구간: {best_interval_key}")
            print(f"🏆 최고 누적 점수: {best_total_score}점")
            print(f"🏆 구간 신뢰도: {best_total_score}/{max_possible_total_score} = {reliability_percentage:.1f}%")
            
            # 디버그 정보 생성
            debug_info = {
                'prediction_count': prediction_count,
                'interval_accumulated_scores': interval_accumulated_scores,
                'best_interval_key': best_interval_key,
                'best_total_score': best_total_score,
                'max_possible_total_score': max_possible_total_score,
                'reliability_percentage': reliability_percentage
            }
            
            return reliability_percentage, debug_info
        else:
            print(f"⚠️ [RELIABILITY] No valid interval scores found")
            return 0.0, {}
            
    except Exception as e:
        print(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0, {'error': str(e)} 

def calculate_accumulated_purchase_reliability_with_debug(accumulated_predictions):
    """
    디버그 정보와 함께 누적 예측의 구매 신뢰도 계산
    """
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        return 0.0, {}
    
    debug_info = {
        'prediction_count': len(accumulated_predictions),
        'individual_scores': [],
        'total_best_score': 0,
        'max_possible_total_score': 0
    }
    
    try:
        total_best_score = 0
        prediction_count = len(accumulated_predictions)
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            pred_date = pred.get('date')
            interval_scores = pred.get('interval_scores', {})
            
            best_score = 0
            capped_score = 0  # ✅ 초기화 추가
            valid_scores = []  # ✅ valid_scores도 외부에서 초기화
            
            if interval_scores and isinstance(interval_scores, dict):
                # 유효한 interval score 찾기
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score_value = score_data.get('score', 0)
                        if isinstance(score_value, (int, float)):
                            valid_scores.append(score_value)
                
                if valid_scores:
                    best_score = max(valid_scores)
                    # 점수를 3점으로 제한 (각 예측의 최대 점수)
                    capped_score = min(best_score, 3.0)
                    total_best_score += capped_score
            
            debug_info['individual_scores'].append({
                'date': pred_date,
                'original_best_score': best_score,
                'actual_score_used': capped_score if valid_scores else 0,
                'max_score_per_prediction': 3,
                'has_valid_scores': len(valid_scores) > 0
            })
        
        # 전체 계산 - 3점이 만점
        max_possible_total_score = prediction_count * 3
        reliability_percentage = (total_best_score / max_possible_total_score) * 100 if max_possible_total_score > 0 else 0.0
        
        debug_info['total_best_score'] = total_best_score
        debug_info['max_possible_total_score'] = max_possible_total_score
        debug_info['reliability_percentage'] = reliability_percentage
        
        logger.info(f"🎯 올바른 누적 구매 신뢰도 계산:")
        logger.info(f"  - 예측 횟수: {prediction_count}회")
        
        # 🔍 개별 점수 디버깅 정보 출력
        for score_info in debug_info['individual_scores']:
            original = score_info.get('original_best_score', 0)
            actual = score_info.get('actual_score_used', 0)
            logger.info(f"📊 날짜 {score_info['date']}: 원본점수={original:.1f}, 적용점수={actual:.1f}, 유효점수있음={score_info['has_valid_scores']}")
        
        logger.info(f"  - 총 획득 점수: {total_best_score:.1f}점")
        logger.info(f"  - 최대 가능 점수: {max_possible_total_score}점 ({prediction_count} × 3)")
        logger.info(f"  - 구매 신뢰도: {reliability_percentage:.1f}%")
        
        # ✅ 추가 검증 로깅
        if reliability_percentage == 100.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 100%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
            for i, score_info in enumerate(debug_info['individual_scores']):
                logger.warning(f"   - 예측 {i+1}: {score_info}")
        elif reliability_percentage == 0.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 0%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
        
        return reliability_percentage, debug_info
            
    except Exception as e:
        logger.error(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0, {'error': str(e)}

def calculate_actual_business_days(predictions):
    """
    예측 결과에서 실제 영업일 수를 계산하는 헬퍼 함수
    """
    if not predictions:
        return 0
    
    try:
        actual_days = len([p for p in predictions 
                          if p.get('Date') and not p.get('is_synthetic', False)])
        return actual_days
    except Exception as e:
        logger.error(f"Error calculating actual business days: {str(e)}")
        return 0

def get_previous_semimonthly_period(semimonthly_period):
    """
    주어진 반월 기간의 이전 반월 기간을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    str
        이전 반월 기간
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월인 경우 이전 월의 하반월로
        if month == 1:
            return f"{year-1}-12-SM2"
        else:
            return f"{year}-{month-1:02d}-SM2"
    else:
        # 하반월인 경우 같은 월의 상반월로
        return f"{year}-{month:02d}-SM1"

#######################################################################
# 시각화 함수
#######################################################################

def get_global_y_range(original_df, test_dates, predict_window):
    """
    테스트 구간의 모든 MOPJ 값을 기반으로 전역 y축 범위를 계산합니다.
    
    Args:
        original_df: 원본 데이터프레임
        test_dates: 테스트 날짜 배열
        predict_window: 예측 기간
    
    Returns:
        tuple: (y_min, y_max) 전역 범위 값
    """
    # 테스트 구간 데이터 추출
    test_values = []
    
    # 테스트 데이터의 실제 값 수집
    for date in test_dates:
        if date in original_df.index and not pd.isna(original_df.loc[date, 'MOPJ']):
            test_values.append(original_df.loc[date, 'MOPJ'])
    
    # 안전장치: 데이터가 없으면 None 반환
    if not test_values:
        return None, None
    
    # 최소/최대 계산 (약간의 마진 추가)
    y_min = min(test_values) * 0.95
    y_max = max(test_values) * 1.05
    
    return y_min, y_max

def visualize_attention_weights(model, features, prev_value, sequence_end_date, feature_names=None, actual_sequence_dates=None):
    """
    모델의 어텐션 가중치를 시각화하는 함수 - 2x2 레이아웃으로 개선
    sequence_end_date: 시퀀스 데이터의 마지막 날짜 (예측 시작일 전날)
    """
    model.eval()
    
    # 특성 이름이 없으면 인덱스로 생성
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[2])]
    else:
        # 특성 수에 맞게 조정
        feature_names = feature_names[:features.shape[2]]
    
    # 텐서가 아니면 변환
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features).to(next(model.parameters()).device)
    
    # prev_value 처리
    if prev_value is not None:
        if not isinstance(prev_value, torch.Tensor):
            try:
                prev_value = float(prev_value)
                prev_value = torch.FloatTensor([prev_value]).to(next(model.parameters()).device)
            except (TypeError, ValueError):
                logger.warning("Warning: prev_value를 숫자로 변환할 수 없습니다. 0으로 대체합니다.")
                prev_value = torch.FloatTensor([0.0]).to(next(model.parameters()).device)
    
    # 시퀀스 길이
    seq_len = features.shape[1]
    
    # 날짜 라벨 생성 - 실제 시퀀스 날짜 사용
    date_labels = []
    if actual_sequence_dates is not None and len(actual_sequence_dates) == seq_len:
        # 실제 날짜 정보가 전달된 경우 사용
        for date in actual_sequence_dates:
            try:
                if isinstance(date, str):
                    date_labels.append(date)
                else:
                    date_labels.append(format_date(date, '%Y-%m-%d'))
            except:
                date_labels.append(str(date))
    else:
        # 실제 날짜 정보가 없으면 기존 방식 사용 (시퀀스 마지막 날짜부터 역순으로)
        for i in range(seq_len):
            try:
                # 시퀀스 마지막 날짜에서 거꾸로 계산
                date = sequence_end_date - timedelta(days=seq_len-i-1)
                date_labels.append(format_date(date, '%Y-%m-%d'))
            except:
                # 날짜 변환 오류 시 인덱스 사용
                date_labels.append(f"T-{seq_len-i-1}")
    
    # GridSpec을 사용한 레이아웃 생성 - 상단 2개, 하단 1개 큰 그래프
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
    # 예측 시작일 계산 (시퀀스 마지막 날짜 다음날)
    prediction_date = sequence_end_date + timedelta(days=1)
    fig.suptitle(f"Attention Weight Analysis for Prediction {format_date(prediction_date, '%Y-%m-%d')}", 
                fontsize=24, fontweight='bold')
    
    # 전체 폰트 크기 설정
    plt.rcParams.update({'font.size': 16})
    
    # 특성 중요도 계산을 위해 데이터 준비
    feature_importance = np.zeros(len(feature_names))
    
    # 특성 중요도를 간단한 방법으로 계산
    # 마지막 시점에서 각 특성의 절대값 사용
    feature_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=0)
    
    # 정규화
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    
    # 특성 중요도를 내림차순으로 정렬
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # 플롯 1: 시간적 중요도 (Time Step Importance) - 상단 왼쪽
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 각 시점의 평균 절대값으로 시간적 중요도 추정
    temporal_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=1)
    if np.sum(temporal_importance) > 0:
        temporal_importance = temporal_importance / np.sum(temporal_importance)
    
    try:
        # 막대그래프로 시간적 중요도 표시
        bars = ax1.bar(range(len(date_labels)), temporal_importance, color='skyblue', alpha=0.7)
        
        # X축 라벨 간격 조정 - 너무 많으면 일부만 표시
        if len(date_labels) > 20:
            # 20개 이상이면 7개 간격으로 표시
            step = max(1, len(date_labels) // 7)
            tick_indices = list(range(0, len(date_labels), step))
            # 마지막 날짜도 포함
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45, ha='right', fontsize=14)
        elif len(date_labels) > 10:
            # 10-20개면 3개 간격으로 표시
            step = max(1, len(date_labels) // 5)
            tick_indices = list(range(0, len(date_labels), step))
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([date_labels[i] for i in tick_indices], rotation=45, ha='right', fontsize=14)
        else:
            # 10개 이하면 모두 표시
            ax1.set_xticks(range(len(date_labels)))
            ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=14)
            
        ax1.set_title("Time Step Importance", fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel("Sequence Dates", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Relative Importance", fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        # 마지막 시점 강조
        ax1.bar(len(date_labels)-1, temporal_importance[-1], color='red', alpha=0.7)
        
        # 그리드 추가
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"시간적 중요도 시각화 오류: {str(e)}")
        ax1.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=16)
    
    # 플롯 2: 특성별 중요도 (Feature Importance) - 상단 오른쪽
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 상위 10개 특성만 표시
    top_n = min(10, len(sorted_features))
    
    try:
        # 수평 막대 차트로 표시
        y_pos = range(top_n)
        bars = ax2.barh(y_pos, sorted_importance[:top_n], color='lightgreen', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_features[:top_n], fontsize=14)
        ax2.set_title("Feature Importance", fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlabel("Relative Importance", fontsize=16, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        # 중요도 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.3f}", va='center', fontsize=13, fontweight='bold')
        
        # 그리드 추가
        ax2.grid(True, alpha=0.3, axis='x')
    except Exception as e:
        logger.error(f"특성 중요도 시각화 오류: {str(e)}")
        ax2.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=16)
    
    # 플롯 3: 상위 특성들의 시계열 그래프 (Top Features Time Series) - 하단 전체
    ax3 = fig.add_subplot(gs[1, :])
    
    try:
        # 상위 8개 특성 사용 (더 많은 특성을 보여줄 수 있음)
        top_n_series = min(8, len(sorted_features))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i in range(top_n_series):
            feature_idx = sorted_idx[i]
            feature_name = feature_names[feature_idx]
            
            # 해당 특성의 시계열 데이터
            feature_data = features[0, :, feature_idx].cpu().numpy()
            
            # min-max 정규화로 모든 특성을 같은 스케일로 표시
            feature_min = feature_data.min()
            feature_max = feature_data.max()
            if feature_max > feature_min:  # 0으로 나누기 방지
                norm_data = (feature_data - feature_min) / (feature_max - feature_min)
            else:
                norm_data = np.zeros_like(feature_data)
            
            # 특성 중요도에 비례하는 선 두께
            line_width = 2 + sorted_importance[i] * 6
            
            # 플롯
            ax3.plot(range(len(date_labels)), norm_data, 
                    label=f"{feature_name[:20]}... ({sorted_importance[i]:.3f})" if len(feature_name) > 20 else f"{feature_name} ({sorted_importance[i]:.3f})",
                    linewidth=line_width, color=colors[i % len(colors)], alpha=0.8, marker='o', markersize=4)
        
        ax3.set_title("Top Features Time Series (Normalized)", fontsize=20, fontweight='bold', pad=25)
        ax3.set_xlabel("Time Steps", fontsize=18, fontweight='bold')
        ax3.set_ylabel("Normalized Value", fontsize=18, fontweight='bold')
        ax3.legend(fontsize=14, loc='best', ncol=2)  # 2열로 범례 표시
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        
        # x축 라벨을 간소화 (너무 많으면 가독성 떨어짐)
        if len(date_labels) > 20:
            # 20개 이상이면 7개 간격으로 표시
            step = max(1, len(date_labels) // 7)
            tick_indices = list(range(0, len(date_labels), step))
            # 마지막 날짜도 포함
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([date_labels[i] for i in tick_indices], 
                              rotation=45, ha='right', fontsize=14)
        elif len(date_labels) > 10:
            # 10-20개면 5개 간격으로 표시
            step = max(1, len(date_labels) // 5)
            tick_indices = list(range(0, len(date_labels), step))
            if tick_indices[-1] != len(date_labels) - 1:
                tick_indices.append(len(date_labels) - 1)
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([date_labels[i] for i in tick_indices], 
                              rotation=45, ha='right', fontsize=14)
        else:
            # 10개 이하면 모두 표시
            ax3.set_xticks(range(len(date_labels)))
            ax3.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=14)
            
    except Exception as e:
        logger.error(f"시계열 시각화 오류: {str(e)}")
        ax3.text(0.5, 0.5, "Visualization error", ha='center', va='center', fontsize=18)
    

    
    plt.tight_layout(pad=3.0)
    
    # 이미지를 메모리에 저장
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    
    # Base64로 인코딩
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    
    # 파일 저장 - 파일별 캐시 디렉토리 사용
    try:
        cache_dirs = get_file_cache_dirs()  # 현재 파일의 캐시 디렉토리 가져오기
        attn_dir = cache_dirs['plots']  # plots 디렉토리에 저장
        
        filename = os.path.join(attn_dir, f"attention_{format_date(prediction_date, '%Y%m%d')}.png")
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_str))
    except Exception as e:
        logger.error(f"Error saving attention image: {str(e)}")
        filename = None
    
    return filename, img_str, {
        'feature_importance': dict(zip(sorted_features, sorted_importance.tolist())),
        'temporal_importance': dict(zip(date_labels, temporal_importance.tolist()))
    }

def plot_prediction_basic(sequence_df, prediction_start_date, start_day_value, 
                         f1, accuracy, mape, weighted_score_pct, 
                         current_date=None,  # 🔑 추가: 데이터 컷오프 날짜
                         save_prefix=None, title_prefix="Basic Prediction Graph",
                         y_min=None, y_max=None, file_path=None):
    """
    기본 예측 그래프 시각화 - 과거/미래 명확 구분
    🔑 current_date 이후는 미래 예측으로만 표시 (데이터 누출 방지)
    """
    
    fig = None
    
    try:
        logger.info(f"Creating prediction graph for prediction starting {format_date(prediction_start_date)}")
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for plots: {str(e)}")
                save_dir = Path("temp_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrame의 날짜 열이 문자열인 경우 날짜 객체로 변환
        if 'Date' in sequence_df.columns and isinstance(sequence_df['Date'].iloc[0], str):
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # ✅ current_date 기준으로 과거/미래 분할
        if current_date is not None:
            current_date = pd.to_datetime(current_date)
            
            # 과거 데이터 (current_date 이전): 실제값과 예측값 모두 표시 가능
            past_df = sequence_df[sequence_df['Date'] <= current_date].copy()
            # 미래 데이터 (current_date 이후): 예측값만 표시
            future_df = sequence_df[sequence_df['Date'] > current_date].copy()
            
            # 과거 데이터에서 실제값이 있는 것만 검증용으로 사용
            valid_df = past_df.dropna(subset=['Actual']) if 'Actual' in past_df.columns else pd.DataFrame()
            
            logger.info(f"  📊 Data split - Past: {len(past_df)}, Future: {len(future_df)}, Validation: {len(valid_df)}")
        else:
            # current_date가 없으면 기존 방식 사용 (하위 호환성)
            valid_df = sequence_df.dropna(subset=['Actual']) if 'Actual' in sequence_df.columns else pd.DataFrame()
            future_df = sequence_df
            past_df = valid_df
        
        pred_df = sequence_df.dropna(subset=['Prediction'])
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 그래프 타이틀과 서브타이틀
        if isinstance(prediction_start_date, str):
            main_title = f"{title_prefix} - Start: {prediction_start_date}"
        else:
            main_title = f"{title_prefix} - Start: {prediction_start_date.strftime('%Y-%m-%d')}"
        
        # ✅ 과거/미래 구분 정보가 포함된 서브타이틀
        if current_date is not None:
            validation_count = len(valid_df)
            future_count = len(future_df)
            subtitle = f"Data Cutoff: {current_date.strftime('%Y-%m-%d')} | Validation: {validation_count} pts | Future: {future_count} pts"
            if validation_count > 0:
                subtitle += f" | F1: {f1:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%"
        else:
            # 기존 방식
            if f1 == 0 and accuracy == 0 and mape == 0 and weighted_score_pct == 0:
                subtitle = "Future Prediction Only (No Validation Data Available)"
            else:
                subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score_pct:.2f}%"

        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # (1) 상단: 가격 예측 그래프
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("Price Prediction: Past Validation vs Future Forecast", fontsize=13)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # 예측 시작 날짜 처리
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        # 시작일 이전 날짜 계산 (연결점용)
        prev_date = start_date - pd.Timedelta(days=1)
        while prev_date.weekday() >= 5 or is_holiday(prev_date):
            prev_date -= pd.Timedelta(days=1)
        
        # ✅ 1. 과거 실제값 (파란색 실선) - 가장 중요한 기준선
        if not valid_df.empty:
            real_dates = [prev_date] + valid_df['Date'].tolist()
            real_values = [start_day_value] + valid_df['Actual'].tolist()
            ax1.plot(real_dates, real_values, marker='o', color='blue', 
                    label='Actual (Past)', linewidth=2.5, markersize=5, zorder=3)
        
        # ✅ 2. 과거 예측값 (회색 점선) - 모델 성능 확인용
        if not valid_df.empty:
            past_pred_dates = [prev_date] + valid_df['Date'].tolist()
            past_pred_values = [start_day_value] + valid_df['Prediction'].tolist()
            ax1.plot(past_pred_dates, past_pred_values, marker='x', color='gray', 
                    label='Predicted (Past)', linewidth=1.5, linestyle=':', markersize=4, alpha=0.8, zorder=2)
        
        # ✅ 3. 미래 예측값 (빨간색 점선) - 핵심 예측
        if not future_df.empty:
            future_dates = future_df['Date'].tolist()
            future_values = future_df['Prediction'].tolist()
            
            # 연결선 (마지막 실제값 → 첫 미래 예측값)
            if not valid_df.empty and future_dates:
                # 마지막 검증 데이터의 실제값에서 첫 미래 예측으로 연결
                connection_x = [valid_df['Date'].iloc[-1], future_dates[0]]
                connection_y = [valid_df['Actual'].iloc[-1], future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            elif start_day_value is not None and future_dates:
                # 검증 데이터가 없으면 시작값에서 연결
                connection_x = [prev_date, future_dates[0]]
                connection_y = [start_day_value, future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            
            ax1.plot(future_dates, future_values, marker='o', color='red', 
                    label='Predicted (Future)', linewidth=2.5, linestyle='--', markersize=5, zorder=3)
        
        # ✅ 4. 데이터 컷오프 라인 (초록색 세로선)
        if current_date is not None:
            ax1.axvline(x=current_date, color='green', linestyle='-', alpha=0.8, 
                       linewidth=2.5, label=f'Data Cutoff', zorder=4)
            
            # 컷오프 날짜 텍스트 추가
            ax1.text(current_date, ax1.get_ylim()[1] * 0.95, 
                    f'{current_date.strftime("%m/%d")}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        else:
            # 예측 시작점에 수직선 표시 (기존 방식)
            ax1.axvline(x=start_date, color='green', linestyle='--', alpha=0.7, 
                       linewidth=2, label='Prediction Start', zorder=4)
        
        # ✅ 5. 배경 색칠 (방향성 일치 여부) - 검증 데이터만
        if not valid_df.empty and len(valid_df) > 1:
            for i in range(len(valid_df) - 1):
                curr_date = valid_df['Date'].iloc[i]
                next_date = valid_df['Date'].iloc[i + 1]
                
                curr_actual = valid_df['Actual'].iloc[i]
                next_actual = valid_df['Actual'].iloc[i + 1]
                curr_pred = valid_df['Prediction'].iloc[i]
                next_pred = valid_df['Prediction'].iloc[i + 1]
                
                # 방향 계산
                actual_dir = np.sign(next_actual - curr_actual)
                pred_dir = np.sign(next_pred - curr_pred)
                
                # 방향 일치 여부에 따른 색상
                color = 'lightblue' if actual_dir == pred_dir else 'lightcoral'
                ax1.axvspan(curr_date, next_date, color=color, alpha=0.15, zorder=0)
        
        ax1.set_xlabel("")
        ax1.set_ylabel("Price (USD/MT)", fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # ✅ (2) 하단: 오차 분석 - 검증 데이터만 또는 변화량
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if not valid_df.empty and len(valid_df) > 0:
            # 검증 데이터의 절대 오차
            error_dates = valid_df['Date'].tolist()
            error_values = [abs(row['Actual'] - row['Prediction']) for _, row in valid_df.iterrows()]
            
            if error_dates and error_values:
                bars = ax2.bar(error_dates, error_values, width=0.6, color='salmon', alpha=0.7, edgecolor='darkred', linewidth=0.5)
                ax2.set_title(f"Prediction Error - Validation Period ({len(error_dates)} points)", fontsize=11)
                
                # 평균 오차 라인
                avg_error = np.mean(error_values)
                ax2.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8, 
                           label=f'Avg Error: {avg_error:.2f}')
                ax2.legend(fontsize=9)
            else:
                ax2.text(0.5, 0.5, "No validation errors to display", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Error Analysis")
        else:
            # 실제값이 없는 경우: 미래 예측의 일일 변화량 표시
            if not future_df.empty and len(future_df) > 1:
                change_dates = future_df['Date'].iloc[1:].tolist()
                change_values = np.diff(future_df['Prediction'].values)
                
                # 상승/하락에 따른 색상 구분
                colors = ['green' if change >= 0 else 'red' for change in change_values]
                
                bars = ax2.bar(change_dates, change_values, width=0.6, color=colors, alpha=0.7)
                ax2.set_title("Daily Price Changes - Future Predictions", fontsize=11)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 범례 추가
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', alpha=0.7, label='Price Up'),
                                 Patch(facecolor='red', alpha=0.7, label='Price Down')]
                ax2.legend(handles=legend_elements, fontsize=9)
            else:
                ax2.text(0.5, 0.5, "Insufficient data for change analysis", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Change Analysis")
        
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 파일 경로 생성
        if isinstance(prediction_start_date, str):
            date_str = pd.to_datetime(prediction_start_date).strftime('%Y%m%d')
        else:
            date_str = prediction_start_date.strftime('%Y%m%d')
        
        filename = f"prediction_start_{date_str}.png"
        full_path = save_dir / filename
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장
        plt.savefig(str(full_path), dpi=300, bbox_inches='tight')
        
        # 메모리 정리
        plt.close(fig)
        plt.clf()
        img_buf.close()
        
        logger.info(f"Enhanced prediction graph saved: {full_path}")
        logger.info(f"  - Past validation points: {len(valid_df) if not valid_df.empty else 0}")
        logger.info(f"  - Future prediction points: {len(future_df) if not future_df.empty else 0}")
        
        return str(full_path), img_str
        
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        plt.close('all')
        plt.clf()
        
        logger.error(f"Error in enhanced graph creation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    
def plot_moving_average_analysis(ma_results, sequence_start_date, save_prefix=None,
                               title_prefix="Moving Average Analysis", y_min=None, y_max=None, file_path=None):
    """이동평균 분석 시각화"""
    try:
        # 입력 데이터 검증
        if not ma_results or len(ma_results) == 0:
            logger.warning("No moving average results to plot")
            return None, None
            
        # ma_results 형식: {'ma5': [{'date': '...', 'prediction': X, 'actual': Y, 'ma': Z}, ...], 'ma10': [...]}
        windows = sorted(ma_results.keys())
        
        if len(windows) == 0:
            logger.warning("No moving average windows found")
            return None, None
        
        # 유효한 윈도우 필터링
        valid_windows = []
        for window_key in windows:
            if window_key in ma_results and ma_results[window_key] and len(ma_results[window_key]) > 0:
                valid_windows.append(window_key)
        
        if len(valid_windows) == 0:
            logger.warning("No valid moving average data found")
            return None, None
        
        fig = plt.figure(figsize=(12, max(4, 4 * len(valid_windows))))
        
        if isinstance(sequence_start_date, str):
            title = f"{title_prefix} Starting {sequence_start_date}"
        else:
            title = f"{title_prefix} Starting {sequence_start_date.strftime('%Y-%m-%d')}"
            
        fig.suptitle(title, fontsize=16)
        
        for idx, window_key in enumerate(valid_windows):
            window_num = window_key.replace('ma', '')
            ax = fig.add_subplot(len(valid_windows), 1, idx+1)
            
            window_data = ma_results[window_key]
            
            # 데이터 검증
            if not window_data or len(window_data) == 0:
                ax.text(0.5, 0.5, f"No data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 날짜, 예측, 실제값, MA 추출
            dates = []
            predictions = []
            actuals = []
            ma_preds = []
            
            for item in window_data:
                try:
                    # 안전한 데이터 추출
                    if isinstance(item['date'], str):
                        dates.append(pd.to_datetime(item['date']))
                    else:
                        dates.append(item['date'])
                    
                    # None 값 처리
                    predictions.append(item.get('prediction', 0))
                    actuals.append(item.get('actual', None))
                    ma_preds.append(item.get('ma', None))
                except Exception as e:
                    logger.warning(f"Error processing MA data item: {str(e)}")
                    continue
            
            if len(dates) == 0:
                ax.text(0.5, 0.5, f"No valid data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # y축 범위 설정
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            # 원본 실제값 vs 예측값 (옅게)
            ax.plot(dates, actuals, marker='o', color='blue', alpha=0.3, label='Actual')
            ax.plot(dates, predictions, marker='o', color='red', alpha=0.3, label='Predicted')
            
            # 이동평균
            # 실제값(actuals)과 이동평균(ma_preds) 모두 None이 아닌 인덱스를 선택
            valid_indices = [
                i for i in range(len(ma_preds))
                if (ma_preds[i] is not None and actuals[i] is not None)
            ]

            if valid_indices:
                valid_dates = [dates[i] for i in valid_indices]
                valid_ma = [ma_preds[i] for i in valid_indices]
                valid_actuals = [actuals[i] for i in valid_indices]
                
                # 배열로 변환
                valid_actuals_arr = np.array(valid_actuals)
                valid_ma_arr = np.array(valid_ma)
                
                # 실제값이 0인 항목은 제외하여 MAPE 계산
                non_zero_mask = valid_actuals_arr != 0
                if np.sum(non_zero_mask) > 0:
                    ma_mape = np.mean(np.abs((valid_actuals_arr[non_zero_mask] - valid_ma_arr[non_zero_mask]) /
                                           valid_actuals_arr[non_zero_mask])) * 100
                else:
                    ma_mape = 0.0
                
                ax.set_title(f"MA-{window_num} Analysis (MAPE: {ma_mape:.2f}%, Count: {len(valid_indices)})")
            else:
                ax.set_title(f"MA-{window_num} Analysis (Insufficient data)")
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
        
        plt.tight_layout()
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['ma_plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for MA plots: {str(e)}")
                save_dir = Path("temp_ma_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(sequence_start_date, str):
            date_str = pd.to_datetime(sequence_start_date).strftime('%Y%m%d')
        else:
            date_str = sequence_start_date.strftime('%Y%m%d')
            
        filename = save_dir / f"ma_analysis_{date_str}.png"
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일 저장
        plt.savefig(str(filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving Average graph saved: {filename}")
        return str(filename), img_str
        
    except Exception as e:
        logger.error(f"Error in moving average visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def compute_performance_metrics_improved(validation_data, start_day_value):
    """
    검증 데이터만을 사용한 성능 지표 계산
    """
    try:
        if not validation_data or len(validation_data) < 1:
            logger.info("No validation data available - this is normal for pure future predictions")
            return None
        
        # start_day_value 안전하게 처리
        if hasattr(start_day_value, 'iloc'):  # pandas Series/DataFrame인 경우
            start_val = float(start_day_value.iloc[0] if len(start_day_value) > 0 else start_day_value)
        elif hasattr(start_day_value, 'item'):  # numpy scalar인 경우
            start_val = float(start_day_value.item())
        else:
            start_val = float(start_day_value)
        
        # 검증 데이터에서 값 추출 (DataFrame/Series를 numpy로 안전하게 변환)
        actual_vals = [start_val]
        pred_vals = [start_val]
        
        for item in validation_data:
            # actual 값 안전하게 추출
            actual_val = item['actual']
            if hasattr(actual_val, 'iloc'):  # pandas Series/DataFrame인 경우
                actual_val = float(actual_val.iloc[0] if len(actual_val) > 0 else actual_val)
            elif hasattr(actual_val, 'item'):  # numpy scalar인 경우
                actual_val = float(actual_val.item())
            else:
                actual_val = float(actual_val)
            actual_vals.append(actual_val)
            
            # prediction 값 안전하게 추출
            pred_val = item['prediction']
            if hasattr(pred_val, 'iloc'):  # pandas Series/DataFrame인 경우
                pred_val = float(pred_val.iloc[0] if len(pred_val) > 0 else pred_val)
            elif hasattr(pred_val, 'item'):  # numpy scalar인 경우
                pred_val = float(pred_val.item())
            else:
                pred_val = float(pred_val)
            pred_vals.append(pred_val)
        
        # F1 점수 계산 (각 단계별 로깅 추가)
        try:
            f1, f1_report = calculate_f1_score(actual_vals, pred_vals)
        except Exception as e:
            logger.error(f"Error in F1 score calculation: {str(e)}")
            f1, f1_report = 0.0, "Error in F1 calculation"
            
        try:
            direction_accuracy = calculate_direction_accuracy(actual_vals, pred_vals)
        except Exception as e:
            logger.error(f"Error in direction accuracy calculation: {str(e)}")
            direction_accuracy = 0.0
            
        try:
            weighted_score, max_score = calculate_direction_weighted_score(actual_vals[1:], pred_vals[1:])
            weighted_score_pct = (weighted_score / max_score) * 100 if max_score > 0 else 0.0
        except Exception as e:
            logger.error(f"Error in weighted score calculation: {str(e)}")
            weighted_score_pct = 0.0
            
        try:
            mape = calculate_mape(actual_vals[1:], pred_vals[1:])
        except Exception as e:
            logger.error(f"Error in MAPE calculation: {str(e)}")
            mape = 0.0
        
        # 코사인 유사도
        cosine_similarity = None
        try:
            if len(actual_vals) > 1:
                # numpy 배열로 변환하여 안전하게 처리
                actual_vals_arr = np.array(actual_vals, dtype=float)
                pred_vals_arr = np.array(pred_vals, dtype=float)
                
                diff_actual = np.diff(actual_vals_arr)
                diff_pred = np.diff(pred_vals_arr)
                norm_actual = np.linalg.norm(diff_actual)
                norm_pred = np.linalg.norm(diff_pred)
                if norm_actual > 0 and norm_pred > 0:
                    cosine_similarity = np.dot(diff_actual, diff_pred) / (norm_actual * norm_pred)
        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}")
            cosine_similarity = None
        
        return {
            'f1': float(f1),
            'accuracy': float(direction_accuracy),
            'mape': float(mape),
            'weighted_score': float(weighted_score_pct),
            'cosine_similarity': float(cosine_similarity) if cosine_similarity is not None else None,
            'f1_report': f1_report,
            'validation_points': len(validation_data)
        }
        
    except Exception as e:
        logger.error(f"Error computing improved metrics: {str(e)}")
        return None

def calculate_f1_score(actual, predicted):
    """방향성 예측의 F1 점수 계산"""
    # 입력을 numpy 배열로 변환
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    actual_directions = np.sign(np.diff(actual))
    predicted_directions = np.sign(np.diff(predicted))

    if len(actual_directions) < 2:
        return 0.0, "Insufficient data for classification report"
        
    try:
        # zero_division=0 파라미터 추가
        f1 = f1_score(actual_directions, predicted_directions, average='macro', zero_division=0)
        report = classification_report(actual_directions, predicted_directions, 
                                    digits=2, zero_division=0)
    except Exception as e:
        logger.error(f"Error in calculating F1 score: {str(e)}")
        return 0.0, "Error in calculation"
        
    return f1, report

def calculate_direction_accuracy(actual, predicted):
    """등락 방향 예측의 정확도 계산"""
    if len(actual) <= 1:
        return 0.0

    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        actual_directions = np.sign(np.diff(actual))
        predicted_directions = np.sign(np.diff(predicted))
        
        correct_predictions = np.sum(actual_directions == predicted_directions)
        total_predictions = len(actual_directions)
        
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy
    except Exception as e:
        logger.error(f"Error in calculating direction accuracy: {str(e)}")
        return 0.0
    
def calculate_direction_weighted_score(actual, predicted):
    """변화율 기반의 가중 점수 계산"""
    if len(actual) <= 1:
        return 0.0, 1.0
        
    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        actual_changes = 100 * np.diff(actual) / actual[:-1]
        predicted_changes = 100 * np.diff(predicted) / predicted[:-1]

        def assign_class(change):
            if change > 6:
                return 1
            elif 4 < change <= 6:
                return 2
            elif 2 < change <= 4:
                return 3
            elif -2 <= change <= 2:
                return 4
            elif -4 <= change < -2:
                return 5
            elif -6 <= change < -4:
                return 6
            else:
                return 7

        actual_classes = np.array([assign_class(x) for x in actual_changes])
        predicted_classes = np.array([assign_class(x) for x in predicted_changes])

        score = 0
        for ac, pc in zip(actual_classes, predicted_classes):
            diff = abs(ac - pc)
            score += max(0, 3 - diff)

        max_score = 3 * len(actual_classes)
        return score, max_score
    except Exception as e:
        logger.error(f"Error in calculating weighted score: {str(e)}")
        return 0.0, 1.0

def calculate_mape(actual, predicted):
    """MAPE 계산 함수"""
    try:
        # 입력을 numpy 배열로 변환
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) == 0:
            return 0.0
        # inf 방지를 위해 0이 아닌 값만 사용
        mask = actual != 0
        if not np.any(mask):  # any() 대신 np.any() 사용
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    except Exception as e:
        logger.error(f"Error in MAPE calculation: {str(e)}")
        return 0.0

def calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ', windows=[5, 10, 23]):
    """예측 데이터와 과거 데이터를 모두 활용한 이동평균 계산"""
    try:
        # 입력 데이터 검증
        if not predictions or len(predictions) == 0:
            logger.warning("No predictions provided for moving average calculation")
            return {}
            
        if historical_data is None or historical_data.empty:
            logger.warning("No historical data provided for moving average calculation")
            return {}
            
        if target_col not in historical_data.columns:
            logger.warning(f"Target column {target_col} not found in historical data")
            return {}
        
        results = {}
        
        # 예측 데이터를 DataFrame으로 변환 및 정렬
        try:
            pred_df = pd.DataFrame(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
            
            # Date 컬럼 검증
            if 'Date' not in pred_df.columns:
                logger.error("Date column not found in predictions")
                return {}
                
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # Prediction 컬럼 검증
            if 'Prediction' not in pred_df.columns:
                logger.error("Prediction column not found in predictions")
                return {}
                
        except Exception as e:
            logger.error(f"Error processing prediction data: {str(e)}")
            return {}
        
        # 예측 시작일 확인
        prediction_start_date = pred_df['Date'].min()
        logger.info(f"MA calculation - prediction start date: {prediction_start_date}")
        
        # 과거 데이터에서 타겟 열 추출 (예측 시작일 이전)
        historical_series = pd.Series(
            data=historical_data.loc[historical_data.index < prediction_start_date, target_col],
            index=historical_data.loc[historical_data.index < prediction_start_date].index
        )
        
        # 최근 30일만 사용 (이동평균 계산에 충분)
        historical_series = historical_series.sort_index().tail(30)
        
        # 예측 데이터에서 시리즈 생성
        prediction_series = pd.Series(
            data=pred_df['Prediction'].values,
            index=pred_df['Date']
        )
        
        # 과거와 예측 데이터 결합
        combined_series = pd.concat([historical_series, prediction_series])
        combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
        combined_series = combined_series.sort_index()
        
        logger.info(f"Combined series for MA: {len(combined_series)} data points "
                   f"({len(historical_series)} historical, {len(prediction_series)} predicted)")
        
        # 각 윈도우 크기별 이동평균 계산
        for window in windows:
            # 전체 데이터에 대해 이동평균 계산
            rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
            
            # 예측 기간에 해당하는 부분만 추출
            window_results = []
            
            for i, date in enumerate(pred_df['Date']):
                # 해당 날짜의 예측 및 실제값
                pred_value = pred_df['Prediction'].iloc[i]
                actual_value = pred_df['Actual'].iloc[i] if 'Actual' in pred_df.columns else None
                
                # 해당 날짜의 이동평균 값
                ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                
                # NaN 값 처리
                if pd.isna(pred_value) or np.isnan(pred_value) or np.isinf(pred_value):
                    pred_value = None
                if pd.isna(actual_value) or np.isnan(actual_value) or np.isinf(actual_value):
                    actual_value = None
                if pd.isna(ma_value) or np.isnan(ma_value) or np.isinf(ma_value):
                    ma_value = None
                
                window_results.append({
                    'date': date,
                    'prediction': pred_value,
                    'actual': actual_value,
                    'ma': ma_value
                })
            
            results[f'ma{window}'] = window_results
            logger.info(f"MA{window} calculated: {len(window_results)} data points")
        
        logger.info(f"Moving average calculation completed with {len(results)} windows")
        return results
        
    except Exception as e:
        logger.error(f"Error calculating moving averages with history: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# 2. 여러 날짜에 대한 누적 예측을 수행하는 함수 추가
def run_accumulated_predictions_with_save(file_path, start_date, end_date=None, save_to_csv=True, use_saved_data=True):
    """
    시작 날짜부터 종료 날짜까지 각 날짜별로 예측을 수행하고 결과를 누적합니다. (수정됨)
    """
    global prediction_state

    try:
        # 상태 초기화
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 5
        prediction_state['prediction_start_time'] = time.time()  # 시작 시간 기록
        prediction_state['error'] = None
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['prediction_dates'] = []
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['current_file'] = file_path  # ✅ 현재 파일 경로 설정
        
        logger.info(f"Running accumulated predictions from {start_date} to {end_date}")

        # 입력 날짜를 datetime 객체로 변환
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is not None and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 저장된 데이터 활용 옵션이 켜져 있으면 먼저 CSV에서 로드 시도
        loaded_predictions = []
        if use_saved_data:
            logger.info("🔍 [CACHE] Attempting to load existing predictions from CSV files...")
            
            # 🔧 인덱스 파일이 없으면 기존 파일들로부터 재생성
            cache_dirs = get_file_cache_dirs(file_path)
            predictions_index_file = cache_dirs['predictions'] / 'predictions_index.csv'
            
            if not predictions_index_file.exists():
                logger.warning("⚠️ [CACHE] predictions_index.csv not found, attempting to rebuild from existing files...")
                if rebuild_predictions_index_from_existing_files():
                    logger.info("✅ [CACHE] Successfully rebuilt predictions index")
                else:
                    logger.warning("⚠️ [CACHE] Failed to rebuild predictions index")
            
            loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date, file_path=file_path)  # ✅ 파일 경로 추가
            logger.info(f"📦 [CACHE] Successfully loaded {len(loaded_predictions)} predictions from CSV cache")
            if len(loaded_predictions) > 0:
                logger.info(f"💡 [CACHE] Using cached predictions will significantly speed up processing!")

        # 데이터 로드 (누적 예측용 - LSTM 모델, 2022년 이전 데이터 제거)
        df = load_data(file_path, model_type='lstm')
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 10

        # 종료 날짜가 지정되지 않으면 데이터의 마지막 날짜 사용
        if end_date is None:
            end_date = df.index.max()

        # 사용 가능한 날짜 추출 후 정렬
        available_dates = [date for date in df.index if start_date <= date <= end_date]
        available_dates.sort()
        
        if not available_dates:
            raise ValueError(f"지정된 기간 내에 사용 가능한 날짜가 없습니다: {start_date} ~ {end_date}")

        total_dates = len(available_dates)
        logger.info(f"Accumulated prediction: {total_dates} dates from {start_date} to {end_date}")

        # 누적 성능 지표 초기화
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }

        # 이미 로드된 예측 결과들을 날짜별 딕셔너리로 변환
        loaded_by_date = {}
        for pred in loaded_predictions:
            loaded_by_date[pred['date']] = pred

        # ✅ 캐시 활용 통계 초기화
        cache_statistics = {
            'total_dates': 0,
            'cached_dates': 0,
            'new_predictions': 0,
            'cache_hit_rate': 0.0
        }

        all_predictions = []
        accumulated_interval_scores = {}

        # 각 날짜별 예측 수행 또는 로드
        for i, current_date in enumerate(available_dates):
            current_date_str = format_date(current_date)
            cache_statistics['total_dates'] += 1
            
            logger.info(f"Processing date {i+1}/{total_dates}: {current_date_str}")
            
            # 이미 로드된 데이터가 있으면 사용
            if current_date_str in loaded_by_date:
                cache_statistics['cached_dates'] += 1  # ✅ 캐시 사용 시 카운터 증가
                logger.info(f"⚡ [CACHE] Using cached prediction for {current_date_str} (skipping computation)")
                date_result = loaded_by_date[current_date_str]
                
                # 🔧 캐시된 metrics 안전성 처리
                metrics = date_result.get('metrics')
                if not metrics or not isinstance(metrics, dict):
                    logger.warning(f"⚠️ [CACHE] Invalid metrics for {current_date_str}, using defaults")
                    metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                
                # 누적 성능 지표 업데이트
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
                
            else:
                # 새로운 예측 수행
                cache_statistics['new_predictions'] += 1
                logger.info(f"🚀 [COMPUTE] Running new prediction for {current_date_str} (not in cache)")
                try:
                    # ✅ 누적 예측에서도 모든 새 예측을 저장하도록 보장
                    results = generate_predictions_with_save(df, current_date, save_to_csv=True, file_path=file_path)
                    
                    # 예측 데이터 타입 안전 확인
                    predictions = results.get('predictions_flat', results.get('predictions', []))
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not predictions or not isinstance(predictions, list):
                        logger.warning(f"No valid predictions found for {current_date_str}: {type(predictions)}")
                        continue
                        
                    # 실제 예측한 영업일 수 계산 (안전한 방식)
                    actual_business_days = 0
                    try:
                        for p in predictions:
                            # p가 딕셔너리인지 확인
                            if isinstance(p, dict):
                                date_key = p.get('Date') or p.get('date')
                                is_synthetic = p.get('is_synthetic', False)
                                if date_key and not is_synthetic:
                                    actual_business_days += 1
                            else:
                                logger.warning(f"Prediction item is not dict for {current_date_str}: {type(p)}")
                    except Exception as calc_error:
                        logger.error(f"Error calculating business days: {str(calc_error)}")
                        actual_business_days = len(predictions)  # 기본값
                    
                    metrics = results.get('metrics', {})
                    if not metrics:
                        # 메트릭이 없으면 기본값 설정
                        metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    # 누적 성능 지표 업데이트
                    accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                    accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                    accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                    accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                    accumulated_metrics['total_predictions'] += 1

                    # 안전한 데이터 구조 생성
                    safe_predictions = predictions if isinstance(predictions, list) else []
                    safe_interval_scores = results.get('interval_scores', {})
                    if not isinstance(safe_interval_scores, dict):
                        safe_interval_scores = {}
                    
                    date_result = {
                        'date': current_date_str,
                        'predictions': safe_predictions,
                        'metrics': metrics,
                        'interval_scores': safe_interval_scores,
                        'actual_business_days': actual_business_days,
                        'next_semimonthly_period': results.get('next_semimonthly_period'),
                        'original_interval_scores': safe_interval_scores,
                        'ma_results': results.get('ma_results', {}),  # 🔑 이동평균 데이터 추가
                        'attention_data': results.get('attention_data', {})  # 🔑 Attention 데이터 추가
                    }
                    
                except Exception as e:
                    logger.error(f"Error in prediction for date {current_date}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # 구간 점수 누적 처리 (안전한 방식)
            interval_scores = date_result.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                        accumulated_interval_scores[interval_key]['avg_price'] = (
                            (accumulated_interval_scores[interval_key]['avg_price'] *
                             (accumulated_interval_scores[interval_key]['count'] - 1) +
                             interval['avg_price']) / accumulated_interval_scores[interval_key]['count']
                        )
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1

            all_predictions.append(date_result)
            prediction_state['prediction_progress'] = 10 + int(90 * (i + 1) / total_dates)

        # 평균 성능 지표 계산
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count

        # 예측 신뢰도 계산
        logger.info("Calculating prediction consistency scores...")
        unique_periods = set()
        for pred in all_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(all_predictions, period)
                accumulated_consistency_scores[period] = consistency_data
                logger.info(f"Consistency score for {period}: {consistency_data.get('consistency_score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")

        # accumulated_interval_scores 처리
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)

        accumulated_purchase_reliability, debug_info = calculate_accumulated_purchase_reliability(all_predictions)
        
        # ✅ 캐시 활용률 계산
        cache_statistics['cache_hit_rate'] = (cache_statistics['cached_dates'] / cache_statistics['total_dates'] * 100) if cache_statistics['total_dates'] > 0 else 0.0
        logger.info(f"🎯 [CACHE] Final statistics: {cache_statistics['cached_dates']}/{cache_statistics['total_dates']} cached ({cache_statistics['cache_hit_rate']:.1f}%), {cache_statistics['new_predictions']} new predictions computed")
        
        # 결과 저장
        prediction_state['accumulated_predictions'] = all_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in all_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['accumulated_purchase_debug'] = debug_info
        prediction_state['cache_statistics'] = cache_statistics  # ✅ 캐시 통계 추가

        if all_predictions:
            latest = all_predictions[-1]
            prediction_state['latest_predictions'] = latest['predictions']
            prediction_state['current_date'] = latest['date']

        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 100
        logger.info(f"Accumulated prediction completed for {len(all_predictions)} dates")
        
    except Exception as e:
        logger.error(f"Error in accumulated prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0
        prediction_state['accumulated_consistency_scores'] = {}

# 3. 백그라운드에서 누적 예측을 수행하는 함수
def background_accumulated_prediction(file_path, start_date, end_date=None):
    """백그라운드에서 누적 예측을 수행하는 함수"""
    thread = Thread(target=run_accumulated_predictions_with_save, args=(file_path, start_date, end_date))
    thread.daemon = True
    thread.start()
    return thread

# 6. 누적 결과 보고서 생성 함수
def generate_accumulated_report():
    """누적 예측 결과 보고서 생성"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None
    
    try:
        metrics = prediction_state['accumulated_metrics']
        all_preds = prediction_state['accumulated_predictions']
        
        # 보고서 파일 이름 생성 - 파일별 캐시 디렉토리 사용
        start_date = all_preds[0]['date']
        end_date = all_preds[-1]['date']
        try:
            cache_dirs = get_file_cache_dirs()
            report_dir = cache_dirs['predictions']
            report_filename = os.path.join(report_dir, f"accumulated_report_{start_date}_to_{end_date}.txt")
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated report: {str(e)}")
            report_filename = f"accumulated_report_{start_date}_to_{end_date}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"===== Accumulated Prediction Report =====\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Total Predictions: {metrics['total_predictions']}\n\n")
            
            # 누적 성능 지표
            f.write("Average Performance Metrics:\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"- Direction Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"- MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"- Weighted Score: {metrics['weighted_score']:.2f}%\n\n")
            
            # 날짜별 상세 정보
            f.write("Performance By Date:\n")
            for pred in all_preds:
                date = pred['date']
                m = pred['metrics']
                f.write(f"\n* {date}:\n")
                f.write(f"  - F1 Score: {m['f1']:.4f}\n")
                f.write(f"  - Accuracy: {m['accuracy']:.2f}%\n")
                f.write(f"  - MAPE: {m['mape']:.2f}%\n")
                f.write(f"  - Weighted Score: {m['weighted_score']:.2f}%\n")
                
                # 구매 구간 정보
                if pred['interval_scores']:
                    best_interval = decide_purchase_interval(pred['interval_scores'])
                    f.write("Best Purchase Interval:\n")
                    f.write(f"- Start Date: {best_interval['start_date']}\n")
                    f.write(f"- End Date: {best_interval['end_date']}\n")
                    f.write(f"- Duration: {best_interval['days']} days\n")
                    f.write(f"- Average Price: {best_interval['avg_price']:.2f}\n")
                    f.write(f"- Score: {best_interval['score']}\n")
                    f.write(f"- Selection Reason: {best_interval.get('selection_reason', '')}\n\n")
        
        return report_filename
    
    except Exception as e:
        logger.error(f"Error generating accumulated report: {str(e)}")
        return None

# 9. 누적 예측 결과 시각화 함수
def visualize_accumulated_metrics():
    """누적 예측 결과 시각화"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None, None
    
    try:
        # 데이터 준비
        dates = []
        f1_scores = []
        accuracies = []
        mapes = []
        weighted_scores = []
        
        for pred in prediction_state['accumulated_predictions']:
            dates.append(pred['date'])
            m = pred['metrics']
            f1_scores.append(m['f1'])
            accuracies.append(m['accuracy'])
            mapes.append(m['mape'])
            weighted_scores.append(m['weighted_score'])
        
        # 날짜를 datetime으로 변환
        dates = [pd.to_datetime(d) for d in dates]
        
        # 그래프 생성
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Accumulated Prediction Metrics', fontsize=16)
        
        # F1 Score
        axs[0, 0].plot(dates, f1_scores, marker='o', color='blue')
        axs[0, 0].set_title('F1 Score')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].grid(True)
        plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Accuracy
        axs[0, 1].plot(dates, accuracies, marker='o', color='green')
        axs[0, 1].set_title('Direction Accuracy (%)')
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].grid(True)
        plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # MAPE
        axs[1, 0].plot(dates, mapes, marker='o', color='red')
        axs[1, 0].set_title('MAPE (%)')
        axs[1, 0].set_ylim(0, max(mapes) * 1.2)
        axs[1, 0].grid(True)
        plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Weighted Score
        axs[1, 1].plot(dates, weighted_scores, marker='o', color='purple')
        axs[1, 1].set_title('Weighted Score (%)')
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True)
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 이미지 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장 - 파일별 캐시 디렉토리 사용
        try:
            cache_dirs = get_file_cache_dirs()
            plots_dir = cache_dirs['plots']
            filename = os.path.join(plots_dir, 'accumulated_metrics.png')
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated metrics: {str(e)}")
            filename = 'accumulated_metrics.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename, img_str
        
    except Exception as e:
        logger.error(f"Error visualizing accumulated metrics: {str(e)}")
        return None, None

#######################################################################
# 예측 및 모델 학습 함수
#######################################################################

def prepare_data(train_data, val_data, sequence_length, predict_window, target_col_idx, augment=False):
    """학습 및 검증 데이터를 시퀀스 형태로 준비"""
    X_train, y_train, prev_train = [], [], []
    for i in range(len(train_data) - sequence_length - predict_window + 1):
        seq = train_data[i:i+sequence_length]
        target = train_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx]
        prev_value = train_data[i+sequence_length-1, target_col_idx]
        X_train.append(seq)
        y_train.append(target)
        prev_train.append(prev_value)
        if augment:
            # 간단한 데이터 증강
            noise = np.random.normal(0, 0.001, seq.shape)
            aug_seq = seq + noise
            X_train.append(aug_seq)
            y_train.append(target)
            prev_train.append(prev_value)
    
    X_val, y_val, prev_val = [], [], []
    for i in range(len(val_data) - sequence_length - predict_window + 1):
        X_val.append(val_data[i:i+sequence_length])
        y_val.append(val_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx])
        prev_val.append(val_data[i+sequence_length-1, target_col_idx])
    
    return map(np.array, [X_train, y_train, prev_train, X_val, y_val, prev_val])



def train_model(features, target_col, current_date, historical_data, device, params):
    """LSTM 모델 학습"""
    try:
        # 일관된 학습 결과를 위한 시드 고정
        set_seed()
        
        # 디바이스 사용 정보 로깅
        log_device_usage(device, "LSTM 모델 학습 시작")
        
        # 특성 이름 확인
        if target_col not in features:
            features.append(target_col)
        
        # 학습 데이터 준비 (현재 날짜까지)
        train_df = historical_data[features].copy()
        target_col_idx = train_df.columns.get_loc(target_col)
        
        # 스케일링
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df)
        
        # 하이퍼파라미터
        sequence_length = params.get('sequence_length', 20)
        hidden_size = params.get('hidden_size', 128)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        num_epochs = params.get('num_epochs', 100)
        batch_size = params.get('batch_size', 32)
        alpha = params.get('loss_alpha', 0.7)  # DirectionalLoss alpha
        beta = params.get('loss_beta', 0.2)    # DirectionalLoss beta
        patience = params.get('patience', 20)   # 조기 종료 인내
        predict_window = params.get('predict_window', 23)  # 예측 기간
        
        # 80/20 분할 (연대순)
        train_size = int(len(train_data) * 0.8)
        train_set = train_data[:train_size]
        val_set = train_data[train_size:]
        
        # 시퀀스 데이터 준비
        X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
            train_set, val_set, sequence_length, predict_window, target_col_idx
        )
        
        # 충분한 데이터가 있는지 확인
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train) // 2)
            logger.warning(f"배치 크기가 데이터 크기보다 커서 조정: {batch_size} (데이터: {len(X_train)})")
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data for training")
        
        logger.info(f"🎯 사용할 배치 크기: {batch_size}")
        
        # 데이터셋 및 로더 생성 (CPU에서 생성, 학습 시 GPU로 이동)
        train_dataset = TimeSeriesDataset(X_train, y_train, torch.device('cpu'), prev_train)
        
        # GPU 활용률 최적화를 위한 DataLoader 설정
        num_workers = 0 if device.type == 'cuda' else 2  # CUDA에서는 멀티프로세싱 비활성화
        pin_memory = device.type == 'cuda'  # GPU 사용 시 pin_memory 활성화
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False if num_workers == 0 else True
        )
        
        val_dataset = TimeSeriesDataset(X_val, y_val, torch.device('cpu'), prev_val)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(batch_size, len(X_val)),  # 검증에서도 배치 크기 최적화
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        logger.info(f"🔧 DataLoader 설정: workers={num_workers}, pin_memory={pin_memory}, train_batch={batch_size}, val_batch={min(batch_size, len(X_val))}")
        
        # 모델 생성
        logger.info("📈 ImprovedLSTMPredictor 사용")
        model = ImprovedLSTMPredictor(
            input_size=train_data.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=predict_window
        ).to(device)
        
        # 모델이 GPU에 올라갔는지 확인
        model_device = next(model.parameters()).device
        logger.info(f"🤖 ImprovedLSTM 모델이 {model_device}에 로드되었습니다")
        log_device_usage(model_device, "모델 로드 완료")
        
        # 손실 함수 생성
        logger.info(f"📈 DirectionalLoss 사용: alpha={alpha}, beta={beta}")
        criterion = DirectionalLoss(alpha=alpha, beta=beta)
        
        # 최적화기 및 스케줄러
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=patience//2
        )
        
        # 학습
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # GPU 최적화 설정
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 성능 향상
            torch.cuda.empty_cache()  # 캐시 정리
            
        log_device_usage(device, "모델 학습 중")
        
        for epoch in range(num_epochs):
            # 학습 모드
            model.train()
            train_loss = 0
            batch_count = 0
            
            for X_batch, y_batch, prev_batch in train_loader:
                optimizer.zero_grad()
                
                # 모델과 같은 디바이스로 데이터 이동
                X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                
                # 모델 예측 및 손실 계산
                y_pred = model(X_batch, prev_batch)
                loss = criterion(y_pred, y_batch, prev_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
                
            # 첫 번째 에포크와 주기적으로 GPU 상태 로깅
            if epoch == 0 or (epoch + 1) % 10 == 0:
                log_device_usage(device, f"에포크 {epoch+1}/{num_epochs}")
            
            # 검증 모드
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch, prev_batch in val_loader:
                    # 모델과 같은 디바이스로 데이터 이동
                    X_batch = X_batch.to(device, non_blocking=device.type=='cuda')
                    y_batch = y_batch.to(device, non_blocking=device.type=='cuda')
                    prev_batch = prev_batch.to(device, non_blocking=device.type=='cuda')
                    
                    # 모델 예측 및 손실 계산
                    y_pred = model(X_batch, prev_batch)
                    loss = criterion(y_pred, y_batch, prev_batch)
                    
                    val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 스케줄러 업데이트
                scheduler.step(val_loss)
                
                # 모델 저장 (최적)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 조기 종료
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최적 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Model training completed with best validation loss: {best_val_loss:.4f}")
        
        # 학습 완료 후 GPU 상태 확인
        log_device_usage(device, "모델 학습 완료")
        
        # GPU 캐시 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🧹 GPU 캐시 정리 완료")
        
        # 모델, 스케일러, 파라미터 반환
        return model, scaler, target_col_idx
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def generate_predictions(df, current_date, predict_window=23, features=None, target_col='MOPJ', file_path=None):
    """
    개선된 예측 수행 함수 - 예측 시작일의 반월 기간 하이퍼파라미터 사용
    🔑 데이터 누출 방지: current_date 이후의 실제값은 사용하지 않음
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_device_usage(device, "모델 학습 시작")
        
        # 현재 날짜가 문자열이면 datetime으로 변환
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 현재 날짜 검증 (데이터 기준일)
        if current_date not in df.index:
            closest_date = df.index[df.index <= current_date][-1]
            logger.warning(f"Current date {current_date} not found in dataframe. Using closest date: {closest_date}")
            current_date = closest_date
        
        # 예측 시작일 계산
        prediction_start_date = current_date + pd.Timedelta(days=1)
        while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
            prediction_start_date += pd.Timedelta(days=1)
        
        # 반월 기간 계산
        data_semimonthly_period = get_semimonthly_period(current_date)
        prediction_semimonthly_period = get_semimonthly_period(prediction_start_date)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 다음 반월 계산
        next_semimonthly_period = get_next_semimonthly_period(prediction_start_date)
        
        logger.info(f"🎯 Prediction Setup:")
        logger.info(f"  📅 Data base date: {current_date} (period: {data_semimonthly_period})")
        logger.info(f"  🚀 Prediction start date: {prediction_start_date} (period: {prediction_semimonthly_period})")
        logger.info(f"  🎯 Purchase interval target period: {next_semimonthly_period}")
        
        # 23일치 예측을 위한 날짜 생성
        all_business_days = get_next_n_business_days(current_date, df, predict_window)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 구매 구간 계산
        semimonthly_business_days, purchase_target_period = get_next_semimonthly_dates(prediction_start_date, df)
        
        logger.info(f"  📊 Total predictions: {len(all_business_days)} days")
        logger.info(f"  🛒 Purchase target period: {purchase_target_period}")
        logger.info(f"  📈 Purchase interval business days: {len(semimonthly_business_days)}")
        
        if not all_business_days:
            raise ValueError(f"No future business days found after {current_date}")

        # ✅ 핵심 수정: LSTM 단기 예측을 위해 2022년 이후 데이터만 사용
        cutoff_date_2022 = pd.to_datetime('2022-01-01')
        available_data = df[df.index <= current_date].copy()
        
        # 2022년 이후 데이터가 충분한 경우 해당 기간만 사용 (단기 예측 정확도 향상)
        recent_data = available_data[available_data.index >= cutoff_date_2022]
        if len(recent_data) >= 50:
            historical_data = recent_data.copy()
            logger.info(f"  🎯 Using recent data for LSTM: 2022+ ({len(historical_data)} records)")
        else:
            historical_data = available_data.copy()
            logger.info(f"  📊 Using full available data: insufficient recent data ({len(recent_data)} < 50)")
        
        logger.info(f"  📊 Training data: {len(historical_data)} records up to {format_date(current_date)}")
        logger.info(f"  📊 Training data range: {format_date(historical_data.index.min())} ~ {format_date(historical_data.index.max())}")
        
        # 최소 데이터 요구사항 확인
        if len(historical_data) < 50:
            raise ValueError(f"Insufficient training data: {len(historical_data)} records (minimum 50 required)")
        
        if features is None:
            selected_features, _ = select_features_from_groups(
                historical_data, 
                variable_groups,
                target_col=target_col,
                vif_threshold=50.0,
                corr_threshold=0.8
            )
        else:
            selected_features = features
            
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        logger.info(f"  🔧 Selected features ({len(selected_features)}): {selected_features}")
        
        # ✅ 핵심 수정: 날짜별 다른 스케일링 보장
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(historical_data[selected_features])
        target_col_idx = selected_features.index(target_col)
        
        logger.info(f"  ⚖️  Scaler fitted on data up to {format_date(current_date)}")
        logger.info(f"  📊 Scaled data shape: {scaled_data.shape}")
        
        # ✅ 핵심: 예측 시작일의 반월 기간 하이퍼파라미터 사용
        optimized_params = optimize_hyperparameters_semimonthly_kfold(
            train_data=scaled_data,
            input_size=len(selected_features),
            target_col_idx=target_col_idx,
            device=device,
            current_period=prediction_semimonthly_period,  # ✅ 예측 시작일의 반월 기간
            file_path=file_path,  # 🔑 파일 경로 전달
            n_trials=30,
            k_folds=10,
            use_cache=True
        )
        
        logger.info(f"✅ Using hyperparameters for prediction start period: {prediction_semimonthly_period}")
        
        # ✅ 핵심 수정: 모델 학습 시 현재 날짜 기준으로 데이터 분할 보장
        logger.info(f"  🚀 Training model with data up to {format_date(current_date)}")
        model, model_scaler, model_target_col_idx = train_model(
            selected_features,
            target_col,
            current_date,
            historical_data,
            device,
            optimized_params
        )
        
        # 스케일러 일관성 확인
        if model_target_col_idx != target_col_idx:
            logger.warning(f"Target column index mismatch: {model_target_col_idx} vs {target_col_idx}")
            target_col_idx = model_target_col_idx
        
        logger.info(f"  ✅ Model trained successfully for prediction starting {format_date(prediction_start_date)}")
        
        # ✅ 핵심 수정: 예측 데이터 준비 시 날짜별 다른 시퀀스 보장 (데이터 누출 방지)
        seq_len = optimized_params['sequence_length']
        
        # 🔑 중요: current_date를 예측하려면 current_date 이전의 데이터만 사용
        available_dates_before_current = [d for d in df.index if d < current_date]
        
        if len(available_dates_before_current) < seq_len:
            logger.warning(f"⚠️  Insufficient historical data before {format_date(current_date)}: {len(available_dates_before_current)} < {seq_len}")
            # 사용 가능한 모든 이전 데이터 사용
            sequence_dates = available_dates_before_current
        else:
            # 마지막 seq_len개의 이전 날짜 사용
            sequence_dates = available_dates_before_current[-seq_len:]
        
        # 시퀀스 데이터 추출 (current_date 제외!)
        sequence = df.loc[sequence_dates][selected_features].values
        
        logger.info(f"  📊 Sequence data: {sequence.shape} from {format_date(sequence_dates[0])} to {format_date(sequence_dates[-1])}")
        logger.info(f"  🚫 Excluded current_date: {format_date(current_date)} (preventing data leakage)")
        
        # 모델에서 반환된 스케일러 사용 (일관성 보장)
        sequence = model_scaler.transform(sequence)
        prev_value = sequence[-1, target_col_idx]
        
        logger.info(f"  📈 Previous value (scaled): {prev_value:.4f}")
        logger.info(f"  📊 Sequence length used: {len(sequence)} (required: {seq_len})")
        
        # 예측 수행
        future_predictions = []  # 미래 예측 (실제값 없음)
        validation_data = []     # 검증 데이터 (실제값 있음)
        
        with torch.no_grad():
            # 23영업일 전체에 대해 예측 수행
            max_pred_days = min(predict_window, len(all_business_days))
            current_sequence = sequence.copy()
            
            # 텐서로 변환
            X = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([prev_value]).to(device)
            
            # 전체 시퀀스 예측
            pred = model(X, prev_tensor).cpu().numpy()[0]
            
            # ✅ 핵심 수정: 각 날짜별 예측 생성 (데이터 누출 방지)
            for j, pred_date in enumerate(all_business_days[:max_pred_days]):
                # ✅ 스케일 역변환 시 일관된 스케일러 사용
                dummy_matrix = np.zeros((1, len(selected_features)))
                dummy_matrix[0, target_col_idx] = pred[j]
                pred_value = model_scaler.inverse_transform(dummy_matrix)[0, target_col_idx]
                
                # 예측값 검증 및 정리
                if np.isnan(pred_value) or np.isinf(pred_value):
                    logger.warning(f"Invalid prediction value for {pred_date}: {pred_value}, skipping")
                    continue
                
                pred_value = float(pred_value)
                
                # ✅ 실제 데이터 마지막 날짜 확인
                last_data_date = df.index.max()
                actual_value = None
                
                # ✅ 실제값 존재 여부 확인 및 설정
                if (pred_date in df.index and 
                    pd.notna(df.loc[pred_date, target_col]) and 
                    pred_date <= last_data_date):
                    
                    actual_value = float(df.loc[pred_date, target_col])
                    
                    if np.isnan(actual_value) or np.isinf(actual_value):
                        actual_value = None
                
                # 기본 예측 정보 (실제값 포함)
                prediction_item = {
                    'date': format_date(pred_date, '%Y-%m-%d'),
                    'prediction': pred_value,
                    'actual': actual_value,  # 🔑 실제값 항상 포함
                    'prediction_from': format_date(current_date, '%Y-%m-%d'),
                    'day_offset': j + 1,
                    'is_business_day': pred_date.weekday() < 5 and not is_holiday(pred_date),
                    'is_synthetic': pred_date not in df.index,
                    'semimonthly_period': data_semimonthly_period,
                    'next_semimonthly_period': next_semimonthly_period
                }
                
                # ✅ 실제값이 있는 경우 검증 데이터에도 추가
                if actual_value is not None:
                    validation_item = {
                        **prediction_item,
                        'error': abs(pred_value - actual_value),
                        'error_pct': abs(pred_value - actual_value) / actual_value * 100 if actual_value != 0 else 0.0
                    }
                    validation_data.append(validation_item)
                    
                    # 📊 검증 타입 구분 로그
                    if pred_date <= current_date:
                        logger.debug(f"  ✅ Training validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                    else:
                        logger.debug(f"  🎯 Test validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                elif pred_date > last_data_date:
                    logger.debug(f"  🔮 Future: {format_date(pred_date)} - Pred: {pred_value:.2f} (no actual - beyond data)")
                
                future_predictions.append(prediction_item)
        
        # 📊 검증 데이터 통계
        training_validation = len([v for v in validation_data if pd.to_datetime(v['date']) <= current_date])
        test_validation = len([v for v in validation_data if pd.to_datetime(v['date']) > current_date])
        
        logger.info(f"📊 Prediction Results:")
        logger.info(f"  📈 Total predictions: {len(future_predictions)}")
        logger.info(f"  ✅ Training validation (≤ {format_date(current_date)}): {training_validation}")
        logger.info(f"  🎯 Test validation (> {format_date(current_date)}): {test_validation}")
        logger.info(f"  📋 Total validation points: {len(validation_data)}")
        logger.info(f"  🔮 Pure future predictions (> {format_date(df.index.max())}): {len(future_predictions) - len(validation_data)}")
        
        if len(validation_data) == 0:
            logger.info("  ℹ️  Pure future prediction - no validation data available")
        
        # ✅ 구간 평균 및 점수 계산 - 올바른 구매 대상 기간 사용
        temp_predictions_for_interval = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            if pred_date in semimonthly_business_days:  # 이제 올바른 다음 반월 날짜들
                temp_predictions_for_interval.append({
                    'Date': pred_date,
                    'Prediction': pred['prediction']
                })
        
        logger.info(f"  🛒 Predictions for interval calculation: {len(temp_predictions_for_interval)} (target period: {purchase_target_period})")
        
        interval_averages, interval_scores, analysis_info = calculate_interval_averages_and_scores(
            temp_predictions_for_interval, 
            semimonthly_business_days
        )

        # 최종 구매 구간 결정
        best_interval = decide_purchase_interval(interval_scores)

        # 성능 메트릭 계산 (검증 데이터가 있을 때만)
        metrics = None
        if validation_data:
            start_day_value = df.loc[current_date, target_col]
            if not (pd.isna(start_day_value) or np.isnan(start_day_value) or np.isinf(start_day_value)):
                try:
                    temp_df_for_metrics = pd.DataFrame([
                        {
                            'Date': pd.to_datetime(item['date']),
                            'Prediction': item['prediction'],
                            'Actual': item['actual']
                        } for item in validation_data
                    ])
                    
                    if not temp_df_for_metrics.empty:
                        metrics = compute_performance_metrics_improved(temp_df_for_metrics, start_day_value)
                        logger.info(f"  📊 Computed metrics from {len(validation_data)} validation points")
                    else:
                        logger.info("  ⚠️  No valid data for metrics computation")
                except Exception as e:
                    logger.error(f"Error computing metrics: {str(e)}")
                    metrics = None
            else:
                logger.warning("Invalid start_day_value for metrics computation")
        else:
            logger.info("  ℹ️  No validation data available - pure future prediction")
        
        # ✅ 이동평균 계산 시 실제값도 포함 (검증 데이터가 있는 경우)
        temp_predictions_for_ma = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_predictions_for_ma.append({
                'Date': pred_date,
                'Prediction': pred['prediction'],
                'Actual': actual_val
            })
        
        logger.info(f"  📈 Calculating moving averages with historical data up to {format_date(current_date)}")
        ma_results = calculate_moving_averages_with_history(
            temp_predictions_for_ma, 
            historical_data,  # 이미 current_date까지로 필터링됨
            target_col=target_col
        )
        
        # 특성 중요도 분석
        attention_data = None
        try:
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([float(prev_value)]).to(device)
            
            # 실제 시퀀스 날짜 정보 전달 (sequence_dates 변수 사용)
            actual_sequence_end_date = current_date  # current_date가 실제 데이터의 마지막 날짜
            attention_file, attention_img, feature_importance = visualize_attention_weights(
                model, sequence_tensor, prev_tensor, actual_sequence_end_date, selected_features, sequence_dates
            )
            
            attention_data = {
                'image': attention_img,
                'file_path': attention_file,
                'feature_importance': feature_importance
            }
        except Exception as e:
            logger.error(f"Error in attention analysis: {str(e)}")
        
        # 시각화 생성
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }

        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, target_col]):
            start_day_value = df.loc[current_date, target_col]

        # 📊 시각화용 데이터 준비 - 실제값 포함
        temp_df_for_plot_data = []
        for item in future_predictions:
            pred_date = pd.to_datetime(item['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_df_for_plot_data.append({
                'Date': pred_date,
                'Prediction': item['prediction'],
                'Actual': actual_val
            })
        
        temp_df_for_plot = pd.DataFrame(temp_df_for_plot_data)

        if metrics:
            f1_score = metrics['f1']
            accuracy = metrics['accuracy']
            mape = metrics['mape']
            weighted_score = metrics['weighted_score']
            visualization_type = "with validation data"
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            visualization_type = "future prediction only"

        if start_day_value is not None and not temp_df_for_plot.empty:
            try:
                basic_plot_file, basic_plot_img = plot_prediction_basic(
                    temp_df_for_plot,
                    prediction_start_date,
                    start_day_value,
                    f1_score,
                    accuracy,
                    mape,
                    weighted_score,
                    current_date=current_date,  # 🔑 데이터 컷오프 날짜 전달
                    save_prefix=None,  # 파일별 캐시 시스템 사용
                    title_prefix=f"Prediction Graph ({visualization_type})",
                    file_path=file_path
                )
                
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results,
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 시스템 사용
                    title_prefix=f"Moving Average Analysis ({visualization_type})",
                    file_path=file_path
                )
                
                plots['basic_plot'] = {'file': basic_plot_file, 'image': basic_plot_img}
                plots['ma_plot'] = {'file': ma_plot_file, 'image': ma_plot_img}
                
                logger.info(f"  📊 Visualizations created ({visualization_type})")
                
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("  ⚠️  No start day value or empty predictions - skipping visualizations")
        
        # 결과 반환 (급등락 모드 정보 포함)
        return {
            'predictions': future_predictions,
            'predictions_flat': future_predictions,  # 호환성을 위한 추가
            'validation_data': validation_data,
            'interval_scores': interval_scores,
            'interval_averages': interval_averages,
            'best_interval': best_interval,
            'ma_results': ma_results,
            'metrics': metrics,
            'selected_features': selected_features,
            'attention_data': attention_data,
            'plots': plots,
            'current_date': format_date(current_date, '%Y-%m-%d'),
            'data_end_date': format_date(current_date, '%Y-%m-%d'),
            'semimonthly_period': data_semimonthly_period,
            'next_semimonthly_period': purchase_target_period,  # ✅ 수정: 올바른 구매 대상 기간
            'prediction_semimonthly_period': prediction_semimonthly_period,
            'hyperparameter_period_used': prediction_semimonthly_period,
            'purchase_target_period': purchase_target_period,  # ✅ 추가
            'model_type': 'ImprovedLSTMPredictor',
            'loss_function': 'DirectionalLoss'
        }
        
    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def generate_predictions_compatible(df, current_date, predict_window=23, features=None, target_col='MOPJ'):
    """
    기존 프론트엔드와 호환되는 예측 함수
    (새로운 구조 + 기존 형태 변환)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        # 새로운 generate_predictions 함수 실행
        new_results = generate_predictions(df, current_date, predict_window, features, target_col)
        
        # 기존 형태로 변환
        if isinstance(new_results.get('predictions'), dict):
            # 새로운 구조인 경우
            future_predictions = new_results['predictions']['future']
            validation_data = new_results['predictions']['validation']
            
            # future와 validation을 합쳐서 기존 형태로 변환
            all_predictions = future_predictions + validation_data
        else:
            # 기존 구조인 경우
            all_predictions = new_results.get('predictions_flat', new_results.get('predictions', []))
        
        # 기존 필드명으로 변환
        compatible_predictions = convert_to_legacy_format(all_predictions)
        
        # 결과에 기존 형태 추가
        new_results['predictions'] = compatible_predictions  # 기존 호환성
        new_results['predictions_new'] = new_results.get('predictions')  # 새로운 구조도 유지
        
        logger.info(f"Generated {len(compatible_predictions)} compatible predictions")
        
        return new_results
        
    except Exception as e:
        logger.error(f"Error in compatible prediction generation: {str(e)}")
        raise e

def generate_predictions_with_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 스마트 캐시 저장이 포함된 함수 (수정됨)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        logger.info(f"Starting prediction with smart cache save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # 스마트 캐시 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with smart cache system...")
            
            # 새로운 스마트 캐시 저장 함수 사용
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"✅ Smart cache save completed successfully")
                logger.info(f"  - Prediction Start Date: {save_result.get('prediction_start_date')}")
                logger.info(f"  - File: {save_result.get('file', 'N/A')}")
                
                # 캐시 정보 추가 (안전한 키 접근)
                results['cache_info'] = {
                    'saved': True,
                    'prediction_start_date': save_result.get('prediction_start_date'),
                    'file': save_result.get('file'),
                    'success': save_result.get('success', False)
                }
            else:
                logger.warning(f"❌ Failed to save prediction with smart cache: {save_result.get('error')}")
                results['cache_info'] = {
                    'saved': False,
                    'error': save_result.get('error')
                }
        else:
            logger.info("Skipping smart cache save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
            results['cache_info'] = {
                'saved': False,
                'reason': 'save_to_csv=False'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            results['cache_info'] = {'saved': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

def generate_predictions_with_attention_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 attention 포함 CSV 저장 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        전체 데이터
    current_date : str or datetime
        현재 날짜 (데이터 기준일)
    predict_window : int
        예측 기간 (기본 23일)
    features : list, optional
        사용할 특성 목록
    target_col : str
        타겟 컬럼명 (기본 'MOPJ')
    save_to_csv : bool
        CSV 저장 여부 (기본 True)
    
    Returns:
    --------
    dict : 예측 결과 (attention 데이터 포함)
    """
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        logger.info(f"Starting prediction with attention save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # attention 포함 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with attention data...")
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"SUCCESS: Prediction with attention saved successfully")
                logger.info(f"  - CSV: {save_result['csv_file']}")
                logger.info(f"  - Metadata: {save_result['meta_file']}")
                logger.info(f"  - Attention: {save_result['attention_file'] if save_result.get('attention_file') else 'Not saved'}")
            else:
                logger.warning(f"❌ Failed to save prediction with attention: {save_result.get('error')}")
        else:
            logger.info("Skipping CSV save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_attention_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

#######################################################################
# 백그라운드 작업 처리
#######################################################################
# 🔧 SyntaxError 수정 - check_existing_prediction 함수 (3987라인 근처)

def check_existing_prediction(current_date, file_path=None):
    """
    파일별 디렉토리 구조에서 저장된 예측을 확인하고 불러오는 함수
    🎯 현재 파일의 디렉토리에서 우선 검색
    """
    try:
        # 현재 날짜(데이터 기준일)에서 첫 번째 예측 날짜 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 다음 영업일 찾기 (현재 날짜의 다음 영업일이 첫 번째 예측 날짜)
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5 or is_holiday(next_date):
            next_date += pd.Timedelta(days=1)
        
        first_prediction_date = next_date
        date_str = first_prediction_date.strftime('%Y%m%d')
        
        # 반월 정보 계산 (캐시 정확성을 위해)
        current_semimonthly = get_semimonthly_period(first_prediction_date)
        
        logger.info(f"🔍 Checking cache for prediction starting: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Data end date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Expected prediction start: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Current semimonthly period: {current_semimonthly}")
        logger.info(f"  📄 Expected filename pattern: prediction_start_{date_str}.*")
        
        # 🎯 1단계: 현재 파일의 캐시 디렉토리에서 정확한 날짜 매치로 캐시 찾기
        try:
            # 🔧 수정: 파일 경로를 명시적으로 전달
            cache_dirs = get_file_cache_dirs(file_path)
            file_predictions_dir = cache_dirs['predictions']
            
            logger.info(f"  📁 Cache directory: {cache_dirs['root']}")
            logger.info(f"  📁 Predictions directory: {file_predictions_dir}")
            logger.info(f"  📁 Directory exists: {file_predictions_dir.exists()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache directories: {str(e)}")
            return None
        
        if file_predictions_dir.exists():
            exact_csv = file_predictions_dir / f"prediction_start_{date_str}.csv"
            exact_meta = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
            
            logger.info(f"  🔍 Looking for: {exact_csv}")
            logger.info(f"  🔍 CSV exists: {exact_csv.exists()}")
            logger.info(f"  🔍 Meta exists: {exact_meta.exists()}")
            
            if exact_csv.exists() and exact_meta.exists():
                logger.info(f"✅ Found exact prediction cache in file directory: {exact_csv.name}")
                return load_prediction_with_attention_from_csv_in_dir(first_prediction_date, file_predictions_dir)
            
            # 해당 파일 디렉토리에서 다른 날짜의 예측 찾기
            logger.info("🔍 Searching for other predictions in file directory...")
            prediction_files = list(file_predictions_dir.glob("prediction_start_*_meta.json"))
            
            logger.info(f"  📋 Found {len(prediction_files)} prediction files:")
            for i, pf in enumerate(prediction_files):
                logger.info(f"    {i+1}. {pf.name}")
            
            if prediction_files:
                # 반월 기간 매칭하는 캐시 찾기
                compatible_cache = None
                
                for meta_file in prediction_files:
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        
                        cached_data_end_date = meta_data.get('data_end_date')
                        if cached_data_end_date:
                            cached_data_end_date = pd.to_datetime(cached_data_end_date)
                            cached_semimonthly = get_semimonthly_period(cached_data_end_date)
                            
                            logger.info(f"    🔍 Checking file cache: {meta_file.name}")
                            logger.info(f"      📅 Current semimonthly: {current_semimonthly}")
                            logger.info(f"      📅 Cached semimonthly:  {cached_semimonthly}")
                            
                            if cached_semimonthly == current_semimonthly:
                                cached_date_str = meta_file.stem.replace('prediction_start_', '').replace('_meta', '')
                                cached_prediction_date = pd.to_datetime(cached_date_str, format='%Y%m%d')
                                
                                logger.info(f"🎯 Found compatible prediction in file directory!")
                                logger.info(f"  📅 Cached prediction date: {cached_prediction_date.strftime('%Y-%m-%d')}")
                                logger.info(f"  📅 Semimonthly period match: {current_semimonthly}")
                                logger.info(f"  📄 Using file: {meta_file.name}")
                                
                                return load_prediction_with_attention_from_csv_in_dir(cached_prediction_date, file_predictions_dir)
                            else:
                                logger.info(f"    ❌ Semimonthly period mismatch - skipping")
                                
                    except Exception as e:
                        logger.debug(f"    ⚠️ Error reading meta file {meta_file}: {str(e)}")
                        continue
                
                logger.info("❌ No compatible cache found in file directory (semimonthly mismatch)")
        else:
            logger.warning(f"❌ Predictions directory does not exist: {file_predictions_dir}")
        
        # 🎯 2단계: 다른 파일들의 캐시에서 호환 가능한 예측 찾기
        current_file_path = file_path or prediction_state.get('current_file', None)
        if current_file_path:
            # 🔧 수정: 모든 기존 파일의 캐시 디렉토리 탐색
            upload_dir = Path(UPLOAD_FOLDER)
            existing_files = [f for f in upload_dir.glob('*.xlsx') if str(f) != current_file_path]
            
            logger.info(f"🔍 [PREDICTION_CACHE] 다른 파일들의 캐시 탐색: {len(existing_files)}개 파일")
            
            for existing_file in existing_files:
                try:
                    # 기존 파일의 캐시 디렉토리 확인
                    existing_cache_dirs = get_file_cache_dirs(str(existing_file))
                    existing_predictions_dir = existing_cache_dirs['predictions']
                    
                    if existing_predictions_dir.exists():
                        # 동일한 반월 기간의 예측 파일 찾기
                        pattern = f"prediction_start_*_meta.json"
                        meta_files = list(existing_predictions_dir.glob(pattern))
                        
                        logger.info(f"    📁 {existing_file.name}: {len(meta_files)}개 예측 파일")
                        
                        for meta_file in meta_files:
                            try:
                                with open(meta_file, 'r', encoding='utf-8') as f:
                                    meta_data = json.load(f)
                                
                                # 예측 시작일로부터 반월 기간 추출
                                cached_date_str = meta_file.stem.replace('prediction_start_', '').replace('_meta', '')
                                cached_prediction_date = pd.to_datetime(cached_date_str, format='%Y%m%d')
                                cached_semimonthly = get_semimonthly_period(cached_prediction_date)
                                
                                if cached_semimonthly == current_semimonthly:
                                    logger.info(f"    🎯 호환 가능한 예측 발견! {existing_file.name} -> {cached_prediction_date.strftime('%Y-%m-%d')}")
                                    logger.info(f"    📅 반월 기간 일치: {current_semimonthly}")
                                    
                                    return load_prediction_with_attention_from_csv_in_dir(cached_prediction_date, existing_predictions_dir)
                                    
                            except Exception as e:
                                logger.debug(f"    ⚠️ 메타 파일 읽기 실패 {meta_file}: {str(e)}")
                                continue
                except Exception as e:
                    logger.debug(f"    ⚠️ 캐시 디렉토리 접근 실패 {existing_file.name}: {str(e)}")
                    continue
                    
            logger.info("❌ 다른 파일들의 캐시에서도 호환 가능한 예측을 찾지 못했습니다")
            
        logger.info("❌ No compatible prediction cache found")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error checking existing prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_visualizations_realtime(predictions, df, current_date, metadata):
    """실시간으로 시각화 생성 (저장하지 않음)"""
    try:
        # DataFrame으로 변환
        sequence_df = pd.DataFrame(predictions)
        if 'Date' in sequence_df.columns:
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # 시작값 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        start_day_value = df.loc[current_date, 'MOPJ'] if current_date in df.index else None
        
        if start_day_value is not None:
            # 성능 메트릭 계산
            metrics = compute_performance_metrics_improved(sequence_df, start_day_value)
            
            # 기본 그래프 생성 (메모리에만)
            _, basic_plot_img = plot_prediction_basic(
                sequence_df, 
                metadata.get('prediction_start_date', current_date),
                start_day_value,
                metrics['f1'],
                metrics['accuracy'], 
                metrics['mape'],
                metrics['weighted_score'],
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
                )
                
            # 이동평균 계산 및 시각화
            historical_data = df[df.index <= current_date].copy()
            ma_results = calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ')
            
            _, ma_plot_img = plot_moving_average_analysis(
                ma_results,
                metadata.get('prediction_start_date', current_date),
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
            )
            
            # 상태에 저장
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': basic_plot_img},
                'ma_plot': {'file': None, 'image': ma_plot_img}
            }
            prediction_state['latest_ma_results'] = ma_results
            prediction_state['latest_metrics'] = metrics
            
        else:
            logger.warning("Cannot generate visualizations: start day value not available")
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': None},
                'ma_plot': {'file': None, 'image': None}
            }
            prediction_state['latest_ma_results'] = {}
            prediction_state['latest_metrics'] = {}
            
    except Exception as e:
        logger.error(f"Error generating realtime visualizations: {str(e)}")
        prediction_state['latest_plots'] = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        prediction_state['latest_ma_results'] = {}
        prediction_state['latest_metrics'] = {}

def regenerate_visualizations_from_cache(predictions, df, current_date, metadata):
    """
    캐시된 데이터로부터 시각화를 재생성하는 함수
    🔑 current_date를 전달하여 과거/미래 구분 시각화 생성
    """
    try:
        logger.info("🎨 Regenerating visualizations from cached data...")
        
        # DataFrame으로 변환 (안전한 방식)
        temp_df_for_plot = pd.DataFrame([
            {
                'Date': pd.to_datetime(item.get('Date') or item.get('date')),
                'Prediction': safe_serialize_value(item.get('Prediction') or item.get('prediction')),
                'Actual': safe_serialize_value(item.get('Actual') or item.get('actual'))
            } for item in predictions if item.get('Date') or item.get('date')
        ])
        
        logger.info(f"  📊 Plot data prepared: {len(temp_df_for_plot)} predictions")
        
        # current_date 처리
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 시작값 계산
        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, 'MOPJ']):
            start_day_value = df.loc[current_date, 'MOPJ']
            logger.info(f"  📈 Start day value: {start_day_value:.2f}")
        else:
            logger.warning(f"  ⚠️  Start day value not available for {current_date}")
        
        # 메타데이터에서 메트릭 가져오기 (안전한 방식)
        metrics = metadata.get('metrics')
        if metrics:
            f1_score = safe_serialize_value(metrics.get('f1', 0.0))
            accuracy = safe_serialize_value(metrics.get('accuracy', 0.0))
            mape = safe_serialize_value(metrics.get('mape', 0.0))
            weighted_score = safe_serialize_value(metrics.get('weighted_score', 0.0))
            logger.info(f"  📊 Metrics loaded - F1: {f1_score:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%")
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            logger.info("  ℹ️  No metrics available - using default values")
        
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        
        # 시각화 생성 (데이터가 충분한 경우만)
        if start_day_value is not None and not temp_df_for_plot.empty:
            logger.info("  🎨 Generating basic prediction plot...")
            
            # 예측 시작일 계산
            prediction_start_date = metadata.get('prediction_start_date')
            if isinstance(prediction_start_date, str):
                prediction_start_date = pd.to_datetime(prediction_start_date)
            elif prediction_start_date is None:
                # 메타데이터에 없으면 current_date 다음 영업일로 계산
                prediction_start_date = current_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)
                logger.info(f"  📅 Calculated prediction start date: {prediction_start_date}")
            
            # ✅ 핵심 수정: current_date 전달하여 과거/미래 구분 시각화
            basic_plot_file, basic_plot_img = plot_prediction_basic(
                temp_df_for_plot,
                prediction_start_date,
                start_day_value,
                f1_score,
                accuracy,
                mape,
                weighted_score,
                current_date=current_date,  # 🔑 핵심 수정: current_date 전달
                save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                title_prefix="Cached Prediction Analysis"
            )
            
            if basic_plot_file:
                logger.info(f"  ✅ Basic plot generated: {basic_plot_file}")
            else:
                logger.warning("  ❌ Basic plot generation failed")
            
            # 이동평균 계산 및 시각화
            logger.info("  📈 Calculating moving averages...")
            historical_data = df[df.index <= current_date].copy()
            
            # 캐시된 예측 데이터를 이동평균 계산용으로 변환
            ma_input_data = []
            for pred in predictions:
                try:
                    ma_item = {
                        'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                        'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                        'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                    }
                    ma_input_data.append(ma_item)
                except Exception as e:
                    logger.warning(f"  ⚠️  Error processing MA data item: {str(e)}")
                    continue
            
            ma_results = calculate_moving_averages_with_history(
                ma_input_data, historical_data, target_col='MOPJ'
            )
            
            if ma_results:
                logger.info(f"  📊 MA calculated for {len(ma_results)} windows")
                
                # 이동평균 시각화
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results,
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                    title_prefix="Cached Moving Average Analysis"
                )
                
                if ma_plot_file:
                    logger.info(f"  ✅ MA plot generated: {ma_plot_file}")
                else:
                    logger.warning("  ❌ MA plot generation failed")
            else:
                logger.warning("  ⚠️  Moving average calculation failed")
                ma_plot_file, ma_plot_img = None, None
            
            plots = {
                'basic_plot': {'file': basic_plot_file, 'image': basic_plot_img},
                'ma_plot': {'file': ma_plot_file, 'image': ma_plot_img}
            }
            
            logger.info("  ✅ Visualizations regenerated from cache successfully")
        else:
            if start_day_value is None:
                logger.warning("  ❌ Cannot regenerate visualizations: start day value not available")
            if temp_df_for_plot.empty:
                logger.warning("  ❌ Cannot regenerate visualizations: no prediction data")
        
        return plots
        
    except Exception as e:
        logger.error(f"❌ Error regenerating visualizations from cache: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }

#######################################################################
# VARMAX 관련 클래스 및 함수
#######################################################################

class VARMAXSemiMonthlyForecaster:
    """VARMAX 기반 반월별 시계열 예측 클래스 - 세 번째 탭용"""
    
    def __init__(self, file_path, result_var='MOPJ', pred_days=50):
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        
        self.file_path = file_path
        self.result_var = result_var
        self.pred_days = pred_days
        self.df1 = None
        self.df_origin = None
        self.target_df = None
        self.df_train = None
        self.df_test = None
        self.ts_exchange = None
        self.exogenous_data = None
        self.varx_result = None
        self.pred_df = None
        self.final_forecast_var = None
        self.final_value = None
        self.final_index = None
        self.filtered_vars = None
        self.var_num = None  # 기본값
        self.r2_train = None
        self.r2_test = None
        self.pred_index = None
        self.selected_vars = []
        self.mape_value = None

    def load_data(self):
        """데이터 로드 (VARMAX 모델용 - 모든 데이터 사용, 최근 800개로 제한)"""
        try:
            # VARMAX 모델은 장기예측이므로 모든 데이터 사용 (2022년 이전 포함)
            df_full = load_data(self.file_path, model_type='varmax')
            # 기존 로직 유지: 최근 800개 데이터만 사용
            self.df_origin = df_full.iloc[-800:]
            logger.info(f"VARMAX data loaded: {self.df_origin.shape} (last 800 records from full dataset)")
            logger.info(f"Date range: {self.df_origin.index.min()} to {self.df_origin.index.max()}")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise e

    def select_variables(self, current_date=None):
        """변수 선택 - 현재 날짜까지의 데이터만 사용하여 데이터 누출 방지"""
        try:
            # 🔑 수정: 현재 날짜까지의 데이터만 사용
            if current_date is not None:
                if isinstance(current_date, str):
                    current_date = pd.to_datetime(current_date)
                recent_data = self.df_origin[self.df_origin.index <= current_date]
                logger.info(f"🔧 Variable selection using data up to {current_date.strftime('%Y-%m-%d')} ({len(recent_data)} records)")
            else:
                recent_data = self.df_origin
                logger.info(f"🔧 Variable selection using all available data ({len(recent_data)} records)")
            
            correlations = recent_data.corr()[self.result_var]
            correlations = correlations.drop(self.result_var)
            correlations = correlations.sort_values(ascending=False)
            select = correlations.index.tolist()
            self.selected_vars = select
            
            # 변수 그룹 정의 (원본 코드와 동일)
            variable_groups = {
                'crude_oil': ['WTI', 'Brent', 'Dubai'],
                'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
                'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
                'lpg': ['C3_LPG', 'C4_LPG'],
                'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
                'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
                'spread': ['Monthly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
                'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
                'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
                'economics': ['Dow_Jones', 'Euro', 'Gold'],
                'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
                'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
                'Flat Rate_Chiba']
            }
            
            # 그룹별 최적 변수 선택
            self.filtered_vars = []
            for group, variables in variable_groups.items():
                filtered_group_vars = [var for var in variables if var in self.selected_vars]
                if filtered_group_vars:
                    best_var = max(filtered_group_vars, key=lambda x: abs(correlations[x]))
                    self.filtered_vars.append(best_var)
            
            self.selected_vars = sorted(self.filtered_vars, key=lambda x: abs(correlations[x]), reverse=True)
            logger.info(f"Selected {len(self.selected_vars)} variables for VARMAX prediction")
            logger.info(f"Top 5 selected variables: {self.selected_vars[:5]}")
            
        except Exception as e:
            logger.error(f"Variable selection failed: {str(e)}")
            raise e

    def prepare_data_for_prediction(self, current_date):
        """예측용 데이터 준비"""
        try:
            # 현재 날짜까지의 데이터만 사용
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # 현재 날짜까지의 데이터 필터링
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            
            filtered_values = self.selected_vars
            input_columns = filtered_values[:self.var_num]
            output_column = [self.result_var]
            
            self.final_value = historical_data.iloc[-1][self.result_var]
            self.final_index = historical_data.index[-1]

            self.target_df = historical_data[input_columns + output_column]
            
            self.df_train = self.target_df
            
            # 외생변수 (환율) 설정
            if 'Exchange' in self.df_origin.columns:
                self.ts_exchange = historical_data['Exchange']
                self.exogenous_data = pd.DataFrame(self.ts_exchange, index=self.ts_exchange.index)
            else:
                self.exogenous_data = None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise e

    def fit_varmax_model(self):
        """VARMAX 모델 학습"""
        try:
            if not VARMAX_AVAILABLE:
                raise ImportError("VARMAX dependencies not available")
                
            logger.info("🔄 [VARMAX_FIT] Starting VARMAX model fitting...")
            logger.info(f"🔄 [VARMAX_FIT] Training data shape: {self.df_train.shape}")
            logger.info(f"🔄 [VARMAX_FIT] Exogenous data available: {self.exogenous_data is not None}")
            
            best_p = 7
            best_q = 0
            
            logger.info(f"🔄 [VARMAX_FIT] Creating VARMAX model with order=({best_p}, {best_q})")
            varx_model = VARMAX(endog=self.df_train, exog=self.exogenous_data, order=(best_p, best_q))
            
            logger.info("🔄 [VARMAX_FIT] Starting model fitting (this may take a while)...")
            
            # 🔑 global prediction_state에 접근하여 진행률 업데이트
            global prediction_state
            prediction_state['varmax_prediction_progress'] = 50
            
            self.varx_result = varx_model.fit(disp=False, maxiter=1000)
            
            if hasattr(self.varx_result, 'converged') and not self.varx_result.converged:
                logger.warning("⚠️ [VARMAX_FIT] VARMAX model did not converge (res.converged=False)")
            else:
                logger.info("✅ [VARMAX_FIT] VARMAX model converged successfully")
                
            logger.info("✅ [VARMAX_FIT] VARMAX model fitted successfully")
            
            # 🔑 진행률 업데이트
            prediction_state['varmax_prediction_progress'] = 60
            
        except Exception as e:
            logger.error(f"❌ [VARMAX_FIT] VARMAX fitting failed: {str(e)}")
            logger.error(f"❌ [VARMAX_FIT] Fitting error traceback: {traceback.format_exc()}")
            
            # 🔑 에러 상태 업데이트
            prediction_state['varmax_error'] = f"Model fitting failed: {str(e)}"
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            
            raise e

    def forecast_varmax(self):
        """VARMAX 예측 수행"""
        try:
            # 미래 외생변수 준비
            if self.exogenous_data is not None:
                # 마지막 값을 예측 기간만큼 반복
                last_exog_value = self.ts_exchange.iloc[-1]
                future_dates = pd.bdate_range(start=self.final_index + pd.Timedelta(days=1), periods=self.pred_days)
                exog_future = pd.DataFrame([last_exog_value] * self.pred_days, 
                                         index=future_dates, 
                                         columns=self.exogenous_data.columns)
            else:
                exog_future = None
                future_dates = pd.bdate_range(start=self.final_index + pd.Timedelta(days=1), periods=self.pred_days)
                
            # VARMAX 예측
            varx_forecast = self.varx_result.forecast(steps=self.pred_days, exog=exog_future)
            self.pred_index = future_dates
            self.pred_df = pd.DataFrame(varx_forecast.values, index=self.pred_index, columns=self.df_train.columns)
            logger.info(f"VARMAX forecast completed for {self.pred_days} days")
            
        except Exception as e:
            logger.error(f"VARMAX forecasting failed: {str(e)}")
            raise e

    def residual_correction(self):
        """랜덤포레스트를 이용한 잔차 보정"""
        try:
            if not VARMAX_AVAILABLE:
                logger.warning("VARMAX not available, skipping residual correction")
                self.final_forecast_var = self.pred_df[[self.result_var]]
                self.r2_train = 0.0
                self.r2_test = 0.0
                return
                
            # 잔차 계산
            residuals_origin = self.df_train - self.varx_result.fittedvalues
            residuals_real = residuals_origin.iloc[1:]
            X = residuals_real.iloc[:, :-1]
            y = residuals_real.iloc[:, -1]
            
            # 테스트 크기 계산
            test_size_value = min(0.3, (self.pred_days + 1) / len(self.target_df))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_value, shuffle=False)
            
            # 랜덤포레스트 모델 학습
            rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr_model.fit(X_train, y_train)
            
            # 성능 평가
            y_train_pred = rfr_model.predict(X_train)
            y_test_pred = rfr_model.predict(X_test)
            self.r2_train = r2_score(y_train, y_train_pred)
            self.r2_test = r2_score(y_test, y_test_pred)
            
            # 예측에 잔차 보정 적용
            var_predictions = self.pred_df[[self.result_var]]
            
            # 최근 잔차 데이터로 예측
            recent_residuals = residuals_real.iloc[-self.pred_days:, :-1]
            if len(recent_residuals) < self.pred_days:
                # 데이터가 부족하면 마지막 행을 반복
                last_residual = residuals_real.iloc[-1:, :-1]
                additional_rows = self.pred_days - len(recent_residuals)
                repeated_residuals = pd.concat([last_residual] * additional_rows, ignore_index=True)
                recent_residuals = pd.concat([recent_residuals, repeated_residuals])[:self.pred_days]
            
            rfr_predictions = rfr_model.predict(recent_residuals.iloc[:len(var_predictions)])
            rfr_pred_df = pd.DataFrame(rfr_predictions, 
                                     index=var_predictions.index, 
                                     columns=var_predictions.columns)
            
            # 최종 예측값 = VARMAX 예측 + 잔차 보정
            self.final_forecast_var = var_predictions.add(rfr_pred_df)
            
            logger.info(f"Residual correction completed. Train R2: {self.r2_train:.4f}, Test R2: {self.r2_test:.4f}")
            
        except Exception as e:
            logger.error(f"Residual correction failed: {str(e)}")
            # 보정 실패 시 원본 VARMAX 예측값 사용
            self.final_forecast_var = self.pred_df[[self.result_var]]
            self.r2_train = 0.0
            self.r2_test = 0.0

    def calculate_performance_metrics(self, actual_data=None):
        """성능 지표 계산 (실제 데이터가 있는 경우)"""
        if actual_data is None:
            return {
                'f1': 0.0,
                'accuracy': 0.0,
                'mape': 0.0,
                'weighted_score': 0.0,
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0
            }
        
        try:
            # 방향성 예측 성능
            pred_series = self.final_forecast_var[self.result_var]
            actual_series = actual_data
            
            pred_trend = (pred_series.diff() > 0).astype(int)[1:]
            actual_trend = (actual_series.diff() > 0).astype(int)[1:]
            
            # 공통 인덱스로 맞춤
            common_idx = pred_trend.index.intersection(actual_trend.index)
            if len(common_idx) > 0:
                pred_trend_common = pred_trend.loc[common_idx]
                actual_trend_common = actual_trend.loc[common_idx]
                
                if VARMAX_AVAILABLE:
                    precision = precision_score(actual_trend_common, pred_trend_common, zero_division=0)
                    recall = recall_score(actual_trend_common, pred_trend_common, zero_division=0)
                    f1 = f1_score(actual_trend_common, pred_trend_common, zero_division=0)
                    accuracy = (actual_trend_common == pred_trend_common).mean() * 100
                else:
                    precision = recall = f1 = accuracy = 0.0
            else:
                precision = recall = f1 = accuracy = 0.0
            
            # MAPE 계산
            common_values_pred = pred_series.loc[common_idx] if len(common_idx) > 0 else pred_series
            common_values_actual = actual_series.loc[common_idx] if len(common_idx) > 0 else actual_series
            
            mask = common_values_actual != 0
            if mask.any():
                mape = np.mean(np.abs((common_values_actual[mask] - common_values_pred[mask]) / common_values_actual[mask])) * 100
            else:
                mape = 0.0
            
            return {
                'f1': f1,
                'accuracy': accuracy,
                'mape': mape,
                'weighted_score': f1 * 100,  # F1 점수를 가중 점수로 사용
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0,
                'precision': precision,
                'recall': recall
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {str(e)}")
            return {
                'f1': 0.0,
                'accuracy': 0.0,
                'mape': 0.0,
                'weighted_score': 0.0,
                'r2_train': self.r2_train or 0.0,
                'r2_test': self.r2_test or 0.0
            }

    def calculate_moving_averages(self, predictions, current_date, windows=[5, 10, 23]):
        """이동평균 계산 (기존 app.py 방식과 동일)"""
        try:
            results = {}
            
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 과거 데이터 추가 (이동평균 계산용)
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            historical_series = historical_data[self.result_var].tail(30)  # 최근 30일
            
            # 예측 시리즈 생성
            prediction_series = pd.Series(
                data=pred_df['Prediction'].values,
                index=pred_df['Date']
            )
            
            # 과거와 예측 데이터 결합
            combined_series = pd.concat([historical_series, prediction_series])
            combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
            combined_series = combined_series.sort_index()
            
            # 각 윈도우별 이동평균 계산
            for window in windows:
                rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
                
                window_results = []
                for i, row in pred_df.iterrows():
                    date = row['Date']
                    pred_value = row['Prediction']
                    actual_value = row['Actual']
                    
                    # 해당 날짜의 이동평균
                    ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                    
                    window_results.append({
                        'date': date,
                        'prediction': pred_value,
                        'actual': actual_value,
                        'ma': ma_value
                    })
                
                results[f'ma{window}'] = window_results
            
            return results
            
        except Exception as e:
            logger.error(f"Moving average calculation failed: {str(e)}")
            return {}

    def calculate_moving_averages_varmax(self, predictions, current_date, windows=[5, 10, 20, 30]):
        try:
            results = {}
            
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 과거 데이터 추가 (이동평균 계산용)
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            historical_series = historical_data[self.result_var].tail(30)  # 최근 30일
            
            # 예측 시리즈 생성
            prediction_series = pd.Series(
                data=pred_df['Prediction'].values,
                index=pred_df['Date']
            )
            
            # 과거와 예측 데이터 결합
            combined_series = pd.concat([historical_series, prediction_series])
            combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
            combined_series = combined_series.sort_index()
            
            # 각 윈도우별 이동평균 계산
            for window in windows:
                rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
                
                window_results = []
                for i, row in pred_df.iterrows():
                    date = row['Date']
                    pred_value = row['Prediction']
                    actual_value = row['Actual']
                    
                    # 해당 날짜의 이동평균
                    ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                    
                    window_results.append({
                        'date': date,
                        'prediction': pred_value,
                        'actual': actual_value,
                        'ma': ma_value
                    })
                
                results[f'ma{window}'] = window_results
            
            return results
            
        except Exception as e:
            logger.error(f"Moving average calculation failed: {str(e)}")
            return {}

    def calculate_half_month_averages(self, predictions, current_date):
        """VarmaxResult 컴포넌트용 반월 평균 데이터 계산"""
        try:
            # 예측 데이터를 DataFrame으로 변환
            pred_df = pd.DataFrame(predictions)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # 반월 기간별로 그룹화
            half_month_groups = {}
            
            for _, row in pred_df.iterrows():
                date = row['Date']
                
                # 반월 라벨 생성 (예: 25_05_1 = 2025년 5월 상반기)
                year = date.year % 100  # 연도 마지막 두 자리
                month = date.month
                half = 1 if date.day <= 15 else 2
                
                half_month_label = f"{year:02d}_{month:02d}_{half}"
                
                if half_month_label not in half_month_groups:
                    half_month_groups[half_month_label] = []
                
                half_month_groups[half_month_label].append(row['Prediction'])
            
            # 각 반월 기간의 평균 계산
            half_month_data = []
            for label, values in half_month_groups.items():
                avg_value = np.mean(values)
                half_month_data.append({
                    'half_month_label': label,
                    'half_month_avg': float(avg_value),
                    'count': len(values)
                })
            
            # 라벨순으로 정렬
            half_month_data.sort(key=lambda x: x['half_month_label'])
            
            logger.info(f"반월 평균 데이터 계산 완료: {len(half_month_data)}개 기간")
            
            return half_month_data
            
        except Exception as e:
            logger.error(f"Half month averages calculation failed: {str(e)}")
            return []

    def prepare_variable_for_prediction(self, current_date):
        """예측용 데이터 준비"""
        try:
            # 현재 날짜까지의 데이터만 사용
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # 현재 날짜까지의 데이터 필터링
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            
            filtered_values = self.selected_vars
            input_columns = filtered_values[:self.var_num]
            output_column = [self.result_var]
            
            self.final_value = historical_data.iloc[-1-self.pred_days][self.result_var]
            self.final_index = historical_data.index[-1-self.pred_days]

            self.target_df = historical_data[input_columns + output_column]
            
            self.df_train = self.target_df[:-self.pred_days]
            
            # 외생변수 (환율) 설정
            if 'Exchange' in self.df_origin.columns:
                self.ts_exchange = historical_data['Exchange']
                self.ts_exchange = self.ts_exchange[:-self.pred_days]
                self.exogenous_data = pd.DataFrame(self.ts_exchange, index=self.ts_exchange.index)
            else:
                self.exogenous_data = None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise e

    def calculate_mape(self, predicted, actual):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return mape 

    def generate_variables_varmax(self, current_date, var_num):
        """변수 수 예측 프로세스 실행"""
        try:
            self.var_num = var_num
            self.load_data()
            self.select_variables(current_date)
            self.prepare_variable_for_prediction(current_date)
            self.fit_varmax_model()
            logger.info("VARMAX 변수 선정 모델 학습 완료")

            self.forecast_varmax()
            logger.info("VARMAX 변수 선정 모델 예측 완료")

            self.residual_correction()
            logger.info(f"잔차 보정 완료 (R2 train={self.r2_train:.3f}, test={self.r2_test:.3f})")
            historical_data = self.df_origin[self.df_origin.index <= current_date]
            test_data = historical_data[-self.pred_days:]
            self.final_forecast_var.index = test_data.index
            self.mape_value = self.calculate_mape(self.final_forecast_var[self.result_var], test_data[self.result_var])

        except Exception as e:
            logger.error(f"VARMAX variables generation failed: {str(e)}")
            raise e

    def generate_predictions_varmax(self, current_date, var_num):
        """VARMAX 예측 수행"""
        try:
            global prediction_state
            
            logger.info(f"🔄 [VARMAX_GEN] Starting VARMAX prediction generation")
            logger.info(f"🔄 [VARMAX_GEN] Parameters: current_date={current_date}, var_num={var_num}")
            
            self.var_num = var_num
            logger.info(f"🔄 [VARMAX_GEN] Step 1: Loading data...")
            prediction_state['varmax_prediction_progress'] = 35
            self.load_data()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 2: Selecting variables...")
            prediction_state['varmax_prediction_progress'] = 40
            self.select_variables(current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 3: Preparing data for prediction...")
            prediction_state['varmax_prediction_progress'] = 45
            self.prepare_data_for_prediction(current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 4: Fitting VARMAX model...")
            # fit_varmax_model 내에서 50→60으로 업데이트됨
            self.fit_varmax_model()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 5: Forecasting...")
            prediction_state['varmax_prediction_progress'] = 65
            self.forecast_varmax()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 6: Residual correction...")
            prediction_state['varmax_prediction_progress'] = 70
            self.residual_correction()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 7: Converting results to standard format...")
            prediction_state['varmax_prediction_progress'] = 75
            # 예측 결과를 표준 형식으로 변환
            predictions = []
            for date, value in self.final_forecast_var.iterrows():
                predictions.append({
                    'Date': format_date(date),
                    'Prediction': float(value[self.result_var]),
                    'Actual': None  # 실제값은 미래이므로 None
                })
            logger.info(f"🔄 [VARMAX_GEN] Converted {len(predictions)} predictions")
            
            logger.info(f"🔄 [VARMAX_GEN] Step 8: Calculating performance metrics...")
            prediction_state['varmax_prediction_progress'] = 80
            # 성능 지표 계산
            metrics = self.calculate_performance_metrics()
            
            logger.info(f"🔄 [VARMAX_GEN] Step 9: Calculating moving averages...")
            prediction_state['varmax_prediction_progress'] = 85
            # 이동평균 계산 (VARMAX용)
            ma_results = self.calculate_moving_averages_varmax(predictions, current_date)
            
            logger.info(f"🔄 [VARMAX_GEN] Step 10: Calculating half-month averages...")
            prediction_state['varmax_prediction_progress'] = 90
            # 반월 평균 데이터 계산 (VarmaxResult 컴포넌트용)
            half_month_data = self.calculate_half_month_averages(predictions, current_date)
            
            logger.info(f"✅ [VARMAX_GEN] All steps completed successfully!")
            logger.info(f"✅ [VARMAX_GEN] Final results: {len(predictions)} predictions, {len(ma_results)} MA windows")
            
            return {
                'success': True,
                'predictions': predictions,  # 원래 예측 데이터 (차트용)
                'half_month_averages': half_month_data,  # 반월 평균 데이터 (VarmaxResult 컴포넌트용)
                'metrics': metrics,
                'ma_results': ma_results,
                'selected_features': self.selected_vars[:var_num],
                'current_date': format_date(current_date),
                'model_info': {
                    'model_type': 'VARMAX',
                    'variables_used': var_num,
                    'prediction_days': self.pred_days,
                    'r2_train': self.r2_train,
                    'r2_test': self.r2_test
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [VARMAX_GEN] VARMAX prediction failed: {str(e)}")
            logger.error(f"❌ [VARMAX_GEN] Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }

def background_prediction_simple_compatible(file_path, current_date, save_to_csv=True, use_cache=True):
    """호환성을 유지하는 백그라운드 예측 함수 - 캐시 우선 사용, JSON 안전성 보장"""
    global prediction_state
    
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 10
        prediction_state['prediction_start_time'] = time.time()  # 시작 시간 기록
        prediction_state['error'] = None
        prediction_state['latest_file_path'] = file_path  # 파일 경로 저장
        prediction_state['current_file'] = file_path  # 캐시 연동용 파일 경로
        
        logger.info(f"🎯 Starting compatible prediction for {current_date}")
        logger.info(f"  🔄 Cache enabled: {use_cache}")
        
        # 데이터 로드 (단일 날짜 예측용 - LSTM 모델, 2022년 이전 데이터 제거)
        df = load_data(file_path, model_type='lstm')
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 20
        
        # 현재 날짜 처리 및 영업일 조정
        if current_date is None:
            current_date = df.index.max()
        else:
            current_date = pd.to_datetime(current_date)
        
        # 🎯 휴일이면 다음 영업일로 조정
        original_date = current_date
        adjusted_date = current_date
        
        # 주말이나 휴일이면 다음 영업일로 이동
        while adjusted_date.weekday() >= 5 or is_holiday(adjusted_date):
            adjusted_date += pd.Timedelta(days=1)
        
        if adjusted_date != original_date:
            logger.info(f"📅 Date adjusted for business day: {original_date.strftime('%Y-%m-%d')} -> {adjusted_date.strftime('%Y-%m-%d')}")
            logger.info(f"  📋 Reason: {'Weekend' if original_date.weekday() >= 5 else 'Holiday'}")
        
        current_date = adjusted_date
        
        # 캐시 확인
        if use_cache:
            logger.info("🔍 Checking for existing prediction cache...")
            prediction_state['prediction_progress'] = 30
            
            try:
                cached_result = check_existing_prediction(current_date, file_path)
                logger.info(f"  📋 Cache check result: {cached_result is not None}")
                if cached_result:
                    logger.info(f"  📋 Cache success status: {cached_result.get('success', False)}")
                else:
                    logger.info("  ❌ No cache result returned")
            except Exception as cache_check_error:
                logger.error(f"  ❌ Cache check failed with error: {str(cache_check_error)}")
                logger.error(f"  📝 Error traceback: {traceback.format_exc()}")
                cached_result = None
        else:
            logger.info("🆕 Cache disabled - running new prediction...")
            cached_result = None
            
        if cached_result and cached_result.get('success'):
            logger.info("🎉 Found existing prediction! Loading from cache...")
            prediction_state['prediction_progress'] = 50
            
            try:
                    # 캐시된 데이터 로드 및 정리
                    predictions = cached_result['predictions']
                    metadata = cached_result['metadata']
                    attention_data = cached_result.get('attention_data')
                    
                    # 데이터 정리 (JSON 안전성 보장)
                    cleaned_predictions = clean_cached_predictions(predictions)
                    
                    # 호환성 유지된 형태로 변환
                    compatible_predictions = convert_to_legacy_format(cleaned_predictions)
                    
                    # JSON 직렬화 테스트
                    try:
                        test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
                        logger.info("✅ JSON serialization test passed for cached data")
                    except Exception as json_error:
                        logger.error(f"❌ JSON serialization failed for cached data: {str(json_error)}")
                        raise Exception("Cached data serialization failed")
                    
                    # 구간 점수 처리 (JSON 안전)
                    interval_scores = metadata.get('interval_scores', {})
                    cleaned_interval_scores = {}
                    for key, value in interval_scores.items():
                        if isinstance(value, dict):
                            cleaned_score = {}
                            for k, v in value.items():
                                cleaned_score[k] = safe_serialize_value(v)
                            cleaned_interval_scores[key] = cleaned_score
                        else:
                            cleaned_interval_scores[key] = safe_serialize_value(value)
                    
                    # 이동평균 재계산
                    prediction_state['prediction_progress'] = 60
                    logger.info("Recalculating moving averages from cached data...")
                    historical_data = df[df.index <= current_date].copy()
                    ma_results = calculate_moving_averages_with_history(
                        cleaned_predictions, historical_data, target_col='MOPJ'
                    )
                    
                    # 시각화 재생성
                    prediction_state['prediction_progress'] = 70
                    logger.info("Regenerating visualizations from cached data...")
                    plots = regenerate_visualizations_from_cache(
                        cleaned_predictions, df, current_date, metadata
                    )
                    
                    # 메트릭 정리
                    metrics = metadata.get('metrics')
                    cleaned_metrics = {}
                    if metrics:
                        for key, value in metrics.items():
                            cleaned_metrics[key] = safe_serialize_value(value)
                    
                    # 어텐션 데이터 정리
                    cleaned_attention = None
                    logger.info(f"📊 [CACHE_ATTENTION] Processing attention data: available={bool(attention_data)}")
                    if attention_data:
                        logger.info(f"📊 [CACHE_ATTENTION] Original keys: {list(attention_data.keys())}")
                        
                        cleaned_attention = {}
                        for key, value in attention_data.items():
                            if key == 'image' and value:
                                cleaned_attention[key] = value  # base64 이미지는 그대로
                                logger.info(f"📊 [CACHE_ATTENTION] Image preserved (length: {len(value)})")
                            elif isinstance(value, dict):
                                cleaned_attention[key] = {}
                                for k, v in value.items():
                                    cleaned_attention[key][k] = safe_serialize_value(v)
                                logger.info(f"📊 [CACHE_ATTENTION] Dict '{key}' processed: {len(cleaned_attention[key])} items")
                            else:
                                cleaned_attention[key] = safe_serialize_value(value)
                                logger.info(f"📊 [CACHE_ATTENTION] Value '{key}' processed: {type(value)}")
                        
                        logger.info(f"📊 [CACHE_ATTENTION] Final cleaned keys: {list(cleaned_attention.keys())}")
                    else:
                        logger.warning(f"📊 [CACHE_ATTENTION] No attention data in cache result")
                    
                    # 상태 설정
                    prediction_state['latest_predictions'] = compatible_predictions
                    prediction_state['latest_attention_data'] = cleaned_attention
                    prediction_state['current_date'] = safe_serialize_value(metadata.get('prediction_start_date'))
                    prediction_state['selected_features'] = metadata.get('selected_features', [])
                    prediction_state['semimonthly_period'] = safe_serialize_value(metadata.get('semimonthly_period'))
                    prediction_state['next_semimonthly_period'] = safe_serialize_value(metadata.get('next_semimonthly_period'))
                    prediction_state['latest_interval_scores'] = cleaned_interval_scores
                    prediction_state['latest_metrics'] = cleaned_metrics
                    prediction_state['latest_plots'] = plots
                    prediction_state['latest_ma_results'] = ma_results
                    
                    # feature_importance 설정
                    if cleaned_attention and 'feature_importance' in cleaned_attention:
                        prediction_state['feature_importance'] = cleaned_attention['feature_importance']
                    else:
                        prediction_state['feature_importance'] = None
                    
                    prediction_state['prediction_progress'] = 100
                    prediction_state['is_predicting'] = False
                    logger.info("✅ Cache prediction completed successfully!")
                    return
                    
            except Exception as cache_error:
                logger.warning(f"⚠️  Cache processing failed: {str(cache_error)}")
                logger.info("🔄 Falling back to new prediction...")
        else:
            logger.info("  📋 No usable cache found - proceeding with new prediction")
        
        # 새로운 예측 수행
        logger.info(f"🤖 Running new prediction...")
        prediction_state['prediction_progress'] = 40
        
        # 예측 수행
        results = generate_predictions_with_save(df, current_date, save_to_csv=save_to_csv, file_path=file_path)
        prediction_state['prediction_progress'] = 80
        
        # 새로운 예측 결과 정리 (JSON 안전성 보장)
        if isinstance(results.get('predictions'), list):
            raw_predictions = results['predictions']
        else:
            raw_predictions = results.get('predictions_flat', [])
        
        # 호환성 유지된 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        # JSON 직렬화 테스트
        try:
            test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
            logger.info("✅ JSON serialization test passed for new prediction")
        except Exception as json_error:
            logger.error(f"❌ JSON serialization failed for new prediction: {str(json_error)}")
            # 데이터 추가 정리 시도
            for pred in compatible_predictions:
                for key, value in pred.items():
                    pred[key] = safe_serialize_value(value)
        
        # 상태 설정
        prediction_state['latest_predictions'] = compatible_predictions
        prediction_state['latest_attention_data'] = results.get('attention_data')
        prediction_state['current_date'] = safe_serialize_value(results.get('current_date'))
        prediction_state['selected_features'] = results.get('selected_features', [])
        prediction_state['semimonthly_period'] = safe_serialize_value(results.get('semimonthly_period'))
        prediction_state['next_semimonthly_period'] = safe_serialize_value(results.get('next_semimonthly_period'))
        prediction_state['latest_interval_scores'] = results.get('interval_scores', {})
        prediction_state['latest_metrics'] = results.get('metrics')
        prediction_state['latest_plots'] = results.get('plots', {})
        prediction_state['latest_ma_results'] = results.get('ma_results', {})
        
        # feature_importance 설정
        if results.get('attention_data') and 'feature_importance' in results['attention_data']:
            prediction_state['feature_importance'] = results['attention_data']['feature_importance']
        else:
            prediction_state['feature_importance'] = None
        
        # 저장
        if save_to_csv:
            logger.info("💾 Saving prediction to cache...")
            save_result = save_prediction_simple(results, current_date)
            if save_result['success']:
                logger.info(f"✅ Cache saved successfully: {save_result.get('prediction_start_date')}")
            else:
                logger.warning(f"⚠️  Cache save failed: {save_result.get('error')}")
        
        prediction_state['prediction_progress'] = 100
        prediction_state['is_predicting'] = False
        logger.info("✅ New prediction completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in compatible prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0


def safe_serialize_value(value):
    """값을 JSON 안전하게 직렬화 (NaN/Infinity 처리 강화)"""
    if value is None:
        return None
    
    # numpy/pandas 배열 타입 먼저 체크
    if isinstance(value, (np.ndarray, pd.Series, list)):
        if len(value) == 0:
            return []
        elif len(value) == 1:
            # 단일 원소 배열인 경우 스칼라로 처리
            return safe_serialize_value(value[0])
        else:
            # 다중 원소 배열인 경우 리스트로 변환
            try:
                return [safe_serialize_value(item) for item in value]
            except:
                return [str(item) for item in value]
    
    # 🔧 강화된 NaN/Infinity 처리
    try:
        # pandas isna 체크 (가장 포괄적)
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    
    # 🔧 NumPy NaN/Infinity 체크
    try:
        if isinstance(value, (int, float, np.number)):
            if np.isnan(value) or np.isinf(value):
                return None
            # 정상 숫자값인 경우
            if isinstance(value, (np.floating, float)):
                return float(value)
            elif isinstance(value, (np.integer, int)):
                return int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    
    # 🔧 문자열 체크 (NaN이 문자열로 변환된 경우)
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['nan', 'inf', '-inf', 'infinity', '-infinity', 'null', 'none']:
            return None
        return value
    
    # 날짜 객체 처리
    if hasattr(value, 'isoformat'):  # datetime/Timestamp
        try:
            return value.strftime('%Y-%m-%d')
        except:
            return str(value)
    elif hasattr(value, 'strftime'):  # 기타 날짜 객체
        try:
            return value.strftime('%Y-%m-%d')
        except:
            return str(value)
    
    # 🔧 최종 JSON 직렬화 테스트 (더 안전하게)
    try:
        # JSON 직렬화 가능한지 확인
        json_str = json.dumps(value)
        # 직렬화된 문자열에 NaN이 포함되어 있는지 확인
        if 'NaN' in json_str or 'Infinity' in json_str:
            return None
        return value
    except (TypeError, ValueError, OverflowError):
        # 직렬화 실패 시 문자열로
        try:
            str_value = str(value)
            # 문자열에도 NaN이 포함된 경우 처리
            if any(nan_str in str_value.lower() for nan_str in ['nan', 'inf', 'infinity']):
                return None
            return str_value
        except:
            return None

def clean_predictions_data(predictions):
    """예측 데이터를 JSON 안전하게 정리"""
    if not predictions:
        return []
    
    cleaned = []
    for pred in predictions:
        cleaned_pred = {}
        for key, value in pred.items():
            if key in ['date', 'prediction_from']:
                # 날짜 필드
                if hasattr(value, 'strftime'):
                    cleaned_pred[key] = value.strftime('%Y-%m-%d')
                else:
                    cleaned_pred[key] = str(value)
            elif key in ['prediction', 'actual', 'error', 'error_pct']:
                # 숫자 필드
                cleaned_pred[key] = safe_serialize_value(value)
            else:
                # 기타 필드
                cleaned_pred[key] = safe_serialize_value(value)
        cleaned.append(cleaned_pred)
    
    return cleaned

def clean_cached_predictions(predictions):
    """캐시에서 로드된 예측 데이터를 정리하는 함수"""
    cleaned_predictions = []
    
    for pred in predictions:
        try:
            # 모든 필드를 안전하게 처리
            cleaned_pred = {}
            for key, value in pred.items():
                if key in ['Date', 'date']:
                    # 날짜 필드 특별 처리
                    if pd.notna(value):
                        if hasattr(value, 'strftime'):
                            cleaned_pred[key] = value.strftime('%Y-%m-%d')
                        else:
                            cleaned_pred[key] = str(value)[:10]
                    else:
                        cleaned_pred[key] = None
                elif key in ['Prediction', 'prediction', 'Actual', 'actual']:
                    # 숫자 필드 처리
                    cleaned_pred[key] = safe_serialize_value(value)
                else:
                    # 기타 필드
                    cleaned_pred[key] = safe_serialize_value(value)
            
            cleaned_predictions.append(cleaned_pred)
            
        except Exception as e:
            logger.warning(f"Error cleaning prediction item: {str(e)}")
            continue
    
    return cleaned_predictions

def clean_interval_scores_safe(interval_scores):
    """구간 점수를 안전하게 정리하는 함수 - 강화된 오류 처리"""
    cleaned_interval_scores = []
    
    try:
        # 입력값 검증
        if interval_scores is None:
            logger.info("📋 interval_scores is None, returning empty list")
            return []
        
        if not isinstance(interval_scores, (dict, list)):
            logger.warning(f"⚠️ interval_scores is not dict or list: {type(interval_scores)}")
            return []
        
        if isinstance(interval_scores, dict):
            if not interval_scores:  # 빈 dict
                logger.info("📋 interval_scores is empty dict, returning empty list")
                return []
                
            for key, value in interval_scores.items():
                try:
                    if isinstance(value, dict):
                        cleaned_score = {}
                        for k, v in value.items():
                            try:
                                # 배열이나 복잡한 타입은 특별 처리
                                if isinstance(v, (np.ndarray, pd.Series, list)):
                                    if hasattr(v, '__len__') and len(v) == 1:
                                        cleaned_score[k] = safe_serialize_value(v[0])
                                    elif hasattr(v, '__len__') and len(v) == 0:
                                        cleaned_score[k] = None
                                    else:
                                        # 다중 원소 배열은 문자열로 변환
                                        cleaned_score[k] = str(v)
                                else:
                                    cleaned_score[k] = safe_serialize_value(v)
                            except Exception as inner_e:
                                logger.warning(f"⚠️ Error processing key {k}: {str(inner_e)}")
                                cleaned_score[k] = None
                        cleaned_interval_scores.append(cleaned_score)
                    else:
                        # dict가 아닌 경우 안전하게 처리
                        cleaned_interval_scores.append(safe_serialize_value(value))
                except Exception as value_e:
                    logger.warning(f"⚠️ Error processing interval_scores key {key}: {str(value_e)}")
                    continue
                    
        elif isinstance(interval_scores, list):
            if not interval_scores:  # 빈 list
                logger.info("📋 interval_scores is empty list, returning empty list")
                return []
                
            for i, score in enumerate(interval_scores):
                try:
                    if isinstance(score, dict):
                        cleaned_score = {}
                        for k, v in score.items():
                            try:
                                # 배열이나 복잡한 타입은 특별 처리
                                if isinstance(v, (np.ndarray, pd.Series, list)):
                                    if hasattr(v, '__len__') and len(v) == 1:
                                        cleaned_score[k] = safe_serialize_value(v[0])
                                    elif hasattr(v, '__len__') and len(v) == 0:
                                        cleaned_score[k] = None
                                    else:
                                        cleaned_score[k] = str(v)
                                else:
                                    cleaned_score[k] = safe_serialize_value(v)
                            except Exception as inner_e:
                                logger.warning(f"⚠️ Error processing score[{i}].{k}: {str(inner_e)}")
                                cleaned_score[k] = None
                        cleaned_interval_scores.append(cleaned_score)
                    else:
                        cleaned_interval_scores.append(safe_serialize_value(score))
                except Exception as score_e:
                    logger.warning(f"⚠️ Error processing interval_scores[{i}]: {str(score_e)}")
                    continue
        
        logger.info(f"✅ Successfully cleaned {len(cleaned_interval_scores)} interval scores")
        return cleaned_interval_scores
        
    except Exception as e:
        logger.error(f"❌ Critical error cleaning interval scores: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def convert_to_legacy_format(predictions_data):
    """
    새·옛 구조를 모두 받아 프론트엔드(대문자) + 백엔드(소문자) 키를 동시 보존.
    JSON 직렬화 안전성 보장
    """
    if not predictions_data:
        return []
    
    legacy_out = []
    actual_values_found = 0  # 실제값이 발견된 수 카운트
    
    for i, pred in enumerate(predictions_data):
        try:
            # 날짜 필드 안전 처리
            date_value = pred.get("date") or pred.get("Date")
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
            elif isinstance(date_value, str):
                date_str = date_value[:10] if len(date_value) > 10 else date_value
            else:
                date_str = str(date_value) if date_value is not None else None
            
            # 예측값 안전 처리
            prediction_value = pred.get("prediction") or pred.get("Prediction")
            prediction_safe = safe_serialize_value(prediction_value)
            
            # 실제값 안전 처리 - 다양한 필드명 확인
            actual_value = (pred.get("actual") or 
                          pred.get("Actual") or 
                          pred.get("actual_value") or 
                          pred.get("Actual_Value"))
            
            # 실제값이 있는지 확인
            if actual_value is not None and actual_value != 'None' and not (
                isinstance(actual_value, float) and (np.isnan(actual_value) or np.isinf(actual_value))
            ):
                actual_safe = safe_serialize_value(actual_value)
                actual_values_found += 1
                if i < 5:  # 처음 5개만 로깅
                    logger.debug(f"  📊 [LEGACY_FORMAT] Found actual value for {date_str}: {actual_safe}")
            else:
                actual_safe = None
            
            # 기타 필드들 안전 처리
            prediction_from = pred.get("prediction_from") or pred.get("Prediction_From")
            if hasattr(prediction_from, 'strftime'):
                prediction_from = prediction_from.strftime('%Y-%m-%d')
            elif prediction_from:
                prediction_from = str(prediction_from)
            
            legacy_item = {
                # ── 프론트엔드 호환 대문자 키 (JSON 안전) ───────────────
                "Date": date_str,
                "Prediction": prediction_safe,
                "Actual": actual_safe,

                # ── 백엔드 후속 함수(소문자 'date' 참조)용 ──
                "date": date_str,
                "prediction": prediction_safe,
                "actual": actual_safe,

                # 기타 필드 안전 처리
                "Prediction_From": prediction_from,
                "SemimonthlyPeriod": safe_serialize_value(pred.get("semimonthly_period")),
                "NextSemimonthlyPeriod": safe_serialize_value(pred.get("next_semimonthly_period")),
                "is_synthetic": bool(pred.get("is_synthetic", False)),
                
                # 추가 메타데이터 (있는 경우)
                "day_offset": safe_serialize_value(pred.get("day_offset")),
                "is_business_day": bool(pred.get("is_business_day", True)),
                "error": safe_serialize_value(pred.get("error")),
                "error_pct": safe_serialize_value(pred.get("error_pct"))
            }
            
            legacy_out.append(legacy_item)
            
        except Exception as e:
            logger.warning(f"Error converting prediction item {i}: {str(e)}")
            continue
    
    # 실제값 통계 로깅
    total_predictions = len(legacy_out)
    logger.info(f"  📊 [LEGACY_FORMAT] Converted {total_predictions} predictions, {actual_values_found} with actual values")
    
    return legacy_out

#######################################################################
# API 엔드포인트
#######################################################################

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'attention_endpoint_available': True
    })

@app.route('/api/test-attention', methods=['GET'])
def test_attention():
    """어텐션 맵 엔드포인트 테스트용"""
    return jsonify({
        'success': True,
        'message': 'Test attention endpoint is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """API 연결 테스트"""
    return jsonify({
        'status': 'ok',
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test/cache-dirs', methods=['GET'])
def test_cache_dirs():
    """캐시 디렉토리 시스템 테스트"""
    try:
        # 현재 상태 확인
        current_file = prediction_state.get('current_file', None)
        
        # 파일 경로가 있으면 해당 파일로, 없으면 기본으로 테스트
        test_file = request.args.get('file_path', current_file)
        
        if test_file and not os.path.exists(test_file):
            return jsonify({
                'error': f'File does not exist: {test_file}',
                'current_file': current_file
            }), 400
        
        # 캐시 디렉토리 생성 테스트
        cache_dirs = get_file_cache_dirs(test_file)
        
        # 디렉토리 존재 여부 확인
        dir_status = {}
        for name, path in cache_dirs.items():
            dir_status[name] = {
                'path': str(path),
                'exists': path.exists(),
                'is_dir': path.is_dir() if path.exists() else False
            }
        
        return jsonify({
            'success': True,
            'test_file': test_file,
            'current_file': current_file,
            'cache_dirs': dir_status,
            'cache_root_exists': Path(CACHE_ROOT_DIR).exists()
        })
        
    except Exception as e:
        logger.error(f"Cache directory test failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def detect_file_type_by_content(file_path):
    """
    파일 내용을 분석하여 실제 파일 타입을 감지하는 함수
    회사 보안으로 인해 확장자가 변경된 파일들을 처리
    """
    try:
        # 파일의 첫 몇 바이트를 읽어서 파일 타입 감지
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        # Excel 파일 시그니처 확인
        if header[:4] == b'PK\x03\x04':  # ZIP 기반 파일 (xlsx)
            return 'xlsx'
        elif header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':  # OLE2 기반 파일 (xls)
            return 'xls'
        
        # CSV 파일인지 확인 (텍스트 기반)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                # CSV 특성 확인: 쉼표나 탭이 포함되어 있고, Date 컬럼이 있는지
                if (',' in first_line or '\t' in first_line) and ('date' in first_line.lower() or 'Date' in first_line):
                    return 'csv'
        except:
            # UTF-8로 읽기 실패시 다른 인코딩 시도
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    first_line = f.readline()
                    if (',' in first_line or '\t' in first_line) and ('date' in first_line.lower() or 'Date' in first_line):
                        return 'csv'
            except:
                pass
        
        # 기본값 반환
        return None
        
    except Exception as e:
        logger.warning(f"File type detection failed: {str(e)}")
        return None

def normalize_security_extension(filename):
    """
    회사 보안정책으로 변경된 확장자를 원래 확장자로 복원
    
    Args:
        filename (str): 원본 파일명
    
    Returns:
        tuple: (정규화된 파일명, 원본 확장자, 보안 확장자인지 여부)
    """
    # 보안 확장자 매핑
    security_extensions = {
        '.cs': '.csv',     # csv -> cs
        '.xl': '.xlsx',    # xlsx -> xl  
        '.xls': '.xlsx',   # 기존 xls도 xlsx로 통일
        '.log': '.xlsx',   # log -> xlsx (보안 정책으로 Excel 파일을 log로 위장)
        '.dat': None,      # 내용 분석 필요
        '.txt': None,      # 내용 분석 필요
    }
    
    filename_lower = filename.lower()
    original_ext = os.path.splitext(filename_lower)[1]
    
    # 보안 확장자인지 확인
    if original_ext in security_extensions:
        if security_extensions[original_ext]:
            # 직접 매핑이 있는 경우
            normalized_ext = security_extensions[original_ext]
            base_name = os.path.splitext(filename)[0]
            normalized_filename = f"{base_name}{normalized_ext}"
            
            logger.info(f"🔒 [SECURITY] Extension normalization: {filename} -> {normalized_filename}")
            return normalized_filename, normalized_ext, True
        else:
            # 내용 분석이 필요한 경우
            return filename, original_ext, True
    
    # 일반 확장자인 경우
    return filename, original_ext, False

def process_security_file(temp_filepath, original_filename):
    """
    보안 정책으로 확장자가 변경된 파일을 처리
    
    Args:
        temp_filepath (str): 임시 파일 경로
        original_filename (str): 원본 파일명
    
    Returns:
        tuple: (처리된 파일 경로, 정규화된 파일명, 실제 확장자)
    """
    # 확장자 정규화
    normalized_filename, detected_ext, is_security_ext = normalize_security_extension(original_filename)
    
    if is_security_ext:
        logger.info(f"🔒 [SECURITY] Processing security file: {original_filename}")
        
        # 파일 내용으로 실제 타입 감지
        if detected_ext is None or detected_ext in ['.dat', '.txt']:
            content_type = detect_file_type_by_content(temp_filepath)
            if content_type:
                detected_ext = f'.{content_type}'
                base_name = os.path.splitext(normalized_filename)[0]
                normalized_filename = f"{base_name}{detected_ext}"
                logger.info(f"📊 [CONTENT_DETECTION] Detected file type: {content_type}")
        
        # 새로운 파일 경로 생성
        new_filepath = temp_filepath.replace(os.path.splitext(temp_filepath)[1], detected_ext)
        
        # 파일 이름 변경 (확장자 수정)
        if new_filepath != temp_filepath:
            try:
                shutil.move(temp_filepath, new_filepath)
                logger.info(f"📝 [SECURITY] File extension corrected: {os.path.basename(temp_filepath)} -> {os.path.basename(new_filepath)}")
                return new_filepath, normalized_filename, detected_ext
            except Exception as e:
                logger.warning(f"⚠️ [SECURITY] Failed to rename file: {str(e)}")
                return temp_filepath, normalized_filename, detected_ext
    
    return temp_filepath, normalized_filename, detected_ext

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """스마트 캐시 기능이 있는 데이터 파일 업로드 API (CSV, Excel 지원)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 🔒 보안 확장자 정규화 처리
    normalized_filename, normalized_ext, is_security_file = normalize_security_extension(file.filename)
    
    # 지원되는 파일 형식 확인 (보안 확장자 포함)
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    security_extensions = ['.cs', '.xl', '.log', '.dat', '.txt']  # 보안 확장자 추가
    
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file and (file_ext in allowed_extensions or file_ext in security_extensions):
        try:
            # 임시 파일명 생성 (원본 확장자 유지)
            original_filename = secure_filename(file.filename)
            temp_filename = secure_filename(f"temp_{int(time.time())}{file_ext}")
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # 임시 파일로 저장
            file.save(temp_filepath)
            logger.info(f"📤 [UPLOAD] File saved temporarily: {temp_filename}")
            
            # 🔒 1단계: 보안 파일 처리 (확장자 복원) - 캐시 비교 전에 먼저 처리
            if is_security_file:
                temp_filepath, normalized_filename, actual_ext = process_security_file(temp_filepath, original_filename)
                file_ext = actual_ext  # 실제 확장자로 업데이트
                logger.info(f"🔒 [SECURITY] File processed: {original_filename} -> {normalized_filename}")
                
                # 처리된 파일이 지원되는 형식인지 재확인
                if file_ext not in allowed_extensions:
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                    return jsonify({'error': f'보안 파일 처리 후 지원되지 않는 형식입니다: {file_ext}'}), 400
            
            # 📊 2단계: 데이터 분석 - 날짜 범위 확인 (보안 처리 완료된 파일로)
            # 🔧 데이터 로딩 캐싱을 위한 변수 초기화
            df_analysis = None
            
            try:
                if file_ext == '.csv':
                    df_analysis = pd.read_csv(temp_filepath)
                else:  # Excel 파일
                    # Excel 파일은 load_data 함수를 사용하여 고급 처리 (🔧 캐시 활성화)
                    logger.info(f"🔍 [UPLOAD] Starting data analysis for {temp_filename}")
                    df_analysis = load_data(temp_filepath, use_cache=True)
                    # 인덱스가 Date인 경우 컬럼으로 복원
                    if df_analysis.index.name == 'Date':
                        df_analysis = df_analysis.reset_index()
                if 'Date' in df_analysis.columns:
                    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
                    start_date = df_analysis['Date'].min()
                    end_date = df_analysis['Date'].max()
                    total_records = len(df_analysis)
                    
                    # 2022년 이후 데이터 확인
                    cutoff_2022 = pd.to_datetime('2022-01-01')
                    recent_data = df_analysis[df_analysis['Date'] >= cutoff_2022]
                    recent_records = len(recent_data)
                    
                    logger.info(f"📊 [DATA_ANALYSIS] Full range: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({total_records} records)")
                    logger.info(f"📊 [DATA_ANALYSIS] 2022+ range: {recent_records} records")
                    
                    data_info = {
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'total_records': total_records,
                        'recent_records_2022plus': recent_records,
                        'has_historical_data': start_date < cutoff_2022,
                        'lstm_recommended_cutoff': '2022-01-01'
                    }
                else:
                    # Date 컬럼이 없는 파일의 경우 (예: holidays.csv)
                    file_type_hint = None
                    if 'holiday' in original_filename.lower():
                        file_type_hint = "휴일 파일로 보입니다. /api/holidays/upload 엔드포인트 사용을 권장합니다."
                    data_info = {
                        'warning': 'No Date column found',
                        'file_type_hint': file_type_hint
                    }
            except Exception as e:
                logger.warning(f"Data analysis failed: {str(e)}")
                data_info = {'warning': f'Data analysis failed: {str(e)}'}
            
            # 🔧 Excel 파일 읽기 완료 후 파일 핸들 강제 해제
            import gc
            gc.collect()  # 가비지 컬렉션으로 pandas가 열어둔 파일 핸들 해제
            
            # 🔍 3단계: 캐시 호환성 확인 (보안 처리 및 데이터 분석 완료 후)
            # 사용자의 의도된 데이터 범위 추정 (기본값: 2022년부터 LSTM, 전체 데이터 VARMAX)
            # end_date가 정의되지 않은 경우를 위한 안전한 fallback
            default_end_date = datetime.now().strftime('%Y-%m-%d')
            intended_range = {
                'start_date': '2022-01-01',  # LSTM 권장 시작점
                'cutoff_date': data_info.get('end_date', default_end_date)
            }
            
            logger.info(f"🔍 [UPLOAD_CACHE] Starting cache compatibility check:")
            logger.info(f"  📁 New file: {temp_filename}")
            logger.info(f"  📅 Data range: {data_info.get('start_date')} ~ {data_info.get('end_date')}")
            logger.info(f"  📊 Total records: {data_info.get('total_records')}")
            logger.info(f"  🎯 Intended range: {intended_range}")
            
            # 🔧 이미 로딩된 데이터를 전달하여 중복 로딩 방지
            cache_result = find_compatible_cache_file(temp_filepath, intended_range, cached_df=df_analysis)
            
            logger.info(f"🎯 [UPLOAD_CACHE] Cache check result:")
            logger.info(f"  ✅ Found: {cache_result['found']}")
            logger.info(f"  🏷️ Type: {cache_result.get('cache_type')}")
            if cache_result.get('cache_files'):
                logger.info(f"  📁 Cache files: {[os.path.basename(f) for f in cache_result['cache_files']]}")
            if cache_result.get('compatibility_info'):
                logger.info(f"  ℹ️ Compatibility info: {cache_result['compatibility_info']}")
            
            response_data = {
                'success': True,
                'filepath': temp_filepath,
                'filename': os.path.basename(temp_filepath),
                'original_filename': original_filename,
                'normalized_filename': normalized_filename if is_security_file else original_filename,
                'data_info': data_info,
                'model_recommendations': {
                    'varmax': '전체 데이터 사용 권장 (장기 트렌드 분석)',
                    'lstm': '2022년 이후 데이터 사용 권장 (단기 정확도 향상)'
                },
                'security_info': {
                    'is_security_file': is_security_file,
                    'original_extension': os.path.splitext(file.filename.lower())[1] if is_security_file else None,
                    'detected_extension': file_ext if is_security_file else None,
                    'message': f"보안 파일이 처리되었습니다: {os.path.splitext(file.filename)[1]} -> {file_ext}" if is_security_file else None
                },
                'cache_info': {
                    'found': cache_result['found'],
                    'cache_type': cache_result.get('cache_type'),
                    'message': None
                }
            }
            
            if cache_result['found']:
                cache_type = cache_result['cache_type']
                cache_files = cache_result.get('cache_files', [])
                compatibility_info = cache_result.get('compatibility_info', {})
                
                if cache_type == 'exact':
                    cache_file = cache_files[0] if cache_files else None
                    response_data['cache_info']['message'] = f"동일한 데이터 발견! 기존 캐시를 활용합니다. ({os.path.basename(cache_file) if cache_file else 'Unknown'})"
                    response_data['cache_info']['compatible_file'] = cache_file
                    logger.info(f"✅ [CACHE] Exact match found: {cache_file}")
                    
                elif cache_type == 'extension':
                    cache_file = cache_files[0] if cache_files else None
                    extension_details = compatibility_info.get('extension_details', {})
                    new_rows = extension_details.get('new_rows_count', compatibility_info.get('new_rows_count', 0))
                    extension_type = extension_details.get('validation_details', {}).get('extension_type', ['데이터 확장'])
                    
                    if isinstance(extension_type, list):
                        extension_desc = ' + '.join(extension_type)
                    else:
                        extension_desc = str(extension_type)
                    
                    response_data['cache_info']['message'] = f"📈 데이터 확장 감지! {extension_desc} (+{new_rows}개 새 행). 기존 하이퍼파라미터와 캐시를 재사용할 수 있습니다."
                    response_data['cache_info']['compatible_file'] = cache_file
                    response_data['cache_info']['extension_info'] = compatibility_info
                    response_data['cache_info']['hyperparams_reusable'] = True  # 하이퍼파라미터 재사용 가능 표시
                    logger.info(f"📈 [CACHE] Extension detected from {cache_file}: {extension_desc} (+{new_rows} rows)")
                    
                elif cache_type in ['partial', 'near_complete', 'multi_cache']:
                    best_coverage = compatibility_info.get('best_coverage', 0)
                    total_caches = compatibility_info.get('total_compatible_caches', len(cache_files))
                    
                    if cache_type == 'near_complete':
                        response_data['cache_info']['message'] = f"🎯 거의 완전한 캐시 매치! ({best_coverage:.1%} 커버리지) 기존 예측 결과를 최대한 활용합니다."
                    elif cache_type == 'multi_cache':
                        response_data['cache_info']['message'] = f"🔗 다중 캐시 발견! {total_caches}개 캐시에서 {best_coverage:.1%} 커버리지로 예측을 가속화합니다."
                    else:  # partial
                        response_data['cache_info']['message'] = f"📊 부분 캐시 매치! ({best_coverage:.1%} 커버리지) 일부 예측 결과를 재활용합니다."
                    
                    response_data['cache_info']['compatible_files'] = cache_files
                    response_data['cache_info']['compatibility_info'] = compatibility_info
                    logger.info(f"🎯 [ENHANCED_CACHE] {cache_type} cache found: {total_caches} caches, {best_coverage:.1%} coverage")
                
                # 🔧 파일 처리 로직 개선: 데이터 확장 시 새 파일 사용
                if cache_type == 'exact' and cache_files:
                    # 정확히 동일한 파일인 경우에만 기존 파일 사용
                    cache_file = cache_files[0]
                    response_data['filepath'] = cache_file
                    response_data['filename'] = os.path.basename(cache_file)
                    
                    # 임시 파일 삭제 (완전히 동일한 경우만)
                    if temp_filepath != cache_file:
                        try:
                            os.remove(temp_filepath)
                            logger.info(f"🗑️ [CLEANUP] Temporary file removed (exact match): {temp_filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file {temp_filename}: {str(e)}")
                            # 실패해도 계속 진행
                            
                elif cache_type == 'extension' and cache_files:
                    # 🔄 데이터 확장의 경우: 새 파일을 사용하되, 캐시 정보는 유지
                    logger.info(f"📈 [EXTENSION] Data extension detected - using NEW file with cache info")
                    
                    # 새 파일을 정식 파일명으로 저장 (원본 확장자 유지)
                    try:
                        content_hash = get_data_content_hash(temp_filepath)
                        final_filename = f"data_{content_hash}{file_ext}" if content_hash else temp_filename
                    except Exception as hash_error:
                        logger.warning(f"⚠️ Hash calculation failed for extended file, using timestamp-based filename: {str(hash_error)}")
                        final_filename = temp_filename  # 해시 실패 시 임시 파일명 유지
                    
                    final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                    
                    if temp_filepath != final_filepath:
                        # 🔧 강화된 파일 이동 로직 (Excel 파일 락 해제 대기)
                        moved_successfully = False
                        for attempt in range(3):  # 최대 3번 시도
                            try:
                                # Excel 파일 읽기 후 파일 락 해제를 위한 충분한 대기
                                import gc
                                gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
                                time.sleep(0.5 + attempt * 0.5)  # 점진적으로 대기 시간 증가
                                
                                shutil.move(temp_filepath, final_filepath)
                                logger.info(f"📝 [UPLOAD] Extended file renamed: {final_filename} (attempt {attempt + 1})")
                                moved_successfully = True
                                break
                            except OSError as move_error:
                                logger.warning(f"⚠️ Extended file move attempt {attempt + 1} failed: {str(move_error)}")
                                if attempt == 2:  # 마지막 시도
                                    logger.warning(f"⚠️ All move attempts failed, keeping original filename: {str(move_error)}")
                                    final_filepath = temp_filepath
                                    final_filename = temp_filename
                        
                        if not moved_successfully:
                            final_filepath = temp_filepath
                            final_filename = temp_filename
                    else:
                        logger.info(f"📝 [UPLOAD] Extended file already has correct name: {final_filename}")
                        
                    response_data['filepath'] = final_filepath
                    response_data['filename'] = final_filename
                    
                    # 확장 정보에 새 파일 정보 추가
                    response_data['cache_info']['new_file_used'] = True
                    response_data['cache_info']['original_cache_file'] = cache_files[0]
                    
                    # 🔑 데이터 확장 표시 - 하이퍼파라미터 재사용 가능
                    response_data['data_extended'] = True
                    response_data['hyperparams_inheritance'] = {
                        'available': True,
                        'source_file': os.path.basename(cache_files[0]),
                        'extension_type': extension_desc if 'extension_desc' in locals() else '데이터 확장',
                        'new_rows_added': new_rows if 'new_rows' in locals() else compatibility_info.get('new_rows_count', 0)
                    }
                    
                else:
                    # 새 파일은 유지 (부분/다중 캐시의 경우, 원본 확장자 유지)
                    try:
                        content_hash = get_data_content_hash(temp_filepath)
                        final_filename = f"data_{content_hash}{file_ext}" if content_hash else temp_filename
                    except Exception as hash_error:
                        logger.warning(f"⚠️ Hash calculation failed, using timestamp-based filename: {str(hash_error)}")
                        final_filename = temp_filename  # 해시 실패 시 임시 파일명 유지
                    
                    final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                    
                    if temp_filepath != final_filepath:
                        # 🔧 강화된 파일 이동 로직 (Excel 파일 락 해제 대기)
                        moved_successfully = False
                        for attempt in range(3):  # 최대 3번 시도
                            try:
                                # Excel 파일 읽기 후 파일 락 해제를 위한 충분한 대기
                                import gc
                                gc.collect()  # 가비지 컬렉션으로 파일 핸들 해제
                                time.sleep(0.5 + attempt * 0.5)  # 점진적으로 대기 시간 증가
                                
                                shutil.move(temp_filepath, final_filepath)
                                logger.info(f"📝 [UPLOAD] File renamed: {final_filename} (attempt {attempt + 1})")
                                moved_successfully = True
                                break
                            except OSError as move_error:
                                logger.warning(f"⚠️ File move attempt {attempt + 1} failed: {str(move_error)}")
                                if attempt == 2:  # 마지막 시도
                                    logger.warning(f"⚠️ All move attempts failed, keeping original filename: {str(move_error)}")
                                    final_filepath = temp_filepath
                                    final_filename = temp_filename
                        
                        if not moved_successfully:
                            final_filepath = temp_filepath
                            final_filename = temp_filename
                    else:
                        logger.info(f"📝 [UPLOAD] File already has correct name: {final_filename}")
                        
                    response_data['filepath'] = final_filepath
                    response_data['filename'] = final_filename
                response_data['cache_info']['message'] = "새로운 데이터입니다. 모델별로 적절한 데이터 범위를 사용하여 예측합니다."
            
            # 🔑 업로드된 파일 경로를 전역 상태에 저장
            prediction_state['current_file'] = response_data['filepath']
            logger.info(f"📁 Set current_file in prediction_state: {response_data['filepath']}")
            
            # 🔧 성공 시 temp 파일 정리 (final_filepath와 다른 경우에만)
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                final_filepath = response_data.get('filepath')
                if final_filepath and temp_filepath != final_filepath:
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"🗑️ [CLEANUP] Success - temp file removed: {os.path.basename(temp_filepath)}")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file after success: {str(cleanup_error)}")
                else:
                    logger.info(f"📝 [CLEANUP] Temp file kept as final file: {os.path.basename(temp_filepath)}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            # 🔧 강화된 temp 파일 정리
            try:
                if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    logger.info(f"🗑️ [CLEANUP] Temp file removed on error: {os.path.basename(temp_filepath)}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [CLEANUP] Failed to remove temp file on error: {str(cleanup_error)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV and Excel files (.csv, .xlsx, .xls) are allowed'}), 400

@app.route('/api/holidays', methods=['GET'])
def get_holidays():
    """휴일 목록 조회 API"""
    try:
        # 휴일을 날짜와 설명이 포함된 딕셔너리 리스트로 변환
        holidays_list = []
        file_holidays = load_holidays_from_file()  # 파일에서 로드
        
        # 현재 전역 휴일에서 파일 휴일과 자동 감지 휴일 구분
        auto_detected = holidays - file_holidays
        
        for holiday_date in file_holidays:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (from file)',
                'source': 'file'
            })
        
        for holiday_date in auto_detected:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (detected from missing data)',
                'source': 'auto_detected'
            })
        
        # 날짜순으로 정렬
        holidays_list.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'holidays': holidays_list,
            'count': len(holidays_list),
            'file_holidays': len(file_holidays),
            'auto_detected_holidays': len(auto_detected)
        })
    except Exception as e:
        logger.error(f"Error getting holidays: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'holidays': [],
            'count': 0
        }), 500

@app.route('/api/holidays/upload', methods=['POST'])
def upload_holidays():
    """휴일 목록 파일 업로드 API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            # 임시 파일명 생성
            filename = secure_filename(f"holidays_{int(time.time())}{os.path.splitext(file.filename)[1]}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 파일 저장
            file.save(filepath)
            
            # 휴일 정보 업데이트 - 보안 우회 기능 사용
            logger.info(f"🏖️ [HOLIDAY_UPLOAD] Processing uploaded holiday file: {filename}")
            new_holidays = update_holidays_safe(filepath)
            
            # 원본 파일을 holidays 디렉토리로 복사
            holidays_dir = 'holidays'
            if not os.path.exists(holidays_dir):
                os.makedirs(holidays_dir)
                logger.info(f"📁 Created holidays directory: {holidays_dir}")
            
            permanent_path = os.path.join(holidays_dir, 'holidays' + os.path.splitext(file.filename)[1])
            shutil.copy2(filepath, permanent_path)
            logger.info(f"📁 Holiday file copied to: {permanent_path}")
            
            # 임시 파일 정리
            try:
                os.remove(filepath)
                logger.info(f"🗑️ Temporary file removed: {filepath}")
            except:
                pass
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded and loaded {len(new_holidays)} holidays',
                'filepath': permanent_path,
                'filename': os.path.basename(permanent_path),
                'holidays': list(new_holidays)
            })
        except Exception as e:
            logger.error(f"Error during holiday file upload: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({
        'error': 'Invalid file type. Only CSV and Excel files are allowed',
        'supported_extensions': {
            'standard': ['.csv', '.xlsx', '.xls'],
            'security': ['.cs (csv)', '.xl (xlsx)', '.log (xlsx)', '.dat (auto-detect)', '.txt (auto-detect)']
        }
    }), 400

@app.route('/api/holidays/reload', methods=['POST'])
def reload_holidays():
    """휴일 목록 재로드 API - 보안 우회 기능 포함"""
    try:
        filepath = request.json.get('filepath') if request.json else None
        
        logger.info(f"🔄 [HOLIDAY_RELOAD] Reloading holidays from: {filepath or 'default file'}")
        
        # 보안 우회 기능을 포함한 안전한 재로드
        new_holidays = update_holidays_safe(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Successfully reloaded {len(new_holidays)} holidays',
            'holidays': list(new_holidays),
            'security_bypass_used': XLWINGS_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"❌ [HOLIDAY_RELOAD] Error reloading holidays: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to reload holidays: {str(e)}'
        }), 500

@app.route('/api/file/metadata', methods=['GET'])
def get_file_metadata():
    """파일 메타데이터 조회 API"""
    filepath = request.args.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 파일 확장자에 따라 읽기 방식 결정
        file_ext = os.path.splitext(filepath.lower())[1]
        
        if file_ext == '.csv':
            # CSV 파일 처리
            df = pd.read_csv(filepath, nrows=5)  # 처음 5행만 읽기
            columns = df.columns.tolist()
            latest_date = None
            
            if 'Date' in df.columns:
                # 날짜 정보를 별도로 읽어서 최신 날짜 확인
                dates_df = pd.read_csv(filepath, usecols=['Date'])
                dates_df['Date'] = pd.to_datetime(dates_df['Date'])
                latest_date = dates_df['Date'].max().strftime('%Y-%m-%d')
        else:
            # Excel 파일 처리 (고급 처리 사용) - 🔧 중복 로딩 방지
            logger.info(f"🔍 [METADATA] Loading Excel data for metadata extraction...")
            df = load_data(filepath)
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                full_df = df.copy()  # 🔧 전체 데이터 저장 (중복 로딩 방지)
                df = df.reset_index()
            else:
                full_df = df.copy()  # 🔧 전체 데이터 저장 (중복 로딩 방지)
            
            # 처음 5행만 선택
            df_sample = df.head(5)
            columns = df.columns.tolist()
            latest_date = None
            
            if 'Date' in df.columns:
                # 🔧 이미 로딩된 데이터에서 최신 날짜 확인 (중복 로딩 방지)
                if full_df.index.name == 'Date':
                    latest_date = pd.to_datetime(full_df.index).max().strftime('%Y-%m-%d')
                else:
                    latest_date = pd.to_datetime(full_df['Date']).max().strftime('%Y-%m-%d')
            
            # 메모리 정리
            df = df_sample
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': columns,
            'latest_date': latest_date,
            'sample': df.head().to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error reading file metadata: {str(e)}")
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500
    
@app.route('/api/data/dates', methods=['GET'])
def get_available_dates():
    filepath = request.args.get('filepath')
    days_limit = int(request.args.get('limit', 999999))  # 기본값을 매우 큰 수로 설정 (모든 날짜)
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'  # 강제 새로고침 옵션
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 🔄 파일의 최신 해시와 수정 시간 확인하여 변경 감지
        current_file_hash = get_data_content_hash(filepath)
        current_file_mtime = os.path.getmtime(filepath)
        
        logger.info(f"🔍 [DATE_REFRESH] Checking file status:")
        logger.info(f"  📁 File: {os.path.basename(filepath)}")
        logger.info(f"  🔑 Current hash: {current_file_hash[:12] if current_file_hash else 'None'}...")
        logger.info(f"  ⏰ Modified time: {datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  🔄 Force refresh: {force_refresh}")
        
        # 파일 데이터 로드 및 분석 (파일 형식에 맞게, 항상 최신 파일 내용 확인)
        # 🔑 단일 날짜 예측용: LSTM 모델 타입 지정하여 2022년 이후 데이터만 로드
        file_ext = os.path.splitext(filepath.lower())[1]
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        else:
            # Excel 파일인 경우 load_data 함수 사용 (LSTM 모델 타입 지정)
            df = load_data(filepath, model_type='lstm')
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                df = df.reset_index()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 🏖️ 데이터를 로드한 후 휴일 정보 자동 업데이트 (빈 평일 감지) - 임시 비활성화
        logger.info(f"🏖️ [HOLIDAYS] Auto-detection temporarily disabled to show more dates...")
        # updated_holidays = update_holidays(df=df)
        updated_holidays = load_holidays_from_file()  # 파일 휴일만 사용
        logger.info(f"🏖️ [HOLIDAYS] Total holidays (file only): {len(updated_holidays)}")
        
        # 📊 실제 파일 데이터 범위 확인 (캐시 무시)
        total_rows = len(df)
        data_start_date = df.iloc[0]['Date']
        data_end_date = df.iloc[-1]['Date']
        
        logger.info(f"📊 [ACTUAL_DATA] File analysis results:")
        logger.info(f"  📈 Total data rows: {total_rows}")
        logger.info(f"  📅 Actual date range: {data_start_date.strftime('%Y-%m-%d')} ~ {data_end_date.strftime('%Y-%m-%d')}")
        
        # 🔍 기존 캐시와 비교 (있는 경우)
        existing_cache_range = find_existing_cache_range(filepath)
        if existing_cache_range and not force_refresh:
            cache_start = pd.to_datetime(existing_cache_range['start_date'])
            cache_cutoff = pd.to_datetime(existing_cache_range['cutoff_date'])
            
            logger.info(f"💾 [CACHE_COMPARISON] Found existing cache range:")
            logger.info(f"  📅 Cached range: {cache_start.strftime('%Y-%m-%d')} ~ {cache_cutoff.strftime('%Y-%m-%d')}")
            
            # 실제 데이터가 캐시된 범위보다 확장되었는지 확인
            data_extended = (
                data_start_date < cache_start or 
                data_end_date > cache_cutoff
            )
            
            if data_extended:
                logger.info(f"📈 [DATA_EXTENSION] Data has been extended!")
                logger.info(f"  ⬅️ Start extension: {data_start_date.strftime('%Y-%m-%d')} vs cached {cache_start.strftime('%Y-%m-%d')}")
                logger.info(f"  ➡️ End extension: {data_end_date.strftime('%Y-%m-%d')} vs cached {cache_cutoff.strftime('%Y-%m-%d')}")
                logger.info(f"  🔄 Using extended data range for date calculation")
            else:
                logger.info(f"✅ [NO_EXTENSION] Data range matches cached range, proceeding with current data")
        else:
            if force_refresh:
                logger.info(f"🔄 [FORCE_REFRESH] Ignoring cache due to force refresh")
            else:
                logger.info(f"📭 [NO_CACHE] No existing cache found, using full data range")
        
        # 데이터 마지막 날짜의 다음 영업일을 계산하여 예측 시작점 설정 (실제 데이터 기준)
        # 최소 100개 행 이상의 히스토리가 있는 경우에만 예측 가능
        min_history_rows = 100
        prediction_start_index = max(min_history_rows, total_rows // 4)  # 25% 지점 또는 최소 100행 중 큰 값
        
        # 실제 예측에 사용할 수 있는 모든 날짜 (충분한 히스토리가 있는 날짜부터)
        predictable_dates = df.iloc[prediction_start_index:]['Date']
        
        # 예측 시작 임계값 계산 (참고용)
        if prediction_start_index < total_rows:
            prediction_threshold_date = df.iloc[prediction_start_index]['Date']
        else:
            prediction_threshold_date = data_end_date
        
        logger.info(f"🎯 [PREDICTION_CALC] Prediction calculation:")
        logger.info(f"  📊 Min history rows: {min_history_rows}")
        logger.info(f"  📍 Start index: {prediction_start_index} (date: {prediction_threshold_date.strftime('%Y-%m-%d')})")
        logger.info(f"  📅 Predictable dates: {len(predictable_dates)} dates available")
        
        # 예측 가능한 모든 날짜를 내림차순으로 반환 (최신 날짜부터)
        # days_limit보다 작은 경우에만 제한 적용
        if len(predictable_dates) <= days_limit:
            dates = predictable_dates.sort_values(ascending=False).dt.strftime('%Y-%m-%d').tolist()
        else:
            dates = predictable_dates.sort_values(ascending=False).head(days_limit).dt.strftime('%Y-%m-%d').tolist()
        
        logger.info(f"🔢 [FINAL_RESULT] Final date calculation:")
        logger.info(f"  📊 Available predictable dates: {len(predictable_dates)}")
        logger.info(f"  📋 Returned dates: {len(dates)}")
        logger.info(f"  📅 Latest available date: {dates[0] if dates else 'None'}")
        
        response_data = {
            'success': True,
            'dates': dates,
            'latest_date': dates[0] if dates else None,  # 첫 번째 요소가 최신 날짜 (내림차순)
            'data_start_date': data_start_date.strftime('%Y-%m-%d'),
            'data_end_date': data_end_date.strftime('%Y-%m-%d'),
            'prediction_threshold': prediction_threshold_date.strftime('%Y-%m-%d'),
            'min_history_rows': min_history_rows,
            'total_rows': total_rows,
            'file_hash': current_file_hash[:12] if current_file_hash else None,  # 추가: 파일 해시 정보
            'file_modified': datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S')  # 추가: 파일 수정 시간
        }
        
        logger.info(f"📡 [API_RESPONSE] Sending enhanced dates response:")
        logger.info(f"  📅 Data range: {response_data['data_start_date']} ~ {response_data['data_end_date']}")
        logger.info(f"  🎯 Prediction threshold: {response_data['prediction_threshold']}")
        logger.info(f"  📅 Available date range: {dates[-1] if dates else 'None'} ~ {dates[0] if dates else 'None'} (최신부터)")
        logger.info(f"  🔑 File signature: {response_data['file_hash']} @ {response_data['file_modified']}")
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error reading dates: {str(e)}")
        return jsonify({'error': f'Error reading dates: {str(e)}'}), 500

@app.route('/api/data/refresh', methods=['POST'])
def refresh_file_data():
    """파일 데이터 새로고침 및 캐시 갱신 API"""
    try:
        filepath = request.json.get('filepath') if request.json else request.args.get('filepath')
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # 파일 해시와 수정 시간 확인
        current_file_hash = get_data_content_hash(filepath)
        current_file_mtime = os.path.getmtime(filepath)
        
        logger.info(f"🔄 [FILE_REFRESH] Starting file data refresh:")
        logger.info(f"  📁 File: {os.path.basename(filepath)}")
        logger.info(f"  🔑 Hash: {current_file_hash[:12] if current_file_hash else 'None'}...")
        
        # 기존 캐시 확인
        existing_cache_range = find_existing_cache_range(filepath)
        refresh_needed = False
        refresh_reason = []
        
        if existing_cache_range:
            # 캐시된 메타데이터와 비교
            meta_file = existing_cache_range.get('meta_file')
            if meta_file and os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    cached_hash = meta_data.get('file_hash')
                    cached_mtime = meta_data.get('file_modified_time')
                    
                    if cached_hash != current_file_hash:
                        refresh_needed = True
                        refresh_reason.append("File content changed")
                        
                    if cached_mtime and cached_mtime != current_file_mtime:
                        refresh_needed = True
                        refresh_reason.append("File modification time changed")
                        
                except Exception as e:
                    logger.warning(f"Error reading cache metadata: {str(e)}")
                    refresh_needed = True
                    refresh_reason.append("Cache metadata error")
            else:
                refresh_needed = True
                refresh_reason.append("No cache metadata found")
        else:
            refresh_needed = True
            refresh_reason.append("No existing cache")
        
        # 파일 데이터 분석 (파일 형식에 맞게)
        file_ext = os.path.splitext(filepath.lower())[1]
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        else:
            # Excel 파일인 경우 load_data 함수 사용
            df = load_data(filepath)
            # 인덱스가 Date인 경우 컬럼으로 복원
            if df.index.name == 'Date':
                df = df.reset_index()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        current_data_range = {
            'start_date': df.iloc[0]['Date'],
            'end_date': df.iloc[-1]['Date'],
            'total_rows': len(df)
        }
        
        # 캐시와 실제 데이터 범위 비교
        if existing_cache_range and not refresh_needed:
            cache_start = pd.to_datetime(existing_cache_range['start_date'])
            cache_cutoff = pd.to_datetime(existing_cache_range['cutoff_date'])
            
            if (current_data_range['start_date'] < cache_start or 
                current_data_range['end_date'] > cache_cutoff):
                refresh_needed = True
                refresh_reason.append("Data range extended")
        
        response_data = {
            'success': True,
            'refresh_needed': refresh_needed,
            'refresh_reasons': refresh_reason,
            'file_info': {
                'hash': current_file_hash[:12] if current_file_hash else None,
                'modified_time': datetime.fromtimestamp(current_file_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'total_rows': current_data_range['total_rows'],
                'date_range': {
                    'start': current_data_range['start_date'].strftime('%Y-%m-%d'),
                    'end': current_data_range['end_date'].strftime('%Y-%m-%d')
                }
            }
        }
        
        if existing_cache_range:
            response_data['cache_info'] = {
                'date_range': {
                    'start': existing_cache_range['start_date'],
                    'end': existing_cache_range['cutoff_date']
                },
                'meta_file': existing_cache_range.get('meta_file')
            }
        
        logger.info(f"📊 [REFRESH_ANALYSIS] File refresh analysis:")
        logger.info(f"  🔄 Refresh needed: {refresh_needed}")
        logger.info(f"  📝 Reasons: {', '.join(refresh_reason) if refresh_reason else 'None'}")
        logger.info(f"  📅 Current range: {response_data['file_info']['date_range']['start']} ~ {response_data['file_info']['date_range']['end']}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in file refresh check: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/compare-files', methods=['POST'])
def debug_compare_files():
    """두 파일을 직접 비교하여 차이점을 분석하는 디버깅 API"""
    try:
        data = request.json
        file1_path = data.get('file1_path')
        file2_path = data.get('file2_path')
        
        if not file1_path or not file2_path:
            return jsonify({'error': 'Both file paths are required'}), 400
            
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            return jsonify({'error': 'One or both files do not exist'}), 404
        
        logger.info(f"🔍 [DEBUG_COMPARE] Comparing files:")
        logger.info(f"  📁 File 1: {file1_path}")
        logger.info(f"  📁 File 2: {file2_path}")
        
        # 파일 기본 정보
        file1_hash = get_data_content_hash(file1_path)
        file2_hash = get_data_content_hash(file2_path)
        file1_size = os.path.getsize(file1_path)
        file2_size = os.path.getsize(file2_path)
        file1_mtime = os.path.getmtime(file1_path)
        file2_mtime = os.path.getmtime(file2_path)
        
        # 데이터 분석 (파일 형식에 맞게)
        def load_file_safely(filepath):
            file_ext = os.path.splitext(filepath.lower())[1]
            if file_ext == '.csv':
                return pd.read_csv(filepath)
            else:
                # Excel 파일인 경우 load_data 함수 사용
                df = load_data(filepath)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if df.index.name == 'Date':
                    df = df.reset_index()
                return df
        
        df1 = load_file_safely(file1_path)
        df2 = load_file_safely(file2_path)
        
        if 'Date' in df1.columns and 'Date' in df2.columns:
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            df1 = df1.sort_values('Date')
            df2 = df2.sort_values('Date')
            
            file1_dates = {
                'start': df1['Date'].min(),
                'end': df1['Date'].max(),
                'count': len(df1)
            }
            
            file2_dates = {
                'start': df2['Date'].min(),
                'end': df2['Date'].max(),
                'count': len(df2)
            }
        else:
            file1_dates = {'error': 'No Date column'}
            file2_dates = {'error': 'No Date column'}
        
        # 확장 체크
        extension_result = check_data_extension(file1_path, file2_path)
        
        # 캐시 호환성 체크
        cache_result = find_compatible_cache_file(file2_path)
        
        response_data = {
            'success': True,
            'comparison': {
                'file1': {
                    'path': file1_path,
                    'hash': file1_hash[:12] if file1_hash else None,
                    'size': file1_size,
                    'modified': datetime.fromtimestamp(file1_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'dates': {
                        'start': file1_dates['start'].strftime('%Y-%m-%d') if isinstance(file1_dates.get('start'), pd.Timestamp) else str(file1_dates.get('start')),
                        'end': file1_dates['end'].strftime('%Y-%m-%d') if isinstance(file1_dates.get('end'), pd.Timestamp) else str(file1_dates.get('end')),
                        'count': file1_dates.get('count')
                    } if 'error' not in file1_dates else file1_dates
                },
                'file2': {
                    'path': file2_path,
                    'hash': file2_hash[:12] if file2_hash else None,
                    'size': file2_size,
                    'modified': datetime.fromtimestamp(file2_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'dates': {
                        'start': file2_dates['start'].strftime('%Y-%m-%d') if isinstance(file2_dates.get('start'), pd.Timestamp) else str(file2_dates.get('start')),
                        'end': file2_dates['end'].strftime('%Y-%m-%d') if isinstance(file2_dates.get('end'), pd.Timestamp) else str(file2_dates.get('end')),
                        'count': file2_dates.get('count')
                    } if 'error' not in file2_dates else file2_dates
                },
                'identical_hash': file1_hash == file2_hash,
                'size_difference': file2_size - file1_size,
                'extension_analysis': extension_result,
                'cache_analysis': cache_result
            }
        }
        
        logger.info(f"📊 [DEBUG_COMPARE] Comparison results:")
        logger.info(f"  🔑 Identical hash: {file1_hash == file2_hash}")
        logger.info(f"  📏 Size difference: {file2_size - file1_size} bytes")
        logger.info(f"  📈 Is extension: {extension_result.get('is_extension', False)}")
        logger.info(f"  💾 Cache found: {cache_result.get('found', False)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in file comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved', methods=['GET'])
def get_saved_predictions():
    """저장된 예측 결과 목록 조회 API"""
    try:
        limit = int(request.args.get('limit', 100))
        predictions_list = get_saved_predictions_list(limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions_list,
            'count': len(predictions_list)
        })
    except Exception as e:
        logger.error(f"Error getting saved predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['GET'])
def get_saved_prediction_by_date(date):
    """특정 날짜의 저장된 예측 결과 조회 API"""
    try:
        result = load_prediction_from_csv(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 404
    except Exception as e:
        logger.error(f"Error loading saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['DELETE'])
def delete_saved_prediction_api(date):
    """저장된 예측 결과 삭제 API"""
    try:
        result = delete_saved_prediction(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>/update-actual', methods=['POST'])
def update_prediction_actual_values_api(date):
    """캐시된 예측의 실제값만 업데이트하는 API - 성능 최적화"""
    try:
        # 요청 파라미터
        data = request.json or {}
        update_latest_only = data.get('update_latest_only', True)
        
        logger.info(f"🔄 [API] Updating actual values for prediction {date}")
        logger.info(f"  📊 Update latest only: {update_latest_only}")
        
        # 실제값 업데이트 실행
        result = update_cached_prediction_actual_values(date, update_latest_only)
        
        if result['success']:
            logger.info(f"✅ [API] Successfully updated {result.get('updated_count', 0)} actual values")
            return jsonify({
                'success': True,
                'updated_count': result.get('updated_count', 0),
                'message': f'Updated {result.get("updated_count", 0)} actual values',
                'predictions': result['predictions']
            })
        else:
            logger.error(f"❌ [API] Failed to update actual values: {result.get('error')}")
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"❌ [API] Error updating actual values: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/export', methods=['GET'])
def export_predictions():
    """저장된 예측 결과들을 하나의 CSV 파일로 내보내기 API"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 날짜 범위에 따른 예측 로드
        if start_date:
            predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        else:
            # 모든 저장된 예측 로드
            predictions_list = get_saved_predictions_list(limit=1000)
            predictions = []
            for pred_info in predictions_list:
                loaded = load_prediction_from_csv(pred_info['prediction_date'])
                if loaded['success']:
                    predictions.extend(loaded['predictions'])
        
        if not predictions:
            return jsonify({'error': 'No predictions found for export'}), 404
        
        # DataFrame으로 변환
        if isinstance(predictions[0], dict) and 'predictions' in predictions[0]:
            # 누적 예측 형식인 경우
            all_predictions = []
            for pred_group in predictions:
                all_predictions.extend(pred_group['predictions'])
            export_df = pd.DataFrame(all_predictions)
        else:
            # 단순 예측 리스트인 경우
            export_df = pd.DataFrame(predictions)
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        export_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # 파일 전송
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 7. API 엔드포인트 수정 - 스마트 캐시 사용
@app.route('/api/predict', methods=['POST'])
def start_prediction_compatible():
    """호환성을 유지하는 예측 시작 API - 캐시 우선 사용 (로그 강화)"""
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    current_date = data.get('date')
    save_to_csv = data.get('save_to_csv', True)
    use_cache = data.get('use_cache', True)  # 기본값 True
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    # 🔑 파일 경로를 전역 상태에 저장 (캐시 연동용)
    prediction_state['current_file'] = filepath
    
    # ✅ 로그 강화
    logger.info(f"🚀 Prediction API called:")
    logger.info(f"  📅 Target date: {current_date}")
    logger.info(f"  📁 Data file: {filepath}")
    logger.info(f"  💾 Save to CSV: {save_to_csv}")
    logger.info(f"  🔄 Use cache: {use_cache}")
    
    # 호환성 유지 백그라운드 함수 실행 (캐시 우선 사용, 단일 예측만)
    thread = Thread(target=background_prediction_simple_compatible, 
                   args=(filepath, current_date, save_to_csv, use_cache))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Compatible prediction started (cache-first)',
        'use_cache': use_cache,
        'cache_priority': 'high',
        'features': ['Cache-first loading', 'Unified file naming', 'Enhanced logging', 'Past/Future visualization split']
    })

@app.route('/api/predict/status', methods=['GET'])
def prediction_status():
    """예측 상태 확인 API (남은 시간 추가)"""
    global prediction_state
    
    status = {
        'is_predicting': prediction_state['is_predicting'],
        'progress': prediction_state['prediction_progress'],
        'error': prediction_state['error']
    }
    
    # 예측 중인 경우 남은 시간 계산
    if prediction_state['is_predicting'] and prediction_state['prediction_start_time']:
        time_info = calculate_estimated_time_remaining(
            prediction_state['prediction_start_time'], 
            prediction_state['prediction_progress']
        )
        status.update(time_info)
    
    # 예측이 완료된 경우 날짜 정보도 반환
    if not prediction_state['is_predicting'] and prediction_state['current_date']:
        status['current_date'] = prediction_state['current_date']
    
    return jsonify(status)

@app.route('/api/results', methods=['GET'])
def get_prediction_results_compatible():
    """호환성을 유지하는 예측 결과 조회 API (오류 수정)"""
    global prediction_state
    
    logger.info(f"=== API /results called (compatible version) ===")
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    if prediction_state['latest_predictions'] is None:
        return jsonify({'error': 'No prediction results available'}), 404

    try:
        # 예측 데이터를 기존 형태로 변환
        if isinstance(prediction_state['latest_predictions'], list):
            raw_predictions = prediction_state['latest_predictions']
        else:
            raw_predictions = prediction_state['latest_predictions']
        
        # 기존 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        logger.info(f"Converted {len(raw_predictions)} predictions to legacy format")
        logger.info(f"Sample converted prediction: {compatible_predictions[0] if compatible_predictions else 'None'}")
        
        # 메트릭 정리
        metrics = prediction_state['latest_metrics']
        cleaned_metrics = {}
        if metrics:
            for key, value in metrics.items():
                cleaned_metrics[key] = safe_serialize_value(value)
        
        # 구간 점수 안전 정리 - 오류 방지 강화
        interval_scores = prediction_state['latest_interval_scores'] or []
        
        # interval_scores 데이터 타입 검증 및 안전 처리
        if interval_scores is None:
            interval_scores = []
        elif not isinstance(interval_scores, (list, dict)):
            logger.warning(f"⚠️ Unexpected interval_scores type: {type(interval_scores)}, converting to empty list")
            interval_scores = []
        elif isinstance(interval_scores, dict) and not interval_scores:
            interval_scores = []
        
        try:
            cleaned_interval_scores = clean_interval_scores_safe(interval_scores)
        except Exception as interval_error:
            logger.error(f"❌ Error cleaning interval_scores: {str(interval_error)}")
            cleaned_interval_scores = []
        
        # MA 결과 정리 및 필요시 재계산
        ma_results = prediction_state['latest_ma_results'] or {}
        cleaned_ma_results = {}
        
        # 이동평균 결과가 없거나 비어있다면 재계산 시도
        if not ma_results or len(ma_results) == 0:
            logger.info("🔄 MA results missing, attempting to recalculate...")
            try:
                # 현재 데이터와 예측 결과를 사용하여 이동평균 재계산
                current_date = prediction_state.get('current_date')
                if current_date and prediction_state.get('latest_file_path'):
                    # 원본 데이터 로드
                    df = load_data(prediction_state['latest_file_path'])
                    if df is not None and not df.empty:
                        # 현재 날짜를 datetime으로 변환
                        if isinstance(current_date, str):
                            current_date_dt = pd.to_datetime(current_date)
                        else:
                            current_date_dt = current_date
                        
                        # 과거 데이터 추출
                        historical_data = df[df.index <= current_date_dt].copy()
                        
                        # 예측 데이터를 이동평균 계산용으로 변환
                        ma_input_data = []
                        for pred in raw_predictions:
                            try:
                                ma_item = {
                                    'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                                    'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                                    'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                                }
                                ma_input_data.append(ma_item)
                            except Exception as e:
                                logger.warning(f"⚠️ Error processing MA data item: {str(e)}")
                                continue
                        
                        # 이동평균 계산
                        if ma_input_data:
                            ma_results = calculate_moving_averages_with_history(
                                ma_input_data, historical_data, target_col='MOPJ'
                            )
                            if ma_results:
                                logger.info(f"✅ MA recalculated successfully with {len(ma_results)} windows")
                                prediction_state['latest_ma_results'] = ma_results
                            else:
                                logger.warning("⚠️ MA recalculation returned empty results")
                        else:
                            logger.warning("⚠️ No valid input data for MA calculation")
                    else:
                        logger.warning("⚠️ Unable to load original data for MA calculation")
                else:
                    logger.warning("⚠️ Missing current_date or file_path for MA calculation")
            except Exception as e:
                logger.error(f"❌ Error recalculating MA: {str(e)}")
        
        # MA 결과 정리
        for key, value in ma_results.items():
            if isinstance(value, list):
                cleaned_ma_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for k, v in item.items():
                            cleaned_item[k] = safe_serialize_value(v)
                        cleaned_ma_results[key].append(cleaned_item)
                    else:
                        cleaned_ma_results[key].append(safe_serialize_value(item))
            else:
                cleaned_ma_results[key] = safe_serialize_value(value)
        
        # 어텐션 데이터 정리
        attention_data = prediction_state['latest_attention_data']
        cleaned_attention = None
        
        logger.info(f"📊 [ATTENTION] Processing attention data: available={bool(attention_data)}")
        if attention_data:
            logger.info(f"📊 [ATTENTION] Original keys: {list(attention_data.keys())}")
            
            cleaned_attention = {}
            for key, value in attention_data.items():
                if key == 'image' and value:
                    cleaned_attention[key] = value  # base64 이미지는 그대로
                    logger.info(f"📊 [ATTENTION] Image data preserved (length: {len(value) if isinstance(value, str) else 'N/A'})")
                elif isinstance(value, dict):
                    cleaned_attention[key] = {}
                    for k, v in value.items():
                        cleaned_attention[key][k] = safe_serialize_value(v)
                    logger.info(f"📊 [ATTENTION] Dict processed for key '{key}': {len(cleaned_attention[key])} items")
                else:
                    cleaned_attention[key] = safe_serialize_value(value)
                    logger.info(f"📊 [ATTENTION] Value processed for key '{key}': {type(value)}")
            
            logger.info(f"📊 [ATTENTION] Final cleaned keys: {list(cleaned_attention.keys())}")
        else:
            logger.warning(f"📊 [ATTENTION] No attention data available in prediction_state")
        
        # 플롯 데이터 정리
        plots = prediction_state['latest_plots'] or {}
        cleaned_plots = {}
        for key, value in plots.items():
            if isinstance(value, dict):
                cleaned_plots[key] = {}
                for k, v in value.items():
                    if k == 'image' and v:
                        cleaned_plots[key][k] = v  # base64 이미지는 그대로
                    else:
                        cleaned_plots[key][k] = safe_serialize_value(v)
            else:
                cleaned_plots[key] = safe_serialize_value(value)
        
        response_data = {
            'success': True,
            'current_date': safe_serialize_value(prediction_state['current_date']),
            'predictions': compatible_predictions,  # 호환성 유지된 형태
            'interval_scores': cleaned_interval_scores,
            'ma_results': cleaned_ma_results,
            'attention_data': cleaned_attention,
            'plots': cleaned_plots,
            'metrics': cleaned_metrics if cleaned_metrics else None,
            'selected_features': prediction_state['selected_features'] or [],
            'feature_importance': safe_serialize_value(prediction_state.get('feature_importance')),
            'semimonthly_period': safe_serialize_value(prediction_state['semimonthly_period']),
            'next_semimonthly_period': safe_serialize_value(prediction_state['next_semimonthly_period'])
        }
        
        # 🔧 강화된 JSON 직렬화 테스트
        try:
            test_json = json.dumps(response_data)
            # 직렬화된 JSON에 NaN이 포함되어 있는지 추가 확인
            if 'NaN' in test_json or 'Infinity' in test_json:
                logger.error(f"JSON contains NaN/Infinity values")
                # NaN 값들을 모두 null로 교체
                test_json_cleaned = test_json.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
                response_data = json.loads(test_json_cleaned)
            logger.info(f"JSON serialization test: SUCCESS (length: {len(test_json)})")
        except Exception as json_error:
            logger.error(f"JSON serialization test: FAILED - {str(json_error)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            
            # 응급 처치: 모든 숫자 필드를 문자열로 변환 시도
            try:
                logger.info("Attempting emergency data cleaning...")
                cleaned_predictions = []
                for pred in compatible_predictions:
                    cleaned_pred = {}
                    for k, v in pred.items():
                        if isinstance(v, (int, float)):
                            if pd.isna(v) or np.isnan(v) or np.isinf(v):
                                cleaned_pred[k] = None
                            else:
                                cleaned_pred[k] = float(v)
                        else:
                            cleaned_pred[k] = safe_serialize_value(v)
                    cleaned_predictions.append(cleaned_pred)
                
                response_data['predictions'] = cleaned_predictions
                
                # 재시도
                test_json = json.dumps(response_data)
                logger.info("Emergency cleaning successful")
            except Exception as emergency_error:
                logger.error(f"Emergency cleaning failed: {str(emergency_error)}")
                return jsonify({
                    'success': False,
                    'error': f'Data serialization error: {str(json_error)}'
                }), 500
        
        logger.info(f"=== Compatible Response Summary ===")
        logger.info(f"Total predictions: {len(compatible_predictions)}")
        logger.info(f"Has metrics: {cleaned_metrics is not None}")
        logger.info(f"Sample prediction fields: {list(compatible_predictions[0].keys()) if compatible_predictions else 'None'}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error creating compatible response: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error creating response: {str(e)}'}), 500

@app.route('/api/results/attention-map', methods=['GET'])
def get_attention_map():
    """어텐션 맵 데이터 조회 API"""
    global prediction_state
    
    logger.info("🔍 [ATTENTION_MAP] API call received - FINAL UPDATE")
    
    # 어텐션 데이터 확인
    attention_data = prediction_state.get('latest_attention_data')
    
    # 테스트용: 데이터가 없으면 더미 데이터 생성
    test_mode = request.args.get('test', '').lower() == 'true'
    
    if not attention_data:
        if test_mode:
            logger.info("🧪 [ATTENTION_MAP] Creating test data")
            attention_data = {
                'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
                'feature_importance': {
                    'Feature_1': 0.35,
                    'Feature_2': 0.25,
                    'Feature_3': 0.20,
                    'Feature_4': 0.15,
                    'Feature_5': 0.05
                },
                'temporal_importance': {
                    '2024-01-01': 0.1,
                    '2024-01-02': 0.2,
                    '2024-01-03': 0.3,
                    '2024-01-04': 0.4
                }
            }
        else:
            logger.warning("⚠️ [ATTENTION_MAP] No attention data available")
            return jsonify({
                'error': 'No attention map data available',
                'message': '예측을 먼저 실행해주세요. 예측 완료 후 어텐션 맵 데이터가 생성됩니다.',
                'suggestion': 'CSV 파일을 업로드하고 예측을 실행한 후 다시 시도해주세요.',
                'test_url': '/api/results/attention-map?test=true'
            }), 404
    
    logger.info(f"📊 [ATTENTION_MAP] Available keys: {list(attention_data.keys())}")
    
    # 어텐션 데이터 정리 및 직렬화
    cleaned_attention = {}
    
    try:
        for key, value in attention_data.items():
            if key == 'image' and value:
                cleaned_attention[key] = value  # base64 이미지는 그대로
                logger.info(f"📊 [ATTENTION_MAP] Image data preserved (length: {len(value) if isinstance(value, str) else 'N/A'})")
            elif isinstance(value, dict):
                cleaned_attention[key] = {}
                for k, v in value.items():
                    cleaned_attention[key][k] = safe_serialize_value(v)
                logger.info(f"📊 [ATTENTION_MAP] Dict processed for key '{key}': {len(cleaned_attention[key])} items")
            else:
                cleaned_attention[key] = safe_serialize_value(value)
                logger.info(f"📊 [ATTENTION_MAP] Value processed for key '{key}': {type(value)}")
        
        response_data = {
            'success': True,
            'attention_data': cleaned_attention,
            'current_date': safe_serialize_value(prediction_state.get('current_date')),
            'feature_importance': safe_serialize_value(prediction_state.get('feature_importance'))
        }
        
        # JSON 직렬화 테스트
        json.dumps(response_data)
        
        logger.info(f"✅ [ATTENTION_MAP] Response ready with keys: {list(cleaned_attention.keys())}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"💥 [ATTENTION_MAP] Error processing attention data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing attention map: {str(e)}'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """선택된 특성 조회 API"""
    global prediction_state
    
    if prediction_state['selected_features'] is None:
        return jsonify({'error': 'No feature information available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'selected_features': prediction_state['selected_features'],
        'feature_importance': prediction_state['feature_importance']
    })

# 정적 파일 제공
@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(os.path.join('static', path))

# 기본 라우트
@app.route('/')
def index():
    return jsonify({
        'app': 'MOPJ Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            '/api/health',
            '/api/upload',
            '/api/holidays',
            '/api/holidays/upload',
            '/api/holidays/reload',
            '/api/file/metadata',
            '/api/data/dates',
            '/api/predict',
            '/api/predict/accumulated',
            '/api/predict/status',
            '/api/results',
            '/api/results/predictions',
            '/api/results/interval-scores',
            '/api/results/moving-averages',
            '/api/results/attention-map',
            '/api/results/accumulated',
            '/api/results/accumulated/interval-scores',
            '/api/results/accumulated/<date>',
            '/api/results/accumulated/report',
            '/api/results/accumulated/visualization',
            '/api/results/reliability',  # 새로 추가된 신뢰도 API
            '/api/features'
        ],
        'new_features': [
            'Prediction consistency scoring (예측 신뢰도)',
            'Purchase reliability percentage (구매 신뢰도)',
            'Holiday management system',
            'Accumulated predictions analysis'
        ]
    })

# 4. API 엔드포인트 추가 - 누적 예측 시작
@app.route('/api/predict/accumulated', methods=['POST'])
def start_accumulated_prediction():
    """여러 날짜에 대한 누적 예측 시작 API (저장/로드 기능 포함)"""
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    save_to_csv = data.get('save_to_csv', True)
    use_saved_data = data.get('use_saved_data', True)  # 저장된 데이터 활용 여부
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not start_date:
        return jsonify({'error': 'Start date is required'}), 400
    
    # 백그라운드에서 누적 예측 실행
    thread = Thread(target=run_accumulated_predictions_with_save, 
                   args=(filepath, start_date, end_date, save_to_csv, use_saved_data))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Accumulated prediction started',
        'save_to_csv': save_to_csv,
        'use_saved_data': use_saved_data,
        'status_url': '/api/predict/status'
    })

# 5. API 엔드포인트 추가 - 누적 예측 결과 조회
@app.route('/api/results/accumulated', methods=['GET'])
def get_accumulated_results():
    global prediction_state
    
    logger.info("🔍 [ACCUMULATED] API call received")
    
    if prediction_state['is_predicting']:
        logger.warning("⚠️ [ACCUMULATED] Prediction still in progress")
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409

    if not prediction_state['accumulated_predictions']:
        logger.error("❌ [ACCUMULATED] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404

    logger.info("✅ [ACCUMULATED] Processing accumulated predictions...")
    
    # 누적 구매 신뢰도 계산 - 올바른 방식 사용
    accumulated_purchase_reliability, _ = calculate_accumulated_purchase_reliability(
        prediction_state['accumulated_predictions']
    )
    
    logger.info(f"💰 [ACCUMULATED] Purchase reliability calculated: {accumulated_purchase_reliability}")
    
    # ✅ 상세 디버깅 로깅 추가
    logger.info(f"🔍 [ACCUMULATED] Purchase reliability debugging:")
    logger.info(f"   - Type: {type(accumulated_purchase_reliability)}")
    logger.info(f"   - Value: {accumulated_purchase_reliability}")
    logger.info(f"   - Repr: {repr(accumulated_purchase_reliability)}")
    if accumulated_purchase_reliability == 100.0:
        logger.warning(f"⚠️ [ACCUMULATED] 100% reliability detected! Detailed analysis:")
        logger.warning(f"   - Total predictions: {len(prediction_state['accumulated_predictions'])}")
        for i, pred in enumerate(prediction_state['accumulated_predictions'][:3]):  # 처음 3개만
            logger.warning(f"   - Prediction {i+1}: date={pred.get('date')}, interval_scores_keys={list(pred.get('interval_scores', {}).keys())}")
    
    # 데이터 안전성 검사
    safe_interval_scores = []
    if prediction_state.get('accumulated_interval_scores'):
        safe_interval_scores = [
            item for item in prediction_state['accumulated_interval_scores'] 
            if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
        ]
        logger.info(f"📊 [ACCUMULATED] Safe interval scores count: {len(safe_interval_scores)}")
    else:
        logger.warning("⚠️ [ACCUMULATED] No accumulated_interval_scores found")
    
    consistency_scores = prediction_state.get('accumulated_consistency_scores', {})
    logger.info(f"🎯 [ACCUMULATED] Consistency scores keys: {list(consistency_scores.keys())}")
    
    # ✅ 캐시 통계 정보 추가
    cache_stats = prediction_state.get('cache_statistics', {
        'total_dates': 0,
        'cached_dates': 0,
        'new_predictions': 0,
        'cache_hit_rate': 0.0
    })
    
    response_data = {
        'success': True,
        'prediction_dates': prediction_state.get('prediction_dates', []),
        'accumulated_metrics': prediction_state.get('accumulated_metrics', {}),
        'predictions': prediction_state['accumulated_predictions'],
        'accumulated_interval_scores': safe_interval_scores,
        'accumulated_consistency_scores': consistency_scores,
        'accumulated_purchase_reliability': accumulated_purchase_reliability,
        'cache_statistics': cache_stats  # ✅ 캐시 통계 추가
    }
    
    # ✅ 최종 응답 데이터 검증 로깅
    logger.info(f"📤 [ACCUMULATED] Final response validation:")
    logger.info(f"   - accumulated_purchase_reliability in response: {response_data['accumulated_purchase_reliability']}")
    logger.info(f"   - Type in response: {type(response_data['accumulated_purchase_reliability'])}")
    
    logger.info(f"📤 [ACCUMULATED] Response summary: predictions={len(response_data['predictions'])}, metrics_keys={list(response_data['accumulated_metrics'].keys())}, reliability={response_data['accumulated_purchase_reliability']}")
    
    return jsonify(response_data)

@app.route('/api/results/accumulated/interval-scores', methods=['GET'])
def get_accumulated_interval_scores():
    global prediction_state
    scores = prediction_state.get('accumulated_interval_scores', [])
    
    # 'days' 속성이 없는 항목 필터링
    safe_scores = [
        item for item in scores 
        if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
    ]
    
    return jsonify(safe_scores)

# 7. 누적 보고서 API 엔드포인트
@app.route('/api/results/accumulated/report', methods=['GET'])
def get_accumulated_report():
    """누적 예측 결과 보고서 생성 및 다운로드 API"""
    global prediction_state
    
    # 예측 결과가 없는 경우
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    report_file = generate_accumulated_report()
    if not report_file:
        return jsonify({'error': 'Failed to generate report'}), 500
    
    return send_file(report_file, as_attachment=True)

def return_prediction_result(pred, date, match_type):
    """
    예측 결과를 API 응답 형식으로 반환하는 헬퍼 함수
    
    Parameters:
    -----------
    pred : dict
        예측 결과 딕셔너리
    date : str
        요청된 날짜
    match_type : str
        매칭 방식 설명
    
    Returns:
    --------
    JSON response
    """
    try:
        logger.info(f"🔄 [API] Returning prediction result for date={date}, match_type={match_type}")
        
        # 예측 데이터 안전하게 추출
        predictions = pred.get('predictions', [])
        if not isinstance(predictions, list):
            logger.warning(f"⚠️ [API] predictions is not a list: {type(predictions)}")
            predictions = []
        
        # 구간 점수 안전하게 추출 및 변환
        interval_scores = pred.get('interval_scores', {})
        if isinstance(interval_scores, dict):
            # 딕셔너리를 리스트로 변환
            interval_scores_list = []
            for key, interval in interval_scores.items():
                if interval and isinstance(interval, dict) and 'days' in interval:
                    interval_scores_list.append(interval)
            interval_scores = interval_scores_list
        elif not isinstance(interval_scores, list):
            logger.warning(f"⚠️ [API] interval_scores is neither dict nor list: {type(interval_scores)}")
            interval_scores = []
        
        # 메트릭 안전하게 추출
        metrics = pred.get('metrics', {})
        if not isinstance(metrics, dict):
            logger.warning(f"⚠️ [API] metrics is not a dict: {type(metrics)}")
            metrics = {}
        
        # 🔄 이동평균 데이터 추출 (캐시된 데이터 또는 파일에서 로드)
        ma_results = pred.get('ma_results', {})
        if not ma_results:
            # 파일별 캐시에서 MA 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                
                if ma_file_path.exists():
                    with open(ma_file_path, 'r', encoding='utf-8') as f:
                        ma_results = json.load(f)
                    logger.info(f"📊 [API] MA results loaded from file for {date}: {len(ma_results)} windows")
                else:
                    logger.info(f"⚠️ [API] No MA file found for {date}: {ma_file_path}")
                    
                    # 파일이 없으면 예측 데이터에서 재계산 (히스토리컬 데이터 없이 제한적으로)
                    if predictions:
                        ma_results = calculate_moving_averages_with_history(
                            predictions, None, target_col='MOPJ', windows=[5, 10, 23]
                        )
                        logger.info(f"📊 [API] MA results recalculated for {date}: {len(ma_results)} windows")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading/calculating MA for {date}: {str(e)}")
                ma_results = {}
        
        # 🎯 Attention 데이터 추출
        attention_data = pred.get('attention_data', {})
        if not attention_data:
            # 파일별 캐시에서 Attention 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                
                if attention_file_path.exists():
                    with open(attention_file_path, 'r', encoding='utf-8') as f:
                        attention_data = json.load(f)
                    logger.info(f"📊 [API] Attention data loaded from file for {date}")
                else:
                    logger.info(f"⚠️ [API] No attention file found for {date}: {attention_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading attention data for {date}: {str(e)}")
        
        # 기본 응답 데이터 구성
        response_data = {
            'success': True,
            'date': date,
            'predictions': predictions,
            'interval_scores': interval_scores,
            'metrics': metrics,
            'ma_results': ma_results,
            'attention_data': attention_data,
            'next_semimonthly_period': pred.get('next_semimonthly_period'),
            'actual_business_days': pred.get('actual_business_days'),
            'match_type': match_type,
            'data_end_date': pred.get('date'),
            'prediction_start_date': pred.get('prediction_start_date')
        }
        
        # 각 필드를 개별적으로 안전하게 직렬화
        safe_response = {}
        for key, value in response_data.items():
            safe_value = safe_serialize_value(value)
            if safe_value is not None:  # None이 아닌 경우에만 추가
                safe_response[key] = safe_value
        
        # success와 date는 항상 포함
        safe_response['success'] = True
        safe_response['date'] = date
        
        logger.info(f"✅ [API] Successfully prepared response for {date}: predictions={len(safe_response.get('predictions', []))}, interval_scores={len(safe_response.get('interval_scores', []))}, ma_windows={len(safe_response.get('ma_results', {}))}, attention_data={bool(safe_response.get('attention_data'))}")
        
        return jsonify(safe_response)
        
    except Exception as e:
        logger.error(f"💥 [API] Error in return_prediction_result for {date}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Error processing prediction result: {str(e)}',
            'date': date
        }), 500

# 8. API 엔드포인트 추가 - 특정 날짜 예측 결과 조회

@app.route('/api/results/accumulated/<date>', methods=['GET'])
def get_accumulated_result_by_date(date):
    """특정 날짜의 누적 예측 결과 조회 API"""
    global prediction_state
    
    logger.info(f"🔍 [API] Searching for accumulated result by date: {date}")
    
    if not prediction_state['accumulated_predictions']:
        logger.warning("❌ [API] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    logger.info(f"📊 [API] Available prediction dates (data_end_date): {[p['date'] for p in prediction_state['accumulated_predictions']]}")
    
    # ✅ 1단계: 정확한 데이터 기준일 매칭 우선 확인
    logger.info(f"🔍 [API] Step 1: Looking for EXACT data_end_date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}")
        
        if data_end_date == date:
            logger.info(f"✅ [API] Found prediction by EXACT DATA END DATE match: {date}")
            logger.info(f"📊 [API] Prediction data preview: predictions={len(pred.get('predictions', []))}, interval_scores={len(pred.get('interval_scores', {}))}")
            return return_prediction_result(pred, date, "exact data end date")
    
    # ✅ 2단계: 정확한 매칭이 없으면 계산된 예측 시작일로 매칭
    logger.info(f"🔍 [API] Step 2: No exact match found. Looking for calculated prediction start date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        prediction_start_date = pred.get('prediction_start_date')  # 예측 시작일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}, prediction_start_date={prediction_start_date}")
        
        if data_end_date:
            try:
                data_end_dt = pd.to_datetime(data_end_date)
                calculated_start_date = data_end_dt + pd.Timedelta(days=1)
                
                # 주말과 휴일 건너뛰기
                while calculated_start_date.weekday() >= 5 or is_holiday(calculated_start_date):
                    calculated_start_date += pd.Timedelta(days=1)
                
                calculated_start_str = calculated_start_date.strftime('%Y-%m-%d')
                
                if calculated_start_str == date:
                    logger.info(f"✅ [API] Found prediction by CALCULATED PREDICTION START DATE: {date} (from data end date: {data_end_date})")
                    return return_prediction_result(pred, date, "calculated prediction start date from data end date")
                    
            except Exception as e:
                logger.warning(f"⚠️ [API] Error calculating prediction start date for {data_end_date}: {str(e)}")
                continue
    
    logger.error(f"❌ [API] No prediction results found for date {date}")
    return jsonify({'error': f'No prediction results for date {date}'}), 404

# 10. 누적 지표 시각화 API 엔드포인트
@app.route('/api/results/accumulated/visualization', methods=['GET'])
def get_accumulated_visualization():
    """누적 예측 지표 시각화 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    filename, img_str = visualize_accumulated_metrics()
    if not filename:
        return jsonify({'error': 'Failed to generate visualization'}), 500
    
    return jsonify({
        'success': True,
        'file_path': filename,
        'image': img_str
    })

# 새로운 API 엔드포인트 추가
@app.route('/api/results/reliability', methods=['GET'])
def get_reliability_scores():
    """신뢰도 점수 조회 API"""
    global prediction_state
    
    # 단일 예측 신뢰도
    single_reliability = {}
    if prediction_state.get('latest_interval_scores') and prediction_state.get('latest_predictions'):
        try:
            # 실제 영업일 수 계산
            actual_business_days = len([p for p in prediction_state['latest_predictions'] 
                                       if p.get('Date') and not p.get('is_synthetic', False)])
            
            single_reliability = {
                'period': prediction_state['next_semimonthly_period']
            }
        except Exception as e:
            logger.error(f"Error calculating single prediction reliability: {str(e)}")
            single_reliability = {'error': 'Unable to calculate single prediction reliability'}
    
    # 누적 예측 신뢰도 (안전한 접근)
    accumulated_reliability = prediction_state.get('accumulated_consistency_scores', {})
    
    return jsonify({
        'success': True,
        'single_prediction_reliability': single_reliability,
        'accumulated_prediction_reliability': accumulated_reliability
    })

@app.route('/api/cache/clear/accumulated', methods=['POST'])
def clear_accumulated_cache():
    """누적 예측 캐시 클리어"""
    global prediction_state
    
    try:
        # 누적 예측 관련 상태 클리어
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['accumulated_interval_scores'] = []
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['accumulated_purchase_reliability'] = 0
        prediction_state['prediction_dates'] = []
        
        logger.info("🧹 [CACHE] Accumulated prediction cache cleared")
        
        return jsonify({
            'success': True,
            'message': 'Accumulated prediction cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE] Error clearing accumulated cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/reliability', methods=['GET'])
def debug_reliability_calculation():
    """구매 신뢰도 계산 디버깅 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    predictions = prediction_state['accumulated_predictions']
    print(f"🔍 [DEBUG] Total predictions: {len(predictions)}")
    
    debug_data = {
        'prediction_count': len(predictions),
        'predictions_details': []
    }
    
    total_score = 0
    
    for i, pred in enumerate(predictions):
        pred_date = pred.get('date')
        interval_scores = pred.get('interval_scores', {})
        
        print(f"📊 [DEBUG] Prediction {i+1} ({pred_date}):")
        print(f"   - interval_scores type: {type(interval_scores)}")
        print(f"   - interval_scores keys: {list(interval_scores.keys()) if isinstance(interval_scores, dict) else 'N/A'}")
        
        pred_detail = {
            'date': pred_date,
            'interval_scores_type': str(type(interval_scores)),
            'interval_scores_keys': list(interval_scores.keys()) if isinstance(interval_scores, dict) else [],
            'individual_scores': [],
            'best_score': 0
        }
        
        if isinstance(interval_scores, dict):
            for key, score_data in interval_scores.items():
                print(f"   - {key}: {score_data}")
                if isinstance(score_data, dict) and 'score' in score_data:
                    score_value = score_data.get('score', 0)
                    pred_detail['individual_scores'].append({
                        'key': key,
                        'score': score_value,
                        'full_data': score_data
                    })
                    print(f"     -> score: {score_value}")
        
        if pred_detail['individual_scores']:
            best_score = max([s['score'] for s in pred_detail['individual_scores']])
            # 점수를 3점으로 제한
            capped_score = min(best_score, 3.0)
            pred_detail['best_score'] = best_score
            pred_detail['capped_score'] = capped_score
            total_score += capped_score
            print(f"   - Best score: {best_score:.1f}, Capped score: {capped_score:.1f}")
        
        debug_data['predictions_details'].append(pred_detail)
    
    max_possible_score = len(predictions) * 3
    reliability = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    debug_data.update({
        'total_score': total_score,
        'max_possible_score': max_possible_score,
        'reliability_percentage': reliability
    })
    
    print(f"🎯 [DEBUG] CALCULATION SUMMARY:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Total score: {total_score}")
    print(f"   - Max possible score: {max_possible_score}")
    print(f"   - Reliability: {reliability:.1f}%")
    
    return jsonify(debug_data)

@app.route('/api/cache/check', methods=['POST'])
def check_cached_predictions():
    """누적 예측 범위에서 캐시된 예측이 얼마나 있는지 확인"""
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date are required'}), 400
    
    try:
        logger.info(f"🔍 [CACHE_CHECK] Checking cache availability for {start_date} to {end_date}")
        
        # 저장된 예측 확인
        cached_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        # 전체 범위 계산 (데이터 기준일 기준)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 사용 가능한 날짜 계산 (데이터 기준일)
        available_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # 영업일만 포함 (주말과 휴일 제외)
            if current_dt.weekday() < 5 and not is_holiday(current_dt):
                available_dates.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += pd.Timedelta(days=1)
        
        # 캐시된 날짜 목록
        cached_dates = [pred['date'] for pred in cached_predictions]
        missing_dates = [date for date in available_dates if date not in cached_dates]
        
        cache_percentage = round(len(cached_predictions) / max(len(available_dates), 1) * 100, 1)
        
        logger.info(f"📊 [CACHE_CHECK] Cache status: {len(cached_predictions)}/{len(available_dates)} ({cache_percentage}%)")
        
        return jsonify({
            'success': True,
            'total_dates_in_range': len(available_dates),
            'cached_predictions': len(cached_predictions),
            'cached_dates': cached_dates,
            'missing_dates': missing_dates,
            'cache_percentage': cache_percentage,
            'will_use_cache': len(cached_predictions) > 0,
            'estimated_time_savings': f"약 {len(cached_predictions) * 3}분 절약 예상" if len(cached_predictions) > 0 else "없음"
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE_CHECK] Error checking cached predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/accumulated/recent', methods=['GET'])
def get_recent_accumulated_results():
    """
    페이지 로드 시 최근 누적 예측 결과를 자동으로 복원하는 API
    """
    try:
        # 저장된 예측 목록 조회 (최근 것부터)
        predictions_list = get_saved_predictions_list(limit=50)
        
        if not predictions_list:
            return jsonify({
                'success': False, 
                'message': 'No saved predictions found',
                'has_recent_results': False
            })
        
        # 날짜별로 그룹화하여 연속된 범위 찾기
        dates_by_groups = {}
        for pred in predictions_list:
            data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
            if data_end_date:
                date_obj = pd.to_datetime(data_end_date)
                # 주차별로 그룹화 (같은 주의 예측들을 하나의 범위로 간주)
                week_key = date_obj.strftime('%Y-W%U')
                if week_key not in dates_by_groups:
                    dates_by_groups[week_key] = []
                dates_by_groups[week_key].append({
                    'date': data_end_date,
                    'date_obj': date_obj,
                    'pred_info': pred
                })
        
        # 가장 최근 그룹 선택
        if not dates_by_groups:
            return jsonify({
                'success': False, 
                'message': 'No valid date groups found',
                'has_recent_results': False
            })
        
        # 최근 주의 예측들 가져오기
        latest_week = max(dates_by_groups.keys())
        latest_group = dates_by_groups[latest_week]
        latest_group.sort(key=lambda x: x['date_obj'])
        
        # 연속된 날짜 범위 찾기
        start_date = latest_group[0]['date_obj']
        end_date = latest_group[-1]['date_obj']
        
        logger.info(f"🔄 [AUTO_RESTORE] Found recent accumulated predictions: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 기존 캐시에서 누적 결과 로드
        loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        if not loaded_predictions:
            return jsonify({
                'success': False, 
                'message': 'Failed to load cached predictions',
                'has_recent_results': False
            })
        
        # 누적 메트릭 계산
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }
        
        for pred in loaded_predictions:
            metrics = pred.get('metrics', {})
            if isinstance(metrics, dict):
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
        
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count
        
        # 구간 점수 계산
        accumulated_interval_scores = {}
        for pred in loaded_predictions:
            interval_scores = pred.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1
        
        # 정렬된 구간 점수 리스트
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)
        
        # 구매 신뢰도 계산
        accumulated_purchase_reliability, _ = calculate_accumulated_purchase_reliability(loaded_predictions)
        
        # 일관성 점수 계산
        unique_periods = set()
        for pred in loaded_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(loaded_predictions, period)
                accumulated_consistency_scores[period] = consistency_data
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")
        
        # 캐시 통계
        cache_statistics = {
            'total_dates': len(loaded_predictions),
            'cached_dates': len(loaded_predictions),
            'new_predictions': 0,
            'cache_hit_rate': 100.0
        }
        
        # 전역 상태 업데이트 (선택적)
        global prediction_state
        prediction_state['accumulated_predictions'] = loaded_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in loaded_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['cache_statistics'] = cache_statistics
        
        logger.info(f"✅ [AUTO_RESTORE] Successfully restored {len(loaded_predictions)} accumulated predictions")
        
        return jsonify({
            'success': True,
            'has_recent_results': True,
            'restored_range': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'prediction_count': len(loaded_predictions)
            },
            'prediction_dates': [p['date'] for p in loaded_predictions],
            'accumulated_metrics': accumulated_metrics,
            'predictions': loaded_predictions,
            'accumulated_interval_scores': accumulated_scores_list,
            'accumulated_consistency_scores': accumulated_consistency_scores,
            'accumulated_purchase_reliability': accumulated_purchase_reliability,
            'cache_statistics': cache_statistics,
            'message': f"최근 누적 예측 결과를 자동으로 복원했습니다 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
        })
        
    except Exception as e:
        logger.error(f"❌ [AUTO_RESTORE] Error restoring recent accumulated results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e),
            'has_recent_results': False
        }), 500

@app.route('/api/cache/rebuild-index', methods=['POST'])
def rebuild_predictions_index_api():
    """예측 인덱스 재생성 API (rebuild_index.py 기능을 통합)"""
    try:
        # 현재 파일의 캐시 디렉토리 가져오기
        current_file = prediction_state.get('current_file')
        if not current_file:
            return jsonify({'success': False, 'error': '현재 업로드된 파일이 없습니다. 먼저 파일을 업로드해주세요.'})
        
        # 🔧 새로운 rebuild 함수 사용
        success = rebuild_predictions_index_from_existing_files()
        
        if success:
            cache_dirs = get_file_cache_dirs(current_file)
            index_file = cache_dirs['predictions'] / 'predictions_index.csv'
            
            # 결과 데이터 읽기
            index_data = []
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    index_data = list(reader)
            
            return jsonify({
                'success': True,
                'message': f'인덱스 파일을 성공적으로 재생성했습니다. ({len(index_data)}개 항목)',
                'file_location': str(index_file),
                'entries_count': len(index_data),
                'rebuilt_entries': [{'date': row.get('prediction_start_date', ''), 'data_end': row.get('data_end_date', '')} for row in index_data]
            })
        else:
            return jsonify({
                'success': False,
                'error': '인덱스 재생성에 실패했습니다. 로그를 확인해주세요.'
            })
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'인덱스 재생성 중 오류 발생: {str(e)}'})

@app.route('/api/cache/clear/semimonthly', methods=['POST'])
def clear_semimonthly_cache():
    """특정 반월 기간의 캐시만 삭제하는 API"""
    try:
        data = request.json
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date parameter is required'}), 400
        
        target_date = pd.to_datetime(target_date)
        target_semimonthly = get_semimonthly_period(target_date)
        
        logger.info(f"🗑️ [API] Clearing cache for semimonthly period: {target_semimonthly}")
        
        # 현재 파일의 캐시 디렉토리에서 해당 반월 캐시 삭제
        cache_dirs = get_file_cache_dirs()
        predictions_dir = cache_dirs['predictions']
        
        deleted_files = []
        
        if predictions_dir.exists():
            # 메타 파일 확인하여 반월 기간이 일치하는 캐시 삭제
            for meta_file in predictions_dir.glob("*_meta.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    cached_data_end_date = meta_data.get('data_end_date')
                    if cached_data_end_date:
                        cached_data_end_date = pd.to_datetime(cached_data_end_date)
                        cached_semimonthly = get_semimonthly_period(cached_data_end_date)
                        
                        if cached_semimonthly == target_semimonthly:
                            # 관련 파일들 삭제
                            base_name = meta_file.stem.replace('_meta', '')
                            files_to_delete = [
                                meta_file,
                                meta_file.parent / f"{base_name}.csv",
                                meta_file.parent / f"{base_name}_attention.json",
                                meta_file.parent / f"{base_name}_ma.json"
                            ]
                            
                            for file_path in files_to_delete:
                                if file_path.exists():
                                    file_path.unlink()
                                    deleted_files.append(str(file_path.name))
                                    logger.info(f"  🗑️ Deleted: {file_path.name}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error processing meta file {meta_file}: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'message': f'Cache cleared for semimonthly period: {target_semimonthly}',
            'target_semimonthly': target_semimonthly,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'deleted_files': deleted_files,
            'deleted_count': len(deleted_files)
        })
        
    except Exception as e:
        logger.error(f"❌ [API] Error clearing semimonthly cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

#######################################################################
# VARMAX 예측 저장/로드 시스템
#######################################################################

def save_varmax_prediction(prediction_results: dict, prediction_date):
    """
    VARMAX 예측 결과를 파일에 저장하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX prediction save")
            return False
            
        # 파일별 캐시 디렉토리 가져오기
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        varmax_dir.mkdir(exist_ok=True)
        
        # 저장할 파일 경로
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        
        # JSON으로 직렬화 가능한 형태로 변환
        clean_results = {}
        for key, value in prediction_results.items():
            try:
                clean_results[key] = safe_serialize_value(value)
            except Exception as e:
                logger.warning(f"Failed to serialize {key}: {e}")
                continue
        
        # 메타데이터 추가
        clean_results['metadata'] = {
            'prediction_date': prediction_date,
            'created_at': datetime.now().isoformat(),
            'file_path': file_path,
            'model_type': 'VARMAX'
        }
        
        # 파일에 저장
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        # 인덱스 업데이트
        update_varmax_predictions_index({
            'prediction_date': prediction_date,
            'file_path': str(prediction_file),
            'created_at': datetime.now().isoformat(),
            'original_file': file_path
        })
        
        logger.info(f"✅ VARMAX prediction saved: {prediction_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save VARMAX prediction: {e}")
        logger.error(traceback.format_exc())
        return False

def load_varmax_prediction(prediction_date):
    """
    저장된 VARMAX 예측 결과를 로드하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX prediction load")
            return None
            
        # 파일별 캐시 디렉토리 가져오기
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        
        # 로드할 파일 경로
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        
        if not prediction_file.exists():
            logger.info(f"VARMAX prediction file not found: {prediction_file}")
            return None
            
        # 파일에서 로드
        with open(prediction_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 🔍 로드된 데이터 타입 및 구조 확인
        logger.info(f"🔍 [VARMAX_LOAD] Loaded data type: {type(results)}")
        if isinstance(results, dict):
            logger.info(f"🔍 [VARMAX_LOAD] Loaded data keys: {list(results.keys())}")
            
            # 🔧 ma_results 필드 타입 확인 및 수정
            if 'ma_results' in results:
                ma_results = results['ma_results']
                logger.info(f"🔍 [VARMAX_LOAD] MA results type: {type(ma_results)}")
                
                if isinstance(ma_results, str):
                    logger.warning(f"⚠️ [VARMAX_LOAD] MA results is string, attempting to parse as JSON...")
                    try:
                        results['ma_results'] = json.loads(ma_results)
                        logger.info(f"🔧 [VARMAX_LOAD] Successfully parsed ma_results from string to dict")
                    except Exception as e:
                        logger.error(f"❌ [VARMAX_LOAD] Failed to parse ma_results string as JSON: {e}")
                        results['ma_results'] = {}
                elif not isinstance(ma_results, dict):
                    logger.warning(f"⚠️ [VARMAX_LOAD] MA results has unexpected type: {type(ma_results)}, setting empty dict")
                    results['ma_results'] = {}
                    
        elif isinstance(results, str):
            logger.warning(f"⚠️ [VARMAX_LOAD] Loaded data is string, not dict: {results[:100]}...")
            # 문자열인 경우 다시 JSON 파싱 시도
            try:
                results = json.loads(results)
                logger.info(f"🔧 [VARMAX_LOAD] Re-parsed string as JSON: {type(results)}")
            except:
                logger.error(f"❌ [VARMAX_LOAD] Failed to re-parse string as JSON")
                return None
        else:
            logger.warning(f"⚠️ [VARMAX_LOAD] Unexpected data type: {type(results)}")
        
        logger.info(f"✅ VARMAX prediction loaded: {prediction_file}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to load VARMAX prediction: {e}")
        logger.error(traceback.format_exc())
        return None

def update_varmax_predictions_index(metadata):
    """
    VARMAX 예측 인덱스를 업데이트하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            return False
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        varmax_dir.mkdir(exist_ok=True)
        
        index_file = varmax_dir / 'varmax_index.json'
        
        # 기존 인덱스 로드
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'predictions': []}
        
        # 새 예측 추가 (중복 제거)
        prediction_date = metadata['prediction_date']
        index['predictions'] = [p for p in index['predictions'] if p['prediction_date'] != prediction_date]
        index['predictions'].append(metadata)
        
        # 날짜순 정렬 (최신순)
        index['predictions'].sort(key=lambda x: x['prediction_date'], reverse=True)
        
        # 최대 100개 유지
        index['predictions'] = index['predictions'][:100]
        
        # 인덱스 저장
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update VARMAX predictions index: {e}")
        return False

def get_saved_varmax_predictions_list(limit=100):
    """
    저장된 VARMAX 예측 목록을 가져오는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            logger.warning("No current file path for VARMAX predictions list")
            return []
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        index_file = varmax_dir / 'varmax_index.json'
        
        if not index_file.exists():
            return []
            
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        predictions = index.get('predictions', [])[:limit]
        
        logger.info(f"✅ Found {len(predictions)} saved VARMAX predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Failed to get saved VARMAX predictions list: {e}")
        return []

def delete_saved_varmax_prediction(prediction_date):
    """
    저장된 VARMAX 예측을 삭제하는 함수
    """
    try:
        file_path = prediction_state.get('current_file', None)
        if not file_path:
            return False
            
        cache_dirs = get_file_cache_dirs(file_path)
        varmax_dir = cache_dirs['root'] / 'varmax'
        
        # 예측 파일 삭제
        prediction_file = varmax_dir / f"varmax_prediction_{prediction_date}.json"
        if prediction_file.exists():
            prediction_file.unlink()
        
        # 인덱스에서 제거
        index_file = varmax_dir / 'varmax_index.json'
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            index['predictions'] = [p for p in index['predictions'] if p['prediction_date'] != prediction_date]
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ VARMAX prediction deleted: {prediction_date}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete VARMAX prediction: {e}")
        return False

#######################################################################
# VARMAX 관련 유틸리티 함수
#######################################################################

def varmax_decision(file_path):
    """Varmax 의사결정 관련"""
    fp = pd.read_csv(file_path)
    df = pd.DataFrame(fp, columns=fp.columns)
    col = df.columns
    # 1) 분석에 사용할 변수 리스트
    vars_pct = ['max_pct2', 'min_pct2', 'mean_pct2', 'max_pct3', 'min_pct3', 'mean_pct3']
    logger.info(f'데이터프레임{df}')
    rename_dict = {
    'max_pct2': '[현 반월 최대 증가율]',
    'min_pct2': '[현 반월 최대 감소율]',
    'mean_pct2': '[현 반월 평균 변동률]',
    'max_pct3': '[이전 반월 최대 증가율]',
    'min_pct3': '[이전 반월 최대 감소율]',
    'mean_pct3': '[이전 반월 평균 변동률]'
    }
    rename_col = list(rename_dict.values())
    df = df.rename(columns=rename_dict)
    logger.info(f'열{col}')
    # 2) Case 정의
    case1 = df['saving_rate'] < 0
    abs_thresh = df['saving_rate'].abs().quantile(0.9)
    case2 = df['saving_rate'].abs() >= abs_thresh

    # 3) 최적 조건 탐색 함수
    def find_best_condition(df, case_mask, var):
        best = None
        for direction in ['greater', 'less']:
            for p in np.linspace(0.1, 0.9, 9):
                th = df[var].quantile(p)
                if direction == 'greater':
                    mask = df[var] > th
                else:
                    mask = df[var] < th
                # 샘플 수가 너무 적은 경우 제외
                if mask.sum() < 5:
                    continue
                prop = case_mask[mask].mean()
                if best is None or prop > best[4]:
                    best = (direction, p, th, mask.sum(), prop)
        return best

    # 5) 각 변수별 최적 조건 찾기
    results_case1 = {var: find_best_condition(df, case1, var) for var in rename_col}
    results_case2 = {var: find_best_condition(df, case2, var) for var in rename_col}

    from itertools import combinations
    # 6) 두 변수 조합을 사용하여 saving_rate < 0 분류 성능 평가 (샘플 수 ≥ 30)
    combi_results_case1 = []

    for var1, var2 in combinations(rename_col, 2):
        for d1 in ['greater', 'less']:
            for d2 in ['greater', 'less']:
                for p1 in np.linspace(0.1, 0.9, 9):
                    for p2 in np.linspace(0.1, 0.9, 9):
                        th1 = df[var1].quantile(p1)
                        th2 = df[var2].quantile(p2)
                        mask1 = df[var1] > th1 if d1 == 'greater' else df[var1] < th1
                        mask2 = df[var2] > th2 if d2 == 'greater' else df[var2] < th2
                        mask = mask1 & mask2
                        n = mask.sum()
                        if n < 30:
                            continue
                        rate = case1[mask].mean()
                        combi_results_case1.append({
                            "조건1": f"{var1} { '>' if d1 == 'greater' else '<' } {round(th1, 3)}%",
                            "조건2": f"{var2} { '>' if d2 == 'greater' else '<' } {round(th2, 3)}%",
                            "샘플 수": n,
                            "음수 비율 [%]": round(rate*100, 3)
                        })
    column_order1 = ["조건1", "조건2", "샘플 수", "음수 비율 [%]"]
    combi_df_case1 = pd.DataFrame(combi_results_case1).sort_values(by="음수 비율 [%]", ascending=False)
    combi_df_case1 = combi_df_case1.reindex(columns=column_order1)

    # 7) 두 변수 조합을 사용하여 절댓값 상위 10% 분류 성능 평가
    combi_results_case2 = []

    for var1, var2 in combinations(rename_col, 2):
        for d1 in ['greater', 'less']:
            for d2 in ['greater', 'less']:
                for p1 in np.linspace(0.1, 0.9, 9):
                    for p2 in np.linspace(0.1, 0.9, 9):
                        th1 = df[var1].quantile(p1)
                        th2 = df[var2].quantile(p2)
                        mask1 = df[var1] > th1 if d1 == 'greater' else df[var1] < th1
                        mask2 = df[var2] > th2 if d2 == 'greater' else df[var2] < th2
                        mask = mask1 & mask2
                        n = mask.sum()
                        if n < 30:
                            continue
                        rate = case2[mask].mean()
                        combi_results_case2.append({
                            "조건1": f"{var1} { '>' if d1 == 'greater' else '<' } {round(th1, 3)}%",
                            "조건2": f"{var2} { '>' if d2 == 'greater' else '<' } {round(th2, 3)}%",
                            "샘플 수": n,
                            "상위 변동성 확률 [%]": round(rate*100, 3)
                        })
    column_order2 = ["조건1", "조건2", "샘플 수", "상위 변동성 확률 [%]"]
    combi_df_case2 = pd.DataFrame(combi_results_case2).sort_values(by="상위 변동성 확률 [%]", ascending=False)
    combi_df_case2 = combi_df_case2.reindex(columns=column_order2)
    return {
        'case_1': combi_df_case1.to_dict(orient='records'),
        'case_2': combi_df_case2.to_dict(orient='records')
    }

def background_varmax_prediction(file_path, current_date, pred_days, use_cache=True):
    """백그라운드에서 VARMAX 예측 작업을 수행하는 함수"""
    global prediction_state
    
    try:
        # 일관된 예측 결과를 위한 시드 고정
        set_seed()
        # 현재 파일 상태 업데이트
        prediction_state['current_file'] = file_path
        
        # 🔍 기존 저장된 예측 확인 (use_cache=True인 경우)
        if use_cache:
            logger.info(f"🔍 [VARMAX_CACHE] Checking for existing prediction for date: {current_date}")
            existing_prediction = load_varmax_prediction(current_date)
            
            if existing_prediction:
                logger.info(f"✅ [VARMAX_CACHE] Found existing VARMAX prediction for {current_date}")
                logger.info(f"🔍 [VARMAX_CACHE] Cached data keys: {list(existing_prediction.keys())}")
                logger.info(f"🔍 [VARMAX_CACHE] MA results available: {bool(existing_prediction.get('ma_results'))}")
                ma_results = existing_prediction.get('ma_results')
                if ma_results:
                    logger.info(f"🔍 [VARMAX_CACHE] MA results type: {type(ma_results)}")
                    if isinstance(ma_results, dict):
                        logger.info(f"🔍 [VARMAX_CACHE] MA results keys: {list(ma_results.keys())}")
                    else:
                        logger.warning(f"⚠️ [VARMAX_CACHE] MA results is not a dict: {type(ma_results)}")
                
                # 🔑 상태 복원 (순차적으로)
                logger.info(f"🔄 [VARMAX_CACHE] Restoring state from cached prediction...")
                
                # 기존 예측 결과를 상태에 로드 (안전한 타입 검사)
                prediction_state['varmax_predictions'] = existing_prediction.get('predictions', [])
                prediction_state['varmax_half_month_averages'] = existing_prediction.get('half_month_averages', [])
                prediction_state['varmax_metrics'] = existing_prediction.get('metrics', {})
                
                # MA results 안전한 로드
                ma_results = existing_prediction.get('ma_results', {})
                if isinstance(ma_results, dict):
                    prediction_state['varmax_ma_results'] = ma_results
                else:
                    logger.warning(f"⚠️ [VARMAX_CACHE] Invalid ma_results type: {type(ma_results)}, setting empty dict")
                    prediction_state['varmax_ma_results'] = {}
                
                prediction_state['varmax_selected_features'] = existing_prediction.get('selected_features', [])
                prediction_state['varmax_current_date'] = existing_prediction.get('current_date', current_date)
                prediction_state['varmax_model_info'] = existing_prediction.get('model_info', {})
                prediction_state['varmax_plots'] = existing_prediction.get('plots', {})
                
                # 즉시 완료 상태로 설정
                prediction_state['varmax_is_predicting'] = False
                prediction_state['varmax_prediction_progress'] = 100
                prediction_state['varmax_error'] = None
                
                logger.info(f"✅ [VARMAX_CACHE] State restoration completed")
                
                logger.info(f"✅ [VARMAX_CACHE] Successfully loaded existing prediction for {current_date}")
                logger.info(f"🔍 [VARMAX_CACHE] State restored - predictions: {len(prediction_state['varmax_predictions'])}, MA results: {len(prediction_state['varmax_ma_results'])}")
                
                # 🔍 최종 검증
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - is_predicting: {prediction_state.get('varmax_is_predicting')}")
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - predictions count: {len(prediction_state.get('varmax_predictions', []))}")
                logger.info(f"🔍 [VARMAX_CACHE] Final verification - ma_results count: {len(prediction_state.get('varmax_ma_results', {}))}")
                
                # 🛡️ 상태 안정화를 위한 짧은 대기
                import time
                time.sleep(1.0)
                
                logger.info(f"🎯 [VARMAX_CACHE] Cache loading process completed for {current_date}")
                return
        
        # 🚀 새로운 예측 수행
        logger.info(f"🚀 [VARMAX_NEW] Starting new VARMAX prediction for {current_date}")
        forecaster = VARMAXSemiMonthlyForecaster(file_path, pred_days=pred_days)
        prediction_state['varmax_is_predicting'] = True
        prediction_state['varmax_prediction_progress'] = 10
        prediction_state['varmax_prediction_start_time'] = time.time()  # VARMAX 시작 시간 기록
        prediction_state['varmax_error'] = None
        
        # VARMAX 예측 수행
        prediction_state['varmax_prediction_progress'] = 30
        logger.info(f"🔄 [VARMAX_NEW] Starting prediction generation (30% complete)")
        
        try:
            min_index = 1 # 임시 인덱스
            logger.info(f"🔄 [VARMAX_NEW] Calling generate_predictions_varmax with current_date={current_date}, var_num={min_index+2}")
            
            # 예측 진행률을 30%로 설정 (모델 초기화 완료)
            prediction_state['varmax_prediction_progress'] = 30

            mape_list=[]
            for var_num in range(2,8):
                mape_value = forecaster.generate_variables_varmax(current_date, var_num)
                mape_list.append(mape_value)
            min_index = mape_list.index(min(mape_list))
            logger.info(f"Var {min_index+2} model is selected, MAPE:{mape_list[min_index]}%")
            
            results = forecaster.generate_predictions_varmax(current_date, min_index+2)
            logger.info(f"✅ [VARMAX_NEW] Prediction generation completed successfully")
            
            # 최종 진행률 95%로 설정 (시각화 생성 전)
            prediction_state['varmax_prediction_progress'] = 95
            
        except Exception as prediction_error:
            logger.error(f"❌ [VARMAX_NEW] Error during prediction generation: {str(prediction_error)}")
            logger.error(f"❌ [VARMAX_NEW] Prediction error traceback: {traceback.format_exc()}")
            
            # 예측 실패 상태로 설정
            prediction_state['varmax_error'] = f"Prediction generation failed: {str(prediction_error)}"
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            logger.error(f"❌ [VARMAX_NEW] Prediction state reset due to error")
            return
        
        if results['success']:
            logger.info(f"🔄 [VARMAX_NEW] Updating state with new prediction results...")
            
            # 상태에 결과 저장 (기존 LSTM 결과와 분리)
            prediction_state['varmax_predictions'] = results['predictions']
            prediction_state['varmax_half_month_averages'] = results.get('half_month_averages', [])
            prediction_state['varmax_metrics'] = results['metrics']
            prediction_state['varmax_ma_results'] = results['ma_results']
            prediction_state['varmax_selected_features'] = results['selected_features']
            prediction_state['varmax_current_date'] = results['current_date']
            prediction_state['varmax_model_info'] = results['model_info']
            
            # 시각화 생성 (기존 app.py 방식 활용)
            plots_info = create_varmax_visualizations(results)
            prediction_state['varmax_plots'] = plots_info
            
            prediction_state['varmax_prediction_progress'] = 100
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_error'] = None
            
            # VARMAX 예측 결과 저장
            save_varmax_prediction(results, current_date)
            
            logger.info("✅ [VARMAX_NEW] Prediction completed successfully")
            logger.info(f"🔍 [VARMAX_NEW] Final state - predictions: {len(prediction_state['varmax_predictions'])}")
        else:
            prediction_state['varmax_error'] = results['error']
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            
    except Exception as e:
        logger.error(f"❌ [VARMAX_BG] Error in background VARMAX prediction: {str(e)}")
        logger.error(f"❌ [VARMAX_BG] Full traceback: {traceback.format_exc()}")
        
        # 에러 상태로 설정하고 자세한 로깅
        prediction_state['varmax_error'] = f"Background prediction failed: {str(e)}"
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 0
        
        logger.error(f"❌ [VARMAX_BG] VARMAX prediction failed completely. Current state reset.")
        logger.error(f"❌ [VARMAX_BG] Error type: {type(e).__name__}")
        logger.error(f"❌ [VARMAX_BG] Error details: {str(e)}")
        
        # 에러 발생 시 모든 VARMAX 관련 상태 초기화
        prediction_state['varmax_predictions'] = []
        prediction_state['varmax_metrics'] = {}
        prediction_state['varmax_ma_results'] = {}
        prediction_state['varmax_selected_features'] = []
        prediction_state['varmax_current_date'] = None
        prediction_state['varmax_model_info'] = {}
        prediction_state['varmax_plots'] = {}
        prediction_state['varmax_half_month_averages'] = []

def plot_varmax_prediction_basic(sequence_df, sequence_start_date, start_day_value, 
                                f1, accuracy, mape, weighted_score, 
                                save_prefix=None, title_prefix="VARMAX Semi-monthly Prediction", file_path=None):
    """VARMAX 기본 예측 그래프 시각화 (기존 plot_prediction_basic과 동일한 스타일)"""
    try:
        logger.info(f"Creating VARMAX prediction graph for {sequence_start_date}")
        
        # 파일별 캐시 디렉토리 가져오기
        if save_prefix is None:
            cache_dirs = get_file_cache_dirs(file_path)
            save_prefix = cache_dirs['plots']
        
        # 예측값만 있는 데이터 처리
        pred_df = sequence_df.dropna(subset=['Prediction'])
        valid_df = sequence_df.dropna(subset=['Actual']) if 'Actual' in sequence_df.columns else pd.DataFrame()
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 제목 설정
        main_title = f"{title_prefix} - {sequence_start_date}"
        subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score:.2f}%"
        
        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # 상단: 예측 vs 실제 (있는 경우)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("VARMAX Long-term Prediction")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 예측값 플롯
        ax1.plot(pred_df['Date'], pred_df['Prediction'],
                marker='o', color='red', label='VARMAX Predicted', linewidth=2)
        
        # 실제값 플롯 (있는 경우)
        if len(valid_df) > 0:
            ax1.plot(valid_df['Date'], valid_df['Actual'],
                    marker='o', color='blue', label='Actual', linewidth=2)
            
            # 방향성 일치 여부 배경 색칠
            for i in range(1, len(valid_df)):
                if i < len(pred_df):
                    actual_dir = np.sign(valid_df['Actual'].iloc[i] - valid_df['Actual'].iloc[i-1])
                    pred_dir = np.sign(pred_df['Prediction'].iloc[i] - pred_df['Prediction'].iloc[i-1])
                    color = 'blue' if actual_dir == pred_dir else 'red'
                    ax1.axvspan(valid_df['Date'].iloc[i-1], valid_df['Date'].iloc[i], alpha=0.1, color=color)
        
        ax1.set_xlabel("Date")
        ax1.set_ylabel("MOPJ Price")
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 하단: 오차 (실제값이 있는 경우만)
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if len(valid_df) > 0:
            # 오차 계산 및 플롯
            errors = valid_df['Actual'] - valid_df['Prediction']
            ax2.bar(valid_df['Date'], errors, alpha=0.7, color='orange', width=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            ax2.set_title(f"Prediction Error (MAE: {abs(errors).mean():.2f})")
        else:
            ax2.text(0.5, 0.5, 'No actual data for error calculation', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Prediction Error (No validation data)")
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Error")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 파일 저장
        os.makedirs(save_prefix, exist_ok=True)
        filename = f"varmax_prediction_{sequence_start_date}.png"
        filepath = os.path.join(save_prefix, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VARMAX prediction graph saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating VARMAX prediction graph: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_varmax_visualizations(results):
    """VARMAX 결과에 대한 시각화 생성"""
    try:
        # 기본 예측 그래프
        sequence_df = pd.DataFrame(results['predictions'])
        sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        metrics = results['metrics']
        current_date = results['current_date']
        start_day_value = sequence_df['Prediction'].iloc[0] if len(sequence_df) > 0 else 0
        
        # 기본 그래프
        basic_plot = plot_varmax_prediction_basic(
            sequence_df, current_date, start_day_value,
            metrics['f1'], metrics['accuracy'], metrics['mape'], metrics['weighted_score']
        )
        
        # 이동평균 그래프
        ma_plot = plot_varmax_moving_average_analysis(
            results['ma_results'], current_date
        )
        
        plots_info = {
            'basic_plot': basic_plot,
            'ma_plot': ma_plot
        }
        
        logger.info("VARMAX visualizations created successfully")
        return plots_info
        
    except Exception as e:
        logger.error(f"Error creating VARMAX visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def plot_varmax_moving_average_analysis(ma_results, sequence_start_date, save_prefix=None,
                                        title_prefix="VARMAX Moving Average Analysis", file_path=None):
    """VARMAX 이동평균 분석 그래프"""
    try:
        logger.info(f"Creating VARMAX moving average analysis for {sequence_start_date}")
        
        # 파일별 캐시 디렉토리 가져오기
        if save_prefix is None:
            cache_dirs = get_file_cache_dirs(file_path)
            save_prefix = cache_dirs['ma_plots']
        
        if not ma_results:
            logger.warning("No moving average results to plot")
            return None
        
        windows = list(ma_results.keys())
        n_windows = len(windows)
        
        if n_windows == 0:
            logger.warning("No moving average windows found")
            return None
        
        # 그래프 생성 (2x2 그리드로 최대 4개 윈도우 표시)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title_prefix} - {sequence_start_date}", fontsize=16, weight='bold')
        axes = axes.flatten()
        
        for i, window in enumerate(windows[:4]):  # 최대 4개까지만
            ax = axes[i]
            ma_data = ma_results[window]
            
            if not ma_data:
                ax.text(0.5, 0.5, f'No data for {window}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{window} (No Data)")
                continue
            
            # 데이터프레임 변환
            df = pd.DataFrame(ma_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 예측값과 이동평균 플롯
            ax.plot(df['date'], df['prediction'], marker='o', color='red', 
                   label='Prediction', linewidth=2, markersize=4)
            ax.plot(df['date'], df['ma'], marker='s', color='blue', 
                   label=f'MA-{window.replace("ma", "")}', linewidth=2, markersize=4)
            
            # 실제값 플롯 (있는 경우)
            actual_data = df.dropna(subset=['actual'])
            if len(actual_data) > 0:
                ax.plot(actual_data['date'], actual_data['actual'], 
                       marker='^', color='green', label='Actual', linewidth=2, markersize=4)
            
            ax.set_title(f"{window.upper()} Moving Average")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 빈 subplot 숨기기
        for i in range(n_windows, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 파일 저장
        os.makedirs(save_prefix, exist_ok=True)
        filename = f"varmax_ma_analysis_{sequence_start_date}.png"
        filepath = os.path.join(save_prefix, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"VARMAX moving average analysis saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating VARMAX moving average analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return None

#######################################################################
# VARMAX API 엔드포인트
#######################################################################

# 1) VARMAX 반월별 예측 시작
@app.route('/api/varmax/predict', methods=['POST', 'OPTIONS'])
def varmax_semimonthly_predict():
    """VARMAX 반월별 예측 시작 API"""
    # 1) 먼저, OPTIONS(preflight) 요청이 들어오면 바로 200을 리턴
    if request.method == 'OPTIONS':
        # CORS(app) 로 설정해뒀으면 이미 Access-Control-Allow-Origin 등이 붙어 있을 것입니다.
        return make_response(('', 200))
    global prediction_state
    
    # 🔧 VARMAX 독립 상태 확인 - hang된 상태면 자동 리셋
    if prediction_state.get('varmax_is_predicting', False):
        current_progress = prediction_state.get('varmax_prediction_progress', 0)
        current_error = prediction_state.get('varmax_error')
        
        logger.warning(f"⚠️ [VARMAX_API] Prediction already in progress (progress: {current_progress}%, error: {current_error})")
        
        # 🔧 개선된 자동 리셋 조건: 에러가 있거나 진행률이 매우 낮은 경우만 리셋
        should_reset = False
        reset_reason = ""
        
        if current_error:
            should_reset = True
            reset_reason = f"error detected: {current_error}"
        elif current_progress > 0 and current_progress < 15:
            should_reset = True  
            reset_reason = f"very low progress stuck: {current_progress}%"
        
        if should_reset:
            logger.warning(f"🔄 [VARMAX_API] Auto-resetting stuck prediction - {reset_reason}")
            
            # 상태 리셋
            prediction_state['varmax_is_predicting'] = False
            prediction_state['varmax_prediction_progress'] = 0
            prediction_state['varmax_error'] = None
            prediction_state['varmax_predictions'] = []
            prediction_state['varmax_half_month_averages'] = []
            prediction_state['varmax_metrics'] = {}
            prediction_state['varmax_ma_results'] = {}
            prediction_state['varmax_selected_features'] = []
            prediction_state['varmax_current_date'] = None
            prediction_state['varmax_model_info'] = {}
            prediction_state['varmax_plots'] = {}
            
            logger.info(f"✅ [VARMAX_API] Stuck state auto-reset completed, proceeding with new prediction")
        else:
            # 정상적으로 진행 중인 경우 409 반환
            return jsonify({
                'success': False,
                'error': 'VARMAX prediction already in progress',
                'progress': current_progress
            }), 409
    
    data = request.get_json(force=True)
    filepath     = data.get('filepath')
    current_date = data.get('date')
    pred_days    = data.get('pred_days', 50)
    use_cache    = data.get('use_cache', True)  # 🆕 캐시 사용 여부 (기본값: True)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    if not current_date:
        return jsonify({'error': 'Date is required'}), 400
    
    logger.info(f"🚀 [VARMAX_API] Starting VARMAX prediction (use_cache={use_cache}) for {current_date}")
    
    thread = Thread(
        target=background_varmax_prediction,
        args=(filepath, current_date, pred_days, use_cache)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'VARMAX semi-monthly prediction started',
        'status_url': '/api/varmax/status',
        'use_cache': use_cache
    })

# 2) VARMAX 예측 상태 조회
@app.route('/api/varmax/status', methods=['GET'])
def varmax_prediction_status():
    """VARMAX 예측 상태 확인 API (남은 시간 추가)"""
    global prediction_state
    
    is_predicting = prediction_state.get('varmax_is_predicting', False)
    progress = prediction_state.get('varmax_prediction_progress', 0)
    error = prediction_state.get('varmax_error', None)
    
    logger.info(f"🔍 [VARMAX_STATUS] Current status - predicting: {is_predicting}, progress: {progress}%, error: {error}")
    
    status = {
        'is_predicting': is_predicting,
        'progress': progress,
        'error': error
    }
    
    # VARMAX 예측 중인 경우 남은 시간 계산
    if is_predicting and prediction_state.get('varmax_prediction_start_time'):
        time_info = calculate_estimated_time_remaining(
            prediction_state['varmax_prediction_start_time'], 
            progress
        )
        status.update(time_info)
    
    if not is_predicting and prediction_state.get('varmax_current_date'):
        status['current_date'] = prediction_state['varmax_current_date']
        logger.info(f"🔍 [VARMAX_STATUS] Prediction completed for date: {status['current_date']}")
    
    return jsonify(status)

# 3) VARMAX 전체 결과 조회
@app.route('/api/varmax/results', methods=['GET'])
def get_varmax_results():
    """VARMAX 예측 결과 조회 API"""
    global prediction_state
    
    # 🔍 상태 디버깅
    logger.info(f"🔍 [VARMAX_API] Current prediction_state keys: {list(prediction_state.keys())}")
    logger.info(f"🔍 [VARMAX_API] varmax_is_predicting: {prediction_state.get('varmax_is_predicting', 'NOT_SET')}")
    logger.info(f"🔍 [VARMAX_API] varmax_predictions available: {bool(prediction_state.get('varmax_predictions'))}")
    logger.info(f"🔍 [VARMAX_API] varmax_ma_results available: {bool(prediction_state.get('varmax_ma_results'))}")
    
    if prediction_state.get('varmax_predictions'):
        logger.info(f"🔍 [VARMAX_API] Predictions count: {len(prediction_state['varmax_predictions'])}")
    
    if prediction_state.get('varmax_ma_results'):
        logger.info(f"🔍 [VARMAX_API] MA results keys: {list(prediction_state['varmax_ma_results'].keys())}")
    
    # 🛡️ 백그라운드 스레드 완료 대기
    if prediction_state.get('varmax_is_predicting', False):
        logger.warning(f"⚠️ [VARMAX_API] Prediction still in progress: {prediction_state.get('varmax_prediction_progress', 0)}%")
        return jsonify({
            'success': False,
            'error': 'VARMAX prediction in progress',
            'progress': prediction_state.get('varmax_prediction_progress', 0)
        }), 409
    
    # 🎯 상태에 데이터가 없으면 캐시에서 직접 로드 (신뢰성 개선)
    if not prediction_state.get('varmax_predictions'):
        logger.warning(f"⚠️ [VARMAX_API] No VARMAX predictions in state, attempting direct cache load")
        logger.info(f"🔍 [VARMAX_API] Current file: {prediction_state.get('current_file')}")
        
        try:
            # 최근 저장된 VARMAX 예측 목록 가져오기
            saved_predictions = get_saved_varmax_predictions_list(limit=1)
            logger.info(f"🔍 [VARMAX_API] Found {len(saved_predictions)} saved predictions")
            
            if saved_predictions:
                latest_date = saved_predictions[0]['prediction_date']
                logger.info(f"🔧 [VARMAX_API] Loading latest prediction: {latest_date}")
                
                # 직접 로드하고 상태 복원
                cached_prediction = load_varmax_prediction(latest_date)
                if cached_prediction and cached_prediction.get('predictions'):
                    logger.info(f"✅ [VARMAX_API] Successfully loaded from cache ({len(cached_prediction.get('predictions', []))} predictions)")
                    
                    # 🔑 즉시 상태 복원 (더 안전하게)
                    prediction_state['varmax_predictions'] = cached_prediction.get('predictions', [])
                    prediction_state['varmax_half_month_averages'] = cached_prediction.get('half_month_averages', [])
                    prediction_state['varmax_metrics'] = cached_prediction.get('metrics', {})
                    prediction_state['varmax_ma_results'] = cached_prediction.get('ma_results', {})
                    prediction_state['varmax_selected_features'] = cached_prediction.get('selected_features', [])
                    prediction_state['varmax_current_date'] = cached_prediction.get('current_date')
                    prediction_state['varmax_model_info'] = cached_prediction.get('model_info', {})
                    prediction_state['varmax_plots'] = cached_prediction.get('plots', {})
                    
                    logger.info(f"🎯 [VARMAX_API] State restored from cache - {len(prediction_state['varmax_predictions'])} predictions")
                    
                    return jsonify({
                        'success': True,
                        'current_date': cached_prediction.get('current_date'),
                        'predictions': cached_prediction.get('predictions', []),
                        'half_month_averages': cached_prediction.get('half_month_averages', []),
                        'metrics': cached_prediction.get('metrics', {}),
                        'ma_results': cached_prediction.get('ma_results', {}),
                        'selected_features': cached_prediction.get('selected_features', []),
                        'model_info': cached_prediction.get('model_info', {}),
                        'plots': cached_prediction.get('plots', {})
                    })
                else:
                    logger.warning(f"⚠️ [VARMAX_API] Cached prediction is empty or invalid")
            else:
                logger.warning(f"⚠️ [VARMAX_API] No saved predictions found")
                
        except Exception as e:
            logger.error(f"❌ [VARMAX_API] Direct cache load failed: {e}")
            import traceback
            logger.error(f"❌ [VARMAX_API] Cache load traceback: {traceback.format_exc()}")
        
        # 캐시 로드도 실패한 경우 명확한 메시지
        logger.error(f"❌ [VARMAX_API] No VARMAX results available in state or cache")
        return jsonify({
            'success': False,
            'error': 'No VARMAX prediction results available. Please run a new prediction.'
        }), 404
    
    logger.info(f"✅ [VARMAX_API] Returning VARMAX results successfully from state")
    return jsonify({
        'success': True,
        'current_date':      prediction_state.get('varmax_current_date'),
        'predictions':       prediction_state.get('varmax_predictions', []),
        'half_month_averages': prediction_state.get('varmax_half_month_averages', []),
        'metrics':           prediction_state.get('varmax_metrics', {}),
        'ma_results':        prediction_state.get('varmax_ma_results', {}),
        'selected_features': prediction_state.get('varmax_selected_features', []),
        'model_info':        prediction_state.get('varmax_model_info', {}),
        'plots':             prediction_state.get('varmax_plots', {})
    })

# 4) VARMAX 예측값만 조회
@app.route('/api/varmax/predictions', methods=['GET'])
def get_varmax_predictions_only():
    """VARMAX 예측 값만 조회 API"""
    global prediction_state
    
    if not prediction_state.get('varmax_predictions'):
        return jsonify({'error': 'No VARMAX prediction results available'}), 404
    
    return jsonify({
        'success': True,
        'current_date':      prediction_state['varmax_current_date'],
        'predictions':       prediction_state['varmax_predictions'],
        'model_info':        prediction_state['varmax_model_info']
    })

# 5) VARMAX 이동평균 조회 - 즉석 계산 방식
@app.route('/api/varmax/moving-averages', methods=['GET'])
def get_varmax_moving_averages():
    """VARMAX 이동평균 조회 API - 예측 결과로 즉석 계산"""
    global prediction_state
    
    # 🎯 상태에 MA 데이터가 있으면 바로 반환
    if prediction_state.get('varmax_ma_results'):
        return jsonify({
            'success': True,
            'current_date': prediction_state['varmax_current_date'],
            'ma_results': prediction_state['varmax_ma_results']
        })
    
    # 🚀 예측 결과가 있으면 즉석에서 MA 계산
    varmax_predictions = prediction_state.get('varmax_predictions')
    current_date = prediction_state.get('varmax_current_date')
    current_file = prediction_state.get('current_file')
    
    # 상태에 예측 결과가 없으면 캐시에서 로드
    if not varmax_predictions or not current_date:
        logger.info(f"🔧 [VARMAX_MA_API] No predictions in state, loading from cache")
        try:
            saved_predictions = get_saved_varmax_predictions_list(limit=1)
            if saved_predictions:
                latest_date = saved_predictions[0]['prediction_date']
                cached_prediction = load_varmax_prediction(latest_date)
                if cached_prediction and cached_prediction.get('predictions'):
                    varmax_predictions = cached_prediction.get('predictions')
                    current_date = cached_prediction.get('current_date', latest_date)
                    # current_file은 prediction_state에서 가져오거나 추정
                    if not current_file:
                        current_file = prediction_state.get('current_file')
                    logger.info(f"✅ [VARMAX_MA_API] Loaded predictions from cache: {len(varmax_predictions)} items")
        except Exception as e:
            logger.error(f"❌ [VARMAX_MA_API] Failed to load from cache: {e}")
    
    # 예측 결과가 없으면 에러
    if not varmax_predictions or not current_date:
        return jsonify({
            'success': False,
            'error': 'No VARMAX predictions available for MA calculation'
        }), 404
    
    # 🎯 즉석에서 MA 계산
    try:
        logger.info(f"🔄 [VARMAX_MA_API] Calculating MA on-the-fly for {len(varmax_predictions)} predictions")
        
        # VARMAX 클래스 인스턴스 생성 (MA 계산용)
        if not current_file or not os.path.exists(current_file):
            logger.error(f"❌ [VARMAX_MA_API] File not found: {current_file}")
            return jsonify({
                'success': False,
                'error': 'Original data file not available for MA calculation'
            }), 404
            
        forecaster = VARMAXSemiMonthlyForecaster(current_file, pred_days=50)
        forecaster.load_data()  # 과거 데이터 로드
        
        # MA 계산
        ma_results = forecaster.calculate_moving_averages_varmax(
            varmax_predictions, 
            current_date, 
            windows=[5, 10, 20, 30]
        )
        
        logger.info(f"✅ [VARMAX_MA_API] MA calculation completed: {len(ma_results)} windows")
        
        # 상태에 저장 (다음번 요청을 위해)
        prediction_state['varmax_ma_results'] = ma_results
        
        return jsonify({
            'success': True,
            'current_date': current_date,
            'ma_results': ma_results
        })
        
    except Exception as e:
        logger.error(f"❌ [VARMAX_MA_API] MA calculation failed: {e}")
        logger.error(f"❌ [VARMAX_MA_API] Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'MA calculation failed: {str(e)}'
        }), 500

# 6) VARMAX 의사결정 조회
@app.route('/api/varmax/saved', methods=['GET'])
def get_saved_varmax_predictions():
    """저장된 VARMAX 예측 목록을 반환하는 API"""
    try:
        limit = request.args.get('limit', 100, type=int)
        predictions = get_saved_varmax_predictions_list(limit=limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Error getting saved VARMAX predictions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/varmax/saved/<date>', methods=['GET'])
def get_saved_varmax_prediction_by_date(date):
    """특정 날짜의 저장된 VARMAX 예측을 반환하는 API"""
    global prediction_state
    
    try:
        prediction = load_varmax_prediction(date)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': f'Prediction not found for date: {date}'
            }), 404
        
        # 🔍 로드된 예측 데이터 타입 확인
        logger.info(f"🔍 [VARMAX_API_LOAD] Prediction data type: {type(prediction)}")
        
        if not isinstance(prediction, dict):
            logger.error(f"❌ [VARMAX_API_LOAD] Prediction is not a dictionary: {type(prediction)}")
            return jsonify({
                'success': False,
                'error': f'Invalid prediction data format: expected dict, got {type(prediction).__name__}'
            }), 500
        
        # 🔧 백엔드 prediction_state 복원
        logger.info(f"🔄 [VARMAX_LOAD] Restoring prediction_state for date: {date}")
        logger.info(f"🔍 [VARMAX_LOAD] Available prediction keys: {list(prediction.keys())}")
        
        # VARMAX 상태 복원 (안전한 접근)
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 100
        prediction_state['varmax_error'] = None
        prediction_state['varmax_current_date'] = prediction.get('current_date', date)
        prediction_state['varmax_predictions'] = prediction.get('predictions', [])
        prediction_state['varmax_half_month_averages'] = prediction.get('half_month_averages', [])
        prediction_state['varmax_metrics'] = prediction.get('metrics', {})
        prediction_state['varmax_ma_results'] = prediction.get('ma_results', {})
        prediction_state['varmax_selected_features'] = prediction.get('selected_features', [])
        prediction_state['varmax_model_info'] = prediction.get('model_info', {})
        prediction_state['varmax_plots'] = prediction.get('plots', {})
        
        logger.info(f"✅ [VARMAX_LOAD] prediction_state restored successfully")
        logger.info(f"🔍 [VARMAX_LOAD] Restored predictions count: {len(prediction_state['varmax_predictions'])}")
        logger.info(f"🔍 [VARMAX_LOAD] MA results keys: {list(prediction_state['varmax_ma_results'].keys()) if prediction_state['varmax_ma_results'] else 'None'}")
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error getting saved VARMAX prediction for {date}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/varmax/saved/<date>', methods=['DELETE'])
def delete_saved_varmax_prediction_api(date):
    """특정 날짜의 저장된 VARMAX 예측을 삭제하는 API"""
    try:
        success = delete_saved_varmax_prediction(date)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'Failed to delete prediction for date: {date}'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Prediction for {date} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting saved VARMAX prediction for {date}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 6) VARMAX 의사결정 조회
# 7) VARMAX 상태 리셋 API (새로 추가)
@app.route('/api/varmax/reset', methods=['POST', 'OPTIONS'])
@cross_origin()
def reset_varmax_state():
    """VARMAX 예측 상태를 리셋하는 API (hang된 예측 해결용)"""
    global prediction_state
    
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        logger.info("🔄 [VARMAX_RESET] Resetting VARMAX prediction state...")
        
        # VARMAX 상태 완전 리셋
        prediction_state['varmax_is_predicting'] = False
        prediction_state['varmax_prediction_progress'] = 0
        prediction_state['varmax_error'] = None
        prediction_state['varmax_predictions'] = []
        prediction_state['varmax_half_month_averages'] = []
        prediction_state['varmax_metrics'] = {}
        prediction_state['varmax_ma_results'] = {}
        prediction_state['varmax_selected_features'] = []
        prediction_state['varmax_current_date'] = None
        prediction_state['varmax_model_info'] = {}
        prediction_state['varmax_plots'] = {}
        
        logger.info("✅ [VARMAX_RESET] VARMAX state reset completed")
        
        return jsonify({
            'success': True,
            'message': 'VARMAX state reset successfully',
            'current_state': {
                'is_predicting': prediction_state.get('varmax_is_predicting', False),
                'progress': prediction_state.get('varmax_prediction_progress', 0),
                'error': prediction_state.get('varmax_error')
            }
        })
        
    except Exception as e:
        logger.error(f"❌ [VARMAX_RESET] Error resetting VARMAX state: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to reset VARMAX state: {str(e)}'
        }), 500

@app.route('/api/varmax/decision', methods=['POST', 'OPTIONS'])
@cross_origin() 
def get_varmax_decision():
    """VARMAX 의사 결정 조회 API"""
    # 1) OPTIONS(preflight) 요청 처리
    if request.method == 'OPTIONS':
        return make_response('', 200)
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    # 파일 저장 경로 설정
    save_dir = '/path/to/models'
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성
    filepath = os.path.join(save_dir, secure_filename(file.filename))
    file.save(filepath)

    logger.info("POST /api/varmax/decision 로 진입")
    #data = request.get_json()
    #filepath = data.get('filepath')
    """# 유효성 검사
    if not filepath or not os.path.exists(os.path.normpath(filepath)):
        return jsonify({'success': False, 'error': 'Invalid file path'}), 400"""

    results = varmax_decision(filepath)
    logger.info("결과 데이터프레임 형성 완료")
    column_order1 = ["조건1", "조건2", "샘플 수", "음수 비율 [%]"]
    column_order2 = ["조건1", "조건2", "샘플 수", "상위 변동성 확률 [%]"]

    return jsonify({
        'success': True,
        'filepath': filepath,  # ← 파일 경로 추가
        'filename': file.filename,
        'columns1': column_order1,
        'columns2': column_order2,
        'case_1':      results['case_1'],
        'case_2':      results['case_2'],
    })

@app.route('/api/market-status', methods=['GET'])
def get_market_status():
    """최근 30일간의 시장 가격 데이터를 카테고리별로 반환하는 API"""
    try:
        # 파일 경로 가져오기
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'File path is required'
            }), 400
        
        # URL 디코딩 및 파일 경로 정규화 (Windows 백슬래시 처리)
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)  # URL 디코딩
        file_path = os.path.normpath(file_path)
        logger.info(f"📊 [MARKET_STATUS] Normalized file path: {file_path}")
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            logger.error(f"❌ [MARKET_STATUS] File not found: {file_path}")
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 400
        
        # 원본 데이터 직접 로드 (Date 컬럼 유지를 위해) - Excel/CSV 파일 모두 지원
        try:
            file_ext = os.path.splitext(file_path.lower())[1]
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"📊 [MARKET_STATUS] CSV data loaded: {df.shape}")
            elif file_ext in ['.xlsx', '.xls']:
                # Excel 파일의 경우 보안 문제를 고려한 안전한 로딩 사용
                df = load_data_safe(file_path, use_cache=True, use_xlwings_fallback=True)
                # 인덱스가 Date인 경우 컬럼으로 복원
                if df.index.name == 'Date':
                    df = df.reset_index()
                logger.info(f"📊 [MARKET_STATUS] Excel data loaded with security bypass: {df.shape}")
            else:
                logger.error(f"❌ [MARKET_STATUS] Unsupported file format: {file_ext}")
                return jsonify({
                    'success': False,
                    'error': f'Unsupported file format: {file_ext}. Only CSV and Excel files are supported.'
                }), 400
        except Exception as e:
            logger.error(f"❌ [MARKET_STATUS] Failed to load data file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load data file: {str(e)}'
            }), 400
        
        if df is None or df.empty:
            logger.error(f"❌ [MARKET_STATUS] No data available or empty dataframe")
            return jsonify({
                'success': False,
                'error': 'No data available'
            }), 400
        
        # 날짜 컬럼 확인 및 정렬
        logger.info(f"📊 [MARKET_STATUS] Columns in dataframe: {list(df.columns)}")
        if 'Date' not in df.columns:
            logger.error(f"❌ [MARKET_STATUS] Date column not found. Available columns: {list(df.columns)}")
            return jsonify({
                'success': False,
                'error': 'Date column not found in data'
            }), 400
        
        # 날짜로 정렬
        df = df.sort_values('Date')
        
        # 휴일 정보 로드
        holidays = get_combined_holidays(df=df)
        holiday_dates = set([h['date'] if isinstance(h, dict) else h for h in holidays])
        
        # 영업일만 필터링
        def is_business_day(date_str):
            date_obj = pd.to_datetime(date_str).date()
            weekday = date_obj.weekday()  # 0=월요일, 6=일요일
            return weekday < 5 and date_str not in holiday_dates  # 월~금 & 휴일 아님
        
        logger.info(f"📊 [MARKET_STATUS] Total rows before business day filtering: {len(df)}")
        logger.info(f"📊 [MARKET_STATUS] Holiday dates count: {len(holiday_dates)}")
        
        business_days_df = df[df['Date'].apply(is_business_day)]
        logger.info(f"📊 [MARKET_STATUS] Business days after filtering: {len(business_days_df)}")
        
        if business_days_df.empty:
            logger.error(f"❌ [MARKET_STATUS] No business days found after filtering")
            return jsonify({
                'success': False,
                'error': 'No business days found in data'
            }), 400
        
        # 최근 30일 영업일 데이터 추출
        recent_30_days = business_days_df.tail(30)

        # 카테고리별 컬럼 분류 (실제 데이터 컬럼명에 맞게 수정)
        categories = {
            '원유 가격': [
                'WTI', 'Brent', 'Dubai'
            ],
            '가솔린 가격': [
                'Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'
            ],
            '나프타 가격': [
                'MOPJ', 'MOPAG', 'MOPS', 'Europe_CIF NWE'
            ],
            'LPG 가격': [
                'C3_LPG', 'C4_LPG'
            ],
            '석유화학 제품 가격': [
                'EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 
                'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2','MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 
                'FO_HSFO 180 CST', 'MTBE_FOB Singapore'
            ]
        }
        
        # 실제 존재하는 컬럼만 필터링
        available_columns = set(recent_30_days.columns)
        filtered_categories = {}
        
        logger.info(f"📊 [MARKET_STATUS] Available columns: {sorted(available_columns)}")
        
        for category, columns in categories.items():
            existing_columns = [col for col in columns if col in available_columns]
            if existing_columns:
                filtered_categories[category] = existing_columns
                logger.info(f"📊 [MARKET_STATUS] Category '{category}': found {len(existing_columns)} columns: {existing_columns}")
            else:
                logger.warning(f"⚠️ [MARKET_STATUS] Category '{category}': no matching columns found from {columns}")
        
        if not filtered_categories:
            logger.error(f"❌ [MARKET_STATUS] No categories found! Expected columns don't match available columns")
            return jsonify({
                'success': False,
                'error': 'No matching columns found for market status categories',
                'debug_info': {
                    'available_columns': sorted(available_columns),
                    'expected_categories': categories
                }
            }), 400
        
        # 카테고리별 데이터 구성
        result = {
            'success': True,
            'date_range': {
                'start_date': recent_30_days['Date'].iloc[0],
                'end_date': recent_30_days['Date'].iloc[-1],
                'total_days': len(recent_30_days)
            },
            'categories': {}
        }
        
        for category, columns in filtered_categories.items():
            category_data = {
                'columns': columns,
                'data': []
            }
            
            for _, row in recent_30_days.iterrows():
                data_point = {
                    'date': row['Date'],
                    'values': {}
                }
                
                for col in columns:
                    if pd.notna(row[col]):
                        data_point['values'][col] = float(row[col])
                    else:
                        data_point['values'][col] = None
                
                category_data['data'].append(data_point)
            
            result['categories'][category] = category_data
        
        logger.info(f"✅ [MARKET_STATUS] Returned {len(recent_30_days)} business days data for {len(filtered_categories)} categories")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ [MARKET_STATUS] Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get market status: {str(e)}'
        }), 500

@app.route('/api/gpu-info', methods=['GET'])
def get_gpu_info():
    """GPU 및 디바이스 정보를 반환하는 API"""
    try:
        # 실시간 GPU 테스트 여부 확인
        run_test = request.args.get('test', 'false').lower() == 'true'
        
        # GPU 정보 수집
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
            'default_device': str(DEFAULT_DEVICE),
            'current_device_info': {},
            'test_performed': False,
            'test_results': {}
        }
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            # 실시간 GPU 활용률 확인 (상세 버전)
            gpu_utilization_stats = get_detailed_gpu_utilization()
            
            device_info.update({
                'gpu_count': gpu_count,
                'current_gpu_device': current_device,
                'cudnn_version': torch.backends.cudnn.version(),
                'cudnn_enabled': torch.backends.cudnn.enabled,
                'detailed_utilization': gpu_utilization_stats,
                'gpus': []
            })
            
            # 각 GPU 정보
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = gpu_props.total_memory / 1024**3
                
                # PyTorch 버전 호환성을 위한 안전한 속성 접근
                gpu_info = {
                    'device_id': i,
                    'name': getattr(gpu_props, 'name', 'Unknown GPU'),
                    'total_memory_gb': round(total, 2),
                    'allocated_memory_gb': round(allocated, 2),
                    'cached_memory_gb': round(cached, 2),
                    'memory_usage_percent': round((allocated / total) * 100, 2),
                    'compute_capability': f"{getattr(gpu_props, 'major', 0)}.{getattr(gpu_props, 'minor', 0)}",
                    'is_current': i == current_device
                }
                
                # 선택적 속성들 (PyTorch 버전에 따라 존재하지 않을 수 있음)
                if hasattr(gpu_props, 'multiprocessor_count'):
                    gpu_info['multiprocessor_count'] = gpu_props.multiprocessor_count
                elif hasattr(gpu_props, 'multi_processor_count'):
                    gpu_info['multiprocessor_count'] = gpu_props.multi_processor_count
                else:
                    gpu_info['multiprocessor_count'] = 'N/A'
                
                # 추가 GPU 속성들 (존재하는 경우에만)
                optional_attrs = {
                    'max_threads_per_block': 'max_threads_per_block',
                    'max_threads_per_multiprocessor': 'max_threads_per_multiprocessor',
                    'warp_size': 'warp_size',
                    'memory_clock_rate': 'memory_clock_rate'
                }
                
                for attr_name, prop_name in optional_attrs.items():
                    if hasattr(gpu_props, prop_name):
                        gpu_info[attr_name] = getattr(gpu_props, prop_name)
                
                device_info['gpus'].append(gpu_info)
            
            # 현재 디바이스 상세 정보
            current_gpu_props = torch.cuda.get_device_properties(current_device)
            device_info['current_device_info'] = {
                'name': current_gpu_props.name,
                'total_memory_gb': round(current_gpu_props.total_memory / 1024**3, 2),
                'allocated_memory_gb': round(torch.cuda.memory_allocated(current_device) / 1024**3, 2),
                'cached_memory_gb': round(torch.cuda.memory_reserved(current_device) / 1024**3, 2)
            }
            
            # GPU 테스트 수행 (요청된 경우)
            if run_test:
                try:
                    logger.info("🧪 API에서 GPU 테스트 수행 중...")
                    
                    # 테스트 전 메모리 상태
                    memory_before = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    # 간단한 GPU 연산 테스트
                    test_size = 500
                    test_tensor = torch.randn(test_size, test_size, device=current_device, dtype=torch.float32)
                    test_result = torch.matmul(test_tensor, test_tensor.T)
                    computation_result = torch.sum(test_result).item()
                    
                    # 테스트 후 메모리 상태
                    memory_after = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    # 메모리 사용량 차이 계산
                    memory_diff = {
                        'allocated_diff': memory_after['allocated'] - memory_before['allocated'],
                        'cached_diff': memory_after['cached'] - memory_before['cached']
                    }
                    
                    device_info['test_performed'] = True
                    device_info['test_results'] = {
                        'test_tensor_size': f"{test_size}x{test_size}",
                        'computation_result': round(computation_result, 4),
                        'memory_before_gb': {
                            'allocated': round(memory_before['allocated'], 4),
                            'cached': round(memory_before['cached'], 4)
                        },
                        'memory_after_gb': {
                            'allocated': round(memory_after['allocated'], 4),
                            'cached': round(memory_after['cached'], 4)
                        },
                        'memory_diff_gb': {
                            'allocated': round(memory_diff['allocated_diff'], 4),
                            'cached': round(memory_diff['cached_diff'], 4)
                        },
                        'test_success': True
                    }
                    
                    # 테스트 텐서 정리
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    
                    # 정리 후 메모리 상태
                    memory_final = {
                        'allocated': torch.cuda.memory_allocated(current_device) / 1024**3,
                        'cached': torch.cuda.memory_reserved(current_device) / 1024**3
                    }
                    
                    device_info['test_results']['memory_after_cleanup_gb'] = {
                        'allocated': round(memory_final['allocated'], 4),
                        'cached': round(memory_final['cached'], 4)
                    }
                    
                    logger.info(f"✅ GPU 테스트 완료: 메모리 사용량 변화 {memory_diff['allocated_diff']:.4f}GB")
                    
                except Exception as test_e:
                    logger.error(f"❌ GPU 테스트 실패: {str(test_e)}")
                    device_info['test_performed'] = True
                    device_info['test_results'] = {
                        'test_success': False,
                        'error': str(test_e)
                    }
        else:
            device_info.update({
                'gpu_count': 0,
                'reason': 'CUDA not available - using CPU'
            })
        
        # 로그에도 정보 출력
        logger.info(f"🔍 GPU Info API 호출:")
        logger.info(f"  🔧 CUDA 사용 가능: {device_info['cuda_available']}")
        logger.info(f"  ⚡ 기본 디바이스: {device_info['default_device']}")
        if device_info['cuda_available']:
            logger.info(f"  🎮 GPU 개수: {device_info.get('gpu_count', 0)}")
            if 'current_gpu_device' in device_info:
                logger.info(f"  🎯 현재 GPU: {device_info['current_gpu_device']}")
        
        # 테스트 결과 로깅
        if device_info.get('test_performed', False):
            test_results = device_info.get('test_results', {})
            if test_results.get('test_success', False):
                logger.info(f"  ✅ GPU 테스트 성공")
            else:
                logger.warning(f"  ❌ GPU 테스트 실패: {test_results.get('error', 'Unknown error')}")
        
        return jsonify({
            'success': True,
            'device_info': device_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ GPU 정보 API 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get GPU info: {str(e)}'
        }), 500

@app.route('/api/gpu-monitoring-comparison', methods=['GET'])
def get_gpu_monitoring_comparison():
    """다양한 GPU 모니터링 방법을 비교하는 API"""
    try:
        comparison_data = compare_gpu_monitoring_methods()
        
        # 추가적인 설명 정보
        explanation = {
            'why_different_readings': [
                "Windows 작업 관리자는 주로 3D 그래픽 엔진 활용률을 표시합니다",
                "nvidia-smi는 CUDA 연산 활용률을 측정하므로 ML/AI 작업에 더 정확합니다",
                "측정 시점의 차이로 인해 순간적인 값이 다를 수 있습니다",
                "GPU는 여러 엔진(Compute, 3D, Encoder, Decoder)을 가지고 있어 각각 다른 활용률을 보입니다"
            ],
            'recommendations': [
                "ML/AI 작업: nvidia-smi의 GPU 활용률 확인",
                "게임/3D 렌더링: Windows 작업 관리자의 3D 활용률 확인", 
                "비디오 처리: nvidia-smi의 Encoder/Decoder 활용률 확인",
                "메모리 사용량: PyTorch CUDA 정보와 nvidia-smi 모두 확인"
            ],
            'task_manager_vs_nvidia_smi': {
                "작업 관리자 GPU": "주로 3D 그래픽 워크로드 (DirectX, OpenGL)",
                "nvidia-smi GPU": "CUDA 연산 워크로드 (ML, AI, GPGPU)",
                "왜 다른가": "서로 다른 GPU 엔진을 측정하기 때문",
                "어느 것이 정확한가": "작업 유형에 따라 다름 - ML/AI는 nvidia-smi가 정확"
            }
        }
        
        # 현재 상황 분석
        current_analysis = {
            'status': 'monitoring_successful',
            'notes': []
        }
        
        if comparison_data.get('nvidia_smi'):
            nvidia_util = comparison_data['nvidia_smi'].get('gpu_utilization', '0')
            try:
                util_value = float(nvidia_util)
                if util_value < 10:
                    current_analysis['notes'].append(f"현재 CUDA 활용률이 매우 낮습니다 ({util_value}%)")
                    current_analysis['notes'].append("이는 정상적일 수 있습니다 - ML 작업이 진행 중이 아닐 때")
                elif util_value > 50:
                    current_analysis['notes'].append(f"현재 CUDA 활용률이 높습니다 ({util_value}%)")
                    current_analysis['notes'].append("ML/AI 작업이 활발히 진행 중입니다")
            except:
                pass
        
        if comparison_data.get('torch_cuda'):
            memory_usage = comparison_data['torch_cuda'].get('memory_usage_percent', 0)
            if memory_usage > 1:
                current_analysis['notes'].append(f"PyTorch가 GPU 메모리를 사용 중입니다 ({memory_usage:.1f}%)")
            else:
                current_analysis['notes'].append("PyTorch가 현재 GPU 메모리를 거의 사용하지 않습니다")
        
        return jsonify({
            'success': True,
            'comparison_data': comparison_data,
            'explanation': explanation,
            'current_analysis': current_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ GPU 모니터링 비교 API 오류: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to compare GPU monitoring methods: {str(e)}'
        }), 500

# 메인 실행 부분 업데이트
if __name__ == '__main__':
    # 필요한 패키지 설치 확인
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 패키지가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 위해 https://inthiswork.com/archives/226539설치가 필요합니다.")
        logger.warning("pip install optuna 명령으로 설치할 수 있습니다.")
    
    # 🎯 파일별 캐시 시스템 - 레거시 디렉토리 및 인덱스 파일 생성 제거
    # 모든 데이터는 이제 파일별 캐시 디렉토리에 저장됩니다
    logger.info("🚀 Starting with file-based cache system - no legacy directories needed")
    
    # 라우트 등록 확인을 위한 디버깅
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule} {list(rule.methods)}")
    
    print("Starting Flask app with attention-map endpoint...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
