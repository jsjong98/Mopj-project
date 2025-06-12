# MOPJ 가격 예측 시스템 - 백엔드

## 📋 개요
LSTM 및 VARMAX 기반 모델을 사용한 MOPJ(일본 나프타 시장) 가격 예측 시스템의 백엔드 API 서버입니다.

## 🔧 환경 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU에서도 동작)
- 최소 8GB RAM 권장

## 📦 설치 방법

### 1. 가상환경 생성 (권장)
```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 의존성 패키지 설치
```bash
# requirements.txt에서 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. CUDA 설정 (GPU 사용 시)
```bash
# CUDA 지원 PyTorch 설치 (GPU 가속)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 실행 방법

### 개발 모드
```bash
python app.py
```

### 프로덕션 모드 (Gunicorn 사용)
```bash
# Gunicorn 설치
pip install gunicorn

# 실행
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 주요 기능

### 1. 파일 업로드 및 처리
- CSV 파일 업로드 지원
- 자동 데이터 검증 및 전처리
- 파일별 캐시 시스템

### 2. 예측 기능
- **LSTM 예측**: 딥러닝 기반 단일/누적 예측
  - 단일 예측: 특정 날짜부터 23일간 예측
  - 누적 예측: 여러 날짜에 대한 배치 예측
- **VARMAX 예측**: 시계열 통계 모델 기반 예측
  - 반월별 예측 (15일 단위)
  - 다변량 시계열 분석
  - 구매 의사결정 지원
- 실시간 예측 진행 상황 추적

### 3. 모델 최적화
- Optuna 기반 하이퍼파라미터 자동 최적화
- K-Fold 교차검증
- 반월별 기간 기반 학습

### 4. 시각화 및 분석
- Attention 메커니즘 시각화 (LSTM)
- 이동평균 분석 (LSTM & VARMAX)
- 성능 지표 계산 (F1 Score, MAPE, 정확도)
- 구매 구간 추천 분석

### 5. 캐시 시스템
- 파일별 독립적 캐시 관리
- 예측 결과 자동 저장/로드
- 캐시 인덱스 시스템

## 🗂️ 디렉토리 구조
```
backend/
├── app.py                 # 메인 Flask 애플리케이션
├── requirements.txt       # Python 의존성 패키지
├── README.md             # 이 파일
├── uploads/              # 업로드된 CSV 파일
├── cache/               # 캐시된 예측 결과
│   └── {file_hash}_{filename}/
│       ├── predictions/  # LSTM 예측 결과 파일
│       ├── varmax/      # VARMAX 예측 결과 파일
│       ├── models/      # 저장된 모델
│       ├── accumulated/ # 누적 예측 결과
│       └── static/      # 생성된 그래프
├── predictions/         # 레거시 예측 파일
├── static/              # 정적 파일 (그래프 이미지)
├── temp/               # 임시 파일
├── holidays/           # 휴일 데이터
└── holidays.json       # 기본 휴일 데이터
```

## 🔍 API 엔드포인트

### 파일 관리
- `POST /api/upload` - CSV 파일 업로드
- `GET /api/file/metadata` - 파일 메타데이터 조회
- `GET /api/data/dates` - 예측 가능한 날짜 목록

### LSTM 예측
- `POST /api/predict` - 단일 예측 시작
- `POST /api/predict/accumulated` - 누적 예측 시작
- `GET /api/predict/status` - 예측 진행 상황 조회

### VARMAX 예측
- `POST /api/varmax/predict` - VARMAX 예측 시작
- `GET /api/varmax/status` - VARMAX 예측 진행 상황 조회
- `POST /api/varmax/decision` - 구매 의사결정 분석

### 결과 조회
- `GET /api/results` - LSTM 단일 예측 결과
- `GET /api/results/accumulated` - LSTM 누적 예측 결과
- `GET /api/varmax/results` - VARMAX 예측 결과
- `GET /api/varmax/predictions` - VARMAX 예측 데이터만 조회
- `GET /api/varmax/moving-averages` - VARMAX 이동평균 분석

### 저장된 예측 관리
- `GET /api/predictions/saved` - 저장된 LSTM 예측 목록
- `GET /api/varmax/saved` - 저장된 VARMAX 예측 목록
- `DELETE /api/predictions/saved/<date>` - LSTM 예측 삭제
- `DELETE /api/varmax/saved/<date>` - VARMAX 예측 삭제

### 캐시 관리
- `POST /api/cache/check` - 캐시 상태 확인
- `POST /api/cache/clear/accumulated` - 누적 예측 캐시 삭제
- `POST /api/cache/clear/semimonthly` - 반월별 캐시 삭제
- `POST /api/cache/rebuild-index` - 캐시 인덱스 재구성

## ⚙️ 환경 설정

### 기본 설정
```python
# app.py에서 설정 가능한 주요 파라미터
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = 'cache'
STATIC_FOLDER = 'static'
TEMP_FOLDER = 'temp'

# LSTM 모델 파라미터
SEQUENCE_LENGTH = 30      # 입력 시퀀스 길이
PREDICT_WINDOW = 23       # 예측 윈도우 크기
HIDDEN_SIZE = 128         # LSTM 히든 크기
NUM_LAYERS = 3            # LSTM 레이어 수

# VARMAX 모델 파라미터
PRED_DAYS = 50           # VARMAX 예측 일수
```

### GPU 메모리 설정
```python
# CUDA 메모리 최적화 (app.py 내)
torch.cuda.empty_cache()
```

## 🐛 문제 해결

### 1. 패키지 설치 오류
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 정리 후 재설치
pip cache purge
pip install -r requirements.txt
```

### 2. CUDA 오류
```bash
# CPU 전용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. VARMAX 모델 오류
```bash
# statsmodels 재설치
pip install --upgrade statsmodels
```

### 4. 메모리 부족
- `SEQUENCE_LENGTH` 값을 줄이기 (30 → 20)
- 배치 크기 조정
- GPU 메모리 정리: `torch.cuda.empty_cache()`

## 📈 성능 최적화

### 1. GPU 가속 활용
```python
# 자동 GPU 감지 및 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2. 캐시 활용
- 동일한 파일에 대한 반복 예측 시 캐시 자동 활용
- 캐시 히트율 모니터링

### 3. 하이퍼파라미터 최적화
- Optuna trials 수 조정 (기본값: 30)
- K-Fold 수 조정 (기본값: 5)

## 📞 지원
문제 발생 시 로그 파일을 확인하거나 개발팀에 문의하세요.

---
© 2025 MOPJ 가격 예측 시스템 