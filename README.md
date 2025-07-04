# 🔮 MOPJ 가격 예측 시스템

> LSTM 딥러닝 및 VARMAX 시계열 모델을 활용한 일본 나프타 시장(MOPJ) 가격 예측 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14.0-orange.svg)

## 📋 프로젝트 개요

MOPJ(Mean Of Platts Japan) 가격 예측을 위한 전문적인 웹 애플리케이션입니다. LSTM 기반 딥러닝 모델과 VARMAX 시계열 통계 모델을 활용하여 정확한 가격 예측을 제공하며, Attention 메커니즘을 통한 해석 가능한 AI를 구현합니다.

### ✨ 주요 기능

- **🧠 이중 모델 예측**: 
  - LSTM + Attention: 딥러닝 기반 23일간 가격 예측
  - VARMAX: 시계열 통계 모델 기반 반월별 예측
- **📊 실시간 분석**: 이동평균, 특성 중요도, 성능 지표 실시간 시각화
- **🎯 구매 의사결정**: VARMAX 기반 구매 타이밍 및 구간 추천
- **⚡ 캐시 시스템**: 파일별 독립적 캐시로 빠른 재예측
- **🔄 누적 분석**: 여러 시점의 예측을 비교 분석
- **📈 인터랙티브 차트**: 예측값과 실제값 비교, 구간별 신뢰도 분석

## 🏗️ 시스템 아키텍처

```
📁 MOPJ 가격 예측 시스템
├── 🖥️ Frontend (React)          # 사용자 인터페이스
│   ├── 📊 통합 대시보드           # LSTM + VARMAX 결과 시각화
│   ├── 📈 차트 컴포넌트          # 인터랙티브 그래프
│   ├── 🎛️ 제어 패널            # 예측 설정 및 실행
│   └── 🤝 의사결정 인터페이스     # 구매 추천 시각화
│
├── ⚙️ Backend (Flask + AI Models) # API 서버 및 ML 엔진
│   ├── 🧠 LSTM 모델             # 딥러닝 예측 모델
│   │   ├── 🔍 Attention 메커니즘 # 특성 중요도 분석
│   │   └── 📋 하이퍼파라미터 최적화 # Optuna 기반 자동 최적화
│   ├── 📈 VARMAX 모델           # 시계열 통계 모델
│   │   ├── 🔢 다변량 분석       # 다중 변수 시계열 예측
│   │   └── 💼 구매 의사결정     # 비즈니스 로직 통합
│   └── 💾 캐시 시스템           # 파일별 예측 결과 저장
│
└── 📊 데이터 처리
    ├── 📁 CSV 파일 업로드       # 시계열 데이터 입력
    ├── 🔄 전처리 및 검증       # 데이터 품질 보장
    ├── 📈 피처 엔지니어링      # 시계열 특성 추출
    └── 🗓️ 반월별 분할         # 비즈니스 사이클 맞춤
```

## 🚀 빠른 시작

### 전체 시스템 실행

1. **저장소 클론**
```bash
git clone <repository-url>
cd Mopj-project
```

2. **백엔드 설정 및 실행**
```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate

# 의존성 설치 (VARMAX 지원 포함)
pip install -r requirements.txt

# 서버 실행
python app.py
```

3. **프론트엔드 설정 및 실행** (새 터미널)
```bash
cd mopj-dashboard

# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

4. **웹 애플리케이션 접속**
```
http://localhost:3000
```

### Docker를 사용한 배포 (권장)

**GPU 지원 환경 (LSTM 가속):**
```bash
# NVIDIA Container Toolkit 설치 후
docker-compose up -d

# 웹 애플리케이션 접속
http://localhost
```

**CPU 전용 환경:**
```bash
docker-compose -f docker-compose.cpu.yml up -d

# 웹 애플리케이션 접속
http://localhost
```

📖 **상세한 Docker 배포 가이드**: [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md)

### WSL2 환경에서 Docker 실행

Windows에서 WSL2를 사용하는 경우 다음 단계를 따라주세요:

#### 1. 사전 준비
```bash
# WSL2가 활성화되어 있는지 확인
wsl --status

# WSL2 배포판 확인
wsl --list --verbose
```

#### 2. Docker Desktop 설정
- **Docker Desktop for Windows** 설치 및 실행
- **Settings** → **General** → **Use the WSL 2 based engine** 체크
- **Settings** → **Resources** → **WSL Integration**:
  - **Enable integration with my default WSL distro** 체크
  - 사용할 WSL2 배포판 활성화

#### 3. WSL2에서 프로젝트 실행
```bash
# WSL2 터미널에서 프로젝트 디렉토리로 이동
cd /mnt/c/path/to/New-mopj-project-main
# 또는 WSL2 파일시스템에 프로젝트 복사 (성능 향상)
cp -r /mnt/c/path/to/New-mopj-project-main ~/mopj-project
cd ~/mopj-project

# Docker Compose 실행
docker-compose up -d
```

#### 4. WSL2 성능 최적화
**`.wslconfig` 파일 생성** (Windows 사용자 홈 디렉토리):
```ini
[wsl2]
memory=8GB          # RAM 할당량 (시스템 RAM의 50-70%)
processors=4        # CPU 코어 수
swap=2GB           # 스왑 메모리
localhostForwarding=true
```

#### 5. 네트워크 접속
- **브라우저에서 접속**: `http://localhost` 또는 `http://127.0.0.1`
- **WSL2 IP로 접속**: `http://$(hostname -I | awk '{print $1}')`

#### ⚠️ WSL2 주의사항
- **파일 성능**: Windows 파일시스템(`/mnt/c/`)보다 WSL2 네이티브 파일시스템(`~/`)에서 더 빠른 성능
- **메모리 관리**: `.wslconfig`로 메모리 사용량 제한 권장
- **Docker 리소스**: GPU 사용 시 WSL2에서 CUDA 지원 확인 필요

## 📊 사용 방법

### 1. 데이터 업로드
- **LSTM용**: 날짜와 MOPJ 가격이 포함된 시계열 CSV 파일
- **VARMAX용**: 구매 의사결정을 위한 다변량 데이터 CSV 파일
- 자동 데이터 검증 및 예측 가능한 날짜 계산

### 2. 예측 실행

**LSTM 예측**
- **단일 예측**: 특정 날짜부터 23일간 예측
- **누적 예측**: 여러 시점의 배치 예측
- 실시간 진행률 표시
- Attention 메커니즘 시각화

**VARMAX 예측**
- **반월별 예측**: 15일 단위 시계열 예측
- **구매 의사결정**: 최적 구매 타이밍 분석
- **다변량 분석**: 여러 변수 동시 고려

### 3. 결과 분석

**LSTM 분석**
- **예측 차트**: 실제값 vs 예측값 비교
- **이동평균 분석**: 5일, 10일, 23일 이동평균
- **특성 중요도**: Attention 가중치 시각화
- **성능 지표**: F1 Score, MAPE, 정확도

**VARMAX 분석**
- **반월별 예측 차트**: 통계 모델 기반 예측
- **구매 구간 추천**: 최적 구매 시점 시각화
- **이동평균 분석**: 5일, 10일, 20일, 30일 이동평균
- **모델 성능 지표**: 통계적 유의성 검증

## 🎯 핵심 기술

### 머신러닝 모델
- **LSTM + Attention**: 시계열 데이터의 장기 의존성 학습 및 해석 가능성
- **VARMAX**: Vector Autoregression with eXogenous variables 통계 모델
- **Optuna**: 하이퍼파라미터 자동 최적화
- **K-Fold 교차검증**: 모델 일반화 성능 보장

### 데이터 처리
- **반월별 분할**: 비즈니스 사이클에 맞춘 학습 기간 설정
- **영업일 필터링**: 휴일 및 주말 제외
- **다중 특성**: 가격, 변화율, 이동평균 등 종합 분석
- **다변량 시계열**: VARMAX를 위한 복합 변수 처리

### 성능 최적화
- **파일별 캐시**: 동일 파일에 대한 빠른 재예측
- **GPU 가속**: CUDA 지원으로 LSTM 학습 속도 향상
- **배치 처리**: 다중 예측의 효율적 처리
- **통계 최적화**: VARMAX 모델의 메모리 효율성

## 📈 성능 지표

### LSTM 모델
| 지표 | 설명 | 목표 값 |
|------|------|---------|
| **F1 Score** | 가격 방향성 예측 정확도 | > 0.7 |
| **MAPE** | 평균 절대 백분율 오차 | < 5% |
| **Direction Accuracy** | 상승/하락 방향 정확도 | > 70% |

### VARMAX 모델
| 지표 | 설명 | 목표 값 |
|------|------|---------|
| **AIC/BIC** | 모델 적합도 지표 | 최소값 |
| **Log-Likelihood** | 통계적 유의성 | 최대값 |
| **구매 신뢰도** | 의사결정 정확도 | > 80% |

### 시스템 성능
| 지표 | 설명 | 목표 값 |
|------|------|---------|
| **Cache Hit Rate** | 캐시 활용률 | > 80% |
| **API 응답시간** | 평균 응답 속도 | < 2초 |

## 🔧 시스템 요구사항

### 최소 요구사항
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04
- **RAM**: 8GB 이상
- **CPU**: Intel i5 또는 AMD Ryzen 5 이상
- **저장공간**: 5GB 이상

### 권장 요구사항
- **RAM**: 16GB 이상 (VARMAX 다변량 분석용)
- **GPU**: NVIDIA GTX 1060 이상 (CUDA 지원, LSTM 가속용)
- **CPU**: Intel i7 또는 AMD Ryzen 7 이상
- **저장공간**: 10GB 이상 (모델 캐시용)

### 소프트웨어 요구사항
- **Python**: 3.8+ (statsmodels 지원)
- **Node.js**: 16.0+
- **Docker**: 20.10+ (선택사항)

## 📁 프로젝트 구조

```
New-mopj-project-main/
├── backend/                    # Flask API 서버
│   ├── app.py                 # 메인 애플리케이션 (LSTM + VARMAX)
│   ├── requirements.txt       # Python 의존성 (statsmodels 포함)
│   ├── README.md             # 백엔드 문서
│   ├── uploads/              # 업로드된 파일
│   ├── cache/               # 캐시된 예측 결과
│   │   └── {hash}_{name}/   # 파일별 캐시 디렉토리
│   │       ├── predictions/ # LSTM 예측 결과
│   │       └── varmax/     # VARMAX 예측 결과
│   ├── predictions/         # 레거시 예측 파일
│   ├── holidays/           # 휴일 데이터
│   ├── static/             # 정적 파일
│   └── temp/              # 임시 파일
│
├── mopj-dashboard/            # React 프론트엔드
│   ├── src/                  # 소스 코드
│   │   ├── components/       # React 컴포넌트
│   │   │   ├── LSTM 관련/   # LSTM 예측 컴포넌트
│   │   │   └── VARMAX 관련/ # VARMAX 예측 컴포넌트
│   │   ├── services/        # API 서비스 (통합)
│   │   └── App.js          # 메인 앱 (탭 기반 UI)
│   ├── public/             # 정적 자원
│   ├── package.json        # Node.js 의존성
│   └── README.md          # 프론트엔드 문서
│
├── docker-compose.yml        # GPU 지원 Docker 구성
├── docker-compose.cpu.yml    # CPU 전용 Docker 구성
├── DOCKER_DEPLOYMENT.md     # Docker 배포 가이드
└── README.md               # 이 파일
```

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원 및 문의

- 📧 이메일: [문의 이메일]

---

**© 2025 MOPJ 가격 예측 시스템. All rights reserved.**
