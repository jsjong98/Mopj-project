# MOPJ 가격 예측 시스템 - 프론트엔드

## 📋 개요
React 기반 MOPJ 가격 예측 시스템의 사용자 인터페이스입니다. LSTM 딥러닝 모델과 VARMAX 시계열 통계 모델을 활용한 예측 결과를 인터랙티브한 차트와 직관적인 대시보드를 통해 시각화합니다.

## 🔧 환경 요구사항
- Node.js 16.0 이상
- npm 8.0 이상 또는 yarn 1.22 이상
- 모던 웹 브라우저 (Chrome, Firefox, Safari, Edge)

## 📦 설치 방법

### 1. 의존성 패키지 설치
```bash
# npm 사용
npm install

# 또는 yarn 사용
yarn install
```

### 2. 환경 설정
```bash
# .env 파일 생성 (선택사항)
REACT_APP_API_URL=http://localhost:5000
REACT_APP_VERSION=1.0.0
```

## 🚀 실행 방법

### 개발 모드
```bash
# npm 사용
npm start

# 또는 yarn 사용
yarn start
```
개발 서버가 `http://localhost:3000`에서 시작됩니다.

### 프로덕션 빌드
```bash
# npm 사용
npm run build

# 또는 yarn 사용
yarn build
```

### 테스트 실행
```bash
# npm 사용
npm test

# 또는 yarn 사용
yarn test
```

## 🎯 주요 기능

### 1. 파일 업로드 인터페이스
- 드래그 앤 드롭으로 CSV 파일 업로드
- 파일 검증 및 미리보기
- 업로드 진행률 표시
- LSTM용 날짜 포함 데이터와 VARMAX용 구매 의사결정 데이터 구분 업로드

### 2. 예측 설정 및 실행
- **LSTM 예측**:
  - 단일 예측: 날짜 선택기를 통한 예측 시작일 설정
  - 누적 예측: 시작일과 종료일 범위 설정
- **VARMAX 예측**:
  - 반월별 시계열 예측 (15일 단위)
  - 다변량 통계 모델 기반 예측
  - 구매 의사결정 지원 분석
- 실시간 예측 진행률 모니터링

### 3. 결과 시각화
- **LSTM 예측 시각화**:
  - Recharts 기반 인터랙티브 차트
  - Attention 메커니즘 히트맵
  - 이동평균 분석 (5일, 10일, 23일)
- **VARMAX 예측 시각화**:
  - 반월별 예측 결과 차트
  - 이동평균 분석 (5일, 10일, 20일, 30일)
  - 구매 구간 추천 시각화
- **공통 기능**:
  - 성능 지표 실시간 표시 (F1 Score, MAPE, 정확도)
  - 반응형 차트 디자인

### 4. 대시보드 및 분석
- **LSTM 누적 분석 대시보드**:
  - 날짜별 예측 비교 테이블
  - 추이 분석 및 일관성 점수
  - 구매 신뢰도 지표
- **VARMAX 의사결정 대시보드**:
  - 구매 추천 구간 분석
  - 시계열 패턴 분석
  - 모델 성능 지표

### 5. 저장된 예측 관리
- 예측 결과 자동 저장 및 불러오기
- 날짜별 예측 기록 관리
- 예측 삭제 및 관리 기능

## 🏗️ 컴포넌트 구조

```
src/
├── App.js                          # 메인 애플리케이션 컴포넌트
├── components/                     # 재사용 가능한 컴포넌트
│   ├── FileUploader.js            # 통합 파일 업로드 컴포넌트
│   ├── DateSelector.js            # 날짜 선택 컴포넌트
│   ├── ProgressBar.js             # 진행률 표시 바
│   │
│   ├── LSTM 관련 컴포넌트/
│   ├── PredictionChart.js         # LSTM 예측 차트
│   ├── MovingAverageChart.js      # LSTM 이동평균 분석 차트
│   ├── AttentionMap.js            # Attention 가중치 시각화
│   ├── IntervalScoresTable.js     # 구간 점수 테이블
│   ├── AccumulatedResultsTable.js # 누적 결과 테이블
│   ├── AccumulatedMetricsChart.js # 누적 지표 차트
│   ├── AccumulatedSummary.js      # 누적 예측 요약
│   │
│   ├── VARMAX 관련 컴포넌트/
│   ├── VarmaxFileUploader.js      # VARMAX 전용 파일 업로드
│   ├── VarmaxPredictionChart.js   # VARMAX 예측 차트
│   ├── VarmaxMovingAverageChart.js # VARMAX 이동평균 분석
│   ├── VarmaxModelInfo.js         # VARMAX 모델 정보
│   ├── VarmaxResult.js            # VARMAX 결과 표시
│   └── VarmaxAlgorithm.js         # VARMAX 알고리즘 분석
│
├── services/                       # API 서비스
│   └── api.js                     # 백엔드 API 호출 (LSTM + VARMAX)
└── utils/                         # 유틸리티 함수
    └── formatting.js              # 데이터 포맷팅 함수
```

## 🎨 사용된 라이브러리

### 핵심 라이브러리
- **React 18.2.0**: UI 프레임워크
- **Recharts 2.15.2**: 차트 및 데이터 시각화
- **Lucide React 0.487.0**: 아이콘 라이브러리
- **Axios 1.8.4**: HTTP 클라이언트

### 유틸리티 라이브러리
- **React Modal 3.16.3**: 모달 다이얼로그
- **HTTP Proxy Middleware 3.0.5**: 개발 서버 프록시

### 테스팅 라이브러리
- **React Testing Library**: 컴포넌트 테스트
- **Jest DOM**: DOM 테스트 유틸리티

## 🎯 주요 상태 관리

### 전역 상태 (App.js)
```javascript
// 파일 및 데이터 상태
const [fileInfo, setFileInfo] = useState(null);
const [predictableStartDates, setPredictableStartDates] = useState([]);

// LSTM 예측 상태
const [isPredicting, setIsPredicting] = useState(false);
const [progress, setProgress] = useState(0);
const [predictionData, setPredictionData] = useState([]);
const [intervalScores, setIntervalScores] = useState([]);
const [maResults, setMaResults] = useState(null);
const [attentionImage, setAttentionImage] = useState(null);

// VARMAX 예측 상태
const [varmaxResults, setVarmaxResults] = useState(null);
const [varmaxPredictions, setVarmaxPredictions] = useState(null);
const [varmaxMaResults, setVarmaxMaResults] = useState(null);
const [isVarmaxPredicting, setIsVarmaxPredicting] = useState(false);
const [varmaxProgress, setVarmaxProgress] = useState(0);

// 누적 예측 상태 (LSTM)
const [accumulatedResults, setAccumulatedResults] = useState(null);
const [selectedAccumulatedDate, setSelectedAccumulatedDate] = useState(null);

// 저장된 예측 관리
const [savedPredictions, setSavedPredictions] = useState([]);
const [savedVarmaxPredictions, setSavedVarmaxPredictions] = useState([]);
```

## 📊 API 통신

### LSTM 관련 API (`services/api.js`)
```javascript
// 파일 업로드 및 예측
export const uploadFile = (file) => { /* ... */ };
export const startPrediction = (filepath, currentDate) => { /* ... */ };
export const startAccumulatedPrediction = (filepath, startDate, endDate) => { /* ... */ };

// 결과 조회
export const getPredictionResults = () => { /* ... */ };
export const getAccumulatedResults = () => { /* ... */ };
export const getPredictionStatus = () => { /* ... */ };
```

### VARMAX 관련 API (`services/api.js`)
```javascript
// VARMAX 파일 업로드 및 예측
export const uploadVarmaxFile = (file) => { /* ... */ };
export const startVarmaxPrediction = (filepath, date, predDays) => { /* ... */ };

// VARMAX 결과 조회
export const getVarmaxResults = () => { /* ... */ };
export const getVarmaxPredictions = () => { /* ... */ };
export const getVarmaxMovingAverages = () => { /* ... */ };
export const getVarmaxStatus = () => { /* ... */ };

// 저장된 VARMAX 예측 관리
export const getSavedVarmaxPredictions = (limit) => { /* ... */ };
export const getSavedVarmaxPredictionByDate = (date) => { /* ... */ };
export const deleteSavedVarmaxPrediction = (date) => { /* ... */ };
```

## 🎨 스타일링

### 인라인 스타일 시스템
```javascript
const styles = {
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    padding: '1rem',
    marginBottom: '1.5rem'
  },
  // VARMAX 전용 스타일
  varmaxCard: {
    backgroundColor: '#f8f9fa',
    border: '1px solid #e2e8f0',
    borderRadius: '0.5rem',
    padding: '1rem'
  }
};
```

### 반응형 디자인
- **모바일 퍼스트**: 768px 브레이크포인트 기준
- **플렉시블 레이아웃**: CSS Flexbox 및 Grid 활용
- **적응형 차트**: 화면 크기에 따른 차트 크기 조정
- **탭 기반 UI**: LSTM과 VARMAX 기능 분리

## 🔧 개발 도구

### 디버깅
```javascript
// 콘솔 로그 시스템
console.log(`🔄 [LSTM] Starting fetchResults...`);
console.log(`📊 [VARMAX] VARMAX results received:`, data);
console.log(`✅ [STATE] States updated successfully`);
console.error(`❌ [ERROR] Prediction failed:`, error);
```

### 성능 모니터링
- React DevTools 호환
- 컴포넌트 렌더링 최적화
- 메모리 사용량 모니터링
- 차트 렌더링 성능 최적화

## 🚀 배포

### 정적 파일 생성
```bash
npm run build
```

### 배포 옵션
1. **Netlify**: `build` 폴더를 드래그 앤 드롭
2. **Vercel**: GitHub 연동 자동 배포
3. **AWS S3**: S3 버킷에 정적 호스팅
4. **Nginx**: 역프록시 설정으로 백엔드와 통합
5. **Docker**: 컨테이너 기반 배포

## 🐛 문제 해결

### 1. 패키지 설치 오류
```bash
# npm 캐시 정리
npm cache clean --force

# node_modules 재설치
rm -rf node_modules package-lock.json
npm install
```

### 2. 프록시 연결 오류
```bash
# 백엔드 서버 실행 확인
curl http://localhost:5000/api/health

# package.json proxy 설정 확인
"proxy": "http://localhost:5000"
```

### 3. 차트 렌더링 문제
- 브라우저 콘솔에서 JavaScript 오류 확인
- Recharts 버전 호환성 확인
- 데이터 구조 검증

### 4. VARMAX 기능 관련 문제
- VARMAX API 연결 상태 확인
- 업로드한 CSV 파일 형식 검증
- 백엔드 statsmodels 패키지 설치 확인

## 📞 지원
프론트엔드 관련 문제 발생 시 브라우저 개발자 도구(F12)의 콘솔 탭에서 오류 메시지를 확인하고 개발팀에 문의하세요.

---
© 2025 MOPJ 가격 예측 시스템