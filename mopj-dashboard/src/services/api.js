// api.js
import axios from 'axios';

// 백엔드 URL 직접 지정
const API_BASE_URL = '/api';

// API 클라이언트 생성
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 60000 // 60초 타임아웃
});

// CSV 파일 업로드 - fetch API 사용
export const uploadCSV = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    // 직접 백엔드 URL로 요청
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
      // CORS 설정
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload failed:', errorText);
      throw new Error('파일 업로드 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    return { error: error.message || '파일 업로드 중 오류가 발생했습니다.' };
  }
};

// 사용 가능한 날짜 조회 (향상된 버전)
export const getAvailableDates = async (filepath, forceRefresh = false) => {
  try {
    // URL과 쿼리 파라미터 구성
    const params = new URLSearchParams();
    params.append('filepath', filepath);
    if (forceRefresh) {
      params.append('force_refresh', 'true');
      params.append('_t', new Date().getTime()); // 캐시 방지
    }
    
    const url = `${API_BASE_URL}/data/dates?${params.toString()}`;
    
    const response = await fetch(url, {
      headers: forceRefresh ? {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      } : {},
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('날짜 정보를 가져오는데 실패했습니다');
    }
    
    const data = await response.json();
    
    // 로깅 추가
    if (forceRefresh) {
      console.log('🔄 [DATE_REFRESH] Forced refresh completed:', {
        filepath: filepath,
        total_rows: data.total_rows,
        date_range: `${data.data_start_date} ~ ${data.data_end_date}`,
        available_dates: data.dates?.length || 0,
        file_hash: data.file_hash,
        file_modified: data.file_modified
      });
    }
    
    return data;
  } catch (error) {
    console.error('Get dates error:', error);
    return { 
      error: error.message || '날짜 정보를 가져오는 중 오류가 발생했습니다.',
      dates: [] 
    };
  }
};

// 파일 데이터 새로고침 체크
export const checkFileRefresh = async (filepath) => {
  try {
    console.log('🔍 [REFRESH_CHECK] Checking if file needs refresh:', filepath);
    
    const response = await fetch(`${API_BASE_URL}/data/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ filepath }),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('파일 새로고침 체크 실패');
    }
    
    const data = await response.json();
    
    console.log('📊 [REFRESH_CHECK] File refresh analysis:', {
      refresh_needed: data.refresh_needed,
      reasons: data.refresh_reasons,
      current_range: data.file_info?.date_range,
      cached_range: data.cache_info?.date_range,
      file_hash: data.file_info?.hash
    });
    
    return data;
  } catch (error) {
    console.error('File refresh check error:', error);
    return { 
      error: error.message || '파일 새로고침 체크 중 오류가 발생했습니다.',
      refresh_needed: false 
    };
  }
};

// 예측 시작
export const startPrediction = async (filepath, date = null) => {
  try {
    const payload = { filepath };
    if (date) payload.date = date;
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('예측 시작 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Start prediction error:', error);
    return { error: error.message || '예측 시작 중 오류가 발생했습니다.' };
  }
};

// 예측 상태 확인 (디버깅 강화)
export const getPredictionStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/status?_t=${new Date().getTime()}`, {
      headers: {
        'Cache-Control': 'no-cache'
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('예측 상태 확인 실패');
    }
    
    const data = await response.json();
    console.log('📊 [STATUS]', data);
    
    return data;
  } catch (error) {
    console.error('💥 [STATUS] Error:', error);
    return { error: error.message || '예측 상태 확인 중 오류가 발생했습니다.' };
  }
};

// 예측 결과 조회 (디버깅 강화)
export const getPredictionResults = async () => {
  try {
    console.log('🔍 [API] Requesting prediction results...');
    
    // 캐시 방지를 위한 타임스탬프 추가
    const timestamp = new Date().getTime();
    const response = await fetch(`${API_BASE_URL}/results?_t=${timestamp}`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    console.log('📡 [API] Response status:', response.status);
    console.log('📡 [API] Response headers:', response.headers);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ [API] Response not OK:', errorText);
      throw new Error(`HTTP ${response.status}: 예측 결과 조회 실패`);
    }
    
    const data = await response.json();
    console.log('✅ [API] Response data received:', data);
    console.log('📊 [API] Data summary:', {
      success: data.success,
      current_date: data.current_date,
      predictions_count: data.predictions ? data.predictions.length : 0,
      has_plots: !!data.plots,
      has_ma_results: !!data.ma_results,
      has_attention: !!data.attention_data,
      interval_scores_count: data.interval_scores ? data.interval_scores.length : 0
    });
    
    return data;
  } catch (error) {
    console.error('💥 [API] Get prediction results error:', error);
    return { 
      error: error.message || '예측 결과를 가져오는 중 오류가 발생했습니다.',
      success: false 
    };
  }
};

// 예측값만 조회
export const getPredictions = async () => {
  try {
    const response = await apiClient.get('/results/predictions');
    return response.data;
  } catch (error) {
    console.error('Get predictions error:', error);
    throw error.response?.data || { error: error.message || '예측값을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 구간 점수 조회
export const getIntervalScores = async () => {
  try {
    const response = await apiClient.get('/results/interval-scores');
    return response.data;
  } catch (error) {
    console.error('Get interval scores error:', error);
    throw error.response?.data || { error: error.message || '구간 점수를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 이동평균 조회
export const getMovingAverages = async () => {
  try {
    const response = await apiClient.get('/results/moving-averages');
    return response.data;
  } catch (error) {
    console.error('Get moving averages error:', error);
    throw error.response?.data || { error: error.message || '이동평균을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 어텐션 맵 조회
export const getAttentionMap = async () => {
  try {
    const response = await apiClient.get('/results/attention-map');
    return response.data;
  } catch (error) {
    console.error('Get attention map error:', error);
    throw error.response?.data || { error: error.message || '어텐션 맵을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 여기서부터 추가된 누적 예측 관련 함수들입니다

// 누적 예측 시작
export const startAccumulatedPrediction = async (filepath, startDate, endDate) => {
  try {
    const payload = { 
      filepath,
      start_date: startDate,
      end_date: endDate
    };
    
    const response = await fetch(`${API_BASE_URL}/predict/accumulated`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('누적 예측 시작 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Start accumulated prediction error:', error);
    return { error: error.message || '누적 예측 시작 중 오류가 발생했습니다.' };
  }
};

// 특정 날짜의 누적 예측 결과 조회
export const getAccumulatedResultByDate = async (date) => {
  try {
    console.log(`🔍 [API] Requesting accumulated result for date: ${date}`);
    const url = `${API_BASE_URL}/results/accumulated/${date}`;
    console.log(`📡 [API] URL: ${url}`);
    
    const response = await fetch(url, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    console.log(`📊 [API] Response status for ${date}:`, response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ [API] Response not OK for ${date}:`, errorText);
      throw new Error(`HTTP ${response.status}: ${date} 날짜의 누적 예측 결과 조회 실패`);
    }
    
    const data = await response.json();
    console.log(`✅ [API] Data received for ${date}:`, data);
    console.log(`📋 [API] Data summary for ${date}:`, {
      success: data.success,
      predictions_count: data.predictions ? data.predictions.length : 0,
      interval_scores_type: typeof data.interval_scores,
      interval_scores_count: data.interval_scores ? 
        (Array.isArray(data.interval_scores) ? data.interval_scores.length : Object.keys(data.interval_scores).length) : 0,
      has_metrics: !!data.metrics,
      date_field: data.date
    });
    
    return data;
  } catch (error) {
    console.error(`💥 [API] Get accumulated result by date error for ${date}:`, error);
    return { 
      error: error.message || '특정 날짜의 누적 예측 결과를 가져오는 중 오류가 발생했습니다.',
      success: false 
    };
  }
};

// 누적 예측 시각화 데이터 조회
export const getAccumulatedVisualization = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/results/accumulated/visualization`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('누적 예측 시각화 데이터 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get accumulated visualization error:', error);
    return { error: error.message || '누적 예측 시각화 데이터를 가져오는 중 오류가 발생했습니다.' };
  }
};

// api.js - getAccumulatedResults 함수 디버깅 로그 추가
export const getAccumulatedResults = async () => {
  try {
    console.log('🔍 [API] Requesting accumulated results...');
    const response = await fetch(`${API_BASE_URL}/results/accumulated`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    console.log('📡 [API] Accumulated response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ [API] Accumulated response not OK:', errorText);
      throw new Error(`HTTP ${response.status}: 누적 예측 결과 조회 실패`);
    }
    
    const data = await response.json();
    console.log('✅ [API] Accumulated results data received:', data);
    console.log('📊 [API] Accumulated data summary:', {
      success: data.success,
      predictions_count: data.predictions ? data.predictions.length : 0,
      has_accumulated_metrics: !!data.accumulated_metrics,
      has_consistency_scores: !!data.accumulated_consistency_scores,
      accumulated_purchase_reliability: data.accumulated_purchase_reliability,
      has_interval_scores: !!data.accumulated_interval_scores,
      interval_scores_count: data.accumulated_interval_scores ? data.accumulated_interval_scores.length : 0
    });
    
    // 확인: accumulated_interval_scores가 있는지
    if (data.success && !data.accumulated_interval_scores) {
      console.warn('⚠️ [API] Warning: No accumulated_interval_scores in response');
    }
    
    return data;
  } catch (error) {
    console.error('💥 [API] Get accumulated results error:', error);
    return { 
      error: error.message || '누적 예측 결과를 가져오는 중 오류가 발생했습니다.',
      success: false
    };
  }
};

// 누적 예측 보고서 URL 가져오기
export const getAccumulatedReportURL = () => {
  return `${API_BASE_URL}/results/accumulated/report`;
};

// =================== 휴일 관리 관련 API 함수 추가 ===================

// 휴일 목록 조회
export const getHolidays = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/holidays`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('휴일 목록 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get holidays error:', error);
    return { 
      error: error.message || '휴일 목록을 가져오는 중 오류가 발생했습니다.',
      holidays: [],
      success: false 
    };
  }
};

// 휴일 파일 업로드
export const uploadHolidayFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/holidays/upload`, {
      method: 'POST',
      body: formData,
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Holiday file upload failed:', errorText);
      throw new Error('휴일 파일 업로드 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Upload holiday file error:', error);
    return { 
      error: error.message || '휴일 파일 업로드 중 오류가 발생했습니다.',
      success: false 
    };
  }
};

// 휴일 정보 재로드
export const reloadHolidays = async (filepath = null) => {
  try {
    const payload = {};
    if (filepath) payload.filepath = filepath;
    
    const response = await fetch(`${API_BASE_URL}/holidays/reload`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('휴일 정보 새로고침 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Reload holidays error:', error);
    return {
      error: error.message || '휴일 정보 새로고침 중 오류가 발생했습니다.',
      success: false
    };
  }
};

// 신뢰도 점수 조회
export const getReliabilityScores = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/results/reliability`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('신뢰도 점수 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get reliability scores error:', error);
    return { error: error.message || '신뢰도 점수를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 캐시된 예측 확인
export const checkCachedPredictions = async (startDate, endDate) => {
  try {
    console.log(`🔍 [API] Checking cached predictions for: ${startDate} ~ ${endDate}`);
    
    const response = await fetch(`${API_BASE_URL}/cache/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        start_date: startDate,
        end_date: endDate
      }),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('캐시 확인 실패');
    }
    
    const result = await response.json();
    console.log(`📊 [API] Cache check result:`, result);
    
    return result;
  } catch (error) {
    console.error('Check cached predictions error:', error);
    return { error: error.message || '캐시 확인 중 오류가 발생했습니다.' };
  }
};

// 누적 예측 캐시 클리어
export const clearAccumulatedCache = async () => {
  try {
    console.log(`🧹 [API] Clearing accumulated prediction cache`);
    
    const response = await fetch(`${API_BASE_URL}/cache/clear/accumulated`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('캐시 클리어 실패');
    }
    
    const result = await response.json();
    console.log(`✅ [API] Cache cleared:`, result);
    
    return result;
  } catch (error) {
    console.error('Clear accumulated cache error:', error);
    return { error: error.message || '캐시 클리어 중 오류가 발생했습니다.' };
  }
};

// 최근 누적 예측 결과 자동 복원
export const getRecentAccumulatedResults = async () => {
  try {
    console.log(`🔄 [API] Fetching recent accumulated results for auto-restore`);
    
    const response = await fetch(`${API_BASE_URL}/results/accumulated/recent`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('최근 누적 결과 조회 실패');
    }
    
    const result = await response.json();
    console.log(`✅ [API] Recent accumulated results:`, result);
    
    return result;
  } catch (error) {
    console.error('Get recent accumulated results error:', error);
    return { error: error.message || '최근 누적 결과 조회 중 오류가 발생했습니다.' };
  }
};

// =================== VARMAX 관련 API 함수 추가 ===================

// VARMAX용 CSV 파일 업로드 (구매 계획 결정용)
export const uploadCSV_2 = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // 파일 정보 출력
  console.log('업로드할 파일:', file);
  console.log('파일 타입:', file.type);
  console.log('파일 이름:', file.name);
  
  // FormData 내용 출력
  for (let pair of formData.entries()) {
    console.log('FormData:', pair[0], pair[1]);
  }
  
  try {
    // 직접 백엔드 URL로 요청
    const response = await fetch(`${API_BASE_URL}/varmax/decision`, {
      method: 'POST',
      body: formData,
      // CORS 설정
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('VARMAX file upload failed:', errorText);
      throw new Error('VARMAX 파일 업로드 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Upload VARMAX file error:', error);
    return { error: error.message || 'VARMAX 파일 업로드 중 오류가 발생했습니다.' };
  }
};

// VARMAX 반월별 예측 시작
export const startVarmaxPrediction = async (filepath, date, predDays = 50, useCache = true) => {
  try {
    const payload = { 
      filepath,
      date,
      pred_days: predDays,
      use_cache: useCache  // 🆕 캐시 사용 여부 추가
    };
    
    console.log('🚀 [VARMAX_API] VARMAX 예측 시작 요청:', payload);
    
    const response = await fetch(`${API_BASE_URL}/varmax/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ [VARMAX_API] VARMAX prediction start failed:', errorText);
      throw new Error('VARMAX 반월별 예측 시작 실패');
    }
    
    const result = await response.json();
    console.log('✅ [VARMAX_API] VARMAX 예측 시작 응답:', result);
    return result;
  } catch (error) {
    console.error('❌ [VARMAX_API] Start VARMAX prediction error:', error);
    return { error: error.message || 'VARMAX 예측 시작 중 오류가 발생했습니다.' };
  }
};

// VARMAX 예측 상태 확인
export const getVarmaxStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/varmax/status?_t=${new Date().getTime()}`, {
      headers: {
        'Cache-Control': 'no-cache'
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('VARMAX 예측 상태 확인 실패');
    }
    
    const data = await response.json();
    console.log('📊 [VARMAX STATUS]', data);
    
    return data;
  } catch (error) {
    console.error('💥 [VARMAX STATUS] Error:', error);
    return { error: error.message || 'VARMAX 예측 상태 확인 중 오류가 발생했습니다.' };
  }
};

// VARMAX 전체 결과 조회
export const getVarmaxResults = async () => {
  try {
    console.log('🔍 [VARMAX API] Requesting VARMAX results...');
    
    // 캐시 방지를 위한 타임스탬프 추가
    const timestamp = new Date().getTime();
    const response = await fetch(`${API_BASE_URL}/varmax/results?_t=${timestamp}`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    console.log('📡 [VARMAX API] Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ [VARMAX API] Response not OK:', errorText);
      throw new Error(`HTTP ${response.status}: VARMAX 예측 결과 조회 실패`);
    }
    
    const data = await response.json();
    console.log('✅ [VARMAX API] Response data received:', data);
    console.log('📊 [VARMAX API] Data summary:', {
      success: data.success,
      current_date: data.current_date,
      predictions_count: data.predictions ? data.predictions.length : 0,
      has_ma_results: !!data.ma_results,
      has_model_info: !!data.model_info
    });
    
    return data;
  } catch (error) {
    console.error('💥 [VARMAX API] Get VARMAX results error:', error);
    return { 
      error: error.message || 'VARMAX 예측 결과를 가져오는 중 오류가 발생했습니다.',
      success: false 
    };
  }
};

// VARMAX 예측값만 조회
export const getVarmaxPredictions = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/varmax/predictions`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('VARMAX 예측값 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get VARMAX predictions error:', error);
    return { error: error.message || 'VARMAX 예측값을 가져오는 중 오류가 발생했습니다.' };
  }
};

// VARMAX 이동평균 조회
export const getVarmaxMovingAverages = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/varmax/moving-averages`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('VARMAX 이동평균 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get VARMAX moving averages error:', error);
    return { error: error.message || 'VARMAX 이동평균을 가져오는 중 오류가 발생했습니다.' };
  }
};

// =================== VARMAX 저장된 예측 관리 API 함수 ===================

// 저장된 VARMAX 예측 목록 조회
export const getSavedVarmaxPredictions = async (limit = 100) => {
  try {
    console.log('🔍 [VARMAX] Fetching saved predictions...');
    const response = await fetch(`${API_BASE_URL}/varmax/saved?limit=${limit}`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('저장된 VARMAX 예측 목록 조회 실패');
    }
    
    const data = await response.json();
    console.log('✅ [VARMAX] Saved predictions fetched:', data);
    return data;
  } catch (error) {
    console.error('❌ [VARMAX] Error fetching saved predictions:', error);
    return { 
      error: error.message || '저장된 VARMAX 예측 목록을 가져오는 중 오류가 발생했습니다.',
      success: false,
      predictions: []
    };
  }
};

// 특정 날짜의 저장된 VARMAX 예측 조회
export const getSavedVarmaxPredictionByDate = async (date) => {
  try {
    console.log(`🔍 [VARMAX] Fetching saved prediction for date: ${date}`);
    const response = await fetch(`${API_BASE_URL}/varmax/saved/${date}`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        return { success: false, error: `${date} 날짜의 저장된 예측을 찾을 수 없습니다.` };
      }
      throw new Error('저장된 VARMAX 예측 조회 실패');
    }
    
    const data = await response.json();
    console.log('✅ [VARMAX] Saved prediction fetched:', data);
    return data;
  } catch (error) {
    console.error(`❌ [VARMAX] Error fetching saved prediction for ${date}:`, error);
    return { 
      error: error.message || `${date} 날짜의 저장된 VARMAX 예측을 가져오는 중 오류가 발생했습니다.`,
      success: false 
    };
  }
};

// 특정 날짜의 저장된 VARMAX 예측 삭제
export const deleteSavedVarmaxPrediction = async (date) => {
  try {
    console.log(`🗑️ [VARMAX] Deleting saved prediction for date: ${date}`);
    const response = await fetch(`${API_BASE_URL}/varmax/saved/${date}`, {
      method: 'DELETE',
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        return { success: false, error: `${date} 날짜의 저장된 예측을 찾을 수 없습니다.` };
      }
      throw new Error('저장된 VARMAX 예측 삭제 실패');
    }
    
    const data = await response.json();
    console.log('✅ [VARMAX] Saved prediction deleted:', data);
    return data;
  } catch (error) {
    console.error(`❌ [VARMAX] Error deleting saved prediction for ${date}:`, error);
    return { 
      error: error.message || `${date} 날짜의 저장된 VARMAX 예측을 삭제하는 중 오류가 발생했습니다.`,
      success: false 
    };
  }
};

// VARMAX 상태 리셋 (hang된 예측 해결용)
export const resetVarmaxState = async () => {
  try {
    console.log('🔄 [VARMAX] Resetting VARMAX state...');
    const response = await fetch(`${API_BASE_URL}/varmax/reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('VARMAX 상태 리셋 실패');
    }
    
    const data = await response.json();
    console.log('✅ [VARMAX] State reset successfully:', data);
    return data;
  } catch (error) {
    console.error('❌ [VARMAX] Error resetting state:', error);
    return { 
      error: error.message || 'VARMAX 상태 리셋 중 오류가 발생했습니다.',
      success: false 
    };
  }
};
