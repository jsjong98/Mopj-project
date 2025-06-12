// src/components/VarmaxResult.js
import React from 'react';

const styles = {
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '1rem'
  },
  card: {
    backgroundColor: '#f8fafc',
    border: '1px solid #e2e8f0',
    borderRadius: '0.5rem',
    padding: '1rem',
    textAlign: 'center'
  },
  period: {
    fontSize: '1rem',
    fontWeight: '500',
    color: '#1e293b',
    marginBottom: '0.5rem'
  },
  avg: {
    fontSize: '1.5rem',
    fontWeight: 'bold'
    // color는 아래 로직에서 동적으로 설정
  },
  noData: {
    color: '#6b7280',
    textAlign: 'center',
    padding: '2rem'
  }
};

const parseLabel = (label) => {
  // 🔧 안전성 검사 추가
  if (!label || typeof label !== 'string') {
    console.warn('parseLabel: Invalid label received:', label);
    return '알 수 없음'; // 기본값 반환
  }

  try {
    const parts = label.split('_');
    if (parts.length !== 3) {
      console.warn('parseLabel: Unexpected label format:', label);
      return label; // 원본 그대로 반환
    }

    const [yy, mm, half] = parts;
    
    // 추가 검증
    if (!yy || !mm || !half) {
      console.warn('parseLabel: Missing parts in label:', label);
      return label;
    }

    const halfKor = half === '1' ? '상반기' : '하반기';
    return `${yy}년 ${mm}월 ${halfKor}`;
  } catch (error) {
    console.error('parseLabel: Error parsing label:', label, error);
    return '파싱 오류';
  }
};

const VarmaxResult = ({ data }) => {
  console.log('🔍 [VarmaxResult] Received data:', data);

  // 🔧 강화된 데이터 검증
  if (!data) {
    console.warn('VarmaxResult: No data provided');
    return <p style={styles.noData}>데이터가 제공되지 않았습니다</p>;
  }

  if (!Array.isArray(data)) {
    console.warn('VarmaxResult: Data is not an array:', typeof data);
    return <p style={styles.noData}>잘못된 데이터 형식입니다</p>;
  }

  if (data.length === 0) {
    console.warn('VarmaxResult: Data array is empty');
    return <p style={styles.noData}>데이터가 없습니다</p>;
  }

  try {
    // 🔧 필터링: 유효한 데이터만 처리
    const validData = data.filter(item => {
      if (!item) return false;
      if (!item.half_month_label || typeof item.half_month_label !== 'string') {
        console.warn('Invalid half_month_label:', item);
        return false;
      }
      if (typeof item.half_month_avg !== 'number' || isNaN(item.half_month_avg)) {
        console.warn('Invalid half_month_avg:', item);
        return false;
      }
      return true;
    });

    if (validData.length === 0) {
      console.warn('VarmaxResult: No valid data after filtering');
      return <p style={styles.noData}>유효한 데이터가 없습니다</p>;
    }

    // 중복 제거
    const unique = Array.from(
      new Map(validData.map(item => [item.half_month_label, item])).values()
    );

    console.log('🔍 [VarmaxResult] Unique data:', unique);

    // 평균값들만 뽑아서 최저/최고 계산
    const avgs = unique.map(item => item.half_month_avg);
    const minAvg = Math.min(...avgs);
    const maxAvg = Math.max(...avgs);

    return (
      <div style={styles.grid}>
        {unique.map((item, index) => {
          // 🔧 안전한 키 생성
          const safeKey = item.half_month_label || `item-${index}`;
          
          // 컬러 결정
          let avgColor = '#000';                    // 기본 검정
          if (item.half_month_avg === minAvg) {
            avgColor = '#3b82f6'; // 파랑
          } else if (item.half_month_avg === maxAvg) {
            avgColor = '#ef4444'; // 빨강
          }

          return (
            <div key={safeKey} style={styles.card}>
              <div style={styles.period}>
                {parseLabel(item.half_month_label)}
              </div>
              <div
                style={{
                  ...styles.avg,
                  color: avgColor
                }}
              >
                {typeof item.half_month_avg === 'number' 
                  ? item.half_month_avg.toFixed(2) 
                  : 'N/A'}
              </div>
            </div>
          );
        })}
      </div>
    );

  } catch (error) {
    console.error('VarmaxResult: Unexpected error:', error);
    return <p style={styles.noData}>데이터 처리 중 오류가 발생했습니다</p>;
  }
};

export default VarmaxResult;
