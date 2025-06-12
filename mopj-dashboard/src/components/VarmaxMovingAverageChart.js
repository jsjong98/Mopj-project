import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const styles = {
  chartContainer: {
    height: '16rem'
  },
  noDataContainer: {
    height: '16rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  noDataText: {
    color: '#6b7280'
  }
};

const VarmaxMovingAverageChart = ({ data }) => {
  // MA 데이터 구조 변환
  const transformData = () => {
    console.log('🔍 [VarmaxMA] Received data:', data);
    
    if (!data || typeof data !== 'object') {
      console.warn('🔍 [VarmaxMA] No data or invalid data type');
      return [];
    }
    
    // 사용 가능한 이동평균 키들 확인
    const availableKeys = Object.keys(data).filter(key => key.startsWith('ma'));
    console.log('🔍 [VarmaxMA] Available MA keys:', availableKeys);
    
    if (availableKeys.length === 0) {
      console.warn('🔍 [VarmaxMA] No moving average data found');
      return [];
    }
    
    // 첫 번째 사용 가능한 키를 기준으로 데이터 구조 생성
    const baseKey = availableKeys[0];
    const baseData = data[baseKey];
    
    if (!Array.isArray(baseData) || baseData.length === 0) {
      console.warn('🔍 [VarmaxMA] Base data is not valid array:', baseData);
      return [];
    }
    
    console.log('🔍 [VarmaxMA] Base data sample:', baseData[0]);
    
    // 동적으로 이동평균 데이터 병합
    return baseData.map((item, index) => {
      const result = {
        date: item.date,
        prediction: item.prediction,
        actual: item.actual,
        ma5: data.ma5 && index < data.ma5.length ? data.ma5[index].ma : null,
        ma10: data.ma10 && index < data.ma10.length ? data.ma10[index].ma : null,
        ma20: data.ma20 && index < data.ma20.length ? data.ma20[index].ma : null,
        ma23: data.ma23 && index < data.ma23.length ? data.ma23[index].ma : null,
        ma30: data.ma30 && index < data.ma30.length ? data.ma30[index].ma : null,
      };
      
      // 첫 번째 아이템만 로깅
      if (index === 0) {
        console.log('🔍 [VarmaxMA] First transformed item:', result);
      }
      
      return result;
    });
  };

  const chartData = transformData();

  if (chartData.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <div style={{ textAlign: 'center' }}>
          <p style={styles.noDataText}>이동평균 데이터가 없습니다</p>
          <p style={{ fontSize: '0.875rem', color: '#9ca3af', marginTop: '0.5rem' }}>
            VARMAX 예측을 실행하거나 저장된 예측을 불러와 주세요
          </p>
        </div>
      </div>
    );
  }

  // 날짜 형식화 함수
  const formatDate = (dateString) => {
    if (!dateString) return '';
    
    // 이미 YYYY-MM-DD 형식이면 그대로 반환
    if (/^\d{4}-\d{2}-\d{2}$/.test(dateString)) {
      return dateString;
    }
    
    // GMT 포함된 문자열이면 파싱하여 변환
    if (dateString.includes('GMT')) {
      const date = new Date(dateString);
      return date.toISOString().split('T')[0];
    }
    
    // 기타 경우 처리
    try {
      const date = new Date(dateString);
      return date.toISOString().split('T')[0];
    } catch (e) {
      console.error('날짜 변환 오류:', e);
      return dateString;
    }
  };

  return (
    <div style={styles.chartContainer}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            tickFormatter={formatDate}
          />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip
            formatter={(value, name) => {
              if (value === null) return ['데이터 없음', name];
              let displayName;
              switch(name) {
                case "prediction":
                  displayName = "예측 가격";
                  break;
                case "ma5":
                  displayName = "5일 이동평균";
                  break;
                case "ma10":
                  displayName = "10일 이동평균";
                  break;
                case "ma20":
                  displayName = "20일 이동평균";
                  break;
                case "ma23":
                  displayName = "23일 이동평균";
                  break;
                case "ma30":
                  displayName = "30일 이동평균";
                  break;
                default:
                  displayName = name;  // 알 수 없는 경우 원본 이름 표시
              }
              return [`${parseFloat(value).toFixed(2)}`, displayName];
            }}
            labelFormatter={(label) => `날짜: ${formatDate(label)}`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="prediction" 
            stroke="#ef4444" 
            strokeWidth={1.5} 
            name="예측 가격" 
            dot={false}
            opacity={0.5}
          />
          <Line 
            type="monotone" 
            dataKey="ma5" 
            stroke="#3b82f6" 
            strokeWidth={2.5} 
            name="5일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma10" 
            stroke="#16a34a" 
            strokeWidth={2.5} 
            name="10일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma20" 
            stroke="#2B2E4A" 
            strokeWidth={2.5} 
            name="20일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma23" 
            stroke="#f59e0b" 
            strokeWidth={2.5} 
            name="23일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma30" 
            stroke="#9333ea" 
            strokeWidth={2.5} 
            name="30일 이동평균" 
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default VarmaxMovingAverageChart;
