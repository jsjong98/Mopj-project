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

const AccumulatedMetricsChart = ({ data }) => {
  if (!data || !data.predictions || data.predictions.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <p style={styles.noDataText}>예측 가격 범위 및 변동성 데이터가 없습니다</p>
      </div>
    );
  }

  // 반월 기간 계산 함수들
  const getSemimonthlyPeriod = (dateString) => {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    
    if (day <= 15) {
      return `${year}-${month.toString().padStart(2, '0')}-SM1`;
    } else {
      return `${year}-${month.toString().padStart(2, '0')}-SM2`;
    }
  };

  const getNextSemimonthlyPeriod = (currentPeriod) => {
    const match = currentPeriod.match(/(\d{4})-(\d{2})-(SM[12])/);
    if (!match) return null;
    
    const [, year, month, sm] = match;
    const yearNum = parseInt(year);
    const monthNum = parseInt(month);
    
    if (sm === 'SM1') {
      return `${year}-${month}-SM2`;
    } else {
      const nextMonth = monthNum + 1;
      if (nextMonth > 12) {
        return `${yearNum + 1}-01-SM1`;
      } else {
        return `${year}-${nextMonth.toString().padStart(2, '0')}-SM1`;
      }
    }
  };

  const getSemimonthlyDateRange = (period) => {
    const match = period.match(/(\d{4})-(\d{2})-(SM[12])/);
    if (!match) return null;
    
    const [, year, month, sm] = match;
    const yearNum = parseInt(year);
    const monthNum = parseInt(month);
    
    if (sm === 'SM1') {
      return {
        start: `${year}-${month}-01`,
        end: `${year}-${month}-15`
      };
    } else {
      const lastDay = new Date(yearNum, monthNum, 0).getDate();
      return {
        start: `${year}-${month}-16`,
        end: `${year}-${month}-${lastDay.toString().padStart(2, '0')}`
      };
    }
  };

  // 예측 시작일 계산 함수
  const calculatePredictionStartDate = (dataEndDate) => {
    const date = new Date(dataEndDate);
    date.setDate(date.getDate() + 1);
    
    // 주말이면 다음 월요일까지 이동
    while (date.getDay() === 0 || date.getDay() === 6) {
      date.setDate(date.getDate() + 1);
    }
    
    return date.toISOString().split('T')[0];
  };

  // 반월별로 데이터 그룹화
  const semimonthlyGroups = {};
  
  data.predictions.forEach(item => {
    const predictionStartDate = item.prediction_start_date || calculatePredictionStartDate(item.date);
    const currentPeriod = getSemimonthlyPeriod(predictionStartDate);
    
    if (!semimonthlyGroups[currentPeriod]) {
      semimonthlyGroups[currentPeriod] = [];
    }
    
    semimonthlyGroups[currentPeriod].push({
      ...item,
      predictionStartDate
    });
  });

  // 각 반월별로 다음 반월의 각 날짜별 통계 계산
  const chartData = [];
  
  Object.entries(semimonthlyGroups).forEach(([period, items]) => {
    const nextPeriod = getNextSemimonthlyPeriod(period);
    const nextPeriodRange = getSemimonthlyDateRange(nextPeriod);
    
    if (!nextPeriodRange) return;
    
    // 다음 반월의 각 날짜별로 모든 예측값들 수집
    const dateWisePredictions = {};
    
    items.forEach(item => {
      const predictions = Array.isArray(item.predictions) ? item.predictions : [];
      
      predictions.forEach(p => {
        const predDate = p.Date || p.date;
        if (!predDate) return;
        
        const predDateStr = String(predDate).split('T')[0];
        
        // 다음 반월 기간에 포함되는지 확인
        if (predDateStr >= nextPeriodRange.start && predDateStr <= nextPeriodRange.end) {
          const price = p.Prediction || p.prediction;
          if (typeof price === 'number' && !isNaN(price)) {
            if (!dateWisePredictions[predDateStr]) {
              dateWisePredictions[predDateStr] = [];
            }
            dateWisePredictions[predDateStr].push(price);
          }
        }
      });
    });
    
    // 각 날짜별로 통계 계산하여 차트 데이터 생성
    Object.entries(dateWisePredictions).forEach(([date, prices]) => {
      if (prices.length > 0) {
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        
        // 변동계수 계산
        let coefficientOfVariation = 0;
        if (prices.length > 1) {
          const mean = prices.reduce((sum, price) => sum + price, 0) / prices.length;
          const variance = prices.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / prices.length;
          const standardDeviation = Math.sqrt(variance);
          coefficientOfVariation = (standardDeviation / mean) * 100;
        }
        
        chartData.push({
          date: date,                     // 예측 날짜 (다음 반월의 특정 날짜)
          sourcePeriod: period,           // 예측 시작 반월
          targetPeriod: nextPeriod,       // 예측 대상 반월
          minPrice,
          maxPrice,
          priceRange: maxPrice - minPrice,
          coefficientOfVariation,
          predictionCount: prices.length, // 해당 날짜에 대한 예측 개수
          minPriceDate: date,             // 최저가 발생일 = 예측 날짜
          maxPriceDate: date              // 최고가 발생일 = 예측 날짜
        });
      }
    });
  });
  
  // 날짜순으로 정렬
  chartData.sort((a, b) => new Date(a.date) - new Date(b.date));

  // 반월 기간 형식화 함수
  const formatSemimonthlyPeriod = (period) => {
    if (!period) return '';
    
    // 2025-06-SM1 -> 25-06-상, 2025-06-SM2 -> 25-06-하
    const match = period.match(/(\d{4})-(\d{2})-(SM[12])/);
    if (match) {
      const [, year, month, sm] = match;
      const shortYear = year.slice(2);
      const smText = sm === 'SM1' ? '상' : '하';
      return `${shortYear}-${month}-${smText}`;
    }
    return period;
  };

  // 툴팁 커스텀 포맷터
  const customTooltipFormatter = (value, name, props) => {
    let displayName = name;
    let formattedValue = value;
    
    if (name === 'minPrice') {
      displayName = '최저가';
      const minDate = props.payload?.minPriceDate;
      formattedValue = `${value.toFixed(2)} (${minDate || 'N/A'})`;
    }
    else if (name === 'maxPrice') {
      displayName = '최고가';
      const maxDate = props.payload?.maxPriceDate;
      formattedValue = `${value.toFixed(2)} (${maxDate || 'N/A'})`;
    }
    else if (name === 'coefficientOfVariation') {
      displayName = '변동계수';
      formattedValue = `${value.toFixed(2)}%`;
    }
    else {
      formattedValue = value.toFixed(2);
    }

    return [formattedValue, displayName];
  };

  // 날짜 형식화 함수
  const formatDate = (dateString) => {
    if (!dateString) return '';
    return dateString;
  };

  // 가격 포맷터 함수 - 10 단위로 반올림
  const formatPrice = (value) => {
    if (typeof value !== 'number' || isNaN(value)) return '';
    return (Math.round(value / 10) * 10).toString();
  };

  // 변동계수 포맷터 함수
  const formatCoefficient = (value) => {
    if (typeof value !== 'number' || isNaN(value)) return '';
    return value.toFixed(1);
  };

  // 툴팁 라벨 포맷터
  const customLabelFormatter = (label, payload) => {
    if (payload && payload.length > 0) {
      const data = payload[0].payload;
      return `${label} | ${formatSemimonthlyPeriod(data.sourcePeriod)} → ${formatSemimonthlyPeriod(data.targetPeriod)} | ${data.predictionCount}개 예측`;
    }
    return label;
  };

  return (
    <div style={styles.chartContainer}>
      <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            interval={0}
            angle={-45}
            textAnchor="end"
            height={60}
            tickFormatter={formatDate}
          />
          <YAxis 
            yAxisId="price"
            domain={['dataMin - 20', 'dataMax + 20']} 
            tick={{ fontSize: 12 }}
            tickCount={7}
            allowDecimals={false}
            tickFormatter={formatPrice}
            label={{ value: '가격 ($/MT)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
          />
          <YAxis 
            yAxisId="coefficient" 
            orientation="right" 
            domain={[0, 'dataMax + 1']}
            tick={{ fontSize: 12 }}
            tickCount={6}
            tickFormatter={formatCoefficient}
            label={{ value: '변동계수 (%)', angle: 90, position: 'insideRight', style: { textAnchor: 'middle' } }}
          />
          <Tooltip 
            formatter={customTooltipFormatter}
            labelFormatter={customLabelFormatter}
            labelStyle={{ fontWeight: 'bold' }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="minPrice" 
            stroke="#10b981" 
            strokeWidth={2.5} 
            name="최저가" 
            dot={{ r: 5 }}
            yAxisId="price"
          />
          <Line 
            type="monotone" 
            dataKey="maxPrice" 
            stroke="#ef4444" 
            strokeWidth={2.5} 
            name="최고가" 
            dot={{ r: 5 }}
            yAxisId="price"
          />
          <Line 
            type="monotone" 
            dataKey="coefficientOfVariation" 
            stroke="#8b5cf6" 
            strokeWidth={2} 
            name="변동계수" 
            dot={{ r: 4 }}
            yAxisId="coefficient"
            strokeDasharray="5 5"
          />

        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AccumulatedMetricsChart;
