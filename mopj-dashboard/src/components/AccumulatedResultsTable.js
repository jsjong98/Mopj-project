import React from 'react';
import { Calendar, TrendingUp, TrendingDown, Minus, DollarSign } from 'lucide-react';

// 통일된 타이포그래피 시스템
const typography = {
  cardTitle: {
    fontSize: '1.125rem',
    fontWeight: '600',
    lineHeight: '1.4'
  },
  content: {
    fontSize: '1rem',
    fontWeight: '400',
    lineHeight: '1.6'
  },
  helper: {
    fontSize: '0.875rem',
    fontWeight: '400',
    lineHeight: '1.5'
  },
  small: {
    fontSize: '0.75rem',
    fontWeight: '400',
    lineHeight: '1.4'
  }
};

const styles = {
  container: {
    maxHeight: '16rem',
    overflowY: 'auto'
  },
  noData: {
    height: '16rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  noDataText: {
    color: '#6b7280',
    ...typography.helper
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    ...typography.helper
  },
  tableHeader: {
    position: 'sticky',
    top: 0,
    backgroundColor: '#f9fafb',
    zIndex: 10
  },
  th: {
    padding: '0.75rem 1rem',
    textAlign: 'left',
    fontWeight: '500',
    color: '#6b7280',
    borderBottom: '1px solid #e5e7eb',
    ...typography.small
  },
  td: {
    padding: '0.5rem 1rem',
    borderBottom: '1px solid #e5e7eb',
    color: '#4b5563',
    ...typography.helper
  },
  date: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  dateIcon: {
    color: '#2563eb'
  },
  good: {
    color: '#10b981',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  warning: {
    color: '#f59e0b',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  bad: {
    color: '#ef4444',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  neutral: {
    color: '#6b7280',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  currentRow: {
    backgroundColor: '#f0fdf4',
  },
  buttonContainer: {
    display: 'flex',
    gap: '0.5rem',
    flexDirection: 'column'
  },
  viewButton: {
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    padding: '0.375rem 0.5rem',
    borderRadius: '0.25rem',
    cursor: 'pointer',
    whiteSpace: 'normal',
    textAlign: 'center',
    lineHeight: '1.2',
    minHeight: '2rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ...typography.small
  }
};

// 가격 경향 표시
const getPriceTrend = (predictions) => {
  if (!Array.isArray(predictions) || predictions.length < 2) {
    return <div style={styles.neutral}><Minus size={14} /> 중립</div>;
  }
  
  const validPredictions = predictions.filter(p => {
    const price = p.Prediction || p.prediction;
    return p && typeof price === 'number' && !isNaN(price);
  });
  
  if (validPredictions.length < 2) {
    return <div style={styles.neutral}><Minus size={14} /> 중립</div>;
  }
  
  const firstPrice = validPredictions[0].Prediction || validPredictions[0].prediction;
  const lastPrice = validPredictions[validPredictions.length - 1].Prediction || validPredictions[validPredictions.length - 1].prediction;
  const change = ((lastPrice - firstPrice) / firstPrice) * 100;
  
  if (change > 3) {
    return <div style={styles.bad}><TrendingUp size={14} /> 상승 ({change.toFixed(1)}%)</div>;
  } else if (change < -3) {
    return <div style={styles.good}><TrendingDown size={14} /> 하락 ({change.toFixed(1)}%)</div>;
  } else {
    return <div style={styles.neutral}><Minus size={14} /> 중립 ({change.toFixed(1)}%)</div>;
  }
};

// 가격 범위 표시
const getPriceRange = (predictions) => {
  if (!Array.isArray(predictions) || predictions.length === 0) {
    return '-';
  }
  
  const prices = predictions.map(p => p.Prediction || p.prediction).filter(price => typeof price === 'number' && !isNaN(price));
  
  if (prices.length === 0) {
    return '-';
  }
  
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  
  return `${minPrice.toFixed(2)} ~ ${maxPrice.toFixed(2)}`;
};

// 최적 구매 기간 표시
const getBestInterval = (intervalScores) => {
  if (!intervalScores) {
    return null;
  }
  
  let intervals = [];
  
  if (Array.isArray(intervalScores)) {
    // 배열인 경우
    intervals = intervalScores.filter(
      interval => interval && typeof interval === 'object' && interval.days != null
    );
  } else if (typeof intervalScores === 'object') {
    // 객체인 경우
    intervals = Object.values(intervalScores).filter(
      interval => interval && typeof interval === 'object' && interval.days != null
    );
  }
  
  if (intervals.length === 0) {
    return null;
  }
  
  // 최고 점수 구간 찾기
  const bestInterval = intervals.reduce((best, current) => 
    (current.score > best.score) ? current : best
  , intervals[0]);
  
  return (
    <div style={{display: 'flex', alignItems: 'center', gap: '0.25rem'}}>
      <DollarSign size={14} style={{color: '#10b981'}} />
      {bestInterval.start_date} ~ {bestInterval.end_date} ({bestInterval.days}일)
    </div>
  );
};

const AccumulatedResultsTable = ({ data, currentDate, onSelectDate, onViewInSingle }) => {
  if (!data || !data.predictions || data.predictions.length === 0) {
    return (
      <div style={styles.noData}>
        <p style={styles.noDataText}>누적 예측 결과 데이터가 없습니다</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <table style={styles.table}>
        <thead style={styles.tableHeader}>
          <tr>
            <th style={styles.th}>예측 시작일</th>
            <th style={styles.th}>가격 범위</th>
            <th style={styles.th}>가격 경향</th>
            <th style={styles.th}>최적 구매 기간</th>
            <th style={styles.th}>작업</th>
          </tr>
        </thead>
        <tbody>
          {data.predictions.map((item) => {
            // 데이터 기준일로부터 예측 시작일 계산
            const calculatePredictionStartDate = (dataEndDate) => {
              const date = new Date(dataEndDate);
              date.setDate(date.getDate() + 1);
              
              // 주말이면 다음 월요일까지 이동
              while (date.getDay() === 0 || date.getDay() === 6) {
                date.setDate(date.getDate() + 1);
              }
              
              return date.toISOString().split('T')[0];
            };
            
            const predictionStartDate = item.prediction_start_date || calculatePredictionStartDate(item.date);
            
            return (
              <tr 
                key={item.date} 
                style={item.date === currentDate ? styles.currentRow : null}
              >
                <td style={styles.td}>
                  <div style={styles.date}>
                    <Calendar size={14} style={styles.dateIcon} />
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <div style={{ ...typography.helper, fontWeight: '500' }}>
                        🚀 예측 시작일: {predictionStartDate}
                      </div>
                      <div style={{ ...typography.small, color: '#6b7280' }}>
                        📊 데이터 기준일: {item.date}
                      </div>
                    </div>
                  </div>
                </td>
                <td style={styles.td}>{getPriceRange(item.predictions)}</td>
                <td style={styles.td}>{getPriceTrend(item.predictions)}</td>
                <td style={styles.td}>{getBestInterval(item.interval_scores)}</td>
                <td style={styles.td}>
                  <div style={styles.buttonContainer}>
                    <button 
                      style={styles.viewButton}
                      onClick={() => onSelectDate(item.date)}
                      title={`데이터 기준일: ${item.date}까지 / 예측 시작일: ${predictionStartDate}부터 23일간 예측 결과 보기`}
                    >
                      상세 보기
                    </button>
                    {onViewInSingle && (
                      <button 
                        style={{
                          ...styles.viewButton,
                          backgroundColor: '#10b981'
                        }}
                        onClick={() => onViewInSingle(item.date)}
                        title={`${item.date} 예측을 단일 예측 탭에서 상세히 보기`}
                      >
                        단일로 보기
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default AccumulatedResultsTable;
