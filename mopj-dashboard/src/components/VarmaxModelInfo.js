// src/components/VarmaxModelInfo.js
import React from 'react';
import { Zap, TrendingUp, BarChart3, Users } from 'lucide-react';

const VarmaxModelInfo = ({ data }) => {
  if (!data) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center', 
        color: '#6b7280',
        backgroundColor: '#f9fafb',
        borderRadius: '0.5rem'
      }}>
        <p>VARMAX 모델 정보가 없습니다.</p>
      </div>
    );
  }

  const styles = {
    container: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem',
      marginBottom: '1rem'
    },
    infoCard: {
      backgroundColor: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: '0.5rem',
      padding: '1rem',
      textAlign: 'center'
    },
    infoTitle: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      fontSize: '0.875rem',
      fontWeight: '500',
      color: '#64748b',
      marginBottom: '0.5rem'
    },
    infoValue: {
      fontSize: '1.5rem',
      fontWeight: 'bold',
      color: '#1e293b'
    },
    infoUnit: {
      fontSize: '0.875rem',
      color: '#64748b',
      marginLeft: '0.25rem'
    },
    description: {
      backgroundColor: '#fef9e7',
      border: '1px solid #fbbf24',
      borderRadius: '0.5rem',
      padding: '1rem',
      marginTop: '1rem'
    },
    descriptionTitle: {
      fontSize: '0.875rem',
      fontWeight: '600',
      color: '#92400e',
      marginBottom: '0.5rem'
    },
    descriptionText: {
      fontSize: '0.875rem',
      color: '#451a03',
      lineHeight: '1.4'
    }
  };

  return (
    <div>
      <div style={styles.container}>
        {/* 사용 변수 수 */}
        <div style={styles.infoCard}>
          <div style={styles.infoTitle}>
            <Users size={16} />
            사용 변수
          </div>
          <div style={styles.infoValue}>
            {data.variables_used || 8}
            <span style={styles.infoUnit}>개</span>
          </div>
        </div>

        {/* 예측 기간 */}
        <div style={styles.infoCard}>
          <div style={styles.infoTitle}>
            <TrendingUp size={16} />
            예측 기간
          </div>
          <div style={styles.infoValue}>
            {data.prediction_days || 50}
            <span style={styles.infoUnit}>일</span>
          </div>
        </div>
      </div>

      {/* 모델 설명 */}
      <div style={styles.description}>
        <div style={styles.descriptionTitle}>
          <Zap size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
          VARMAX 모델 특징
        </div>
        <div style={styles.descriptionText}>
          Vector Autoregression with eXogenous variables (VARMAX) 모델은 다변량 시계열 예측에 특화된 
          경제 통계학 모델입니다. 여러 변수들 간의 상호작용을 고려하여 시계열 예측을 수행합니다. 
          또한, 잔차 보정을 위해 RandomForest (RF) 모델을 
          추가로 적용하여 예측 정확도를 향상시켰습니다.
        </div>
      </div>
    </div>
  );
};

export default VarmaxModelInfo;