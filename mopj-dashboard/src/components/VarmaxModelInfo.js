// src/components/VarmaxModelInfo.js
import React from 'react';
import { Zap, TrendingUp, Users } from 'lucide-react';

// 통일된 타이포그래피 시스템
const typography = {
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
  large: {
    fontSize: '1.5rem',
    fontWeight: '600',
    lineHeight: '1.3'
  }
};

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
      display: 'flex',
      flexDirection: 'row',
      gap: '1rem',
      marginBottom: '1.5rem'
    },
    section: {
      backgroundColor: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: '0.5rem',
      padding: '1rem',
      flex: '1',
      minWidth: '0'
    },
    sectionTitle: {
      fontWeight: '600',
      color: '#1e293b',
      marginBottom: '0.5rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      ...typography.helper
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem'
    },
    statCard: {
      backgroundColor: 'white',
      padding: '1rem',
      borderRadius: '0.375rem',
      textAlign: 'center'
    },
    statValue: {
      fontWeight: 'bold',
      color: '#059669',
      ...typography.large
    },
    statLabel: {
      color: '#6b7280',
      marginTop: '0.25rem',
      ...typography.helper
    },
    equationContainer: {
      backgroundColor: 'white',
      padding: '1rem',
      borderRadius: '0.375rem'
    },
    equation: {
      fontFamily: 'monospace',
      color: '#1e293b',
      textAlign: 'center',
      ...typography.helper
    },
    noData: {
      color: '#6b7280',
      textAlign: 'center',
      ...typography.helper
    },
    modelSection: {
      backgroundColor: '#f8fafc',
      border: '1px solid #e2e8f0',
      borderRadius: '0.5rem',
      padding: '1rem',
      marginTop: '1rem'
    }
  };

  return (
    <div>
      <div style={styles.container}>
        {/* 사용 변수 수 */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>
            <Users size={16} />
            사용 변수
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>
              {data.variables_used || 8}
            </div>
            <div style={styles.statLabel}>개</div>
          </div>
        </div>

        {/* 예측 기간 */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>
            <TrendingUp size={16} />
            예측 기간
          </div>
          <div style={styles.statCard}>
            <div style={styles.statValue}>
              {data.prediction_days || 50}
            </div>
            <div style={styles.statLabel}>일</div>
          </div>
        </div>
      </div>

      {/* 모델 설명 */}
      <div style={styles.modelSection}>
        <div style={styles.sectionTitle}>
          <Zap size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
          VARMAX 모델 특징
        </div>
        <div style={styles.equationContainer}>
          <div style={styles.equation}>
            Vector Autoregression with eXogenous variables (VARMAX) 모델은 다변량 시계열 예측에 특화된 
            경제 통계학 모델입니다. 여러 변수들 간의 상호작용을 고려하여 시계열 예측을 수행합니다. 
            또한, 잔차 보정을 위해 RandomForest (RF) 모델을 
            추가로 적용하여 예측 정확도를 향상시켰습니다.
          </div>
        </div>
      </div>
    </div>
  );
};

export default VarmaxModelInfo;
