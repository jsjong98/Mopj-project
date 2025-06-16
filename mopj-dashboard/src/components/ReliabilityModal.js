import React from 'react';
import { X, Target, ShoppingCart, TrendingUp, Award, AlertCircle } from 'lucide-react';

const ReliabilityModal = ({ isOpen, onClose, type }) => {
  if (!isOpen) return null;

  // 통일된 타이포그래피 시스템
  const typography = {
    sectionTitle: {
      fontSize: '1.5rem',
      fontWeight: '600',
      lineHeight: '1.3'
    },
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
    }
  };

  const modalContent = {
    prediction: {
      title: '예측 신뢰도 (Prediction Reliability)',
      icon: <Target size={24} style={{ color: '#3b82f6' }} />,
      description: 'AI 모델의 예측 일관성(안정성)을 나타내는 지표입니다. 변동계수를 기반으로 계산됩니다.',
      sections: [
        {
          title: '📊 계산 방법',
          content: [
            '• 일마다 예측한 결과들의 평균값을 계산합니다',
            '• 예측 결과들의 표준편차를 구합니다',
            '• 변동계수(CV) = 표준편차 / 평균 을 계산합니다',
            '• 변동계수를 기반으로 일관성 점수를 산출합니다',
            '• 점수 범위: 0~100점 (높을수록 우수, 변동계수가 낮을수록 높은 점수)',
            '• 예시: 평균 1000, 표준편차 50 → CV = 0.05 (5%) → 높은 일관성'
          ]
        },
        {
          title: '🎯 평가 기준',
          content: [
            '• 96점 이상: 예측 결과 신뢰 가능',
            '• 96점 미만: 예측 결과 활용 불권장',
            '• 기준값은 백테스팅을 통해 도출된 임계값입니다'
          ]
        },
        {
          title: '💡 해석 방법',
          content: [
            '• 높은 예측 신뢰도: 낮은 변동계수 → 예측이 일관되고 안정적',
            '• 낮은 예측 신뢰도: 높은 변동계수 → 예측이 불안정하고 산발적',
            '• 변동계수가 낮다 = 예측 결과들이 평균 주변에 일관되게 분포',
            '• 변동계수가 높다 = 예측 결과들이 크게 흩어져 있어 불안정',
            '• 예측의 정확도보다는 일관성(안정성)을 평가하는 지표입니다'
          ]
        }
      ]
    },
    purchase: {
      title: '구매 신뢰도 (Purchase Reliability)',
      icon: <ShoppingCart size={24} style={{ color: '#10b981' }} />,
      description: '구매 의사결정 추천의 신뢰성을 나타내는 지표입니다.',
      sections: [
        {
          title: '📊 계산 방법',
          content: [
            '• 구간별 점수(1점~3점)를 기반으로 계산됩니다',
            '• 모든 구간의 점수를 누적하여 계산합니다',
            '• 실제 획득 점수를 최대 가능 점수로 나눈 비율입니다',
            '• 계산식: (누적 점수 / 최대 점수) × 100',
            '• 예시: 10개 구간에서 각각 2, 3, 1, 3, 2... 점수를 받으면 → (총합/30) × 100'
          ]
        },
        {
          title: '🎯 평가 기준',
          content: [
            '• 63.7% 이상: 구매 타이밍 추천 활용 가능',
            '• 63.7% 미만: 구매 타이밍 추천 신중 검토 필요',
            '• 기준값은 과거 데이터 분석을 통해 도출되었습니다'
          ]
        },
        {
          title: '💡 해석 방법',
          content: [
            '• 높은 구매 신뢰도: 전반적으로 높은 점수(2~3점)를 많이 획득',
            '• 낮은 구매 신뢰도: 대부분 낮은 점수(1~2점) 구간으로 분류됨',
            '• 구매 신뢰도가 높을 때 구체적인 구매 계획 수립 가능',
            '• 모든 구간의 점수가 종합적으로 반영되어 전체적인 신뢰성 평가'
          ]
        },
        {
          title: '📈 구간 점수 체계',
          content: [
            '• 3점 (최적): 적극적 구매 추천 구간',
            '• 2점 (보통): 일반적인 구매 시점',
            '• 1점 (주의): 구매 보류 권장 구간'
          ]
        }
      ]
    }
  };

  const content = modalContent[type];
  if (!content) return null;

  const styles = {
    overlay: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    },
    modal: {
      backgroundColor: 'white',
      borderRadius: '0.75rem',
      padding: '1.5rem',
      maxWidth: '600px',
      width: '90%',
      maxHeight: '80vh',
      overflowY: 'auto',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '1.5rem',
      paddingBottom: '1rem',
      borderBottom: '1px solid #e5e7eb'
    },
         title: {
       display: 'flex',
       alignItems: 'center',
       gap: '0.75rem',
       ...typography.cardTitle,
       fontSize: '1.25rem', // 모달 제목은 약간 크게
       color: '#1f2937'
     },
    closeButton: {
      padding: '0.5rem',
      borderRadius: '0.375rem',
      backgroundColor: '#f3f4f6',
      border: 'none',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: 'background-color 0.2s'
    },
         description: {
       ...typography.content,
       color: '#6b7280',
       marginBottom: '1.5rem'
     },
    section: {
      marginBottom: '1.5rem'
    },
         sectionTitle: {
       ...typography.cardTitle,
       color: '#374151',
       marginBottom: '0.75rem',
       display: 'flex',
       alignItems: 'center',
       gap: '0.5rem'
     },
    sectionContent: {
      backgroundColor: '#f9fafb',
      borderRadius: '0.5rem',
      padding: '1rem',
      border: '1px solid #e5e7eb'
    },
         listItem: {
       ...typography.helper,
       color: '#4b5563',
       marginBottom: '0.5rem'
     },
    highlight: {
      backgroundColor: '#dbeafe',
      color: '#1e40af',
      padding: '0.75rem',
      borderRadius: '0.5rem',
      border: '1px solid #bfdbfe',
      marginTop: '1rem'
    },
    warningBox: {
      backgroundColor: '#fef3c7',
      color: '#92400e',
      padding: '0.75rem',
      borderRadius: '0.5rem',
      border: '1px solid #fcd34d',
      marginTop: '1rem',
      display: 'flex',
      alignItems: 'flex-start',
      gap: '0.5rem'
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <div style={styles.title}>
            {content.icon}
            {content.title}
          </div>
          <button 
            style={styles.closeButton}
            onClick={onClose}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#e5e7eb'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#f3f4f6'}
          >
            <X size={20} style={{ color: '#6b7280' }} />
          </button>
        </div>

        <div style={styles.description}>
          {content.description}
        </div>

        {content.sections.map((section, index) => (
          <div key={index} style={styles.section}>
            <div style={styles.sectionTitle}>
              {section.title}
            </div>
            <div style={styles.sectionContent}>
              {section.content.map((item, itemIndex) => (
                <div key={itemIndex} style={styles.listItem}>
                  {item}
                </div>
              ))}
            </div>
          </div>
        ))}

        {type === 'prediction' && (
          <div style={styles.highlight}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <TrendingUp size={16} />
              <strong>실무 활용 팁</strong>
            </div>
                         <div style={{ ...typography.helper }}>
               예측 신뢰도가 96점 이상일 때만 예측 결과를 기반으로 한 의사결정을 권장합니다. 
               낮은 신뢰도는 예측이 일관되지 않음을 의미하므로, 추가적인 시장 분석과 함께 신중히 검토하세요.
             </div>
          </div>
        )}

        {type === 'purchase' && (
          <>
            <div style={styles.highlight}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <Award size={16} />
                <strong>실무 활용 팁</strong>
              </div>
                             <div style={{ ...typography.helper }}>
                 구매 신뢰도가 높을 때는 전반적으로 좋은 점수를 받은 것이므로, 
                 구간별 점수표를 신뢰하고 활용하여 구매 계획을 수립하면 비용 절감 효과를 기대할 수 있습니다.
               </div>
            </div>
            
            <div style={styles.warningBox}>
              <AlertCircle size={16} style={{ marginTop: '0.125rem', flexShrink: 0 }} />
                             <div style={{ ...typography.helper }}>
                 <strong>주의사항:</strong> 구매 신뢰도가 낮더라도 예측 신뢰도가 높다면 
                 가격 추이 자체는 참고할 수 있습니다. 다만 구체적인 구매 타이밍 결정은 
                 다른 시장 요인들과 함께 종합적으로 검토하세요.
               </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ReliabilityModal; 
