import React from 'react';
// VarmaxAlgorithm은 테이블 기반으로 차트 컴포넌트가 필요하지 않음

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

const VarmaxAlgorithm = ({ data, columns, title }) => {
  if (!data || data.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <span style={styles.noDataText}>데이터가 없습니다.</span>
      </div>
    );
  }
  const topRows = data.slice(0, 3);

  // "샘플 수"의 최대값 구하기
  const maxSample = Math.max(...topRows.map(row => Number(row['샘플 수'] ?? 0)));

  return (
    <div style={styles.chartContainer}>
      <h3 style={{ marginBottom: '0.5rem' }}>{title}</h3>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr>
            {columns.map((header) => (
              <th key={header} style={{ border: '1px solid #ccc', padding: '8px', background: '#dbeafe' }}>
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {topRows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((header) => {
                // "샘플 수" 컬럼에만 바 시각화 적용 예시
                if (header === '샘플 수' && maxSample > 0) {
                  const widthPercent = (Number(row[header]) / maxSample) * 100;
                  return (
                    <td key={header} style={{ border: '1px solid #eee', padding: '8px', position: 'relative' }}>
                      <div style={{
                        background: '#60a5fa',
                        height: '1.5em',
                        width: `${widthPercent}%`,
                        position: 'absolute',
                        left: 0,
                        top: '50%',
                        transform: 'translateY(-50%)',
                        zIndex: 0,
                        opacity: 0.3,
                        borderRadius: '4px'
                      }} />
                      <span style={{ position: 'relative', zIndex: 1 }}>{row[header]}</span>
                    </td>
                  );
                }
                // 그 외 컬럼은 기본 스타일
                return (
                  <td key={header} style={{ border: '1px solid #eee', padding: '8px' }}>
                    {row[header]}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};



export default VarmaxAlgorithm;
