import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, RefreshCw, Activity, Database, Upload } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import FileUploader from './FileUploader';

const MarketStatus = ({ fileInfo, windowWidth }) => {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedCategories, setExpandedCategories] = useState({});
  
  // 자체 파일 관리 상태
  const [localFileInfo, setLocalFileInfo] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  // 실제 사용할 파일 정보 (prop으로 받은 것 or 자체 업로드한 것)
  const activeFileInfo = localFileInfo || fileInfo;

  // 카테고리 아이콘 매핑
  const categoryIcons = {
    '원유 가격': '🛢️',
    '가솔린 가격': '⛽',
    '나프타 가격': '🏭',
    'LPG 가격': '🔥',
    '석유화학 제품 가격': '🧪'
  };

  // 파일 업로드 성공 핸들러
  const handleUploadSuccess = (data) => {
    console.log('✅ [MARKET_STATUS] File uploaded:', data);
    setLocalFileInfo(data);
    setError(null);
  };

  // 데이터 로드
  const loadMarketData = useCallback(async () => {
    if (!activeFileInfo?.file_path && !activeFileInfo?.filepath) {
      console.log('📋 [MARKET_STATUS] No file path available');
      return;
    }

    let filePath = activeFileInfo.file_path || activeFileInfo.filepath;
    
    // 파일 경로 정규화 (Windows 백슬래시를 슬래시로 변환)
    filePath = filePath.replace(/\\/g, '/');
    console.log('📋 [MARKET_STATUS] Normalized file path:', filePath);
    
    setLoading(true);
    setError(null);

    try {
      const { getMarketStatus } = await import('../services/api');
      const response = await getMarketStatus(filePath);

      if (response.success) {
        setMarketData(response);
        console.log('✅ [MARKET_STATUS] Data loaded successfully');
      } else {
        setError(response.error || '데이터를 불러오는데 실패했습니다.');
      }
    } catch (err) {
      console.error('❌ [MARKET_STATUS] Load error:', err);
      setError('시장 시황 데이터 로드 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  }, [activeFileInfo?.file_path, activeFileInfo?.filepath]);

  // 컴포넌트 마운트 시 및 파일 변경 시 데이터 로드
  useEffect(() => {
    loadMarketData();
  }, [loadMarketData]);

  // 카테고리 확장/축소 토글
  const toggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  // 가격 변화 계산
  const calculatePriceChange = (data, column) => {
    if (!data || data.length < 2) return null;

    const latestValue = data[data.length - 1]?.values[column];
    const previousValue = data[data.length - 2]?.values[column];

    if (latestValue == null || previousValue == null) return null;

    const change = latestValue - previousValue;
    const changePercent = (change / previousValue) * 100;

    return {
      absolute: change,
      percent: changePercent,
      isPositive: change >= 0
    };
  };

  // 차트 데이터 변환
  const transformChartData = (categoryData, columns) => {
    return categoryData.data.map(item => {
      const chartPoint = { date: item.date };
      columns.forEach(column => {
        if (item.values[column] != null) {
          chartPoint[column] = item.values[column];
        }
      });
      return chartPoint;
    });
  };

  // Y축 범위 계산 (각 컬럼별로)
  const calculateYAxisRange = (categoryData, column) => {
    const values = categoryData.data
      .map(item => item.values[column])
      .filter(val => val != null && !isNaN(val));
    
    if (values.length === 0) return { min: 0, max: 100 };
    
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const padding = range * 0.1; // 10% 여백
    
    return {
      min: Math.max(0, min - padding), // 최소값은 0 이상
      max: max + padding
    };
  };

  // 색상 팔레트
  const colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c', '#0891b2'];

  const styles = {
    container: {
      backgroundColor: 'white',
      borderRadius: '0.5rem',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
      padding: '1.5rem',
      marginBottom: '1.5rem'
    },
    header: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '1.5rem'
    },
    title: {
      fontSize: '1.25rem',
      fontWeight: '600',
      color: '#1f2937',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem'
    },
    refreshButton: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.25rem',
      backgroundColor: '#f3f4f6',
      color: '#374151',
      padding: '0.5rem 1rem',
      borderRadius: '0.375rem',
      cursor: 'pointer',
      border: 'none',
      fontSize: '0.875rem',
      transition: 'background-color 0.2s'
    },
    periodInfo: {
      backgroundColor: '#f9fafb',
      padding: '1rem',
      borderRadius: '0.5rem',
      marginBottom: '1.5rem',
      border: '1px solid #e5e7eb'
    },
    categoryCard: {
      border: '1px solid #e5e7eb',
      borderRadius: '0.5rem',
      marginBottom: '1rem',
      overflow: 'hidden',
      transition: 'box-shadow 0.2s'
    },
    categoryHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '1rem',
      backgroundColor: '#f9fafb',
      cursor: 'pointer',
      transition: 'background-color 0.2s'
    },
    categoryTitle: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      fontSize: '1rem',
      fontWeight: '500',
      color: '#374151'
    },
    categoryContent: {
      padding: '1rem',
      backgroundColor: 'white'
    },
    priceGrid: {
      display: 'grid',
      gridTemplateColumns: windowWidth < 768 ? '1fr' : 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem',
      marginBottom: '1.5rem'
    },
    priceCard: {
      backgroundColor: '#f8fafc',
      padding: '1rem',
      borderRadius: '0.375rem',
      border: '1px solid #e2e8f0'
    },
    priceLabel: {
      fontSize: '0.875rem',
      color: '#64748b',
      marginBottom: '0.25rem'
    },
    priceValue: {
      fontSize: '1.125rem',
      fontWeight: '600',
      color: '#1e293b',
      marginBottom: '0.25rem'
    },
    priceChange: (isPositive) => ({
      fontSize: '0.75rem',
      color: isPositive ? '#059669' : '#dc2626',
      display: 'flex',
      alignItems: 'center',
      gap: '0.25rem'
    }),
    chartContainer: {
      height: '240px',
      marginTop: '0.25rem',
      marginBottom: '0.25rem',
      padding: '0.5rem'
    },
    loadingSpinner: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      color: '#6b7280'
    },
    errorMessage: {
      backgroundColor: '#fef2f2',
      color: '#dc2626',
      padding: '1rem',
      borderRadius: '0.5rem',
      border: '1px solid #fecaca'
    },
    uploadSection: {
      backgroundColor: '#f9fafb',
      padding: '2rem',
      borderRadius: '0.5rem',
      border: '2px dashed #d1d5db',
      textAlign: 'center',
      marginBottom: '1.5rem'
    },
    uploadIcon: {
      color: '#9ca3af',
      marginBottom: '1rem'
    },
    uploadText: {
      fontSize: '1.125rem',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '0.5rem'
    },
    uploadSubtext: {
      fontSize: '0.875rem',
      color: '#6b7280',
      marginBottom: '1.5rem'
    },
    fileInfoCard: {
      backgroundColor: '#f0f9ff',
      padding: '1rem',
      borderRadius: '0.5rem',
      border: '1px solid #bae6fd',
      marginBottom: '1.5rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem'
    }
  };

  // 파일이 업로드되지 않은 경우
  if (!activeFileInfo) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <h3 style={styles.title}>
            <Activity size={20} style={{ color: '#2563eb' }} />
            최근 시황 (영업일 기준 최근 30일)
          </h3>
        </div>
        
        <div style={styles.uploadSection}>
          <Upload size={48} style={styles.uploadIcon} />
          <div style={styles.uploadText}>시장 가격 데이터 파일을 업로드하세요</div>
          <div style={styles.uploadSubtext}>
            데이터 파일(CSV, Excel)을 업로드하면 영업일 기준 최근 30일간의 가격 동향을 분석해드립니다.
          </div>
          <FileUploader 
            onUploadSuccess={handleUploadSuccess}
            isLoading={isUploading}
            setIsLoading={setIsUploading}
          />
        </div>
        
        {fileInfo && (
          <div style={{
            backgroundColor: '#fef3c7',
            padding: '1rem',
            borderRadius: '0.5rem',
            border: '1px solid #fbbf24',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '0.875rem', color: '#92400e' }}>
              💡 <strong>팁:</strong> 예측 시스템에서 이미 업로드한 파일이 있습니다. 
              해당 파일의 시장 데이터를 보려면 페이지를 새로고침하세요.
            </div>
          </div>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loadingSpinner}>
          <RefreshCw className="animate-spin" size={20} style={{ marginRight: '0.5rem' }} />
          최근 시황 데이터를 불러오는 중...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.errorMessage}>
          <strong>오류:</strong> {error}
        </div>
      </div>
    );
  }

  if (!marketData) {
    return (
      <div style={styles.container}>
        <div style={styles.header}>
          <h3 style={styles.title}>
            <Activity size={20} style={{ color: '#2563eb' }} />
            최근 시황 (영업일 기준 최근 30일)
          </h3>
        </div>
        
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: '#6b7280'
        }}>
          <Database size={32} style={{ marginBottom: '1rem' }} />
          <p>시장 데이터를 분석하고 있습니다...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>
          <Activity size={20} style={{ color: '#2563eb' }} />
          최근 시황 (영업일 기준 최근 30일)
        </h3>
        <button 
          style={styles.refreshButton}
          onClick={loadMarketData}
          onMouseEnter={(e) => e.target.style.backgroundColor = '#e5e7eb'}
          onMouseLeave={(e) => e.target.style.backgroundColor = '#f3f4f6'}
        >
          <RefreshCw size={14} />
          새로고침
        </button>
      </div>

      {/* 현재 사용 중인 파일 정보 */}
      <div style={styles.fileInfoCard}>
        <Database size={16} style={{ color: '#2563eb' }} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#1f2937' }}>
            현재 분석 파일: {activeFileInfo.original_filename || activeFileInfo.filename || '파일명 미확인'}
          </div>
          {localFileInfo && (
            <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
              ✨ 최근 시황 전용으로 업로드된 파일
            </div>
          )}
        </div>
      </div>

      {/* 기간 정보 */}
      <div style={styles.periodInfo}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <span style={{ fontWeight: '500', color: '#374151' }}>기간: </span>
            <span style={{ color: '#6b7280' }}>
              {marketData.date_range?.start_date} ~ {marketData.date_range?.end_date}
            </span>
          </div>
          <div>
            <span style={{ fontWeight: '500', color: '#374151' }}>총 영업일: </span>
            <span style={{ color: '#6b7280' }}>{marketData.date_range?.total_days}일</span>
          </div>
        </div>
      </div>

      {/* 카테고리별 데이터 */}
      {Object.entries(marketData.categories || {}).map(([category, categoryData]) => (
        <div key={category} style={styles.categoryCard}>
          <div 
            style={styles.categoryHeader}
            onClick={() => toggleCategory(category)}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#f3f4f6'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#f9fafb'}
          >
            <div style={styles.categoryTitle}>
              <span style={{ fontSize: '1.2rem' }}>{categoryIcons[category] || '📊'}</span>
              <span>{category}</span>
              <span style={{ 
                backgroundColor: '#e5e7eb', 
                color: '#374151', 
                padding: '0.125rem 0.5rem', 
                borderRadius: '1rem', 
                fontSize: '0.75rem' 
              }}>
                {categoryData.columns?.length || 0}개 지표
              </span>
            </div>
            {expandedCategories[category] ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>

          {expandedCategories[category] && (
            <div style={styles.categoryContent}>
              {/* 최신 가격 정보 */}
              <div style={styles.priceGrid}>
                {categoryData.columns?.map(column => {
                  const latestData = categoryData.data?.[categoryData.data.length - 1];
                  const latestValue = latestData?.values[column];
                  const priceChange = calculatePriceChange(categoryData.data, column);

                  return (
                    <div key={column} style={styles.priceCard}>
                      <div style={styles.priceLabel}>{column}</div>
                      <div style={styles.priceValue}>
                        {latestValue != null ? `$${latestValue.toFixed(2)}` : 'N/A'}
                      </div>
                      {priceChange && (
                        <div style={styles.priceChange(priceChange.isPositive)}>
                          {priceChange.isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                          {priceChange.absolute > 0 ? '+' : ''}{priceChange.absolute.toFixed(2)} 
                          ({priceChange.percent > 0 ? '+' : ''}{priceChange.percent.toFixed(2)}%)
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* 개별 차트들 */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: windowWidth < 768 ? '1fr' : windowWidth < 1200 ? 'repeat(2, 1fr)' : 'repeat(4, 1fr)',
                gap: '1rem',
                paddingBottom: '1rem'
              }}>
                {categoryData.columns?.map((column, index) => {
                  const yAxisRange = calculateYAxisRange(categoryData, column);
                  return (
                    <div key={column} style={{
                      ...styles.chartContainer,
                      border: '1px solid #e5e7eb',
                      borderRadius: '0.5rem',
                      backgroundColor: '#fafafa'
                    }}>
                      <h4 style={{
                        fontSize: windowWidth < 1200 ? '1rem' : '0.875rem',
                        fontWeight: '500',
                        color: '#374151',
                        marginBottom: '0.25rem',
                        textAlign: 'center'
                      }}>
                        {column}
                      </h4>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={transformChartData(categoryData, [column])}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis 
                            dataKey="date" 
                            tick={{ fontSize: 12 }}
                            tickFormatter={(value) => {
                              const date = new Date(value);
                              return `${date.getMonth() + 1}/${date.getDate()}`;
                            }}
                          />
                          <YAxis 
                            tick={{ fontSize: 12 }}
                            domain={[yAxisRange.min.toFixed(0), yAxisRange.max.toFixed(0)]}
                            tickFormatter={(value) => `$${value.toFixed(0)}`}
                          />
                          <Tooltip 
                            labelFormatter={(value) => `날짜: ${value}`}
                            formatter={(value, name) => [`$${value?.toFixed(2) || 'N/A'}`, name]}
                          />
                          <Line
                            type="monotone"
                            dataKey={column}
                            stroke={colors[index % colors.length]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default MarketStatus; 
