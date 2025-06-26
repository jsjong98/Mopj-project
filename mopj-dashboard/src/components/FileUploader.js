import React, { useState, useEffect } from 'react';
import { Upload, AlertTriangle, RefreshCw, Shield } from 'lucide-react';
import { uploadCSV, getAvailableDates, checkFileRefresh } from '../services/api';

const styles = {
  container: (dragActive) => ({
    border: '2px dashed',
    borderColor: dragActive ? '#3b82f6' : '#d1d5db',
    borderRadius: '0.5rem',
    padding: '1.5rem',
    textAlign: 'center',
    transition: 'colors 0.3s',
    backgroundColor: dragActive ? '#eff6ff' : 'transparent'
  }),
  icon: {
    height: '3rem',
    width: '3rem',
    color: '#9ca3af',
    margin: '0 auto 0.5rem auto'
  },
  text: {
    color: '#4b5563',
    marginBottom: '0.5rem'
  },
  smallText: {
    fontSize: '0.875rem',
    color: '#6b7280'
  },
  hidden: {
    display: 'none'
  },
  button: (isLoading) => ({
    marginTop: '1rem',
    display: 'inline-block',
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '0.5rem 1rem',
    borderRadius: '0.375rem',
    transition: 'background-color 0.3s',
    cursor: isLoading ? 'not-allowed' : 'pointer',
    opacity: isLoading ? 0.5 : 1,
  }),
  errorContainer: {
    marginTop: '0.75rem',
    color: '#ef4444',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  errorIcon: {
    marginRight: '0.25rem'
  },
  securityContainer: {
    marginTop: '0.75rem',
    color: '#059669',
    backgroundColor: '#ecfccb',
    border: '1px solid #a7f3d0',
    borderRadius: '0.375rem',
    padding: '0.75rem',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'center',
    textAlign: 'left'
  },
  securityIcon: {
    marginRight: '0.5rem',
    marginTop: '0.125rem',
    flexShrink: 0
  }
};

const FileUploader = ({ 
  onUploadSuccess, 
  onUploadNoDates, // VARMAX용 날짜 없는 CSV 업로드 콜백 추가
  isLoading, 
  setIsLoading, 
  acceptedFormats = '.csv,.xlsx,.xls,.cs,.xl,.log,.dat,.txt', // 🔒 보안 확장자 추가
  fileType = '데이터' 
}) => {
  const [error, setError] = useState(null);
  const [securityInfo, setSecurityInfo] = useState(null); // 🔒 보안 파일 정보 상태 추가
  const [dragActive, setDragActive] = useState(false);
  const inputId = `file-upload-${Math.random().toString(36).substr(2, 9)}`;

  // 브라우저 기본 드래그앤드롭 동작 방지
  useEffect(() => {
    const preventDefaults = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };

    // 전체 문서에 대해 기본 드래그앤드롭 방지
    document.addEventListener('dragover', preventDefaults);
    document.addEventListener('drop', preventDefaults);
    document.addEventListener('dragenter', preventDefaults);
    document.addEventListener('dragleave', preventDefaults);

    return () => {
      document.removeEventListener('dragover', preventDefaults);
      document.removeEventListener('drop', preventDefaults);
      document.removeEventListener('dragenter', preventDefaults);
      document.removeEventListener('dragleave', preventDefaults);
    };
  }, []);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('🔄 [DRAG] Event type:', e.type);
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    console.log('📁 [DROP] Drop event:', e);
    console.log('📁 [DROP] DataTransfer:', e.dataTransfer);
    console.log('📁 [DROP] Files:', e.dataTransfer.files);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      console.log('📁 [DROP] Processing file:', e.dataTransfer.files[0].name);
      await handleFileUpload(e.dataTransfer.files[0]);
    } else {
      console.warn('📁 [DROP] No files found in drop event');
    }
  };

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      await handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    // 🔒 보안 확장자를 포함한 파일 형식 확인
    const validExtensions = acceptedFormats.split(',').map(ext => ext.trim());
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    // 보안 확장자 정보
    const securityExtensions = {
      '.cs': 'CSV 파일 (보안 확장자)',
      '.xl': 'Excel 파일 (보안 확장자)', 
      '.log': 'Excel 파일 (보안 확장자)',
      '.dat': '자동 감지 (보안 확장자)',
      '.txt': '자동 감지 (보안 확장자)'
    };
    
    if (!validExtensions.includes(fileExtension)) {
      const securityInfo = securityExtensions[fileExtension] 
        ? `\n💡 ${fileExtension} 확장자는 ${securityExtensions[fileExtension]}로 처리됩니다.`
        : '';
      
      setError(`지원되지 않는 파일 형식입니다. 지원 형식: ${acceptedFormats}${securityInfo}`);
      return;
    }
    
    // 보안 파일 업로드 시 사용자에게 알림
    const isSecurityFile = Object.keys(securityExtensions).includes(fileExtension);
    if (isSecurityFile) {
      console.log(`🔒 [SECURITY] 보안 파일 업로드: ${file.name} (${securityExtensions[fileExtension]})`);
    }
    
    setIsLoading(true);
    setError(null);
    setSecurityInfo(null); // 🔒 보안 정보 초기화
    
    try {
      // 파일 업로드
      const uploadResult = await uploadCSV(file);
      
      // 오류 확인
      if (uploadResult.error) {
        setError(uploadResult.error);
        return;
      }
      
      // 🔒 보안 파일 처리 결과 처리
      if (uploadResult.security_info && uploadResult.security_info.is_security_file) {
        console.log('🔒 [SECURITY] 보안 파일 처리 완료:', uploadResult.security_info.message);
        setSecurityInfo(uploadResult.security_info); // 사용자에게 표시할 보안 정보 설정
      }
      
      // 데이터 파일인 경우 날짜 정보 요청
      if (fileType.toLowerCase() === 'csv' || fileType.toLowerCase() === '데이터' || fileType.toLowerCase() === 'excel') {
        // 🔍 Step 1: 파일 확장 여부 확인
        console.log('🔍 [FILE_UPLOAD] Checking if file needs refresh...');
        const refreshCheck = await checkFileRefresh(uploadResult.filepath);
        
        let datesResult;
        let dataExtended = false;
        
        if (refreshCheck.refresh_needed) {
          console.log('🔄 [FILE_UPLOAD] File refresh needed:', refreshCheck.refresh_reasons);
          
          // 데이터 확장 감지
          if (refreshCheck.refresh_reasons.includes('Data range extended')) {
            dataExtended = true;
            console.log('📈 [FILE_UPLOAD] Data extension detected! Forcing date refresh...');
          }
          
          // 강제 새로고침으로 날짜 정보 요청
          datesResult = await getAvailableDates(uploadResult.filepath, true);
        } else {
          console.log('✅ [FILE_UPLOAD] No refresh needed, getting dates normally...');
          // 일반적인 날짜 정보 요청
          datesResult = await getAvailableDates(uploadResult.filepath);
        }
        
        // 날짜 정보 오류 확인
        if (datesResult.error) {
          setError(datesResult.error);
          return;
        }
        
        // 🎯 날짜 배열이 비어 있으면 onUploadNoDates 콜백 호출 (VARMAX용)
        if ((!datesResult.dates || datesResult.dates.length === 0) && onUploadNoDates) {
          console.log('📅 [FileUploader] 날짜 없는 데이터 파일 - onUploadNoDates 콜백 호출');
          onUploadNoDates({
            filepath: uploadResult.filepath,
            filename: uploadResult.filename,
            original_filename: uploadResult.original_filename,
            file: file,
            // VARMAX 구매 계획 결정 데이터 포함 (백엔드에서 제공하는 경우)
            case_1: uploadResult.case_1 || [],
            case_2: uploadResult.case_2 || []
          });
          return;
        }
        
        // 성공 콜백 호출 - 데이터 확장 정보 포함
        onUploadSuccess({
          filepath: uploadResult.filepath,
          dates: datesResult.dates || [],
          latestDate: datesResult.latest_date,
          file: file,
          // 🎯 기존 정보 유지
          prediction_threshold: datesResult.prediction_threshold,
          halfway_point: datesResult.halfway_point,
          halfway_semimonthly: datesResult.halfway_semimonthly,
          target_semimonthly: datesResult.target_semimonthly,
          // 캐시 정보 추가 (있는 경우)
          cache_info: uploadResult.cache_info,
          // 🔄 새로운 확장 정보 추가
          data_extended: dataExtended,
          refresh_info: refreshCheck,
          // 파일 메타데이터 추가
          file_hash: datesResult.file_hash,
          file_modified: datesResult.file_modified,
          total_rows: datesResult.total_rows,
          data_start_date: datesResult.data_start_date,
          data_end_date: datesResult.data_end_date
        });
      } else {
        // 휴일 파일 등 다른 용도의 파일
        onUploadSuccess({
          filepath: uploadResult.filepath,
          filename: uploadResult.filename,
          file: file
        });
      }
    } catch (err) {
      console.error('File upload failed:', err);
      setError(err.error || err.message || '파일 업로드 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div 
      style={styles.container(dragActive)}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      data-file-uploader="true"
    >
      <Upload style={styles.icon} />
      <p style={styles.text}>{fileType} 파일을 드래그하여 업로드하거나 클릭하여 선택하세요</p>
      <p style={styles.smallText}>지원 형식: {acceptedFormats}</p>
      
      <input
        type="file"
        id={inputId}
        style={styles.hidden}
        accept={acceptedFormats}
        onChange={handleFileChange}
        disabled={isLoading}
        ref={(input) => {
          if (input) {
            input.onclick = () => input.value = '';
          }
        }}
      />
      
      <label 
        htmlFor={inputId}
        style={styles.button(isLoading)}
      >
        {isLoading ? '처리 중...' : `${fileType} 파일 업로드`}
      </label>
      
      {error && (
        <div style={styles.errorContainer}>
          <AlertTriangle size={16} style={styles.errorIcon} />
          {error}
        </div>
      )}
      
      {securityInfo && (
        <div style={styles.securityContainer}>
          <Shield size={16} style={styles.securityIcon} />
          <div>
            <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
              🔒 보안 파일 처리 완료
            </div>
            <div style={{ fontSize: '0.875rem' }}>
              {securityInfo.message}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
