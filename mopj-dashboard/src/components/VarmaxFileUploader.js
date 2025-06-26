import React, { useState, useEffect } from 'react';
import { Upload, AlertTriangle } from 'lucide-react';
import { uploadCSV_2} from '../services/api';

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
  }
};

const VarmaxFileUploader = ({ 
  onUploadNoDates,      // 새로 추가: 날짜 없는 CSV
  isLoading, 
  setIsLoading, 
  acceptedFormats = '.csv,.xlsx,.xls,.cs,.xl,.log,.dat,.txt', // 🔒 보안 확장자 추가
  fileType = '데이터' 
}) => {
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputId = `varmax-file-upload-${Math.random().toString(36).substr(2, 9)}`;

  // 브라우저 기본 드래그앤드롭 동작 방지 (FileUploader와 중복 방지를 위해 조건부)
  useEffect(() => {
    // VarmaxFileUploader가 마운트될 때만 추가적인 보호
    const handleDocumentDrop = (e) => {
      if (!e.target.closest('[data-file-uploader]')) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    document.addEventListener('drop', handleDocumentDrop);

    return () => {
      document.removeEventListener('drop', handleDocumentDrop);
    };
  }, []);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('🔄 [VARMAX_DRAG] Event type:', e.type);
    
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
    
    console.log('📁 [VARMAX_DROP] Drop event:', e);
    console.log('📁 [VARMAX_DROP] DataTransfer:', e.dataTransfer);
    console.log('📁 [VARMAX_DROP] Files:', e.dataTransfer.files);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      console.log('📁 [VARMAX_DROP] Processing file:', e.dataTransfer.files[0].name);
      await handleFileUpload(e.dataTransfer.files[0]);
    } else {
      console.warn('📁 [VARMAX_DROP] No files found in drop event');
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
      console.log(`🔒 [VARMAX_SECURITY] 보안 파일 업로드: ${file.name} (${securityExtensions[fileExtension]})`);
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // 파일 업로드
      const uploadResult = await uploadCSV_2(file);
      
      // 오류 확인
      if (uploadResult.error) {
        setError(uploadResult.error);
        return;
      }
      
      // 🔒 보안 파일 처리 결과 로깅
      if (uploadResult.security_info && uploadResult.security_info.is_security_file) {
        console.log('🔒 [VARMAX_SECURITY] 보안 파일 처리 완료:', uploadResult.security_info.message);
        console.log('🔒 [VARMAX_SECURITY] 원본 확장자:', uploadResult.security_info.original_extension);
        console.log('🔒 [VARMAX_SECURITY] 변환된 확장자:', uploadResult.security_info.detected_extension);
      }
      // 날짜 없는 CSV 전용 콜백
      onUploadNoDates({
        filepath: uploadResult.filepath,
        filename: uploadResult.filename,
        file: file,
        columns1: uploadResult.columns1,
        columns2: uploadResult.columns2,
        case_1: uploadResult.case_1,  // 추가
        case_2: uploadResult.case_2   // 추가
      });
      return;
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
    </div>
  );
};

export default VarmaxFileUploader;
