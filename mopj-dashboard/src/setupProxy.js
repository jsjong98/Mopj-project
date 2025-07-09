const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  console.log('🔧 [PROXY] setupProxy.js is loading...');
  
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
      secure: false,
      logLevel: 'debug',
      onError: (err, req, res) => {
        console.error('🚨 [PROXY] 프록시 오류:', err.message);
        console.error('🚨 [PROXY] 요청 URL:', req.url);
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log('📡 [PROXY] 프록시 요청:', req.method, req.url, '→', proxyReq.path);
      },
      onProxyRes: (proxyRes, req, res) => {
        console.log('📡 [PROXY] 프록시 응답:', proxyRes.statusCode, req.url);
      }
    })
  );
  
  console.log('✅ [PROXY] setupProxy.js loaded successfully - /api → http://localhost:5000');
};
