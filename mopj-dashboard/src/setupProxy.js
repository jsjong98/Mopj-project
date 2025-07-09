const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
      logLevel: 'debug', // 디버그 로깅 활성화
      onError: (err, req, res) => {
        console.error('프록시 오류:', err);
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log('프록시 요청:', req.method, req.path);
      }
    })
  );
};
