const markdown = require('markdown-it')();

module.exports = {
  transpileDependencies: ['vue'],
  publicPath: process.env.NODE_ENV === 'production' ? '/blogs/' : '/',
  chainWebpack: config => {
    config.module
      .rule('markdown')
      .test(/\.md$/)
      .use('raw-loader')
      .loader('raw-loader')
      .end();
  }
};
