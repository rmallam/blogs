const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  publicPath: process.env.NODE_ENV === 'production' ? '/blogs/' : '/',
  transpileDependencies: true,
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.md$/,
          type: 'asset/source'
        }
      ]
    }
  }
})
