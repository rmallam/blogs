

export const JSONBIN_CONFIG = {
  BIN_ID: process.env.VUE_APP_JSONBIN_BIN_ID,
  API_KEY: process.env.VUE_APP_JSONBIN_API_KEY,
  BASE_URL: process.env.VUE_APP_JSONBIN_BASE_URL
};

// Debug logging
console.log('Environment variables present:', {
  BIN_ID: !!process.env.VUE_APP_JSONBIN_BIN_ID,
  API_KEY: process.env.VUE_APP_JSONBIN_API_KEY,
  BASE_URL: !!process.env.VUE_APP_JSONBIN_BASE_URL
});
