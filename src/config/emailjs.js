export const EMAILJS_CONFIG = {
  SERVICE_ID: process.env.VUE_APP_EMAILJS_SERVICE_ID || '',
  TEMPLATE_ID: process.env.VUE_APP_EMAILJS_TEMPLATE_ID || '',
  PUBLIC_KEY: process.env.VUE_APP_EMAILJS_PUBLIC_KEY || ''
};

// Add console log for debugging
console.log('EmailJS Config:', {
  SERVICE_ID: !!EMAILJS_CONFIG.SERVICE_ID,
  TEMPLATE_ID: !!EMAILJS_CONFIG.TEMPLATE_ID,
  PUBLIC_KEY: EMAILJS_CONFIG.PUBLIC_KEY ? 'Present' : 'Missing'
});
