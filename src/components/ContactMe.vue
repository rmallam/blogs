<template>
  <article class="contact-form">
    <div class="contact-container">
      <div class="contact-info">
        <h2>Let's Connect</h2>
        <p>Feel free to reach out for collaborations, questions, or just to say hello!</p>
        <div class="social-links">
          <a href="https://github.com/rmallam" target="_blank" class="social-link">
            <img src="@/assets/github.png" alt="GitHub" />
          </a>
          <a href="https://www.linkedin.com/in/rakeshkumarmallam/" target="_blank" class="social-link">
            <img src="@/assets/linkedin.png" alt="LinkedIn" />
          </a>
          <a href="mailto:mallamrakesh@gmail.com" class="social-link">
            <img src="@/assets/email.png" alt="Email" />
          </a>
        </div>
      </div>
      <div class="form-wrapper">
        <form @submit.prevent="handleSubmit" class="animated-form">
          <div class="form-group">
            <input 
              id="name" 
              v-model.trim="form.name" 
              type="text" 
              required 
              maxlength="100"
            />
            <label for="name">Your Name</label>
            <span v-if="errors.name" class="error-text">{{ errors.name }}</span>
          </div>
          <div class="form-group">
            <input id="email" v-model="form.email" type="email" required />
            <label for="email">Your Email</label>
            <span v-if="errors.email" class="error-text">{{ errors.email }}</span>
          </div>
          <div class="form-group">
            <textarea id="message" v-model="form.message" required></textarea>
            <label for="message">Message</label>
            <span v-if="errors.message" class="error-text">{{ errors.message }}</span>
          </div>
          <button type="submit" :disabled="sending">
            {{ sending ? 'Sending...' : 'Send Message' }}
          </button>
          <p v-if="status === 'success'" class="success-message">
            Message sent successfully!
          </p>
          <p v-if="status === 'error'" class="error-message">
            Failed to send message. Please try again.
          </p>
        </form>
      </div>
    </div>
  </article>
</template>

<script>
import emailjs from '@emailjs/browser';
import { EMAILJS_CONFIG } from '../config/emailjs';

export default {
  name: "ContactMe",
  data() {
    return {
      form: {
        name: "",
        email: "",
        message: ""
      },
      sending: false,
      status: "",
      lastSubmissionTime: 0,
      submissionCount: 0,
      errors: {
        name: '',
        email: '',
        message: ''
      }
    };
  },
  created() {
    emailjs.init(EMAILJS_CONFIG.PUBLIC_KEY);
  },
  methods: {
    validateForm() {
      let isValid = true;
      this.errors = {
        name: '',
        email: '',
        message: ''
      };

      // Name validation
      if (!this.form.name.trim()) {
        this.errors.name = 'Name is required';
        isValid = false;
      } else if (this.form.name.length > 100) {
        this.errors.name = 'Name is too long';
        isValid = false;
      }

      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!this.form.email.trim()) {
        this.errors.email = 'Email is required';
        isValid = false;
      } else if (!emailRegex.test(this.form.email)) {
        this.errors.email = 'Invalid email format';
        isValid = false;
      }

      // Message validation
      if (!this.form.message.trim()) {
        this.errors.message = 'Message is required';
        isValid = false;
      } else if (this.form.message.length > 1000) {
        this.errors.message = 'Message is too long';
        isValid = false;
      }

      return isValid;
    },

    sanitizeInput(input) {
      return input
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    },

    checkRateLimit() {
      const now = Date.now();
      const oneMinute = 60 * 1000;

      // Reset count if more than a minute has passed
      if (now - this.lastSubmissionTime > oneMinute) {
        this.submissionCount = 0;
      }

      // Allow maximum 3 submissions per minute
      if (this.submissionCount >= 3) {
        return false;
      }

      this.submissionCount++;
      this.lastSubmissionTime = now;
      return true;
    },

    async handleSubmit() {
      if (!this.validateForm()) {
        return;
      }

      if (!this.checkRateLimit()) {
        this.status = 'error';
        alert('Too many attempts. Please try again later.');
        return;
      }

      this.sending = true;
      this.status = "";

      try {
        const templateParams = {
          from_name: this.sanitizeInput(this.form.name),
          from_email: this.sanitizeInput(this.form.email),
          message: this.sanitizeInput(this.form.message),
          to_name: 'Rakesh Kumar Mallam',
          reply_to: this.sanitizeInput(this.form.email),
        };

        await emailjs.send(
          EMAILJS_CONFIG.SERVICE_ID,
          EMAILJS_CONFIG.TEMPLATE_ID,
          templateParams,
          EMAILJS_CONFIG.PUBLIC_KEY
        );

        this.status = 'success';
        this.form.name = "";
        this.form.email = "";
        this.form.message = "";
      } catch (error) {
        console.error('Email error:', error);
        this.status = 'error';
      } finally {
        this.sending = false;
      }
    }
  }
};
</script>

<style scoped>
.contact-form {
  max-width: 1200px;
  margin: 3rem auto;
  padding: 2rem;
}

.contact-container {
  display: flex;
  background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.contact-info {
  flex: 1;
  padding: 4rem;
  background: linear-gradient(135deg, #2c3e50 0%, #EE0000 100%);
  color: white;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.contact-info h2 {
  font-size: 2.5rem;
  margin-bottom: 1.5rem;
  font-weight: 700;
}

.contact-info p {
  font-size: 1.1rem;
  margin-bottom: 2rem;
  line-height: 1.6;
}

.social-links {
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
}

.social-link img {
  width: 32px;
  height: 32px;
  transition: transform 0.3s ease;
}

.social-link:hover img {
  transform: translateY(-5px);
}

.form-wrapper {
  flex: 1.5;
  padding: 4rem;
  background: white;
}

.animated-form .form-group {
  position: relative;
  margin-bottom: 2rem;
}

.animated-form input,
.animated-form textarea {
  width: 100%;
  padding: 0.8rem;
  border: none;
  border-bottom: 2px solid #e1e1e1;
  background: transparent;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.animated-form label {
  position: absolute;
  left: 0;
  top: 0.8rem;
  color: #999;
  transition: all 0.3s ease;
  pointer-events: none;
}

.animated-form input:focus,
.animated-form textarea:focus,
.animated-form input:valid,
.animated-form textarea:valid {
  border-color: #EE0000;
  outline: none;
}

.animated-form input:focus + label,
.animated-form textarea:focus + label,
.animated-form input:valid + label,
.animated-form textarea:valid + label {
  top: -1.2rem;
  font-size: 0.8rem;
  color: #EE0000;
}

.animated-form textarea {
  min-height: 100px;
  resize: vertical;
}

.file-upload {
  margin-top: 2rem;
}

.file-label {
  display: inline-block;
  padding: 0.8rem 1.5rem;
  background: #f3f4f6;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.3s ease;
}

.file-label:hover {
  background: #e2e4e7;
}

.file-label input {
  display: none;
}

button {
  width: 100%;
  padding: 1rem;
  background: #EE0000;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.3s ease, background 0.3s ease;
}

button:hover {
  background: #CC0000;
  transform: translateY(-2px);
}

.error-text {
  color: #f44336;
  font-size: 0.8rem;
  margin-top: 0.5rem;
  display: block;
}

@media (max-width: 768px) {
  .contact-container {
    flex-direction: column;
  }
  
  .contact-info,
  .form-wrapper {
    padding: 2rem;
  }
}
</style>
