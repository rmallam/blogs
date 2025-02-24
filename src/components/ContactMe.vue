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
            <input id="email" v-model="form.email" type="email" required />
            <label for="email">Your Email</label>
          </div>
          <div class="form-group">
            <input id="subject" v-model="form.subject" type="text" required />
            <label for="subject">Subject</label>
          </div>
          <div class="form-group">
            <textarea id="description" v-model="form.description" required></textarea>
            <label for="description">Message</label>
          </div>
          <button type="submit">Send Message</button>
        </form>
      </div>
    </div>
  </article>
</template>

<script>
export default {
  name: "ContactMe",
  data() {
    return {
      form: {
        email: "",
        subject: "",
        description: ""
      }
    };
  },
  methods: {
    handleSubmit() {
      // Create email content
      const emailSubject = encodeURIComponent(this.form.subject);
      const emailBody = encodeURIComponent(
        `From: ${this.form.email}\n\n${this.form.description}`
      );
      
      // Create mailto link
      const mailtoLink = `mailto:mallamrakesh@gmail.com?subject=${emailSubject}&body=${emailBody}`;
      
      // Open email client
      window.location.href = mailtoLink;
      
      // Reset form
      this.form.email = "";
      this.form.subject = "";
      this.form.description = "";
      
      // Show success message
      alert("Opening your email client...");
    }
  }
}
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
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
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
  border-color: #3498db;
  outline: none;
}

.animated-form input:focus + label,
.animated-form textarea:focus + label,
.animated-form input:valid + label,
.animated-form textarea:valid + label {
  top: -1.2rem;
  font-size: 0.8rem;
  color: #3498db;
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
  background: #3498db;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.3s ease, background 0.3s ease;
}

button:hover {
  background: #2980b9;
  transform: translateY(-2px);
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
