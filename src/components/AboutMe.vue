<template>
  <article class="about-me">
    <div class="about-content">
      <div class="text-content">
        <h2>Welcome to My Tech Blog</h2>
        <p>
          This is where I share my thoughts, experiences, and insights on cloud-native technologies, DevSecOps practices, and enterprise architecture. 
          With over 15 years in the industry, I've learned a lot and want to give back to the tech community through detailed technical articles and practical guides.
        </p>
        <p>
          <strong>What you'll find here:</strong>
        </p>
        <ul style="text-align: left; max-width: 600px; margin: 1rem auto; line-height: 1.8;">
          <li>🚀 <strong>Cloud Native Technologies:</strong> Kubernetes, OpenShift, containers, and orchestration</li>
          <li>🔐 <strong>DevSecOps Practices:</strong> CI/CD pipelines, GitOps, infrastructure as code</li>
          <li>☁️ <strong>Multi-Cloud Strategies:</strong> AWS, Azure, and hybrid cloud architectures</li>
          <li>🤖 <strong>AI & Machine Learning:</strong> Practical applications and emerging trends</li>
          <li>📚 <strong>How-To Guides:</strong> Step-by-step tutorials and best practices</li>
        </ul>
        <p>
          For my full professional background and portfolio, check out my <a href="/" style="color: #3498db; text-decoration: underline;">main portfolio site</a>.
        </p>
        <div class="action-buttons">
          <button class="action-button" @click="$emit('navigate', 'blogs')">
            <span class="button-text">Read My Blogs</span>
            <span class="button-icon">→</span>
          </button>
          <button class="action-button" @click="$emit('navigate', 'contact')">
            <span class="button-text">Contact Me</span>
            <span class="button-icon">✉</span>
          </button>
        </div>
      </div>

      <div class="image-content">
        <img :src="imageSrc" alt="Rakesh Kumar Mallam" @click="expandImage" />
        <div class="icons">
          <a href="mailto:mallamrakesh@gmail.com">
            <img src="@/assets/email.png" alt="Email" class="icon" />
          </a>
          <a href="https://www.linkedin.com/in/rakeshkumarmallam/" target="_blank">
            <img src="@/assets/linkedin.png" alt="LinkedIn" class="icon" />
          </a>
          <a href="https://github.com/rmallam" target="_blank">
            <img src="@/assets/github.png" alt="GitHub" class="icon" />
          </a>
        </div>
      </div>
    </div>
    <div v-if="isImageExpanded" class="image-modal" @click="closeImage">
      <img :src="imageSrc" alt="Rakesh Kumar Mallam" />
    </div>
    <div class="visitor-counter">
      <span>Visitors: {{ visitorCount }}</span>
    </div>
  </article>
</template>

<script>
export default {
  name: "AboutMe",
  data() {
    return {
      isImageExpanded: false,
      imageSrc: require('@/assets/rakesh.jpg'),
      visitorCount: 0
    };
  },
  methods: {
    expandImage() {
      this.isImageExpanded = true;
    },
    closeImage() {
      this.isImageExpanded = false;
    },
    async incrementVisitorCount() {
      // For demonstration, we'll use localStorage
      // In a real app, this should use a backend API
      const count = parseInt(localStorage.getItem('visitorCount') || '0');
      const newCount = count + 1;
      localStorage.setItem('visitorCount', newCount);
      this.visitorCount = newCount;
    }
  },
  mounted() {
    // Get existing count
    this.visitorCount = parseInt(localStorage.getItem('visitorCount') || '0');
    // Increment count for new visit
    this.incrementVisitorCount();
  }
};
</script>
