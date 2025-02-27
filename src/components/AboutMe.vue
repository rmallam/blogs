<template>
  <article class="about-me">
    <div class="about-content">
      <div class="text-content">
        <h2>About Me</h2>
        <p>
          Hey there! I’m Rakesh – a passionate architect, curious traveler, and lover of all things tech.
          By day, I’m a senior consulting architect based in Melbourne, helping businesses design and implement cutting-edge solutions. By night, I’m a storyteller, sharing insights about architecture, technology trends, and the little things that inspire me.
          I started this blog as a space to document my experiences, share knowledge, and connect with like-minded folks. Whether it’s exploring the latest in cloud infrastructure or reflecting on life’s adventures, you’ll find a mix of technical deep-dives and personal musings here.
          When I’m not architecting solutions, I’m likely planning my next travel adventure with my family, discovering new cuisines, or chasing after my little ones.
          I’d love to hear from you – feel free to drop a comment or connect with me on my social medial below. Let’s learn and grow together!
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
}
</script>

<style scoped>
.about-me {
  max-width: 1800px;
  margin: 6rem auto;       /* center horizontally */
  padding: 2rem;
  font-size: 18px;
  line-height: 1.8;
  font-family: 'Georgia', serif;
  color: #1a1a1a;
  background-color: transparent; /* remove box background */
  border-radius: 0;              /* remove rounded corners */
  box-shadow: none;              /* remove shadow */
  border: none;                  /* remove any border */
  text-align: center;
}

.about-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.text-content {
  flex: 1;
  margin-right: 2rem;
}

.nav-links {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin: 30px 0;
  width: 100%;
}

.nav-links button {
  padding: 12px 24px;
  font-size: 1.1rem;
  color: white;
  background: #333;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.nav-links button:hover {
  background-color: #555;
}

.image-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-content img {
  max-width: 40%;
  height: auto;
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: transform 0.3s ease;
}

.image-content img:hover {
  transform: scale(1.05);
}

.icons {
  margin-top: 1rem;
  display: flex;
  justify-content: center;
}

.icon {
  width: 50px;
  height: 50px;
  margin: 0 8px;
  vertical-align: middle;
}

.image-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.image-modal img {
  max-width: 90%;
  max-height: 90%;
  border-radius: 10px;
}

.about-me h2 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-family: 'Merriweather', serif;
  font-weight: 700;
  line-height: 1.2;
}

.about-me p {
  margin-bottom: 1.5rem;
}

.visitor-counter {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  animation: slideIn 0.5s ease-out;
  z-index: 100;
}

@keyframes slideIn {
  from {
    transform: translateY(100px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.action-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin: 2rem 0;
}

.action-button {
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
  border: none;
  border-radius: 25px;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
}

.action-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
  background: linear-gradient(135deg, #3498db 20%, #2980b9 100%);
}

.action-button:active {
  transform: translateY(0);
}

.button-text {
  font-weight: 500;
}

.button-icon {
  font-size: 1.2rem;
  transition: transform 0.3s ease;
}

.action-button:hover .button-icon {
  transform: translateX(3px);
}

/* Responsive Design */
@media screen and (max-width: 1024px) {
  .about-content {
    flex-direction: column;
    gap: 2rem;
  }

  .text-content {
    margin-right: 0;
  }

  .image-content img {
    max-width: 60%;
  }
}

@media screen and (max-width: 768px) {
  .about-me {
    margin: 3rem auto;
    padding: 1rem;
  }

  .action-buttons {
    flex-direction: column;
    gap: 1rem;
  }

  .action-button {
    width: 100%;
    justify-content: center;
  }

  .about-me h2 {
    font-size: 2rem;
  }

  .visitor-counter {
    bottom: 10px;
    right: 10px;
    font-size: 0.8rem;
  }
}

@media screen and (max-width: 480px) {
  .image-content img {
    max-width: 80%;
  }

  .icons {
    flex-wrap: wrap;
  }

  .icon {
    width: 40px;
    height: 40px;
  }
}
</style>
