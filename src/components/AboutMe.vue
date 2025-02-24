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
  </article>
</template>

<script>
export default {
  name: "AboutMe",
  data() {
    return {
      isImageExpanded: false,
      imageSrc: require('@/assets/rakesh.jpg')
    };
  },
  methods: {
    expandImage() {
      this.isImageExpanded = true;
    },
    closeImage() {
      this.isImageExpanded = false;
    }
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
</style>
