<template>
  <div class="blog-catalog">
    <h2>Catalog</h2>
    <ul>
      <li v-for="(heading, index) in headings" :key="index" @click="navigateTo(heading.id)">
        {{ heading.text }}
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'BlogCatalog',
  props: {
    headings: Array
  },
  methods: {
    navigateTo(id) {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }
}
</script>

<style scoped>
.blog-catalog {
  max-width: 1080px;
  margin: 3rem auto;
  padding: 2rem;
  font-size: 18px;
  line-height: 1.8;
  font-family: 'Georgia', serif;
  color: #1a1a1a;
  background-color: transparent;
  border-radius: 0;
  box-shadow: none;
  border: none;
  text-align: center;
}

.blog-catalog h2 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-family: 'Merriweather', serif;
  font-weight: 700;
  line-height: 1.2;
}

.blog-catalog ul {
  list-style-type: none;
  padding: 0;
}

.blog-catalog li {
  cursor: pointer;
  padding: 0.75rem 1rem;
  background-color: #f9f9f9;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease;
}

.blog-catalog li:hover {
  background-color: #f1f1f1;
}
</style>
