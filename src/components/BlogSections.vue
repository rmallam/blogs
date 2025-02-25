<template>
  <div class="blog-sections">
    <h2>Blog Categories</h2>
    <ul>
      <li v-for="(section, index) in localSections" :key="index">
        <div @click="selectSection(section)" class="section-item">
          {{ section.name }}
        </div>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'BlogSections',
  props: {
    sections: Array
  },
  data() {
    return {
      localSections: this.sections.map(section => ({ ...section, expanded: false }))
    };
  },
  methods: {
    toggleSection(index) {
      this.localSections[index].expanded = !this.localSections[index].expanded;
    },
    selectSection(section) {
      this.$emit('select-section', section);
    }
  }
}
</script>

<style scoped>
.blog-sections {
  width: 250px;
  padding: 2rem;
  background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
  border-radius: 20px;
  color: white;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.blog-sections h2 {
  font-size: 1.8rem;
  margin-bottom: 1.5rem;
  font-weight: 700;
  color: white;
}

.blog-sections ul {
  list-style-type: none;
  padding: 0;
}

.section-item {
  cursor: pointer;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  margin-bottom: 0.5rem;
  border-radius: 8px;
  transition: all 0.3s ease;
  font-size: 1.1rem;
}

.section-item:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}
</style>
