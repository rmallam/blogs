<template>
  <div class="blog-sections-wrapper">
    <div class="blog-sections">
      <h2>Blog Categories</h2>
      <div v-if="!sections || sections.length === 0" class="no-blogs">
        <p>No blog posts found</p>
        <p class="debug-info">Available sections: {{ sections ? sections.length : 0 }}</p>
      </div>
      <div v-else class="sections-list">
        <div 
          v-for="section in sections" 
          :key="section.name"
          class="section-item"
          @click="$emit('select-section', section)"
        >
          <h3>{{ formatSectionName(section.name) }}</h3>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'BlogSections',
  props: {
    sections: {
      type: Array,
      required: true,
      default: () => []
    }
  },
  mounted() {
    console.log('BlogSections mounted with sections:', this.sections);
  },
  methods: {
    formatSectionName(name) {
      return name.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    }
  }
}
</script>

<style scoped>
.blog-sections-wrapper {
  width: 100%;
}

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

.debug-info {
  font-size: 0.8rem;
  color: #666;
  margin-top: 0.5rem;
}

.no-blogs {
  padding: 1rem;
  text-align: center;
  color: #666;
}
</style>