<template>
  <div class="sidebar" :class="{ collapsed: isMenuCollapsed }">
    <div class="hamburger" @click="toggleMenu">
      &#9776;
    </div>
    <div class="blog-list">
      <h3>Blogs</h3>
      <ul>
        <li v-for="(section, index) in blogSections" :key="index">
          <div @click="toggleSection(index)">
            {{ section.name }}
          </div>
          <ul v-if="section.expanded">
            <li v-for="(blog, blogIndex) in section.blogs" :key="blogIndex" @click="selectPage(blog)">
              {{ blog.title }}
            </li>
          </ul>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
const requireBlog = require.context('../blogs', true, /\.md$/);

export default {
  name: 'BlogSidebar',
  data() {
    return {
      blogSections: this.getBlogSections(),
      isMenuCollapsed: false
    };
  },
  methods: {
    getBlogSections() {
      const sections = {};
      requireBlog.keys().forEach(file => {
        const pathParts = file.replace('./', '').split('/');
        if (pathParts.length < 2) return;
        const sectionName = pathParts[0];
        const blogTitle = pathParts[1] ? pathParts[1].replace('.md', '') : '';
        if (!sections[sectionName]) {
          sections[sectionName] = {
            name: sectionName,
            expanded: false,
            blogs: []
          };
        }
        sections[sectionName].blogs.push({
          title: blogTitle,
          content: requireBlog(file).default
        });
      });
      return Object.values(sections);
    },
    toggleSection(index) {
      this.blogSections[index].expanded = !this.blogSections[index].expanded;
    },
    selectPage(page) {
      this.$emit('page-selected', page);
    },
    toggleMenu() {
      this.isMenuCollapsed = !this.isMenuCollapsed;
    }
  }
}
</script>

<style scoped>
.sidebar {
  width: 250px;
  height: 100%;
  background-color: #333;
  color: white;
  position: fixed;
  top: 0;
  left: 0;
  transition: width 0.3s;
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar button {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  margin: 10px;
}

.sidebar nav ul {
  list-style-type: none;
  padding: 0;
}

.sidebar nav ul li {
  padding: 10px;
}

.sidebar nav ul li a {
  color: white;
  text-decoration: none;
}

.hamburger {
  font-size: 2rem;
  cursor: pointer;
}
.blog-list {
  padding-right: 2rem;
}
.blog-list h3 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
}
.blog-list ul {
  list-style-type: none;
  padding: 0;
}
.blog-list li {
  cursor: pointer;
  padding: 0.75rem;
  background-color: #fff;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease;
}
.blog-list li:hover {
  background-color: #f1f1f1;
}
</style>
