<template>
  <div class="sidebar" :class="{ collapsed: isMenuCollapsed }">
    <!-- New sidebar heading -->
    <div class="sidebar-heading" @click="navigateAboutMe">
      <span v-if="!isMenuCollapsed">Rakesh's World</span>
      <span v-else>R</span>
    </div>
    <!-- Added hamburger element -->
    <div class="hamburger" @click="toggleMenu">
      &#9776;
    </div>
    <!-- New search bar -->
    <div v-if="!isMenuCollapsed" class="search-bar">
      <input type="text" v-model="searchQuery" placeholder="Search blogs" />
      <!-- Suggestion list -->
      <div v-if="searchQuery && suggestions.length" class="suggestions">
        <ul>
          <li v-for="(suggestion, index) in suggestions" :key="index" @click="selectSuggestion(suggestion)">
            {{ suggestion }}
          </li>
        </ul>
      </div>
    </div>
    <div class="blog-list" v-show="!isMenuCollapsed">
      <ul>
        <li @click="selectPage('about')">
          Home
        </li>
        <!-- New Contact Me navigation -->
        <li @click="selectPage('contact')">
          Contact Me
        </li>
      </ul> 
      <h3>Blogs</h3>
      <ul>
        <li v-for="(section, index) in filteredBlogSections" :key="index">
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
      isMenuCollapsed: true, // Set initial state to collapsed
      searchQuery: ''
    };
  },
  computed: {
    filteredBlogSections() {
      if (!this.searchQuery) {
        return this.blogSections;
      }
      const query = this.searchQuery.toLowerCase();
      // Filter sections by matching section name or any blog title
      return this.blogSections
        .map(section => {
          // Check if the section name matches query
          const sectionMatches = section.name.toLowerCase().includes(query);
          // Also filter blogs in this section
          const filteredBlogs = section.blogs.filter(blog =>
            blog.title.toLowerCase().includes(query)
          );
          // If either section or any blog matches, return the section with filtered blogs
          if (sectionMatches || filteredBlogs.length > 0) {
            return { ...section, blogs: filteredBlogs };
          }
          return null;
        })
        .filter(section => section !== null);
    },
    suggestions() {
      if (!this.searchQuery) return [];
      const query = this.searchQuery.toLowerCase();
      let sugg = [];
      // Add matching blog titles
      this.blogSections.forEach(section => {
        section.blogs.forEach(blog => {
          if (blog.title.toLowerCase().includes(query)) {
            sugg.push(blog.title);
          }
        });
      });
      // Optionally add section name suggestions
      this.blogSections.forEach(section => {
        if (section.name.toLowerCase().includes(query)) {
          sugg.push(section.name);
        }
      });
      // Remove duplicates
      return [...new Set(sugg)];
    }
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
      this.isMenuCollapsed = true;
    },
    // Added toggleMenu to update collapse state
    toggleMenu() {
      this.isMenuCollapsed = !this.isMenuCollapsed;
    },
    selectSuggestion(suggestion) {
      // Fill search query with selected suggestion
      this.searchQuery = suggestion;
    },
    navigateAboutMe() {
      this.$emit('page-selected', 'about');
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
  overflow: hidden;
  cursor: pointer; /* Add cursor pointer for clickable elements */
}

.sidebar.collapsed {
  width: 60px;
}

.hamburger {
  font-size: 2rem;
  cursor: pointer;
  padding: 10px;
  text-align: center;
}

/* New styling for the heading */
.sidebar-heading {
  padding: 1rem;
  text-align: center;
  font-size: 1.6rem;
  font-weight: bold;
  border-bottom: 1px solid rgba(255,255,255,0.2);
}

/* New styling for the search bar to mimic Medium's minimalist style */
.search-bar {
  padding: 1rem;
}
.search-bar input {
  width: 100%;
  box-sizing: border-box; /* include padding and border in the element's total width */
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 50px;
  background-color: #f1f3f5; /* subtle light gray */
  font-size: 1rem;
  outline: none;
  transition: background-color 0.3s ease;
}
.search-bar input::placeholder {
  color: #888;
}
.search-bar input:focus {
  background-color: #e9ecef;
}

.blog-list {
  padding-right: 1rem;
  padding-left: 1rem;
}

.blog-list h5 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
}

.blog-list ul {
  list-style-type: none;
  padding: 0;
}

/* Updated styles for menu items */
.blog-list li {
  cursor: pointer;
  padding: 0.75rem 1rem; /* adjust vertical space */
  background-color: #444;
  color: white;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease;
  display: block;
  width: 100%;
  min-width: 21ch; /* ensure at least 12 characters in width */
  box-sizing: border-box;
  font-size: 1.2rem;
  text-align: center;
  white-space: normal;
  word-break: break-word;
}

.blog-list li:hover {
  background-color: #7b6666;
}

/* New rule for blog folder names to appear larger */
.blog-list > ul > li > div {
  font-size: 1.2rem;  /* larger than default */
  font-weight: bold;
}

/* New rule for sub-items under blog items updated */
.blog-list li ul li {
  font-size: 1rem;     /* smaller font-size */
  padding: 0.5rem;     /* reduced padding */
  background-color: #333; /* changed from blue to near-black */
  color: #fff;         /* update text to white for contrast */
  margin-bottom: 0.25rem;  /* slightly smaller margin */
  border-radius: 6px;
}

/* New styling for suggestions dropdown */
.suggestions {
  background-color: #f1f3f5;
  border-radius: 4px;
  margin-top: 0.5rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.suggestions ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.suggestions li {
  padding: 0.5rem 1rem;
  cursor: pointer;
  color: #333;
}
.suggestions li:hover {
  background-color: #e9ecef;
}

/* Responsive adjustments */
@media (max-width: 767px) {
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
  }
  .sidebar.collapsed {
    width: 100%;
  }
  .hamburger {
    text-align: left;
  }
  .blog-list {
    padding: 0;
  }
  .blog-list li {
    width: 100%;
  }
  .blog-list > ul > li > div {
    width: 100%;
  }
}
</style>
