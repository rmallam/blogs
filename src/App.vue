<template>
  <div id="app">
    <div class="sidebar" :class="{ collapsed: isSidebarCollapsed }">
      <BlogSidebar @page-selected="selectPage" @toggle-sidebar="toggleSidebar" />
    </div>
    <div class="main-content" :class="{ collapsed: isSidebarCollapsed }">
      <div class="content">
        <main>
          <div class="content">
            <div class="blog-content" :class="{ fullWidth: selectedPage === 'about' || selectedPage==='contact' }">
              <!-- About Me page now rendered via AboutMe component -->
              <AboutMe v-if="selectedPage === 'about'" />
              <ContactMe v-else-if="selectedPage === 'contact'" />
              <article v-else-if="selectedPage !== null && selectedPage !== 'blogs'">
                <h2>{{ selectedPage.title }}</h2>
                <div class="markdown-content" v-html="renderMarkdown(selectedPage.content)"></div>
              </article>
            </div>
          </div>
        </main>
      </div>
    </div>
  </div>
</template>

<script>
import BlogSidebar from './components/BlogSidebar.vue';
import AboutMe from './components/AboutMe.vue';
import ContactMe from './components/ContactMe.vue';
import markdown from 'markdown-it'
const md = markdown()

const requireBlog = require.context('./blogs', true, /\.md$/);

export default {
  name: 'App',
  components: { BlogSidebar, AboutMe, ContactMe },
  data() {
    return {
      blogSections: this.getBlogSections(),
      selectedPage: 'about',
      isSidebarCollapsed: false
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
    selectPage(page) {
      this.selectedPage = page;
    },
    renderMarkdown(content) {
      return md.render(content)
    },
    toggleSidebar() {
      this.isSidebarCollapsed = !this.isSidebarCollapsed;
    }
  },
  mounted() {
    console.log('Blog website loaded');
  }
}
</script>

<style>
#app {
  display: flex;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.header {
  background-color: #fff;
  color: #333;
  padding: 1rem 0;
  text-align: center;
  width: 100%;
  margin: 0;
  transition: none;
}

.header h1 {
  margin: 0;
  font-size: 2rem;
}

.main-content {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100vh; /* Full viewport height */
  overflow-y: auto; /* Enable scrolling if needed */
  text-align: left; /* align main content to left */
  transition: margin-left 0.3s, width 0.3s; /* Smooth transition */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.main-content.collapsed {
  margin-left: 60px; /* Adjust margin when sidebar is collapsed */
  width: calc(100% - 60px); /* Adjust width when sidebar is collapsed */
}

.content {
  margin-left: 0; /* Remove left margin */
  width: 100%; /* Full width */
  height: 100%; /* Full height */
  padding: 20px;
  transition: margin-left 0.3s, width 0.3s; /* Smooth transition */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.content.collapsed {
  margin-left: 0; /* Remove left margin when sidebar is collapsed */
  width: 100%; /* Full width when sidebar is collapsed */
}

body {
  font-family: 'Roboto', sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f4f4f4;
  color: #333;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

main {
  padding: 2rem;
  display: flex;
  justify-content: flex-start; /* Align content to the left */
  flex-wrap: wrap;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.content {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 2400px; /* Doubled max-width to fit screen size */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.blog-list {
  width: 100%;
  padding-right: 2rem;
  margin-bottom: 2rem;
  transition: transform 0.3s ease;
}

.blog-list.collapsed {
  transform: translateX(-100%);
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

.blog-content {
  width: 100%;
  height: 100%;
  margin: 0;  /* align the blog content to the left */
  display: flex;
  justify-content: flex-start; /* Align content to the left */
  flex-direction: column;
  align-items: flex-start; /* Align its child components to the left */
  padding: 0 20px;
  box-sizing: border-box;
  overflow-y: auto; /* Enable scrolling for content if needed */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.blog-content.fullWidth {
  width: 100%;
  margin-left: 0 !important;
}

/* Updated article style for aligning to the left */
article {
  max-width: 1080px; /* Increase max-width for wider content */
  width: 100%;
  margin: 3rem 0;  /* Center horizontally */
  padding: 2rem;
  font-size: 18px;
  line-height: 1.8;
  font-family: 'Georgia', serif;
  color: #1a1a1a;
  background-color: transparent;
  border-radius: 0;
  box-shadow: none;
  border: none;
  text-align: left; /* explicitly align text to the left */
  white-space: normal;            /* ensure normal wrapping */
  overflow-wrap: break-word;      /* break long words */
  box-sizing: border-box;
  overflow-y: visible; /* Remove scrollbar */
  overflow-x: hidden; /* Remove horizontal scrollbar */
}

/* Updated markdown-content styling for aligning to the left */
.markdown-content {
  width: 100%;
  max-width: 1080px; /* Increase max-width for wider content */
  margin: 0;  /* align the markdown-content block to the left */
  text-align: left;
  background-color: transparent;
  white-space: normal;
  overflow-wrap: break-word;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

footer {
  background-color: #fff;
  color: #333;
  text-align: center;
  padding: 1rem 0;
  position: fixed;
  bottom: 0;
  box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.1);
  /* Updated to span full width */
  width: 100%;
  margin: 0;
  transition: none;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

@media (min-width: 768px) {
  .content {
    flex-direction: row;
  }

  .blog-list {
    width: 30%;
    margin-bottom: 0;
    transform: translateX(0);
  }

  .blog-content {
    width: 100%;
  }
}

@media (max-width: 767px) {
  .blog-list {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background-color: #fff;
    z-index: 1000;
  }

  .blog-content {
    margin-top: 5rem;
  }
}

/* Sidebar styles added */
.sidebar {
  width: 250px;
  transition: width 0.3s;
}

.sidebar.collapsed {
  width: 60px;
}

.header,
.main-content,
.content,
article {
  text-align: left;
  margin: 0;
}

/* Default layout when sidebar is open */
.content {
  margin-left: 0; /* Remove left margin */
  transition: margin-left 0.3s, width 0.3s;
}

footer {
  width: 100%; /* Full width */
  transition: width 0.3s;
}

/* Responsive adjustments when sidebar is collapsed or on smaller screens */
@media screen and (max-width: 768px) {
  .header, footer {
    width: 100%;
    margin: 0;
  }

  .content {
    margin-left: 0;
    width: 100%;
    padding: 10px;
  }

  .main-content {
    height: calc(100vh - 60px); /* Account for mobile browser chrome */
  }

  .blog-content {
    padding: 0 10px;
  }

  article {
    padding: 1rem;
  }

  .sidebar.collapsed + .main-content .content {
    margin-left: 0; /* Remove left margin when sidebar is collapsed */
    width: 100%; /* Full width when sidebar is collapsed */
  }
}

/* Optionally, force centering when sidebar collapse state changes */
/* If the sidebar component had a global class indicating collapse,
   rules similar to below might be used:
   
.sidebar.collapsed + .main-content .content {
  margin-left: 0;
}
*/
</style>
