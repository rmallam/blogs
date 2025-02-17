<template>
  <div id="app">
    <header>
      <h1>Welcome to Rakesh's Blog</h1>
      <nav>
        <ul>
          <li @click="selectPage('about')">About Me</li>
          <li @click="selectPage('blogs')">Blogs</li>
        </ul>
      </nav>
    </header>
    <main>
      <div class="content">
        <div class="blog-list" v-if="selectedPage !== 'about'">
          <h3>Blog Titles</h3>
          <ul>
            <li v-for="(blog, index) in blogs" :key="index" @click="selectPage(index)">
              {{ blog.title }}
            </li>
          </ul>
        </div>
        <div class="blog-content" :class="{ fullWidth: selectedPage === 'about' }">
          <article v-if="selectedPage === 'about'">
            <h2>About Me</h2>
            <p>Hi, I'm Rakesh Kumar Mallam, a Senior Architect at Red Hat. As a passionate technologist, I love exploring the ever-changing world of technology. Outside of my professional life, I enjoy cooking and playing tennis. I'm also a proud dad of two wonderful boys. Welcome to my blog where I share my insights and experiences in AI and ML.</p>
          </article>
          <article v-else-if="selectedPage !== null && selectedPage !== 'blogs'">
            <h2>{{ blogs[selectedPage].title }}</h2>
            <div v-html="renderMarkdown(blogs[selectedPage].content)"></div>
          </article>
        </div>
      </div>
    </main>
    <footer>
      <p>&copy; 2023 Rakesh's Blog</p>
    </footer>
  </div>
</template>

<script>
import markdown from 'markdown-it'
const md = markdown()

const requireBlog = require.context('./blogs', false, /\.md$/);

export default {
  name: 'App',
  data() {
    return {
      blogs: requireBlog.keys().map(file => ({
        title: file.replace('./', '').replace('.md', ''),
        content: requireBlog(file).default
      })),
      selectedPage: 'about'
    };
  },
  methods: {
    selectPage(page) {
      this.selectedPage = page;
    },
    renderMarkdown(content) {
      return md.render(content)
    }
  },
  mounted() {
    console.log('Blog website loaded');
  }
}
</script>

<style>
body {
  font-family: 'Roboto', sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f4f4f4;
  color: #333;
}

header {
  background-color: #fff;
  color: #333;
  padding: 1rem 0;
  text-align: center;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

header h1 {
  margin: 0;
  font-size: 2rem;
}

nav {
  margin-top: 1rem;
}

nav ul {
  list-style-type: none;
  padding: 0;
  display: flex;
  justify-content: center;
}

nav ul li {
  margin: 0 1rem;
  cursor: pointer;
  font-size: 1.1rem;
  position: relative;
}

main {
  padding: 2rem;
  display: flex;
  justify-content: center;
}

.content {
  display: flex;
  width: 80%;
  max-width: 1200px;
}

.blog-list {
  width: 30%;
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

.blog-content {
  width: 70%;
}

.blog-content.fullWidth {
  width: 100%;
}

article {
  background-color: #fff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  line-height: 1.6;
}

article h2 {
  font-size: 2rem;
  margin-bottom: 1rem;
}

article p {
  font-size: 1.1rem;
  margin-bottom: 1rem;
}

footer {
  background-color: #fff;
  color: #333;
  text-align: center;
  padding: 1rem 0;
  position: fixed;
  width: 100%;
  bottom: 0;
  box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.1);
}
</style>
