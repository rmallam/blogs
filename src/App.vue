/* eslint-disable */
<template>
  <div id="app">
    <header class="header">
      <nav>
        <ul>
          <li class="brand" @click="selectPage('about')">R's Blog</li>
          <li @click="selectPage('about')">Home</li>
          <li @click="selectPage('blogs')">Blogs</li>
          <li @click="selectPage('contact')">Contact Me</li>
        </ul>
      </nav>
    </header>
    <div class="main-content">
      <div class="content">
        <main>
          <div class="content">
            <div class="page-content" :class="{ fullWidth: selectedPage === 'about' || selectedPage==='contact' }">
              <!-- About Me page now rendered via AboutMe component -->
              <AboutMe 
                v-if="selectedPage === 'about'" 
                @navigate="selectPage"
              />
              <ContactMe v-else-if="selectedPage === 'contact'" />
              <div v-else-if="selectedPage === 'blogs'" class="blogs-page">
                <BlogSections :sections="blogSections" @select-section="selectSection" />
                <div class="blog-list" v-if="selectedSection">
                  <h2>{{ selectedSection.name }}</h2>
                  <ul>
                    <li v-for="(blog, index) in selectedSection.blogs" :key="index" @click="selectPage(blog)">
                      <h3>{{ formatTitle(blog.title) }}</h3>
                      <p>{{ blog.summary }}</p>
                      <div class="tags">
                        <span v-for="(tag, index) in blog.tags" :key="index" class="tag">{{ tag }}</span>
                      </div>
                    </li>
                  </ul>
                </div>
              </div>
              <div v-else-if="selectedPage !== null && selectedPage !== 'blogs'">
                <article>
                  <h2>{{ formatTitle(selectedPage.title) }}</h2>
                  <div class="markdown-content" v-html="renderMarkdownWithIds(selectedPage.content)"></div>
                  <BlogComments 
                    v-if="selectedPage.title"
                    :blogId="selectedPage.title"
                  />
                </article>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  </div>
</template>

<script>
import AboutMe from './components/AboutMe.vue';
import ContactMe from './components/ContactMe.vue';
import BlogSections from './components/BlogSections.vue';
import BlogComments from './components/BlogComments.vue';
import markdown from 'markdown-it'
const md = markdown()

export default {
  name: 'App',
  components: { 
    AboutMe, 
    ContactMe, 
    BlogSections,
    BlogComments
  },
  data() {
    const sections = this.getBlogSections();
    console.log('Initial blog sections:', sections); // Debug log
    return {
      blogSections: sections,
      selectedPage: 'about',
      isSidebarCollapsed: false,
      headings: [],
      selectedSection: null,
      copyCodeHandler: null // Add this to store the event handler
    };
  },
  methods: {
    getBlogSections() {
      const sections = {};
      try {
        const blogFiles = require.context('./blogs', true, /\.md$/);
        console.log('Available blog files:', blogFiles.keys()); // Debug log
        
        blogFiles.keys().forEach(file => {
          console.log('Processing file:', file); // Debug log
          const cleanPath = file.replace(/^\.\//, '');
          const [section, fileName] = cleanPath.split('/');
          
          if (!section || !fileName) {
            console.log('Skipping invalid file:', file); // Debug log
            return;
          }
          
          if (!sections[section]) {
            sections[section] = {
              name: section,
              blogs: []
            };
          }
          
          const blogContent = blogFiles(file).default || blogFiles(file);
          console.log('Blog content loaded:', !!blogContent); // Debug log
          
          sections[section].blogs.push({
            title: fileName.replace('.md', ''),
            content: blogContent,
            summary: this.summarizeContent(blogContent),
            tags: this.extractTags(blogContent)
          });
        });
        
        console.log('Final processed sections:', sections); // Debug log
        return Object.values(sections);
      } catch (error) {
        console.error('Error loading blogs:', error);
        return [];
      }
    },
    selectPage(page) {
      if (typeof page === 'string') {
        this.selectedPage = page;
      } else {
        this.selectedPage = page;
        if (page && page.content) {
          this.headings = this.extractHeadings(page.content);
        }
      }
    },
    selectSection(section) {
      this.selectedSection = section;
    },
    goBackToBlogs() {
      this.selectedPage = 'blogs';
    },
    renderMarkdown(content) {
      return md.render(content)
    },
    renderMarkdownWithIds(content) {
      const renderedContent = md.render(content);
      const parser = new DOMParser();
      const doc = parser.parseFromString(renderedContent, 'text/html');
      
      // Process code blocks
      const codeBlocks = doc.querySelectorAll('pre');
      codeBlocks.forEach((pre, index) => {
        // Create wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = 'Copy';
        copyButton.setAttribute('onclick', `document.dispatchEvent(new CustomEvent('copyCode', { detail: { index: ${index} } }))`);
        
        // Add button and wrap pre
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(copyButton);
        wrapper.appendChild(pre);
      });
      
      // Process images to make them responsive
      const images = doc.querySelectorAll('img');
      images.forEach(img => {
        // Add responsive image class
        img.classList.add('responsive-image');
        
        // Set image to fit container
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        
        // Create a figure container for better image display
        const figure = document.createElement('figure');
        figure.className = 'markdown-image';
        img.parentNode.insertBefore(figure, img);
        figure.appendChild(img);
        
        // If image has alt text, add it as a caption
        if (img.alt && !img.alt.startsWith('http')) {
          const caption = document.createElement('figcaption');
          caption.textContent = img.alt;
          figure.appendChild(caption);
        }
      });

      return doc.body.innerHTML;
    },
    copyCode(code) {
      navigator.clipboard.writeText(code).then(() => {
        // Optional: Show feedback that code was copied
        const button = event.target;
        button.innerHTML = 'Copied!';
        setTimeout(() => {
          button.innerHTML = 'Copy';
        }, 2000);
      });
    },
    extractHeadings(content) {
      const renderedContent = md.render(content);
      const parser = new DOMParser();
      const doc = parser.parseFromString(renderedContent, 'text/html');
      const headings = doc.querySelectorAll('h1, h2, h3, h4, h5, h6');
      return Array.from(headings).map((heading, index) => ({
        id: `heading-${index}`,
        text: heading.textContent
      }));
    },
    extractTags(content) {
      const tagPattern = /#(\w+)/g;
      const tags = [];
      let match;
      while ((match = tagPattern.exec(content)) !== null) {
        tags.push(match[1]);
      }
      return tags;
    },
    summarizeContent(content) {
      // Remove markdown syntax
      const cleanContent = content.replace(/\[(.*?)\]\((.*?)\)/g, '$1')  // Remove links
                                .replace(/[#*`]/g, '')                    // Remove markdown symbols
                                .replace(/\n+/g, ' ')                    // Replace newlines with spaces
                                .trim();

      // Split into sentences
      const sentences = cleanContent.match(/[^.!?]+[.!?]+/g) || [];
      
      if (sentences.length === 0) return content.slice(0, 150) + '...';

      // Calculate sentence importance (basic)
      const importantWords = ['how', 'what', 'why', 'this', 'guide', 'tutorial', 'learn', 'understand', 'explore'];
      const sentenceScores = sentences.map(sentence => {
        const words = sentence.toLowerCase().split(' ');
        const score = words.reduce((acc, word) => {
          if (importantWords.includes(word)) acc += 2;
          if (word.length > 7) acc += 1; // Technical terms are often longer
          return acc;
        }, 0);
        return { sentence, score };
      });

      // Sort by importance and take top 2 sentences
      const topSentences = sentenceScores
        .sort((a, b) => b.score - a.score)
        .slice(0, 2)
        .map(item => item.sentence)
        .join(' ');

      // Ensure summary isn't too long
      if (topSentences.length > 200) {
        return topSentences.slice(0, 200) + '...';
      }

      return topSentences;
    },
    formatTitle(title) {
      return title.replace(/_/g, ' ');
    }
  },
  mounted() {
    // Create a bound handler function
    this.copyCodeHandler = (e) => {
      const index = e.detail.index;
      const codeBlock = document.querySelectorAll('.code-block-wrapper pre')[index];
      if (codeBlock) {
        const code = codeBlock.textContent;
        navigator.clipboard.writeText(code).then(() => {
          const button = codeBlock.parentNode.querySelector('.copy-button');
          button.textContent = 'Copied!';
          setTimeout(() => {
            button.textContent = 'Copy';
          }, 2000);
        });
      }
    };

    // Add the event listener using the stored handler
    document.addEventListener('copyCode', this.copyCodeHandler);
  },
  beforeUnmount() {
    // Remove the event listener with the same handler reference
    if (this.copyCodeHandler) {
      document.removeEventListener('copyCode', this.copyCodeHandler);
    }
  }
}
</script>

<style>
#app {
  display: flex;
  flex-direction: column;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.header {
  background-color: #333;
  color: white;
  padding: 1rem 0;
  text-align: center;
  width: 100%;
  margin: 0;
  transition: none;
}

.header nav ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
  display: flex;
  justify-content: center;
}

.header nav ul li {
  margin: 0 1rem;
  cursor: pointer;
  font-size: 1.2rem;
}

.header nav ul li:hover {
  text-decoration: underline;
}

.header nav ul .brand {
  font-weight: bold;
  font-size: 1.5rem;
  margin-right: auto; /* Align to the left */
}

.main-content {
  display: flex;
  flex-direction: column;
  width: 100%;
  overflow-y: auto; /* Enable scrolling if needed */
  text-align: center; /* Center align main content */
  transition: margin-left 0.3s, width 0.3s; /* Smooth transition */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.content {
  margin-left: 0; /* Remove left margin */
  margin-right: 0;
  width: 100%; /* Full width */
  height: 100%; /* Full height */
  transition: margin-left 0.3s, width 0.3s; /* Smooth transition */
  overflow-x: hidden; /* Remove horizontal scrolling */
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
  justify-content: center; /* Center content horizontally */
  flex-wrap: wrap;
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.content {
  display: flex;
  flex-direction: column;
  justify-content: center;
  width: 100%;
  max-width: 2400px; /* Doubled max-width to fit screen size */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.page-content {
  width: 100%;
  height: 100%;
  margin: 0 auto;  /* Center the page content */
  display: flex;
  justify-content: center; /* Center content horizontally */
  flex-direction: column;
  align-items: center; /* Center its child components */
  padding: 0 20px;
  box-sizing: border-box;
  overflow-y: auto; /* Enable scrolling for content if needed */
  overflow-x: hidden; /* Remove horizontal scrolling */
}

.page-content.fullWidth {
  width: 150%;
  margin-left: 0 !important;
}

.blogs-page {
  display: flex;
  gap: 2rem;
  padding: 2rem;
  max-width: 100%;
  margin: 0 auto;
  align-items: flex-start;
}

.blog-list {
  flex: 3;  /* Increased from 1 to 3 to give more space to blog content */
  margin-left: 2rem;
  padding: 2rem;
  background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  width: auto;  /* Remove fixed width */
  min-width: 0; /* Allow content to shrink if needed */
  word-wrap: break-word;
  overflow-wrap: break-word;
}

.blog-list h2 {
  font-size: 2rem;
  margin-bottom: 1.5rem;
  color: #2c3e50;
}

.blog-list ul {
  list-style-type: none;
  padding: 0;
}

.blog-list li {
  cursor: pointer;
  padding: 1.5rem;
  background: white;
  margin-bottom: 1rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  max-width: 100%; /* Ensure content doesn't overflow */
}

.blog-list li:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
}

.blog-list h3 {
  color: #2c3e50;
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.blog-list p {
  color: #666;
  line-height: 1.6;
  white-space: normal; /* Allow text to wrap */
  overflow-wrap: break-word;
  word-wrap: break-word;
  hyphens: auto;
}

.tags {
  margin-top: 0.5rem;
}

.tag {
  display: inline-block;
  background: #3498db;
  color: white;
  padding: 0.4rem 0.8rem;
  border-radius: 20px;
  font-size: 0.85rem;
  margin-right: 0.5rem;
  transition: background 0.3s ease;
}

.tag:hover {
  background: #2980b9;
}

.back-button {
  margin-bottom: 1rem;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  color: #fff;
  background-color: #333;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.back-button:hover {
  background-color: #555;
}

/* Updated article style for aligning to the center */
article {
  max-width: 1090px; /* Increase max-width for wider content */
  width: 100%;
  margin: 3rem auto;  /* Center horizontally */
  padding: 1rem;
  font-size: 18px;
  line-height: 1.8;
  font-family: 'Georgia', serif;
  color: #1a1a1a;
  background-color: transparent;
  border-radius: 0;
  box-shadow: none;
  border: none;
  text-align: center; /* Center align text */
  white-space: normal;            /* ensure normal wrapping */
  overflow-wrap: break-word;      /* break long words */
  box-sizing: border-box;
  overflow-y: visible; /* Remove scrollbar */
  overflow-x: hidden; /* Remove horizontal scrollbar */
}

/* Updated markdown-content styling for aligning to the center */
.markdown-content {
  width: 90%;
  max-width: 1200px; /* Increase max-width for wider content */
  margin: 0 auto;  /* Center the markdown-content block */
  text-align: center;
  background-color: transparent;
  white-space: normal;
  overflow-wrap: break-word;
  overflow-x: hidden; /* Remove horizontal scrolling */
  padding-left: 1rem;
  padding-right: 1rem;
  background: white;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  text-align: left;
  margin-bottom: 2rem;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3 {
  color: #2c3e50;
  margin: 1.5rem 0 1rem;
}

.markdown-content p {
  color: #666;
  line-height: 1.8;
  margin-bottom: 1rem;
}

.markdown-content code {
  font-family: 'Fira Code', 'Consolas', monospace;
  background: #f3f4f6;
  color: #e83e8c;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
  font-size: 0.9em;
}

.markdown-content pre {
  position: relative;
  background: #1e1e1e; /* VS Code dark theme color */
  color: #d4d4d4; /* Light gray text */
  padding: 2.5rem 1.5rem 1.5rem;
  border-radius: 8px;
  overflow-x: auto;
  margin: 1.5rem 0;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.markdown-content pre code {
  font-family: 'Fira Code', 'Consolas', monospace;
  font-size: 0.95rem;
  line-height: 1.6;
  display: block;
  background: transparent;
  color: inherit;
  padding: 0;
  border-radius: 0;
}

.code-block-wrapper {
  position: relative;
  margin: 1.5rem 0;
}

.copy-button {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  padding: 0.4rem 0.8rem;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: #d4d4d4;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 1;
  font-family: 'Roboto', sans-serif;
}

.copy-button:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-1px);
}

.copy-button:active {
  transform: translateY(0);
}

/* Add a subtle indication of the command prompt */
.markdown-content pre code::before {
  content: "$ ";
  color: #6a9955; /* Light green color */
  user-select: none;
}

/* Style shell output differently */
.markdown-content pre code:not(:first-line) {
  color: #9cdcfe; /* Light blue color for output */
}

/* Highlight specific commands */
.markdown-content pre code strong {
  color: #569cd6; /* Bright blue for emphasis */
  font-weight: normal;
}

/* Enhanced styling for markdown images */
.markdown-content img {
  max-width: 100%;
  height: auto;
  display: block;
  border-radius: 8px;
  margin: 1.5rem auto;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
}

.markdown-content .responsive-image:hover {
  transform: scale(1.01);
}

.markdown-content figure.markdown-image {
  margin: 2rem 0;
  max-width: 100%;
  text-align: center;
}

.markdown-content figcaption {
  color: #666;
  font-style: italic;
  font-size: 0.9rem;
  margin-top: 0.75rem;
}

/* Disable command prompt styling for code blocks that might interfere with formatting */
.markdown-content pre code::before {
  content: none;
}

/* Responsiveness for various screen sizes */
@media screen and (max-width: 768px) {
  .markdown-content figure.markdown-image {
    margin: 1.5rem 0;
  }
  
  .markdown-content img {
    margin: 1rem auto;
  }
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
}

@media (max-width: 767px) {
  .page-content {
    margin-top: 5rem;
  }
}

/* Responsive Design */
@media screen and (max-width: 1024px) {
  .page-content.fullWidth {
    width: 100%;
  }

  .blogs-page {
    flex-direction: column;
  }

  .blog-list {
    margin-left: 0;
  }
}

@media screen and (max-width: 768px) {
  .header nav ul {
    padding: 0 1rem;
  }

  .header nav ul li {
    font-size: 1rem;
    margin: 0 0.5rem;
  }

  .header nav ul .brand {
    font-size: 1.2rem;
  }

  main {
    padding: 1rem;
  }

  .markdown-content {
    width: 95%;
    padding: 1rem;
  }
}

@media screen and (max-width: 480px) {
  .header nav ul {
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: 0.5rem;
  }

  .header nav ul .brand {
    width: 100%;
    margin-bottom: 0.5rem;
  }

  .blog-list li {
    padding: 1rem;
  }

  .blog-list h3 {
    font-size: 1.2rem;
  }

  .tag {
    font-size: 0.75rem;
    padding: 0.3rem 0.6rem;
  }

  .copy-button {
    font-size: 0.75rem;
    padding: 0.3rem 0.6rem;
  }
}

/* Fix for mobile viewport height issues */
@supports (-webkit-touch-callout: none) {
  .main-content {
    min-height: -webkit-fill-available;
  }
}

/* Improve touch targets on mobile */
@media (hover: none) and (pointer: coarse) {
  .header nav ul li,
  .blog-list li,
  .action-button,
  .copy-button {
    min-height: 44px;
    min-width: 44px;
  }
}
</style>

