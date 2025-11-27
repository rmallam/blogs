/* eslint-disable */
<template>
  <div id="app">
    <header class="header">
      <nav>
        <ul>
          <li class="brand" @click="navigateTo('about')">R's Blog</li>
          <li><a href="/" style="color: inherit; text-decoration: none;">← Portfolio</a></li>
          <li @click="navigateTo('about')">About</li>
          <li @click="navigateTo('blogs')">Blogs</li>
          <li @click="navigateTo('contact')">Contact</li>
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
                @navigate="navigateTo"
              />
              <ContactMe v-else-if="selectedPage === 'contact'" />
              <div v-else-if="selectedPage === 'blogs'" class="blogs-page">
                <BlogSections :sections="blogSections" @select-section="selectSection" />
                <div class="blog-list" v-if="selectedSection">
                  <h2>{{ selectedSection.name }}</h2>
                  <ul>
                    <li v-for="(blog, index) in selectedSection.blogs" :key="index" @click="navigateToBlog(selectedSection.name, blog.title)">
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
                  <h2>{{ formatTitle(currentBlog.title) }}</h2>
                  <div class="share-link">
                    <button @click="copyShareLink" class="share-button">
                      <span>{{ shareButtonText }}</span>
                    </button>
                  </div>
                  <div class="markdown-content" v-html="renderMarkdownWithIds(currentBlog.content)"></div>
                  <BlogComments 
                    v-if="currentBlog.title"
                    :blogId="currentBlog.title"
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
      copyCodeHandler: null, // Add this to store the event handler
      currentBlog: {},
      shareButtonText: 'Copy Share Link'
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
        this.currentBlog = page;
        if (page && page.content) {
          this.headings = this.extractHeadings(page.content);
        }
      }
    },
    selectSection(section) {
      this.selectedSection = section;
      // Update URL when selecting a section
      history.pushState(
        { page: 'blogs', section: section.name }, 
        `Blogs: ${section.name}`, 
        `#/blogs/${encodeURIComponent(section.name)}`
      );
    },
    navigateTo(page) {
      if (page === 'about') {
        history.pushState({ page: 'about' }, 'About', '#/about');
      } else if (page === 'blogs') {
        history.pushState({ page: 'blogs' }, 'Blogs', '#/blogs');
      } else if (page === 'contact') {
        history.pushState({ page: 'contact' }, 'Contact', '#/contact');
      }
      this.selectPage(page);
    },
    navigateToBlog(section, blogTitle) {
      const blog = this.findBlogByTitleAndSection(section, blogTitle);
      if (blog) {
        this.currentBlog = blog;
        this.selectPage(blog);
        
        // Create a URL-friendly version of the blog title
        const urlFriendlyTitle = encodeURIComponent(blogTitle.replace(/\s+/g, '-').toLowerCase());
        
        // Update the URL with the blog info
        history.pushState(
          { page: 'blog', section, title: blogTitle }, 
          blogTitle, 
          `#/blogs/${encodeURIComponent(section)}/${urlFriendlyTitle}`
        );
      }
    },
    findBlogByTitleAndSection(sectionName, blogTitle) {
      const section = this.blogSections.find(s => s.name === sectionName);
      if (section) {
        return section.blogs.find(blog => blog.title === blogTitle);
      }
      return null;
    },
    copyShareLink() {
      const url = window.location.href;
      navigator.clipboard.writeText(url).then(() => {
        this.shareButtonText = 'Link Copied!';
        setTimeout(() => {
          this.shareButtonText = 'Copy Share Link';
        }, 2000);
      });
    },
    goBackToBlogs() {
      this.navigateTo('blogs');
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
    },
    handleInitialNavigation() {
      // Parse the current URL hash to determine what to show
      const hash = window.location.hash;
      
      if (!hash || hash === '#/' || hash === '#/about') {
        this.navigateTo('about');
      } else if (hash === '#/blogs') {
        this.navigateTo('blogs');
      } else if (hash === '#/contact') {
        this.navigateTo('contact');
      } else if (hash.startsWith('#/blogs/')) {
        // Handle blog section or specific blog post
        const parts = hash.substring(8).split('/');
        
        if (parts.length >= 1) {
          const sectionName = decodeURIComponent(parts[0]);
          const section = this.blogSections.find(s => s.name === sectionName);
          
          if (section) {
            this.selectPage('blogs');
            this.selectSection(section);
            
            // If there's a specific blog post
            if (parts.length >= 2) {
              const urlFriendlyTitle = parts[1];
              // Find the blog by converting all titles to URL-friendly format and comparing
              const blog = section.blogs.find(b => 
                b.title.replace(/\s+/g, '-').toLowerCase() === decodeURIComponent(urlFriendlyTitle).toLowerCase()
              );
              
              if (blog) {
                this.navigateToBlog(sectionName, blog.title);
              }
            }
          }
        }
      }
    },
    
    handlePopState(event) {
      if (event.state) {
        if (event.state.page === 'about' || event.state.page === 'blogs' || event.state.page === 'contact') {
          this.selectPage(event.state.page);
          
          if (event.state.page === 'blogs' && event.state.section) {
            const section = this.blogSections.find(s => s.name === event.state.section);
            if (section) {
              this.selectSection(section);
            }
          }
        } else if (event.state.page === 'blog') {
          const blog = this.findBlogByTitleAndSection(event.state.section, event.state.title);
          if (blog) {
            this.currentBlog = blog;
            this.selectPage(blog);
          }
        }
      }
    }
  },
  mounted() {
    // Handle existing event listeners
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
    document.addEventListener('copyCode', this.copyCodeHandler);
    
    // Handle URL-based navigation on page load
    this.handleInitialNavigation();
    
    // Listen for back/forward navigation
    window.addEventListener('popstate', this.handlePopState);
  },
  beforeUnmount() {
    // Remove event listeners
    if (this.copyCodeHandler) {
      document.removeEventListener('copyCode', this.copyCodeHandler);
    }
    window.removeEventListener('popstate', this.handlePopState);
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

.share-link {
  display: flex;
  justify-content: center;
  margin: 1rem 0;
}

.share-button {
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background-color 0.3s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.share-button:hover {
  background-color: #2980b9;
}

/* Add responsive CSS variables at the root level */
:root {
  --content-width: 90%;
  --content-max-width: 1200px;
  --content-padding: 1rem;
  --font-size-base: 1rem;
  --line-height-base: 1.8;
  --border-radius: 8px;
  --shadow-default: 0 10px 30px rgba(0, 0, 0, 0.1);
}

/* Updated markdown-content styling with responsive values */
.markdown-content {
  width: var(--content-width);
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: var(--content-padding);
  background: white;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-default);
  text-align: left;
  margin-bottom: 2rem;
  overflow-wrap: break-word;
  overflow-x: hidden;
}

/* Responsive code blocks */
.markdown-content pre {
  position: relative;
  background: #1e1e1e;
  color: #d4d4d4;
  padding: calc(1.5 * var(--content-padding)) var(--content-padding) var(--content-padding);
  border-radius: calc(var(--border-radius) - 2px);
  overflow-x: auto;
  margin: var(--content-padding) 0;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  width: calc(100% + 0px);
  box-sizing: border-box;
  font-size: calc(var(--font-size-base) * 0.95);
}

.markdown-content pre code {
  font-size: inherit;
  line-height: var(--line-height-base);
  white-space: pre;
  overflow-wrap: normal;
  padding: 0;
}

/* Enhanced image responsiveness */
.markdown-content img {
  max-width: 100%;
  height: auto;
  display: block;
  border-radius: var(--border-radius);
  margin: var(--content-padding) auto;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Media queries using relative sizing */
@media screen and (max-width: 1024px) {
  :root {
    --content-width: 95%;
    --content-padding: 0.9rem;
    --font-size-base: 0.95rem;
  }
}

@media screen and (max-width: 768px) {
  :root {
    --content-width: 98%;
    --content-padding: 0.8rem;
    --font-size-base: 0.95rem;
    --border-radius: 6px;
  }
  
  .markdown-content {
    text-align: left;
  }
  
  .markdown-content pre {
    margin: calc(var(--content-padding) * 0.8) calc(var(--content-padding) * -0.5);
    padding: calc(var(--content-padding) * 2) calc(var(--content-padding) * 0.75) calc(var(--content-padding) * 0.75);
    width: calc(100% + var(--content-padding));
    border-radius: calc(var(--border-radius) * 0.8);
    font-size: calc(var(--font-size-base) * 0.9);
  }
  
  /* Use viewport-relative units for spacing */
  .markdown-content figure.markdown-image {
    margin: 3vh 0;
  }
  
  .copy-button {
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.3rem 0.6rem;
    font-size: calc(var(--font-size-base) * 0.75);
  }
}

@media screen and (max-width: 480px) {
  :root {
    --content-width: 100%;
    --content-padding: 0.7rem;
    --font-size-base: 0.9rem;
    --line-height-base: 1.6;
    --border-radius: 5px;
  }
  
  .markdown-content {
    padding: var(--content-padding);
  }
  
  .markdown-content pre {
    font-size: calc(var(--font-size-base) * 0.85);
    padding-top: calc(var(--content-padding) * 1.8);
  }
  
  /* Responsive typography */
  .markdown-content h1 { font-size: calc(var(--font-size-base) * 1.7); }
  .markdown-content h2 { font-size: calc(var(--font-size-base) * 1.45); }
  .markdown-content h3 { font-size: calc(var(--font-size-base) * 1.2); }
  .markdown-content p, 
  .markdown-content li { font-size: var(--font-size-base); }
  
  /* Tables need to be handled specially */
  .markdown-content table {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
}

/* Viewport-height based adjustments */
@media screen and (max-height: 700px) {
  .markdown-content {
    margin-bottom: 1.5rem;
  }
  
  .markdown-content figure.markdown-image {
    margin: 2vh 0;
  }
}

/* Dynamic viewport-based sizing for small screens */
@media screen and (max-width: 360px) {
  :root {
    --content-padding: 0.6rem;
    --font-size-base: 0.85rem;
  }
  
  .markdown-content pre {
    font-size: calc(var(--font-size-base) * 0.8);
    padding-top: calc(var(--content-padding) * 1.3);
  }
  
  .copy-button {
    padding: 0.25rem 0.5rem;
  }
}

/* Resolution-based adjustments for high-density displays */
@media 
(-webkit-min-device-pixel-ratio: 2), 
(min-resolution: 192dpi) {
  .markdown-content {
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  }
  
  .markdown-content img {
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.08);
  }
}

/* Orientation adjustments */
@media screen and (orientation: landscape) and (max-height: 500px) {
  .markdown-content {
    margin-top: 1rem;
    margin-bottom: 1rem;
  }
  
  .markdown-content pre {
    max-height: 60vh;
  }
}
</style>

