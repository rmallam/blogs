require('dotenv').config();
const fs = require('fs').promises;
const path = require('path');
const LocalLLM = require('../src/services/LocalLLMService');

// Log environment variable (for debugging)
console.log('API Key available:', !!process.env.VUE_APP_OPENAI_API_KEY);

const topics = [
  'Kubernetes Operators',
  'Service Mesh Architecture',
  'GitOps Best Practices',
  'openshift',
  'Aritifical intelligence',
  'Machine Learning',
  'Deep Learning',    
  'Natural Language Processing',
  'Computer Vision',
  'Reinforcement Learning',
  'Generative Adversarial Networks',
  'Transformer Networks',
  'TESLA',
  'Cloud Native Security',
  'Container Orchestration',
  'Serverless Architecture'
];

async function generateDailyBlog() {
  try {
    // Pick a random topic
    const topic = topics[Math.floor(Math.random() * topics.length)];
    console.log('Generating blog for topic:', topic);
    
    console.log('Attempting to generate blog...');
    // Generate blog content
    const blog = await LocalLLM.generateBlogPost(topic);
    
    // Create blog file path
    const blogPath = path.join(
      __dirname,
      '../src/blogs/AI_Generated',
      `${blog.title}.md`
    );
    
    // Ensure directory exists
    await fs.mkdir(path.dirname(blogPath), { recursive: true });
    
    // Save blog content
    await fs.writeFile(blogPath, blog.content);
    
    console.log(`Successfully generated blog: ${blog.title}`);
  } catch (error) {
    console.error('Failed to generate blog:', error.message);
    process.exit(1);
  }
}

generateDailyBlog();
