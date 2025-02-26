require('dotenv').config();
const { OpenAI } = require('openai');

const openai = new OpenAI({
  apiKey: process.env.VUE_APP_OPENAI_API_KEY || process.env.OPENAI_API_KEY
});

async function generateBlogPost(topic) {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo", // Changed from gpt-4 to gpt-3.5-turbo
      messages: [
        {
          role: "system",
          content: "You are a technical blog writer specializing in cloud native technologies, Kubernetes, and modern software architecture."
        },
        {
          role: "user",
          content: `Write a detailed technical blog post about ${topic}. Include code examples where relevant. Format in markdown.`
        }
      ],
      temperature: 0.7,
      max_tokens: 2500
    });

    const blogContent = completion.choices[0].message.content;
    const title = topic.toLowerCase().replace(/\s+/g, '_');
    const date = new Date().toISOString().split('T')[0];
    
    return {
      title: `${title}_${date}`,
      content: blogContent,
      tags: ['ai_generated', topic.toLowerCase()]
    };
  } catch (error) {
    console.error('Error generating blog post:', error);
    throw error;
  }
}

module.exports = {
  generateBlogPost
};
